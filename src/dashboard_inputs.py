"""Bounded in-memory inputs for the public dashboard; never server file paths."""

import csv
from datetime import datetime, timedelta, timezone
from io import StringIO
import ipaddress
import random
import re

import numpy as np
import pandas as pd

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from src.web_traffic import add_traffic_features


MAX_UPLOAD_BYTES = 2 * 1024 * 1024
MAX_CONNECTIONS = 500
MAX_LOG_LINES = 5000
BINARY_FIELDS = {"land", "logged_in", "root_shell", "is_host_login", "is_guest_login"}
PATTERN = re.compile(
    r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] '
    r'"(?P<method>[A-Z]{1,16}) (?P<path>[^"\s]{1,4096}) (?P<protocol>HTTP/[0-9.]+)" '
    r'(?P<status>\d{3}) (?P<size>\d+|-)(?: "(?P<referer>[^"\r\n]*)" "(?P<user_agent>[^"\r\n]*)")?'
)


def decode_upload(content: bytes) -> str:
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise ValueError("Upload must contain data and be no larger than 2 MiB.")
    try:
        text = content.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as error:
        raise ValueError("Upload must be valid UTF-8 text.") from error
    if "\x00" in text:
        raise ValueError("NUL bytes are not accepted in text uploads.")
    return text


def validate_connections(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.columns.duplicated().any() or set(frame.columns) != set(FEATURE_COLUMNS):
        raise ValueError("Connection data must contain exactly the 41 NSL-KDD feature columns, without labels or duplicates.")
    if not 1 <= len(frame) <= MAX_CONNECTIONS:
        raise ValueError("Connection inputs must contain between 1 and 500 rows.")
    result = frame[FEATURE_COLUMNS].copy().reset_index(drop=True)
    if result.isna().any().any():
        raise ValueError("Every connection feature must be supplied; missing values are not imputed.")
    for name in NUMERIC_FEATURES:
        try:
            values = pd.to_numeric(result[name], errors="raise").astype(float)
        except (ValueError, TypeError) as error:
            raise ValueError(f"{name} must contain numeric values.") from error
        if not np.isfinite(values).all() or not values.between(0, 1e12).all():
            raise ValueError(f"{name} must contain finite, non-negative values no larger than 1e12.")
        if name.endswith("_rate"):
            if values.gt(1).any(): raise ValueError(f"{name} must be between zero and one.")
        elif values.mod(1).ne(0).any():
            raise ValueError(f"{name} must contain integer counts or indicators.")
        if name in BINARY_FIELDS and not values.isin([0, 1]).all():
            raise ValueError(f"{name} must be zero or one.")
        if name == "su_attempted" and not values.isin([0, 1, 2]).all():
            raise ValueError("su_attempted must be zero, one, or two.")
        result[name] = values
    for name in CATEGORICAL_FEATURES:
        values = result[name].astype(str)
        if not values.str.fullmatch(r"[A-Za-z0-9_./-]{1,64}").all():
            raise ValueError(f"{name} must be a nonempty category token of at most 64 characters.")
        result[name] = values
    if not result.protocol_type.isin(["tcp", "udp", "icmp"]).all():
        raise ValueError("protocol_type must be tcp, udp, or icmp.")
    return result


def parse_connection_csv(content: bytes) -> pd.DataFrame:
    text = decode_upload(content)
    try:
        reader = csv.reader(StringIO(text), strict=True)
        records = []
        for row in reader:
            records.append(row)
            if len(records) > MAX_CONNECTIONS + 1:
                raise ValueError("Connection inputs must contain between 1 and 500 rows.")
    except csv.Error as error:
        raise ValueError("Malformed CSV quoting or field structure.") from error
    if len(records) < 2 or any(len(row) != len(records[0]) for row in records):
        raise ValueError("CSV must contain a header and consistently sized data rows.")
    if len(records[0]) != len(set(records[0])):
        raise ValueError("Duplicate CSV column names are not accepted.")
    return validate_connections(pd.DataFrame(records[1:], columns=records[0]))


def parse_log_upload(content: bytes) -> tuple[pd.DataFrame, dict]:
    lines = []
    for line in StringIO(decode_upload(content)):
        if line.strip(): lines.append(line.rstrip("\r\n"))
        if len(lines) > MAX_LOG_LINES:
            raise ValueError("A log upload may contain at most 5,000 nonblank lines.")
    records, rejected = [], 0
    for line in lines:
        try:
            if len(line) > 8192: raise ValueError("Long line")
            match = PATTERN.fullmatch(line)
            if match is None: raise ValueError("Unsupported format")
            row = match.groupdict()
            if "%" in row["ip"]: raise ValueError("Scoped client addresses are unsupported")
            ipaddress.ip_address(row["ip"])
            row["timestamp"] = datetime.strptime(row["timestamp"], "%d/%b/%Y:%H:%M:%S %z")
            if not 1970 <= row["timestamp"].year <= 2100: raise ValueError("Unsupported log date range")
            row["status"] = int(row["status"])
            if not 100 <= row["status"] <= 599: raise ValueError("Invalid HTTP status")
            row["size"] = 0 if row["size"] == "-" else int(row["size"])
            if row["size"] > 1e12: raise ValueError("Invalid byte count")
            row["referer"], row["user_agent"] = row["referer"] or "-", row["user_agent"] or "-"
            records.append(row)
        except (ValueError, OverflowError):
            rejected += 1
    if not records:
        raise ValueError("No valid Apache/Nginx common or combined access-log rows were found.")
    frame = pd.DataFrame(records)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame = add_traffic_features(frame.sort_values("timestamp").reset_index(drop=True))
    return frame, {"accepted": len(frame), "rejected": rejected, "total": len(lines)}


def demo_log(seed: int = 42, now: datetime | None = None) -> bytes:
    rng = random.Random(seed)
    end = (now or datetime.now(timezone.utc)).replace(second=0, microsecond=0)
    lines = []
    for index in range(180):
        risky = index % 9 == 0 or 95 <= index < 115
        stamp = end - timedelta(seconds=(179 - index) * 4)
        ip = "198.51.100.23" if risky else f"192.0.2.{rng.randint(1, 24)}"
        path = rng.choice(["/.env", "/wp-login.php", "/admin", "/config.php"]) if risky else rng.choice(["/", "/docs", "/products", "/blog"])
        status = rng.choice([401, 403, 404]) if risky else rng.choice([200, 200, 200, 301, 404])
        method, agent = ("POST", "python-requests/2.32") if risky else ("GET", "Mozilla/5.0")
        lines.append(f'{ip} - - [{stamp.strftime("%d/%b/%Y:%H:%M:%S %z")}] "{method} {path} HTTP/1.1" {status} {rng.randint(100, 9000)} "-" "{agent}"')
    return ("\n".join(lines) + "\n").encode("utf-8")


def mask_ip(value: str) -> str:
    address = ipaddress.ip_address(value)
    return str(ipaddress.ip_network(f"{address}/{24 if address.version == 4 else 48}", strict=False))


def safe_csv(frame: pd.DataFrame) -> bytes:
    result = frame.copy()
    for name in result.select_dtypes(include=["object", "string"]).columns:
        result[name] = result[name].map(lambda value: "'" + value if isinstance(value, str) and (
            value.lstrip().startswith(("=", "+", "-", "@")) or value.startswith(("\t", "\r"))) else value)
    return result.to_csv(index=False, float_format="%.17g").encode("utf-8")
