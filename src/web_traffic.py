from __future__ import annotations

import random
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


LOG_PATTERN = re.compile(
    r'(?P<ip>\S+) \S+ \S+ \[(?P<timestamp>[^\]]+)\] '
    r'"(?P<method>\S+) (?P<path>\S+) (?P<protocol>[^"]+)" '
    r'(?P<status>\d{3}) (?P<size>\S+) "(?P<referer>[^"]*)" "(?P<user_agent>[^"]*)"'
)

SUSPICIOUS_PATH_KEYWORDS = [
    "admin",
    "login",
    "wp-admin",
    "wp-login",
    ".env",
    "phpmyadmin",
    "config",
    "shell",
    "cmd",
    "passwd",
    "etc",
]

BOT_KEYWORDS = ["bot", "crawler", "spider", "scrapy", "curl", "wget", "python-requests"]

REPEATING_ATTACKER_IPS = [
    "45.83.12.90",
    "103.21.244.8",
    "185.220.101.42",
    "198.51.100.23",
]


def parse_access_log_line(line: str) -> dict | None:
    match = LOG_PATTERN.match(line.strip())
    if not match:
        return None

    row = match.groupdict()
    row["timestamp"] = datetime.strptime(row["timestamp"], "%d/%b/%Y:%H:%M:%S %z")
    row["status"] = int(row["status"])
    row["size"] = 0 if row["size"] == "-" else int(row["size"])
    return row


def load_access_log(path: Path, max_lines: int | None = None) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if max_lines:
        lines = lines[-max_lines:]

    rows = [parse_access_log_line(line) for line in lines]
    rows = [row for row in rows if row is not None]
    if not rows:
        return pd.DataFrame(columns=["ip", "timestamp", "method", "path", "protocol", "status", "size", "referer", "user_agent"])

    df = pd.DataFrame(rows).sort_values("timestamp")
    return add_traffic_features(df)


def add_traffic_features(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    output = df.copy()
    output["minute"] = output["timestamp"].dt.floor("min")
    output["is_error"] = output["status"] >= 400
    output["is_server_error"] = output["status"] >= 500
    output["is_suspicious_path"] = output["path"].str.lower().apply(
        lambda path: any(keyword in path for keyword in SUSPICIOUS_PATH_KEYWORDS)
    )
    output["is_bot_user_agent"] = output["user_agent"].str.lower().apply(
        lambda agent: any(keyword in agent for keyword in BOT_KEYWORDS)
    )
    output["risk_score"] = output.apply(score_request, axis=1)
    output["risk_level"] = pd.cut(
        output["risk_score"],
        bins=[-1, 19, 49, 100],
        labels=["low", "medium", "high"],
    )
    return output


def score_request(row: pd.Series) -> int:
    score = 0
    if row["status"] == 401 or row["status"] == 403:
        score += 20
    elif row["status"] == 404:
        score += 12
    elif row["status"] >= 500:
        score += 10

    if row["is_suspicious_path"]:
        score += 35
    if row["is_bot_user_agent"]:
        score += 15
    if row["method"] not in {"GET", "POST", "HEAD"}:
        score += 15

    return min(score, 100)


def summarize_traffic(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "requests": 0,
            "unique_ips": 0,
            "error_rate": 0.0,
            "high_risk_requests": 0,
        }

    return {
        "requests": int(len(df)),
        "unique_ips": int(df["ip"].nunique()),
        "error_rate": float(df["is_error"].mean()),
        "high_risk_requests": int((df["risk_level"] == "high").sum()),
    }


def top_values(df: pd.DataFrame, column: str, n: int = 10) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[column, "count"])
    return df[column].value_counts().head(n).rename_axis(column).reset_index(name="count")


def requests_per_minute(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["minute", "requests"])
    return df.groupby("minute").size().rename("requests").reset_index()


def generate_log_line(timestamp: datetime, risky: bool = False) -> str:
    normal_paths = ["/", "/pricing", "/docs", "/api/products", "/blog/ml-projects", "/contact"]
    risky_paths = ["/admin", "/wp-login.php", "/.env", "/phpmyadmin", "/config.php", "/etc/passwd"]
    methods = ["GET", "POST", "HEAD"] if not risky else ["GET", "POST", "PUT"]
    agents = [
        "Mozilla/5.0",
        "Chrome/125.0 Safari/537.36",
        "curl/8.1.2",
        "python-requests/2.32",
        "Googlebot/2.1",
    ]

    if risky and random.random() < 0.7:
        ip = random.choice(REPEATING_ATTACKER_IPS)
    else:
        ip = ".".join(str(random.randint(1, 255)) for _ in range(4))
    method = random.choice(methods)
    path = random.choice(risky_paths if risky else normal_paths)
    status = random.choice([401, 403, 404]) if risky else random.choice([200, 200, 200, 301, 404])
    size = random.randint(300, 9000)
    user_agent = random.choice(agents if risky else agents[:2])
    formatted_time = timestamp.strftime("%d/%b/%Y:%H:%M:%S %z")
    return f'{ip} - - [{formatted_time}] "{method} {path} HTTP/1.1" {status} {size} "-" "{user_agent}"'


def append_simulated_traffic(path: Path, rows: int = 20, risky_rate: float = 0.18) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    now = datetime.now().astimezone()
    lines = []
    for index in range(rows):
        timestamp = now - timedelta(seconds=(rows - index) * random.randint(1, 4))
        lines.append(generate_log_line(timestamp, risky=random.random() < risky_rate))

    with path.open("a", encoding="utf-8") as file:
        for line in lines:
            file.write(line + "\n")
