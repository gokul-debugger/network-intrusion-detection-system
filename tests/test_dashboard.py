from datetime import datetime, timezone
from io import StringIO
import ipaddress
import json
from pathlib import Path
import shutil

import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import FEATURE_COLUMNS
from src.dashboard_bundle import DEFAULT_BUNDLE, load_bundle
from src import dashboard_inputs as inputs


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def release():
    return load_bundle()


def test_bundled_models_have_stable_reference_predictions(release):
    result, unknown = release.predict(release.examples.iloc[[0, 12, 22, 31, 44]][FEATURE_COLUMNS])
    np.testing.assert_allclose(result.attack_probability,
        [.000457649183416, .999995193924616, .002138688032633, .000438217406732, .66073843485836], atol=1e-12, rtol=0)
    assert result.binary_prediction.tolist() == ["Normal", "Attack", "Normal", "Normal", "Attack"]
    assert result.attack_category.tolist() == ["normal", "dos", "normal", "normal", "normal"]
    assert unknown == []


def test_user_threshold_does_not_modify_the_release_policy(release):
    original = release.manifest["binary_threshold"]
    frame = release.examples.iloc[[0]][FEATURE_COLUMNS]
    a, _ = release.predict(frame, threshold=0.)
    b, _ = release.predict(frame, threshold=1.)
    assert a.binary_prediction.item() == "Attack" and b.binary_prediction.item() == "Normal"
    assert release.manifest["binary_threshold"] == original


@pytest.mark.parametrize("threshold", [True, "bad", np.nan, np.inf, -.1, 1.1])
def test_invalid_session_thresholds_are_rejected(release, threshold):
    with pytest.raises(ValueError, match="threshold"):
        release.predict(release.examples.iloc[:1][FEATURE_COLUMNS], threshold)


def test_unseen_service_is_disclosed_without_refitting(release):
    frame = release.examples.iloc[:1][FEATURE_COLUMNS].copy()
    frame["service"] = "unseen_demo_service"
    result, unknown = release.predict(frame)
    assert unknown == ["service"] and len(result) == 1


@pytest.mark.parametrize("column,value", [("duration", -1), ("src_bytes", np.inf), ("src_bytes", 1e13),
    ("duration", .5), ("serror_rate", 1.1), ("logged_in", 2), ("su_attempted", 3),
    ("protocol_type", "http"), ("service", "=HYPERLINK(test)"), ("flag", ""), ("service", None)])
def test_invalid_connection_values_fail(release, column, value):
    frame = release.examples.iloc[:1][FEATURE_COLUMNS].copy()
    frame[column] = frame[column].astype(object)
    frame.loc[frame.index[0], column] = value
    with pytest.raises(ValueError): inputs.validate_connections(frame)


@pytest.mark.parametrize("change", ["label", "missing", "duplicate", "empty", "too_many"])
def test_connection_schema_and_row_limits(release, change):
    frame = release.examples.iloc[:1][FEATURE_COLUMNS].copy()
    if change == "label": frame["attack_type"] = "normal"
    elif change == "missing": frame = frame.drop(columns="flag")
    elif change == "duplicate": frame.columns = [*frame.columns[:-1], frame.columns[0]]
    elif change == "empty": frame = frame.iloc[:0]
    else: frame = pd.concat([frame] * 501, ignore_index=True)
    with pytest.raises(ValueError): inputs.validate_connections(frame)


def test_csv_accepts_reordered_features_but_not_duplicate_headers(release):
    frame = release.examples.iloc[:3][FEATURE_COLUMNS].copy()
    actual = inputs.parse_connection_csv(frame[FEATURE_COLUMNS[::-1]].to_csv(index=False).encode())
    pd.testing.assert_frame_equal(actual, inputs.validate_connections(frame))
    csv = frame.to_csv(index=False).replace("duration,", "flag,", 1)
    with pytest.raises(ValueError, match="Duplicate"):
        inputs.parse_connection_csv(csv.encode())


@pytest.mark.parametrize("content", [b"", b"\xff", b"a\x00b", b"x" * (inputs.MAX_UPLOAD_BYTES + 1)])
def test_binary_empty_and_oversize_uploads_are_rejected(content):
    with pytest.raises(ValueError): inputs.decode_upload(content)


def test_csv_row_limit_is_enforced_during_parsing(release):
    text = release.examples.iloc[:1][FEATURE_COLUMNS].to_csv(index=False)
    header, row = text.splitlines()
    with pytest.raises(ValueError, match="500"):
        inputs.parse_connection_csv((header + "\n" + (row + "\n") * 1000).encode())


def test_demo_traffic_uses_only_reserved_ips_and_is_reproducible():
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    first = inputs.demo_log(42, now)
    assert first == inputs.demo_log(42, now)
    frame, audit = inputs.parse_log_upload(first)
    assert audit == {"accepted": 180, "rejected": 0, "total": 180}
    networks = [ipaddress.ip_network("192.0.2.0/24"), ipaddress.ip_network("198.51.100.0/24")]
    assert all(any(ipaddress.ip_address(ip) in network for network in networks) for ip in frame.ip)
    assert frame.is_suspicious_path.any() and frame.risk_score.max() >= 50
    assert str(frame.timestamp.dtype) == "datetime64[us, UTC]" or str(frame.timestamp.dtype) == "datetime64[ns, UTC]"


def test_common_combined_invalid_and_mixed_timezone_log_rows():
    common = '2001:db8::1 - - [01/Jan/2026:06:00:00 +0530] "GET / HTTP/1.1" 200 -'
    combined = '192.0.2.1 - - [01/Jan/2026:00:30:00 +0000] "POST /login HTTP/1.1" 403 10 "-" "curl/8"'
    frame, audit = inputs.parse_log_upload((common + "\n" + combined + "\nnot a valid row\n").encode())
    assert audit == {"accepted": 2, "rejected": 1, "total": 3}
    assert frame.timestamp.nunique() == 1
    assert frame.user_agent.isna().sum() == 0
    assert inputs.mask_ip("2001:db8::1") == "2001:db8::/48"
    assert inputs.mask_ip("192.0.2.4") == "192.0.2.0/24"


def test_invalid_log_values_do_not_crash_a_valid_upload():
    good = inputs.demo_log(now=datetime(2026, 11, 22, tzinfo=timezone.utc)).decode().splitlines()[0]
    bad = [good.replace("198.51.100.23", "999.999.999.999"), good.replace("/Nov/", "/Bad/"), "x" * 9000]
    frame, audit = inputs.parse_log_upload((good + "\n" + "\n".join(bad)).encode())
    assert len(frame) >= 1 and audit["rejected"] >= 2


def test_log_line_cap_and_no_parseable_data():
    with pytest.raises(ValueError, match="5,000"):
        inputs.parse_log_upload(b"x\n" * 5001)
    with pytest.raises(ValueError, match="No valid"):
        inputs.parse_log_upload(b"not a log\n")


def test_csv_export_neutralizes_formula_cells_without_changing_source():
    frame = pd.DataFrame({"cell": ["=1+1", "  @cmd", "+SUM(1)", "-2", "\tunsafe", "/safe", "Mozilla"], "number": [1] * 7})
    before = frame.copy(deep=True)
    restored = pd.read_csv(StringIO(inputs.safe_csv(frame).decode()))
    assert restored.cell.iloc[:5].str.startswith("'").all()
    assert restored.cell.iloc[5:].tolist() == ["/safe", "Mozilla"]
    pd.testing.assert_frame_equal(frame, before)


def test_corrupt_model_is_rejected_before_joblib_load(monkeypatch, tmp_path):
    target = tmp_path / "bundle"
    shutil.copytree(DEFAULT_BUNDLE, target)
    (target / "binary.joblib").write_bytes(b"corrupt")
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("Corrupted model must not load"))
    with pytest.raises(ValueError, match="checksum"):
        load_bundle(target)


def test_loading_bundle_does_not_modify_release_files():
    from src.evaluation_v2 import digest
    before = {p.name: digest(p) for p in DEFAULT_BUNDLE.iterdir() if p.is_file()}
    release = load_bundle()
    release.predict(release.examples[FEATURE_COLUMNS])
    assert before == {p.name: digest(p) for p in DEFAULT_BUNDLE.iterdir() if p.is_file()}


def test_all_dashboard_views_render_and_session_results_are_private():
    from streamlit.testing.v1 import AppTest
    first = AppTest.from_file(str(ROOT / "app/modern_app.py"), default_timeout=30).run()
    assert not first.exception and not first.error
    assert any(metric.label == "Attack score" for metric in first.metric)
    first.session_state["private_marker"] = "only this session"
    for page in ("pages/traffic.py", "pages/evidence.py", "pages/explanations.py"):
        first.switch_page(page).run()
        assert not first.exception and not first.error
    second = AppTest.from_file(str(ROOT / "app/modern_app.py"), default_timeout=30).run()
    assert "private_marker" not in second.session_state


def test_manual_form_and_harder_benchmark_render():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app/modern_app.py"), default_timeout=30).run()
    app.segmented_control(key="connection_mode").set_value("Manual input").run()
    assert not app.exception and len(app.number_input) == 38
    next(button for button in app.button if button.label == "Run detection").click().run()
    assert not app.exception and not app.error
    app.switch_page("pages/evidence.py").run()
    next(control for control in app.segmented_control if control.label == "Benchmark").set_value("KDDTest-21").run()
    assert not app.exception and any("dependent subset" in warning.value for warning in app.warning)


def test_dynamic_shap_uses_only_session_cache():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(ROOT / "app/modern_app.py"), default_timeout=30).run()
    app.switch_page("pages/explanations.py").run()
    next(button for button in app.button if button.label == "Compute explanation").click().run(timeout=30)
    assert not app.exception and not app.error
    assert len(app.session_state["private_shap"]) == 1
    assert any(metric.label == "Reconstruction error" for metric in app.metric)
    next(control for control in app.segmented_control if control.label == "Explanation task").set_value("Attack category").run()
    next(button for button in app.button if button.label == "Compute explanation").click().run(timeout=30)
    assert not app.exception and not app.error
    assert len(app.session_state["private_shap"]) == 2
