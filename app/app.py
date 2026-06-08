import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import FEATURE_COLUMNS, MODELS_DIR, REPORTS_DIR
from src.web_traffic import (
    append_simulated_traffic,
    load_access_log,
    requests_per_minute,
    summarize_traffic,
    top_values,
)


@st.cache_resource
def load_models():
    binary_path = MODELS_DIR / "binary_intrusion_model.pkl"
    multiclass_path = MODELS_DIR / "multiclass_attack_model.pkl"

    if not binary_path.exists() or not multiclass_path.exists():
        return None, None

    return joblib.load(binary_path), joblib.load(multiclass_path)


@st.cache_data
def load_examples():
    examples_path = REPORTS_DIR / "example_connections.csv"
    if examples_path.exists():
        return pd.read_csv(examples_path)
    return pd.DataFrame()


def probability_table(model, row):
    if not hasattr(model, "predict_proba"):
        return pd.DataFrame()

    probabilities = model.predict_proba(row)[0]
    return pd.DataFrame(
        {
            "class": [str(label) for label in model.classes_],
            "probability": probabilities,
        }
    ).sort_values("probability", ascending=False)


st.set_page_config(page_title="Network Intrusion Detection", layout="wide")

st.title("Network Intrusion Detection System")
st.caption("NSL-KDD benchmark classifier plus live web access-log anomaly dashboard")

model_tab, traffic_tab = st.tabs(["NSL-KDD Model Demo", "Live Web Traffic Monitor"])

with model_tab:
    binary_model, multiclass_model = load_models()
    examples = load_examples()

    if binary_model is None or multiclass_model is None:
        st.warning("Saved models were not found. Run notebooks 01-05 first.")
    elif examples.empty:
        st.warning("Example connection file was not found. Run notebook 05 first.")
    else:
        selected_index = st.selectbox(
            "Example connection",
            options=list(examples.index),
            format_func=lambda idx: f"Row {idx} - actual: {examples.loc[idx, 'attack_category']}",
        )

        row = examples.loc[[selected_index], FEATURE_COLUMNS]
        actual_binary = int(examples.loc[selected_index, "binary_label"])
        actual_category = examples.loc[selected_index, "attack_category"]
        actual_attack = examples.loc[selected_index, "attack_type"]

        binary_prediction = int(binary_model.predict(row)[0])
        multiclass_prediction = multiclass_model.predict(row)[0]

        left, right = st.columns(2)

        with left:
            st.subheader("Binary Detection")
            st.metric("Prediction", "Attack" if binary_prediction == 1 else "Normal")
            st.caption(f"Actual: {'Attack' if actual_binary == 1 else 'Normal'}")
            binary_probs = probability_table(binary_model, row)
            if not binary_probs.empty:
                st.bar_chart(binary_probs, x="class", y="probability")

        with right:
            st.subheader("Attack Category")
            st.metric("Prediction", str(multiclass_prediction).upper())
            st.caption(f"Actual category: {actual_category} | Attack type: {actual_attack}")
            multiclass_probs = probability_table(multiclass_model, row)
            if not multiclass_probs.empty:
                st.bar_chart(multiclass_probs, x="class", y="probability")

        with st.expander("Connection Features"):
            st.dataframe(row.T.rename(columns={selected_index: "value"}), use_container_width=True)

with traffic_tab:
    st.subheader("Live Web Access-Log Monitor")
    st.caption("Reads Apache/Nginx-style access logs and scores suspicious requests with transparent rules.")

    log_path = st.text_input("Log file path", str(REPORTS_DIR / "live_access.log"))
    max_lines = st.slider("Read last N lines", min_value=100, max_value=5000, value=1000, step=100)

    controls = st.columns([1, 1, 4])
    with controls[0]:
        if st.button("Append demo traffic"):
            append_simulated_traffic(Path(log_path), rows=25)
            st.rerun()
    with controls[1]:
        st.button("Refresh")

    log_file = Path(log_path)
    if not log_file.exists():
        st.info("No log file found yet. Click 'Append demo traffic' or run `python scripts/simulate_web_traffic.py`.")
    else:
        traffic_df = load_access_log(log_file, max_lines=max_lines)
        summary = summarize_traffic(traffic_df)

        metric_cols = st.columns(4)
        metric_cols[0].metric("Requests", summary["requests"])
        metric_cols[1].metric("Unique IPs", summary["unique_ips"])
        metric_cols[2].metric("Error Rate", f"{summary['error_rate']:.1%}")
        metric_cols[3].metric("High Risk", summary["high_risk_requests"])

        st.divider()
        left, right = st.columns(2)

        with left:
            st.markdown("#### Requests Per Minute")
            rpm = requests_per_minute(traffic_df)
            if not rpm.empty:
                st.line_chart(rpm, x="minute", y="requests")

            st.markdown("#### Top IPs")
            st.dataframe(top_values(traffic_df, "ip"), hide_index=True, use_container_width=True)

        with right:
            st.markdown("#### Risk Levels")
            risk_counts = top_values(traffic_df, "risk_level")
            if not risk_counts.empty:
                st.bar_chart(risk_counts, x="risk_level", y="count")

            st.markdown("#### Suspicious Paths")
            suspicious = traffic_df[traffic_df["is_suspicious_path"]]
            st.dataframe(
                suspicious[["timestamp", "ip", "method", "path", "status", "risk_score"]].tail(15),
                hide_index=True,
                use_container_width=True,
            )

        with st.expander("Recent Requests"):
            st.dataframe(
                traffic_df[["timestamp", "ip", "method", "path", "status", "user_agent", "risk_score", "risk_level"]].tail(100),
                hide_index=True,
                use_container_width=True,
            )
