"""Streamlit views for the reference dashboard; visitor data stays session-local."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from src.dashboard_bundle import DEFAULT_BUNDLE, load_bundle
from src.dashboard_inputs import BINARY_FIELDS, demo_log, mask_ip, parse_connection_csv, parse_log_upload, safe_csv, validate_connections
from src.web_traffic import requests_per_minute, summarize_traffic, top_values


COLORS = {"normal": "#16836b", "dos": "#da6458", "probe": "#7061ad", "r2l": "#b47c16", "u2r": "#397ca8"}
REPO = "https://github.com/gokul-debugger/network-intrusion-detection-system"


@st.cache_resource(show_spinner=False)
def _resource(manifest_sha: str):
    return load_bundle()


def bundle():
    return _resource(hashlib.sha256((DEFAULT_BUNDLE / "manifest.json").read_bytes()).hexdigest())


def chart(figure, key: str):
    figure.update_layout(template="plotly_white", font=dict(family="sans-serif", size=14, color="#233137"),
                         margin=dict(l=12, r=12, t=24, b=16), paper_bgcolor="rgba(0,0,0,0)",
                         plot_bgcolor="rgba(0,0,0,0)", legend_title_text="")
    st.plotly_chart(figure, width="stretch", key=key, config={"displaylogo": False})


def heading(title: str, detail: str):
    st.title(title)
    st.caption(detail)


def _save_result(features: pd.DataFrame, source: str, threshold: float):
    result, unknown = bundle().predict(features, threshold)
    st.session_state["private_result"] = result
    st.session_state["private_features"] = features.reset_index(drop=True)
    st.session_state["private_result_source"] = source
    st.session_state["private_unknown"] = unknown


def _result_view():
    result = st.session_state.get("private_result")
    if result is None: return
    st.divider()
    st.subheader("Detection result")
    st.caption(st.session_state["private_result_source"])
    if st.session_state.get("private_unknown"):
        st.warning("Unseen categories were encoded as all-zero indicators: " + ", ".join(st.session_state["private_unknown"]))
    if len(result) == 1:
        row = result.iloc[0]
        cols = st.columns(3)
        cols[0].metric("Binary detection", row.binary_prediction)
        cols[1].metric("Attack score", f"{row.attack_probability:.1%}")
        cols[2].metric("Category model", row.attack_category.upper())
        st.caption(f"Decision threshold {row.decision_threshold:.6f} · model score, not calibrated real-world risk")
        if (row.binary_prediction == "Normal") != (row.attack_category == "normal"):
            st.warning("The independently trained binary and category models disagree on normal versus attack.")
        probs = pd.DataFrame({"Category": list(COLORS), "Probability": [row[f"p_{name}"] for name in COLORS]})
        fig = px.bar(probs, x="Probability", y="Category", orientation="h", color="Category", color_discrete_map=COLORS)
        fig.update_layout(height=260, showlegend=False, yaxis=dict(autorange="reversed"), xaxis=dict(range=[0, 1], tickformat=".0%"))
        chart(fig, "prediction_probabilities")
    else:
        cols = st.columns(3)
        cols[0].metric("Connections", len(result))
        cols[1].metric("Flagged as attack", int(result.binary_prediction.eq("Attack").sum()))
        cols[2].metric("Normal", int(result.binary_prediction.eq("Normal").sum()))
        st.dataframe(result, hide_index=True, width="stretch")
    st.download_button("Results CSV", safe_csv(result), "intrusion-results.csv", "text/csv", icon=":material/download:")


def connections():
    release = bundle()
    heading("Connection analysis", "NSL-KDD reference · LightGBM · training-selected decision policy")
    st.caption("Reference benchmark, not a production IDS. Newer-dataset training remains pending.")
    mode = st.segmented_control("Connection source", ["Examples", "Manual input", "CSV upload"], default="Examples", required=True, key="connection_mode")
    if st.session_state.get("private_connection_mode") != mode:
        st.session_state.pop("private_result", None)
        st.session_state["private_connection_mode"] = mode
    threshold = release.manifest["binary_threshold"]
    with st.expander("Decision policy"):
        override = st.toggle("Session threshold override", value=False)
        if override: threshold = st.slider("Attack threshold", 0.01, 0.99, round(threshold, 2), .01)
        st.caption(f"Frozen reference threshold: {release.manifest['binary_threshold']:.6f} · published benchmark metrics remain fixed")
    template_id = 0
    if mode in ("Examples", "Manual input"):
        template_id = st.selectbox("Reference connection", release.examples.index,
                                  format_func=lambda index: f"{release.examples.loc[index, 'attack_category'].upper()} · source row {int(release.examples.loc[index, 'source_row'])}")
    template = release.examples.loc[[template_id], FEATURE_COLUMNS]
    if mode == "Examples":
        _save_result(template, f"Example · actual {release.examples.loc[template_id, 'attack_category']} · category-balanced demonstration sample", threshold)
    elif mode == "Manual input":
        with st.form(f"manual_connection_{template_id}"):
            values = {}
            basic, rates, indicators = st.tabs(["Connection", "Rates", "Access indicators"])
            with basic:
                cols = st.columns(3)
                encoder = release.models["binary"].named_steps["preprocessor"].named_transformers_["categorical"]
                for index, (name, categories) in enumerate(zip(CATEGORICAL_FEATURES, encoder.categories_)):
                    choices = list(categories)
                    current = template.iloc[0][name]
                    if current not in choices: choices.append(current)
                    values[name] = cols[index].selectbox(name.replace("_", " ").capitalize(), choices, index=choices.index(current))
            primary = {"duration", "src_bytes", "dst_bytes", "count", "srv_count", "dst_host_count", "dst_host_srv_count"}
            for container, fields in ((basic, [n for n in NUMERIC_FEATURES if n in primary]),
                                       (rates, [n for n in NUMERIC_FEATURES if n.endswith("_rate")]),
                                       (indicators, [n for n in NUMERIC_FEATURES if n not in primary and not n.endswith("_rate")])):
                with container:
                    cols = st.columns(3)
                    for index, name in enumerate(fields):
                        rate = name.endswith("_rate")
                        maximum = 1 if name in BINARY_FIELDS else 2 if name == "su_attempted" else int(1e12)
                        values[name] = cols[index % 3].number_input(name.replace("_", " ").capitalize(),
                            min_value=0.0 if rate else 0, max_value=1.0 if rate else maximum,
                            value=float(template.iloc[0][name]) if rate else int(template.iloc[0][name]), step=.01 if rate else 1)
            submitted = st.form_submit_button("Run detection", icon=":material/radar:", type="primary")
        if submitted:
            try: _save_result(validate_connections(pd.DataFrame([values])), "Manual connection · no known ground truth", threshold)
            except ValueError as error:
                st.session_state.pop("private_result", None)
                st.error(str(error))
    elif mode == "CSV upload":
        example = release.examples.groupby("attack_category", sort=False).head(1)[FEATURE_COLUMNS]
        st.download_button("CSV template", safe_csv(example), "connection-template.csv", "text/csv", icon=":material/download:")
        upload = st.file_uploader("Connection CSV · 41 features · up to 500 rows / 2 MiB", type=["csv"], key=f"connection_upload_{st.session_state.get('upload_epoch', 0)}", max_upload_size=2)
        if upload is not None:
            try: _save_result(parse_connection_csv(upload.getvalue()), f"Uploaded connections · {upload.name}", threshold)
            except ValueError as error:
                st.session_state.pop("private_result", None)
                st.error(str(error))
    _result_view()
    if mode == "Examples":
        with st.expander("Connection features"):
            st.dataframe(template.T.rename(columns={template_id: "Value"}).astype(str), width="stretch")


def traffic():
    heading("Access-log monitor", "Transparent request-scoring rules · separate from the NSL-KDD classifiers")
    source = st.segmented_control("Log source", ["Synthetic traffic", "Upload log"], default="Synthetic traffic", required=True, key="log_source")
    if "private_demo_log" not in st.session_state: st.session_state["private_demo_log"] = demo_log()
    content = st.session_state["private_demo_log"]
    if source == "Synthetic traffic":
        if st.button("Refresh demo", icon=":material/refresh:"):
            st.session_state["private_demo_log"] = demo_log(seed=43 + st.session_state.get("demo_revision", 0))
            st.session_state["demo_revision"] = st.session_state.get("demo_revision", 0) + 1
            content = st.session_state["private_demo_log"]
        st.caption("Synthetic requests · documentation-only IP addresses")
    else:
        upload = st.file_uploader("Common or combined access log · up to 5,000 lines / 2 MiB", type=["log", "txt"], key=f"log_upload_{st.session_state.get('upload_epoch', 0)}", max_upload_size=2)
        if upload is None: return
        content = upload.getvalue()
        st.caption("Session-only upload · no server file-path access or saved log file")
    try: frame, audit = parse_log_upload(content)
    except ValueError as error:
        st.error(str(error))
        return
    if audit["rejected"]: st.warning(f"{audit['accepted']} accepted lines · {audit['rejected']} rejected lines")
    masked = st.toggle("Mask IP addresses", value=True)
    summary = summarize_traffic(frame)
    cols = st.columns(4)
    cols[0].metric("Requests", summary["requests"])
    cols[1].metric("Unique addresses", summary["unique_ips"], help="Distinct addresses before display masking.")
    cols[2].metric("HTTP error rate", f"{summary['error_rate']:.1%}")
    cols[3].metric("High rule score", summary["high_risk_requests"])
    left, right = st.columns([2, 1])
    with left:
        rpm = requests_per_minute(frame)
        fig = px.area(rpm, x="minute", y="requests", color_discrete_sequence=["#16836b"])
        fig.update_layout(height=280, xaxis_title="UTC", yaxis_title="Requests per minute")
        chart(fig, "traffic_volume")
    with right:
        statuses = frame.status.astype(str).value_counts().rename_axis("HTTP status").reset_index(name="Requests")
        fig = px.bar(statuses, x="HTTP status", y="Requests", color_discrete_sequence=["#397ca8"])
        fig.update_layout(height=280)
        chart(fig, "http_status")
    view = st.segmented_control("Request filter", ["All", "Flagged paths", "HTTP errors"], default="All", required=True)
    minimum = st.slider("Minimum rule score", 0, 100, 0, 5)
    visible = frame.loc[frame.risk_score.ge(minimum)].copy()
    if view == "Flagged paths": visible = visible.loc[visible.is_suspicious_path]
    elif view == "HTTP errors": visible = visible.loc[visible.is_error]
    if masked: visible["ip"] = visible.ip.map(mask_ip)
    columns = ["timestamp", "ip", "method", "path", "status", "risk_score", "risk_level", "user_agent"]
    st.subheader("Requests")
    st.caption(f"{len(visible)} visible requests · rule scores are not attack probabilities")
    st.dataframe(visible[columns], hide_index=True, width="stretch", height=360)
    st.download_button("Filtered log CSV", safe_csv(visible[columns]), "log-analysis.csv", "text/csv", icon=":material/download:")
    with st.expander("Rule weights and limits"):
        st.dataframe(pd.DataFrame({"Condition": ["Suspicious path keyword", "HTTP 401 / 403", "HTTP 404", "HTTP 5xx", "Bot-like user agent", "Method outside GET / POST / HEAD"],
                                   "Points": [35, 20, 12, 10, 15, 15]}), hide_index=True, width="stretch")
        st.caption("Maximum score 100. Legitimate requests may match these rules; unflagged requests are not guaranteed safe. No automatic blocking.")


def evidence():
    release = bundle()
    heading("Model evidence", "Retrospective v3 benchmark · training-data model and threshold selection")
    dataset = st.segmented_control("Benchmark", ["Full test", "KDDTest-21"], default="Full test", required=True)
    key = "historical_test" if dataset == "Full test" else "harder_test"
    if key == "harder_test": st.warning("KDDTest-21 is a harder dependent subset of the full test, not independent external validation.")
    metrics = release.metrics[key]
    cols = st.columns(4)
    for col, label, metric in zip(cols, ["Attack precision", "Attack recall", "Attack F1", "Category macro-F1"],
                                  [metrics["binary"]["metrics"][n] for n in ("precision", "recall", "f1")] + [metrics["multiclass"]["metrics"]["macro_f1"]]):
        col.metric(label, f"{metric:.2%}")
    st.caption(f"{release.audit[key]['rows']:,} connections · threshold {release.manifest['binary_threshold']:.6f} · binary: {release.selection['binary']['selected_model']} · category: {release.selection['multiclass']['selected_model']}")
    task_label = st.segmented_control("Task", ["Binary", "Attack category"], default="Binary", required=True)
    task = "binary" if task_label == "Binary" else "multiclass"
    data = metrics[task]["metrics"]
    matrix = np.asarray(data["confusion_matrix"])
    normalized = st.toggle("Normalize by actual class", value=False)
    display = np.divide(matrix, matrix.sum(axis=1, keepdims=True), out=np.zeros_like(matrix, dtype=float), where=matrix.sum(axis=1, keepdims=True) != 0) if normalized else matrix
    labels = ["Normal", "Attack"] if task == "binary" else data["labels"]
    annotations = np.asarray([[f"{value:.1%}" for value in row] for row in display]) if normalized else matrix
    chart(go.Figure(go.Heatmap(z=display, x=labels, y=labels, colorscale="Teal", text=annotations, texttemplate="%{text}", hovertemplate="Actual %{y}<br>Predicted %{x}<br>Value %{z}<extra></extra>"))
          .update_layout(height=340, xaxis_title="Predicted", yaxis_title="Actual", yaxis=dict(autorange="reversed")), "confusion_matrix")
    report = data["classification_report"]
    rows = [{"Class": ("Normal" if label == 0 else "Attack" if label == 1 else str(label).upper()),
             "Precision": f"{report[str(label)]['precision']:.2%}", "Recall": f"{report[str(label)]['recall']:.2%}",
             "F1": f"{report[str(label)]['f1-score']:.2%}", "Support": int(report[str(label)]["support"])} for label in data["labels"]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    with st.expander("Validation threshold tradeoff"):
        curve = release.curve
        fig = go.Figure()
        for name, color in (("precision", "#397ca8"), ("recall", "#b47c16"), ("f1", "#16836b")):
            fig.add_scatter(x=curve.threshold, y=curve[name], name=name.capitalize(), line=dict(color=color))
        fig.add_vline(x=release.manifest["binary_threshold"], line_dash="dot")
        fig.update_layout(height=320, xaxis_title="Attack threshold", yaxis_title="Validation score", yaxis=dict(range=[0, 1.02]))
        chart(fig, "threshold_tradeoff")
        st.caption("Display subsample of the training-validation curve; selected threshold and original metrics are unchanged. Shared validation selects both model and threshold.")
    st.download_button("Benchmark JSON", json.dumps(release.metrics, indent=2), "reference-metrics.json", "application/json", icon=":material/download:")
    with st.expander("Provenance and limitations"):
        st.markdown("NSL-KDD is an older laboratory benchmark. Historical test data were previously inspected, and some test feature vectors overlap training. Rare R2L/U2R performance and distribution shift remain limitations. These scores do not validate modern production traffic.")
        st.json({"study_manifest_sha256": release.manifest["study_manifest_sha256"], "input_sha256": release.manifest["input_sha256"], "data_audit": release.audit})


def explanations():
    release = bundle()
    heading("Model explanations", "SHAP contributions · explicitly labelled model-output scales")
    task_label = st.segmented_control("Explanation task", ["Binary", "Attack category"], default="Binary", required=True)
    task = "binary" if task_label == "Binary" else "multiclass"
    importance = pd.read_csv(release.path / f"{task}_global_importance.csv")
    labels = importance["class"].astype(str).unique().tolist()
    selected = st.selectbox("Explained class", labels)
    units = "log-odds" if task == "binary" else "raw class margin"
    selected_importance = importance.loc[importance["class"].astype(str).eq(selected)].nlargest(12, "mean_abs_grouped_shap").sort_values("mean_abs_grouped_shap")
    fig = px.bar(selected_importance, x="mean_abs_grouped_shap", y="feature", orientation="h", color_discrete_sequence=["#397ca8"])
    fig.update_layout(height=420, xaxis_title=f"Mean |SHAP| ({'log-odds' if task == 'binary' else 'margin'})", yaxis_title=None)
    chart(fig, "global_shap")
    st.caption("Global reference: 32 uniformly sampled historical test rows, 64 training background rows. Signed one-hot contributions are summed before absolute averaging. Not causal attribution.")
    st.subheader("Local explanation")
    local_source = st.radio("Connection", ["Reference example", "Last analyzed connection"], horizontal=True)
    if local_source == "Last analyzed connection":
        frame = st.session_state.get("private_features")
        if frame is None:
            st.caption("No analyzed connection in this session.")
            return
        position = st.number_input("Connection row", min_value=1, max_value=len(frame), value=1)
        row = frame.iloc[[int(position) - 1]][FEATURE_COLUMNS]
    else:
        index = st.selectbox("Explanation example", release.examples.index,
                              format_func=lambda i: f"{release.examples.loc[i, 'attack_category'].upper()} · source row {int(release.examples.loc[i, 'source_row'])}")
        row = release.examples.loc[[index], FEATURE_COLUMNS]
    cache_key = task + hashlib.sha256(row.to_csv(index=False).encode()).hexdigest()
    cache = st.session_state.setdefault("private_shap", {})
    if st.button("Compute explanation", icon=":material/insights:", type="primary"):
        if cache_key not in cache:
            from src.explainability import explain_pipeline
            with st.spinner("Computing SHAP contributions..."), release.lock:
                result = explain_pipeline(release.models[task], release.background, row, task, workers=2)
            if len(cache) >= 4: cache.pop(next(iter(cache)))
            cache[cache_key] = result
    if cache_key in cache:
        result = cache[cache_key]
        axis = [str(value) for value in result["output_labels"]].index(selected)
        values = result["grouped_values"][0, :, axis]
        cols = st.columns(3)
        cols[0].metric("Background score", f"{result['base_values'][0, axis]:.4f}")
        cols[1].metric("Model output", f"{result['model_outputs'][0, axis]:.4f}")
        cols[2].metric("Reconstruction error", f"{result['max_reconstruction_error']:.2e}")
        from matplotlib import pyplot as plt
        import shap
        with release.lock:
            figure = plt.figure()
            try:
                explanation = shap.Explanation(values=values, base_values=result["base_values"][0, axis],
                                               data=row.iloc[0][FEATURE_COLUMNS].to_numpy(), feature_names=FEATURE_COLUMNS)
                shap.plots.waterfall(explanation, max_display=12, show=False)
                st.pyplot(figure, width="stretch")
            finally: plt.close(figure)
        st.caption(f"Contributions explain {units}, not probability points or the truth of the attack label. Background + signed contributions = model output.")
        table = pd.DataFrame({"feature": FEATURE_COLUMNS, "contribution": values, "units": units})
        st.download_button("Contributions CSV", safe_csv(table), "shap-contributions.csv", "text/csv", icon=":material/download:")
