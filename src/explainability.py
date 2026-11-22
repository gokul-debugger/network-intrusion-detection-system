"""Bounded, verified SHAP exports for trusted, completed v3 reference studies."""

from copy import copy
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import platform
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import expit, softmax
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.config import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from src import evaluation_v2 as v2


SOURCE_PATHS = ("src/__init__.py", "src/config.py", "src/evaluation_v2.py", "src/explainability.py",
                "scripts/explain_models.py", "requirements-explainability.txt", "requirements-upgrade.txt",
                "requirements-evaluation.txt", "docs/explainability.md")
ATOL, RTOL = 1e-5, 1e-5


@dataclass(frozen=True)
class ExplanationProtocol:
    background_rows: int = 64
    explain_rows: int = 32
    seed: int = 42
    workers: int = 2

    def __post_init__(self):
        for name, maximum in (("background_rows", 100), ("explain_rows", 128), ("workers", 16)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
                raise ValueError(f"{name} must be an integer between 1 and {maximum}.")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")


def sample_positions(length: int, requested: int, seed: int) -> np.ndarray:
    if length < 1:
        raise ValueError("Cannot explain an empty source.")
    return np.sort(np.random.default_rng(seed).choice(length, size=min(length, requested), replace=False))


def read_study(study: Path) -> tuple[dict, bytes]:
    """Hash checks detect corruption, not malicious authors or unsafe pickle code."""
    study = study.resolve()
    content = (study / "manifest.json").read_bytes()
    manifest = json.loads(content)
    if not isinstance(manifest, dict) or manifest.get("evaluation_version") != 3 or not isinstance(manifest.get("files_sha256"), dict):
        raise ValueError("A completed v3 evaluation manifest is required.")
    required = {"models/binary_estimator.joblib", "models/multiclass_estimator.joblib", "selection.json", "decision_policy.json"}
    if not required <= manifest["files_sha256"].keys():
        raise ValueError("The study does not record all required models and decision metadata.")
    for name, expected in manifest["files_sha256"].items():
        relative = PurePosixPath(name)
        path = (study / name).resolve()
        if (relative.is_absolute() or ".." in relative.parts or "\\" in name or path == study
                or study not in path.parents or not path.is_file()):
            raise ValueError("Manifest artifact paths must resolve to files inside the study.")
        if not isinstance(expected, str) or len(expected) != 64 or v2.digest(path) != expected:
            raise ValueError(f"Study artifact hash mismatch: {name}")
    if platform.python_version_tuple()[:2] != tuple(str(manifest.get("python", "")).split(".")[:2]):
        raise ValueError("Use the study's recorded Python major/minor version to load its models.")
    packages = manifest.get("packages", {})
    if not isinstance(packages, dict) or not {"numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl"} <= packages.keys():
        raise ValueError("Study dependency versions are missing.")
    for name, expected in packages.items():
        if version(name) != expected:
            raise ValueError(f"Study dependency mismatch for {name}: expected {expected}.")
    policy_record = manifest.get("decision_policy", {})
    if not isinstance(policy_record, dict) or not isinstance(manifest.get("input_sha256"), dict):
        raise ValueError("Study input/decision metadata are missing.")
    threshold = policy_record.get("binary_threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("The study needs a finite binary threshold in [0, 1].")
    policy = json.loads((study / "decision_policy.json").read_bytes())
    selection = json.loads((study / "selection.json").read_bytes())
    if policy != manifest["decision_policy"] or selection.get("binary", {}).get("binary_threshold") != threshold:
        raise ValueError("Study decision-policy records disagree.")
    return manifest, content


def feature_mapping(pipeline: Pipeline) -> tuple[np.ndarray, np.ndarray]:
    """Map encoded positions from fitted transformer structure, never string prefixes."""
    if not isinstance(pipeline, Pipeline) or list(pipeline.named_steps) != ["preprocessor", "model"]:
        raise ValueError("Expected a fitted preprocessor/model evaluation pipeline.")
    preprocessor = pipeline.named_steps["preprocessor"]
    if not isinstance(preprocessor, ColumnTransformer) or list(preprocessor.feature_names_in_) != FEATURE_COLUMNS:
        raise ValueError("The fitted input schema is not the declared 41-feature NSL-KDD schema.")
    transformers = [(name, transform, list(columns)) for name, transform, columns in preprocessor.transformers_
                    if len(columns)]
    if len(transformers) != 2 or [item[0] for item in transformers] != ["numeric", "categorical"]:
        raise ValueError("Unsupported transformer layout; declare its feature mapping explicitly.")
    _, scaler, numeric = transformers[0]
    _, encoder, categorical = transformers[1]
    if not isinstance(scaler, StandardScaler) or numeric != NUMERIC_FEATURES or categorical != CATEGORICAL_FEATURES:
        raise ValueError("Numeric/categorical transformer columns do not match the source schema.")
    if (not isinstance(encoder, OneHotEncoder) or encoder.drop is not None
            or encoder.min_frequency is not None or encoder.max_categories is not None):
        raise ValueError("Explainability currently requires full, non-grouped one-hot encoding.")
    names = np.asarray(preprocessor.get_feature_names_out(), dtype=str)
    owners = np.asarray(numeric + [column for column, values in zip(categorical, encoder.categories_)
                                  for _ in values], dtype=str)
    if (len(names) != len(owners) or len(set(names)) != len(names)
            or set(owners) != set(FEATURE_COLUMNS)):
        raise ValueError("Transformed feature names and source-feature mapping are misaligned.")
    return names, owners


def _model_outputs(estimator, transformed: np.ndarray, task: str) -> tuple[np.ndarray, list, str, str, np.ndarray]:
    classes = np.asarray(estimator.classes_).tolist()
    if len(classes) != len(v2.TASKS[task][1]) or set(classes) != set(v2.TASKS[task][1]):
        raise ValueError("Fitted classifier classes do not match the declared task.")
    probabilities = np.asarray(estimator.predict_proba(transformed), dtype=float)
    if (probabilities.shape != (len(transformed), len(classes)) or not np.isfinite(probabilities).all()
            or (probabilities < 0).any() or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-7, rtol=0)):
        raise ValueError("Classifier returned invalid probabilities.")
    lightgbm = type(estimator).__module__ == "lightgbm.sklearn" and type(estimator).__name__ == "LGBMClassifier"
    if isinstance(estimator, RandomForestClassifier):
        return probabilities, classes, "probability", "TreeExplainer", probabilities
    if isinstance(estimator, LogisticRegression) and task != "binary":
        raise ValueError("Only the v3 binary logistic-regression configuration is supported.")
    if not lightgbm and not isinstance(estimator, (HistGradientBoostingClassifier, LogisticRegression)):
        raise ValueError("Unsupported selected model; no substitute explainer or model is used.")
    output = np.asarray(estimator.predict(transformed, raw_score=True) if lightgbm
                        else estimator.decision_function(transformed), dtype=float)
    if task == "binary":
        if output.shape != (len(transformed),):
            raise ValueError("Binary raw scores must have exactly one value per row.")
        expected_probability = expit(output)
        matches = probabilities[:, 1]
        labels, units = [classes[1]], "log_odds"
        output = output[:, None]
    else:
        if output.shape != probabilities.shape:
            raise ValueError("Multiclass raw-score axes do not match estimator classes_.")
        expected_probability, matches = softmax(output, axis=1), probabilities
        labels, units = classes, "class_margin"
    if not np.isfinite(output).all() or not np.allclose(expected_probability, matches, atol=1e-10, rtol=1e-8):
        raise ValueError("Raw model output does not reconstruct its declared probability link.")
    explainer = "LinearExplainer" if isinstance(estimator, LogisticRegression) else "TreeExplainer"
    return output, labels, units, explainer, probabilities


def checked_values(explanation, expected: np.ndarray, features: int) -> tuple[np.ndarray, np.ndarray, float]:
    rows, outputs = expected.shape
    values, bases = np.asarray(explanation.values, dtype=float), np.asarray(explanation.base_values, dtype=float)
    if outputs == 1 and values.shape == (rows, features):
        values = values[:, :, None]
    if values.shape != (rows, features, outputs):
        raise ValueError("SHAP feature/output axes do not match the fitted model.")
    if outputs == 1 and bases.shape == (rows,):
        bases = bases[:, None]
    elif bases.shape == (outputs,):
        bases = np.broadcast_to(bases, (rows, outputs)).copy()
    elif outputs == 1 and bases.ndim == 0:
        bases = np.full((rows, 1), bases.item())
    if bases.shape != (rows, outputs) or not np.isfinite(values).all() or not np.isfinite(bases).all():
        raise ValueError("SHAP values/base values are non-finite or misaligned.")
    reconstruction = bases + values.sum(axis=1)
    if not np.allclose(reconstruction, expected, atol=ATOL, rtol=RTOL):
        raise ValueError("SHAP contributions failed independent model-output reconstruction.")
    return values, bases, float(np.max(np.abs(reconstruction - expected)))


def explain_pipeline(pipeline: Pipeline, background: pd.DataFrame, examples: pd.DataFrame,
                     task: str, workers: int = 2) -> dict:
    import shap

    if task not in v2.TASKS:
        raise ValueError("Unknown explanation task.")
    if not 1 <= len(background) <= 100 or not 1 <= len(examples) <= 128:
        raise ValueError("Background/example rows exceed the bounded explanation protocol.")
    ExplanationProtocol(background_rows=len(background), explain_rows=len(examples), workers=workers)
    names, owners = feature_mapping(pipeline)
    # Override execution parallelism on a shallow wrapper, not on the caller's fitted object.
    pipeline = copy(pipeline)
    pipeline.steps = [(name, copy(step) if name == "model" else step) for name, step in pipeline.steps]
    preprocessor, estimator = pipeline.named_steps["preprocessor"], pipeline.named_steps["model"]
    if "n_jobs" in estimator.get_params():
        estimator.set_params(n_jobs=workers)
    with threadpool_limits(limits=workers), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        bg = np.asarray(preprocessor.transform(background[FEATURE_COLUMNS]), dtype=float)
        X = np.asarray(preprocessor.transform(examples[FEATURE_COLUMNS]), dtype=float)
        if (X.shape != (len(examples), len(names)) or bg.shape != (len(background), len(names))
                or not np.isfinite(X).all() or not np.isfinite(bg).all()):
            raise ValueError("The fitted preprocessor returned invalid dense feature matrices.")
        expected, labels, units, kind, probabilities = _model_outputs(estimator, X, task)
        pipeline_probabilities = np.asarray(pipeline.predict_proba(examples[FEATURE_COLUMNS]), dtype=float)
        if not np.allclose(pipeline_probabilities, probabilities, atol=1e-10, rtol=1e-8):
            raise ValueError("Pipeline and transformed-estimator predictions differ.")
        if kind == "LinearExplainer":
            explainer = shap.LinearExplainer(estimator, shap.maskers.Independent(bg, max_samples=len(bg)))
            explanation = explainer(X)
        else:
            explainer = shap.TreeExplainer(estimator, data=bg, model_output="raw", feature_perturbation="interventional")
            explanation = explainer(X, check_additivity=True)
        if not np.array_equal(explainer.masker.data, bg):
            raise ValueError("SHAP silently changed the explicitly selected training background.")
    values, bases, error = checked_values(explanation, expected, len(names))
    grouped = np.stack([values[:, owners == name, :].sum(axis=1) for name in FEATURE_COLUMNS], axis=1)
    if not np.allclose(bases + grouped.sum(axis=1), expected, atol=ATOL, rtol=RTOL):
        raise ValueError("Original-feature aggregation lost SHAP additivity.")
    return {"values": values, "grouped_values": grouped, "base_values": bases, "model_outputs": expected,
            "probabilities": probabilities, "transformed_features": names, "feature_owners": owners,
            "output_labels": labels, "estimator_classes": np.asarray(estimator.classes_).tolist(),
            "units": units, "explainer": kind, "max_reconstruction_error": error,
            "warnings": [{"category": item.category.__name__, "message": str(item.message)} for item in caught]}


def _save_plots(output: Path, task: str, result: dict, examples: pd.DataFrame) -> None:
    from matplotlib import pyplot as plt
    import shap

    for axis, label in enumerate(result["output_labels"]):
        group = result["grouped_values"][:, :, axis]
        existing_figures = set(plt.get_fignums())
        units = result["units"].replace("_", " ")
        try:
            plt.figure()
            summary = shap.Explanation(values=group, feature_names=FEATURE_COLUMNS)
            ax = shap.plots.bar(summary, max_display=15, show=False)
            ax.set_title(f"{task.title()}: class {label} | mean absolute grouped SHAP")
            ax.set_xlabel(f"Mean absolute contribution ({units})")
            ax.figure.savefig(output / f"{task}_output_{axis}_global.png", dpi=150, bbox_inches="tight")
            plt.close(ax.figure)
            plt.figure()
            local = shap.Explanation(values=group[0], base_values=result["base_values"][0, axis],
                                     data=examples.iloc[0][FEATURE_COLUMNS].to_numpy(), feature_names=FEATURE_COLUMNS)
            ax = shap.plots.waterfall(local, max_display=12, show=False)
            ax.set_title(f"{task.title()}: class {label} | first sampled connection ({units})")
            ax.figure.savefig(output / f"{task}_output_{axis}_local.png", dpi=150, bbox_inches="tight")
        finally:
            for number in set(plt.get_fignums()) - existing_figures:
                plt.close(number)


def build_explanations(study: Path, train_path: Path, example_path: Path, output: Path, project_root: Path,
                       protocol: ExplanationProtocol = ExplanationProtocol(), dataset: str = "historical_test",
                       trust_models: bool = False) -> Path:
    """Read trusted study models without refitting; reserve only a new sibling artifact directory."""
    if trust_models is not True:
        raise ValueError("Explicitly trust this local study before loading executable joblib model files.")
    if dataset not in ("historical_test", "harder_test", "training"):
        raise ValueError("Explanation data must be a recorded study input.")
    study, train_path, example_path = study.resolve(), train_path.resolve(), example_path.resolve()
    output = v2.validate_destination(output, project_root, (study, train_path, example_path))
    manifest, manifest_bytes = read_study(study)
    inputs = {"training": v2.digest(train_path), "examples": v2.digest(example_path)}
    if (inputs["training"] != manifest["input_sha256"].get("training")
            or inputs["examples"] != manifest["input_sha256"].get(dataset)):
        raise ValueError("Training background and example source must match the recorded study hashes.")
    source = {name: (project_root / name).read_bytes() for name in SOURCE_PATHS}
    training, frame = v2.load_evaluation_data(train_path), v2.load_evaluation_data(example_path)
    if training.attrs["source_sha256"] != inputs["training"] or frame.attrs["source_sha256"] != inputs["examples"]:
        raise ValueError("Explanation source changed while loading.")
    background_positions = sample_positions(len(training), protocol.background_rows, protocol.seed)
    example_positions = sample_positions(len(frame), protocol.explain_rows, protocol.seed + 1)
    background, examples = training.iloc[background_positions], frame.iloc[example_positions]
    selection = json.loads((study / "selection.json").read_bytes())
    results = {}
    for task in v2.TASKS:
        name = f"models/{task}_estimator.joblib"
        content = (study / name).read_bytes()
        if hashlib.sha256(content).hexdigest() != manifest["files_sha256"][name]:
            raise ValueError("Model bytes changed after study verification.")
        model = joblib.load(BytesIO(content))
        print(f"SHAP {task}: explaining {len(examples)} sampled rows...", flush=True)
        results[task] = explain_pipeline(model, background, examples, task, protocol.workers)
    checked, checked_bytes = read_study(study)
    if (checked_bytes != manifest_bytes or checked != manifest or inputs != {
        "training": v2.digest(train_path), "examples": v2.digest(example_path)
    } or any((project_root / name).read_bytes() != data for name, data in source.items())):
        raise ValueError("Study, source data, or explanation implementation changed; no export was written.")
    output = v2.validate_destination(output, project_root, (study, train_path, example_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    background[FEATURE_COLUMNS].assign(source_row=background_positions).to_csv(output / "background_rows.csv", index=False, float_format="%.17g")
    examples[FEATURE_COLUMNS].assign(source_row=example_positions).to_csv(output / "explained_rows.csv", index=False, float_format="%.17g")
    diagnostics = {}
    for task, result in results.items():
        np.savez_compressed(output / f"{task}_shap.npz", values=result["values"], grouped_values=result["grouped_values"],
                            base_values=result["base_values"], model_outputs=result["model_outputs"],
                            transformed_features=result["transformed_features"], feature_owners=result["feature_owners"],
                            original_features=np.asarray(FEATURE_COLUMNS), output_labels=np.asarray(result["output_labels"], dtype=str))
        rows = []
        for axis, label in enumerate(result["output_labels"]):
            for name, magnitude in zip(FEATURE_COLUMNS, np.abs(result["grouped_values"][:, :, axis]).mean(axis=0)):
                rows.append({"class": label, "feature": name, "mean_abs_grouped_shap": float(magnitude), "units": result["units"]})
        pd.DataFrame(rows).to_csv(output / f"{task}_global_importance.csv", index=False, float_format="%.17g")
        predictions = pd.DataFrame({"source_row": example_positions})
        classes = result["estimator_classes"]
        for axis, label in enumerate(classes):
            predictions[f"p_{label}"] = result["probabilities"][:, axis]
        predictions["prediction"] = ((predictions.p_1 >= manifest["decision_policy"]["binary_threshold"]).astype(int)
                                      if task == "binary" else np.asarray(classes)[result["probabilities"].argmax(axis=1)])
        predictions.to_csv(output / f"{task}_predictions.csv", index=False, float_format="%.17g")
        diagnostics[task] = {key: result[key] for key in ("output_labels", "estimator_classes", "units", "explainer", "max_reconstruction_error", "warnings")}
        diagnostics[task]["selected_model"] = selection[task]["selected_model"]
        _save_plots(output, task, result, examples)
    (output / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, data in source.items():
        path = output / "source_code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output / "README.md").write_bytes(source["docs/explainability.md"])
    (output / "study_manifest.json").write_bytes(manifest_bytes)
    packages = {name: version(name) for name in (*manifest["packages"], "shap", "matplotlib")}
    (output / "requirements.txt").write_text("\n".join(f"{name}=={value}" for name, value in packages.items()) + "\n", encoding="utf-8")
    completed = {"explanation_version": 1, "generated_at_utc": datetime.now(UTC).isoformat(),
                 "protocol": asdict(protocol), "dataset": dataset, "input_sha256": inputs,
                 "study_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                 "background_source_rows": background_positions.tolist(), "explained_source_rows": example_positions.tolist(),
                 "python": platform.python_version(), "packages": packages, "decision_policy": manifest["decision_policy"],
                 "model_sha256": {task: manifest["files_sha256"][f"models/{task}_estimator.joblib"] for task in v2.TASKS},
                 "additivity_tolerance": {"atol": ATOL, "rtol": RTOL},
                 "interpretation": "Sample-conditional model attributions, not causal explanations, calibrated risk, "
                                   "newer-dataset results, or evidence of production security effectiveness.",
                 "files_sha256": {str(path.relative_to(output)): v2.digest(path) for path in sorted(output.rglob("*")) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(completed, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output
