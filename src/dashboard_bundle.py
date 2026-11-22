"""Integrity-checked, repository-owned dashboard release assets and inference."""

from dataclasses import dataclass, field
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import platform
from threading import RLock

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.config import FEATURE_COLUMNS
from src.dashboard_inputs import validate_connections
from src.explainability import feature_mapping
from src import evaluation_v2 as v2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BUNDLE = PROJECT_ROOT / "dashboard_assets/reference_v1"


@dataclass
class DashboardBundle:
    path: Path
    manifest: dict
    models: dict
    examples: pd.DataFrame
    background: pd.DataFrame
    metrics: dict
    selection: dict
    audit: dict
    curve: pd.DataFrame
    lock: RLock = field(default_factory=RLock)

    def predict(self, frame: pd.DataFrame, threshold: float | None = None) -> tuple[pd.DataFrame, list[str]]:
        frame = validate_connections(frame)
        threshold = self.manifest["binary_threshold"] if threshold is None else threshold
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)) or not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Decision threshold must be finite and between zero and one.")
        result, unknown = pd.DataFrame({"row": np.arange(len(frame)) + 1}), []
        with self.lock, threadpool_limits(limits=2):
            for task, model in self.models.items():
                classes = list(model.classes_)
                probabilities = np.asarray(model.predict_proba(frame), dtype=float)
                if (probabilities.shape != (len(frame), len(classes)) or not np.isfinite(probabilities).all()
                        or (probabilities < 0).any() or (probabilities > 1).any()
                        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-7, rtol=0)):
                    raise ValueError("The release model returned invalid probabilities.")
                if task == "binary":
                    result["attack_probability"] = probabilities[:, classes.index(1)]
                    result["binary_prediction"] = np.where(result.attack_probability >= threshold, "Attack", "Normal")
                else:
                    result["attack_category"] = np.asarray(classes)[probabilities.argmax(axis=1)]
                    for index, name in enumerate(classes): result[f"p_{name}"] = probabilities[:, index]
                preprocessor = model.named_steps["preprocessor"]
                encoder = preprocessor.named_transformers_["categorical"]
                for name, categories in zip(("protocol_type", "service", "flag"), encoder.categories_):
                    if not frame[name].isin(categories).all(): unknown.append(name)
        result["decision_threshold"] = threshold
        return result, sorted(set(unknown))


def load_bundle(path: Path = DEFAULT_BUNDLE) -> DashboardBundle:
    """Load only repository-owned assets. This is not an untrusted-model upload API."""
    path = path.resolve()
    manifest = json.loads((path / "manifest.json").read_bytes())
    if manifest.get("bundle_version") != 1 or manifest.get("feature_columns") != FEATURE_COLUMNS:
        raise ValueError("Unsupported dashboard bundle or feature schema.")
    if tuple(platform.python_version_tuple()[:2]) != tuple(manifest["python"].split(".")[:2]):
        raise ValueError("Dashboard requires the release's recorded Python major/minor version.")
    for name, expected in manifest["model_packages"].items():
        if version(name) != expected: raise ValueError(f"Install app/requirements.txt: release requires {name}=={expected}.")
    files = manifest.get("files_sha256", {})
    required = {"binary.joblib", "multiclass.joblib", "examples.csv", "background.csv", "test_metrics.json", "selection.json", "data_audit.json", "threshold_curve.csv", "binary_global_importance.csv", "multiclass_global_importance.csv"}
    if not required <= files.keys(): raise ValueError("Dashboard release assets are incomplete.")
    for name, expected in files.items():
        target = (path / name).resolve()
        if PurePosixPath(name).is_absolute() or ".." in PurePosixPath(name).parts or path not in target.parents or not target.is_file():
            raise ValueError("Dashboard asset path escapes its release directory.")
        if v2.digest(target) != expected: raise ValueError(f"Dashboard asset checksum failed: {name}")
    models = {}
    for task in v2.TASKS:
        content = (path / f"{task}.joblib").read_bytes()
        if hashlib.sha256(content).hexdigest() != files[f"{task}.joblib"]: raise ValueError("Dashboard model changed during verification.")
        model = joblib.load(BytesIO(content))
        feature_mapping(model)
        if list(model.classes_) != manifest["model_classes"][task]: raise ValueError("Dashboard class order does not match release metadata.")
        if set(model.classes_) != set(v2.TASKS[task][1]): raise ValueError("Dashboard release has unknown classes.")
        if "n_jobs" in model.named_steps["model"].get_params(): model.named_steps["model"].set_params(n_jobs=2)
        models[task] = model
    examples = pd.read_csv(path / "examples.csv", float_precision="round_trip")
    background = pd.read_csv(path / "background.csv", float_precision="round_trip")
    validate_connections(examples[FEATURE_COLUMNS])
    validate_connections(background[FEATURE_COLUMNS])
    if not 1 <= len(background) <= 100: raise ValueError("Dashboard SHAP background exceeds its bounded protocol.")
    return DashboardBundle(path, manifest, models, examples, background[FEATURE_COLUMNS],
                           json.loads((path / "test_metrics.json").read_bytes()),
                           json.loads((path / "selection.json").read_bytes()),
                           json.loads((path / "data_audit.json").read_bytes()),
                           pd.read_csv(path / "threshold_curve.csv", float_precision="round_trip"))
