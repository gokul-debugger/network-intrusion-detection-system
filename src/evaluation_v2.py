"""Training-only model selection and write-once NSL-KDD benchmark evaluation."""

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
from io import BytesIO
import json
from pathlib import Path
import platform
from time import perf_counter
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from src.config import (
    ALL_COLUMNS, ATTACK_CATEGORY_MAP, ATTACK_CATEGORY_ORDER,
    CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES,
)


TASKS = {"binary": ("binary_label", [0, 1], "f1"),
         "multiclass": ("attack_category", ATTACK_CATEGORY_ORDER, "macro_f1")}
SOURCE_PATHS = ("src/__init__.py", "src/config.py", "src/evaluation_v2.py",
                "scripts/evaluate_models_v2.py", "requirements-evaluation.txt", "docs/evaluation_v2.md")


@dataclass(frozen=True)
class EvaluationProtocol:
    folds: int = 5
    seed: int = 42
    workers: int = 2
    binary_threshold: float = .5

    def __post_init__(self):
        for name in ("folds", "seed", "workers"):
            value = getattr(self, name)
            minimum = 2 if name == "folds" else 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"Invalid protocol count: {name}.")
        if self.binary_threshold != .5 or isinstance(self.binary_threshold, bool):
            raise ValueError("V2 fixes the binary threshold at 0.5; threshold tuning is not part of this study.")
        if self.seed >= 2**32:
            raise ValueError("Random seeds must be smaller than 2**32.")


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_evaluation_data(path: Path) -> pd.DataFrame:
    """Reject malformed rows; derive targets, never use label/difficulty as features."""
    content = path.read_bytes()
    frame = pd.read_csv(BytesIO(content), header=None)
    if frame.empty or frame.shape[1] != len(ALL_COLUMNS):
        raise ValueError("NSL-KDD evaluation requires nonempty 43-column source files.")
    frame.columns = ALL_COLUMNS
    if frame.isna().any().any():
        raise ValueError("NSL-KDD evaluation does not silently impute missing source values.")
    for name in NUMERIC_FEATURES:
        frame[name] = pd.to_numeric(frame[name], errors="raise").astype(float)
        if not np.isfinite(frame[name]).all() or frame[name].lt(0).any():
            raise ValueError(f"Feature {name} must contain finite non-negative values.")
        if name.endswith("_rate") and frame[name].gt(1).any():
            raise ValueError(f"Rate feature {name} must be between zero and one.")
    for name in [*CATEGORICAL_FEATURES, "attack_type"]:
        frame[name] = frame[name].astype(str)
        if frame[name].str.strip().eq("").any() or frame[name].ne(frame[name].str.strip()).any():
            raise ValueError(f"Empty or whitespace-padded values in {name}.")
    difficulty = pd.to_numeric(frame.difficulty, errors="raise")
    if not np.isfinite(difficulty).all() or not difficulty.between(0, 21).all() or difficulty.mod(1).ne(0).any():
        raise ValueError("Difficulty metadata must contain integers between 0 and 21.")
    frame["attack_category"] = frame.attack_type.map(ATTACK_CATEGORY_MAP)
    if frame.attack_category.isna().any():
        raise ValueError("Source contains unknown attack labels.")
    frame["binary_label"] = frame.attack_type.ne("normal").astype(int)
    frame.attrs["source_sha256"] = hashlib.sha256(content).hexdigest()
    return frame


def feature_groups(frame: pd.DataFrame) -> pd.Series:
    """Identical predictor vectors always stay together, regardless of their labels."""
    features = frame[FEATURE_COLUMNS].copy()
    numeric = features[NUMERIC_FEATURES].astype(float)
    features[NUMERIC_FEATURES] = numeric.mask(numeric.eq(0), 0.0)
    return pd.util.hash_pandas_object(features, index=False)


def training_partition(frame: pd.DataFrame, protocol: EvaluationProtocol) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    if set(frame.attack_category) != set(ATTACK_CATEGORY_ORDER):
        raise ValueError("Training data must contain all five attack categories.")
    groups = feature_groups(frame)
    support = pd.DataFrame({"category": frame.attack_category.to_numpy(), "group": groups.to_numpy()})
    if support.groupby("category")["group"].nunique().min() < protocol.folds:
        raise ValueError("Each category needs at least one distinct feature group per configured fold.")
    splitter = StratifiedGroupKFold(n_splits=protocol.folds, shuffle=True, random_state=protocol.seed)
    fit, validation = next(splitter.split(frame[FEATURE_COLUMNS], frame.attack_category, groups=groups))
    if any(set(frame.iloc[rows].attack_category) != set(ATTACK_CATEGORY_ORDER) for rows in (fit, validation)):
        raise ValueError("The fixed first fold lacks a category; revise the protocol explicitly, not by test performance.")
    if set(groups.iloc[fit]) & set(groups.iloc[validation]):
        raise ValueError("Feature groups overlap across the training/validation boundary.")
    membership = pd.DataFrame({"source_row": np.arange(len(frame)), "partition": "fit",
                               "feature_group": [f"{int(value):016x}" for value in groups],
                               "attack_category": frame.attack_category.to_numpy()})
    membership.loc[validation, "partition"] = "validation"
    return fit, validation, membership


def candidate_models(task: str, protocol: EvaluationProtocol) -> dict[str, Pipeline]:
    if task not in TASKS:
        raise ValueError("Unknown evaluation task.")
    models = {"Prior baseline": DummyClassifier(strategy="prior")}
    if task == "binary":
        models["Logistic Regression"] = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=protocol.seed,
        )
    models["Random Forest"] = RandomForestClassifier(
        n_estimators=300 if task == "binary" else 350,
        class_weight="balanced" if task == "binary" else "balanced_subsample",
        random_state=protocol.seed, n_jobs=protocol.workers,
    )
    models["HistGradientBoosting"] = HistGradientBoostingClassifier(
        learning_rate=.08, max_iter=220 if task == "binary" else 260, random_state=protocol.seed,
    )
    return {name: Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ], remainder="drop", sparse_threshold=0)),
        ("model", estimator),
    ]) for name, estimator in models.items()}


def _fit(model, X, y, workers: int) -> tuple[object, list[dict], bool, float]:
    start = perf_counter()
    with threadpool_limits(limits=workers), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fitted = clone(model).fit(X, y)
    diagnostics = [{"category": item.category.__name__, "message": str(item.message)} for item in caught]
    converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
    return fitted, diagnostics, converged, perf_counter() - start


def score_model(model, frame: pd.DataFrame, task: str, protocol: EvaluationProtocol) -> tuple[dict, pd.DataFrame]:
    target, labels, _ = TASKS[task]
    y = frame[target].to_numpy()
    if len(y) == 0 or not set(y) <= set(labels):
        raise ValueError("Evaluation targets must be nonempty and use known labels.")
    classes = list(model.classes_)
    if len(classes) != len(labels) or set(classes) != set(labels):
        raise ValueError("Fitted model classes do not match the fixed task labels.")
    with threadpool_limits(limits=protocol.workers):
        raw = np.asarray(model.predict_proba(frame[FEATURE_COLUMNS]), dtype=float)
    if (raw.shape != (len(frame), len(classes)) or not np.isfinite(raw).all()
            or (raw < 0).any() or (raw > 1).any() or not np.allclose(raw.sum(axis=1), 1, atol=1e-7, rtol=0)):
        raise ValueError("A classifier returned invalid or misaligned probabilities.")
    probabilities = raw[:, [classes.index(label) for label in labels]]
    prediction = ((probabilities[:, 1] >= protocol.binary_threshold).astype(int) if task == "binary"
                  else np.asarray(classes)[raw.argmax(axis=1)])
    metrics = {"accuracy": float(accuracy_score(y, prediction)),
               "macro_f1": float(f1_score(y, prediction, labels=labels, average="macro", zero_division=0)),
               "weighted_f1": float(f1_score(y, prediction, labels=labels, average="weighted", zero_division=0)),
               "labels": labels, "confusion_matrix": confusion_matrix(y, prediction, labels=labels).tolist(),
               "classification_report": classification_report(y, prediction, labels=labels, output_dict=True, zero_division=0)}
    if task == "binary":
        metrics.update(
            f1=float(f1_score(y, prediction, pos_label=1, zero_division=0)),
            precision=float(precision_score(y, prediction, pos_label=1, zero_division=0)),
            recall=float(recall_score(y, prediction, pos_label=1, zero_division=0)),
            roc_auc=float(roc_auc_score(y, probabilities[:, 1])) if len(set(y)) == 2 else None,
            average_precision=float(average_precision_score(y, probabilities[:, 1])) if 1 in y else None,
        )
    rows = pd.DataFrame({"source_row": frame.index.to_numpy(), "actual": y, "prediction": prediction})
    for offset, label in enumerate(labels):
        rows[f"p_{label}"] = probabilities[:, offset]
    return metrics, rows


def select_models(training: pd.DataFrame, protocol: EvaluationProtocol) -> tuple[dict, dict, dict, pd.DataFrame]:
    """This function cannot access official test inputs: it accepts training data only."""
    fit, validation, membership = training_partition(training, protocol)
    selected, reports, tables = {}, {}, {}
    for task, (target, _, primary) in TASKS.items():
        print(f"V2 {task}: fitting candidates on training-only partitions...", flush=True)
        candidates = candidate_models(task, protocol)
        scored, candidate_rows = [], []
        for name, candidate in candidates.items():
            model, notes, converged, seconds = _fit(
                candidate, training.iloc[fit][FEATURE_COLUMNS], training.iloc[fit][target], protocol.workers,
            )
            metrics, predictions = score_model(model, training.iloc[validation], task, protocol)
            predictions.insert(0, "model", name)
            candidate_rows.append(predictions)
            scored.append({"model": name, "metrics": metrics, "eligible": converged,
                           "fit_seconds": seconds, "warnings": notes})
        eligible = [row for row in scored if row["eligible"]]
        if not eligible:
            raise ValueError(f"No converged candidate is eligible for {task} selection.")
        winner = min(eligible, key=lambda row: (-row["metrics"][primary], row["model"]))
        selected[task] = clone(candidates[winner["model"]])
        reports[task] = {"selected_model": winner["model"], "primary_metric": primary, "candidates": scored}
        tables[f"validation_{task}_predictions.csv"] = pd.concat(candidate_rows, ignore_index=True)
    return selected, reports, tables, membership


def validate_destination(output: Path, project_root: Path, inputs: tuple[Path, ...]) -> Path:
    output, root = output.resolve(), project_root.resolve()
    roots = {root}
    for ancestor in (*root.parents, *output.parents):
        if (ancestor / "src/config.py").is_file():
            roots.add(ancestor)
    protected = [path.resolve() for path in inputs]
    for directory in roots:
        protected += [directory / name for name in ("Models", "reports", "notebooks", "Dataset", "src", "scripts", "app", "tests", "docs", ".github", ".git")]
        if output == directory or output in directory.parents:
            raise ValueError("Evaluation outputs must not replace the project or its ancestors.")
    if any(output == path or path in output.parents or output in path.parents for path in protected):
        raise ValueError("Evaluation outputs must be separate from existing data, models, results, notebooks, and source files.")
    if any((ancestor / "manifest.json").is_file() for ancestor in output.parents):
        raise ValueError("Cannot write inside an existing evaluation study.")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing evaluation: {output}")
    return output


def build_evaluation(
    train_path: Path, test_path: Path, output: Path, project_root: Path,
    protocol: EvaluationProtocol = EvaluationProtocol(),
) -> Path:
    """Select and refit both models before loading the historical test data."""
    train_path, test_path = train_path.resolve(), test_path.resolve()
    output = validate_destination(output, project_root, (train_path, test_path))
    inputs = {"training": digest(train_path), "historical_test": digest(test_path)}
    if inputs["training"] == inputs["historical_test"]:
        raise ValueError("Training and historical test files must not be identical.")
    sources = {name: (project_root / name).read_bytes() for name in SOURCE_PATHS}
    training = load_evaluation_data(train_path)
    if training.attrs["source_sha256"] != inputs["training"]:
        raise ValueError("Training input changed while loading; no study was written.")
    selected, selection, tables, membership = select_models(training, protocol)
    final_models, refits = {}, {}
    for task, (target, _, _) in TASKS.items():
        model, notes, converged, seconds = _fit(selected[task], training[FEATURE_COLUMNS], training[target], protocol.workers)
        if not converged:
            raise ValueError(f"The selected {task} model did not converge on refit; no study was written.")
        final_models[task], refits[task] = model, {"fit_seconds": seconds, "warnings": notes}
    # Choices are fixed and full-training refits are complete before test labels are loaded.
    historical_test = load_evaluation_data(test_path)
    if historical_test.attrs["source_sha256"] != inputs["historical_test"]:
        raise ValueError("Historical test input changed while loading; no study was written.")
    final = {}
    for task, model in final_models.items():
        metrics, predictions = score_model(model, historical_test, task, protocol)
        final[task] = {"selected_model": selection[task]["selected_model"], "metrics": metrics, "refit": refits[task]}
        tables[f"historical_test_{task}_predictions.csv"] = predictions
    train_groups, test_groups = feature_groups(training), feature_groups(historical_test)
    audit = {"training_rows": len(training), "historical_test_rows": len(historical_test),
             "training_duplicate_feature_rows": int(train_groups.duplicated().sum()),
             "train_test_overlapping_feature_rows": int(test_groups.isin(set(train_groups)).sum()),
             "test_only_attack_types": sorted(set(historical_test.attack_type) - set(training.attack_type)),
             "category_support": {name: frame.attack_category.value_counts().to_dict()
                                  for name, frame in (("fit", training.iloc[membership.index[membership.partition.eq("fit")]]),
                                                      ("validation", training.iloc[membership.index[membership.partition.eq("validation")]]),
                                                      ("historical_test", historical_test))}}
    if inputs != {"training": digest(train_path), "historical_test": digest(test_path)} or any(
        (project_root / name).read_bytes() != content for name, content in sources.items()
    ):
        raise ValueError("Input data or implementation changed during evaluation; no study was written.")
    packages = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")}
    output = validate_destination(output, project_root, (train_path, test_path))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    (output / "models").mkdir()
    for task, model in final_models.items():
        joblib.dump(model, output / "models" / f"{task}.joblib")
    for name, table in {"split_membership.csv": membership, **tables}.items():
        table.to_csv(output / name, index=False, float_format="%.17g")
    for name, data in (("selection.json", selection), ("historical_test_metrics.json", final), ("data_audit.json", audit)):
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, content in sources.items():
        destination = output / "source_code" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    (output / "README.md").write_bytes(sources["docs/evaluation_v2.md"])
    (output / "requirements.txt").write_text("\n".join(f"{name}=={value}" for name, value in packages.items()) + "\n", encoding="utf-8")
    manifest = {"evaluation_version": 2, "generated_at_utc": datetime.now(UTC).isoformat(),
                "protocol": asdict(protocol), "validation_fold": 0,
                "selection": {task: {name: report[name] for name in ("selected_model", "primary_metric")} for task, report in selection.items()},
                "input_sha256": inputs, "input_files": {"training": train_path.name, "historical_test": test_path.name},
                "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
                "prediction_policy": {"binary": "attack probability >= 0.5", "multiclass": "argmax in estimator classes_ order"},
                "source_code_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
                "interpretation": "Retrospective NSL-KDD benchmark. Historical test data were inspected in earlier work; "
                                  "this is not a newly independent holdout, modern traffic validation, or a production IDS.",
                "files_sha256": {str(path.relative_to(output)): digest(path) for path in sorted(output.rglob("*")) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output
