"""Validation-selected thresholds and optional LightGBM in a separate NSL-KDD study."""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_recall_curve, precision_score, recall_score

from src.config import ALL_COLUMNS, FEATURE_COLUMNS
from src import evaluation_v2 as v2


SOURCE_PATHS = (*v2.SOURCE_PATHS, "src/evaluation_v3.py", "scripts/evaluate_models_v3.py",
                "requirements-upgrade.txt", "docs/evaluation_v3.md")


@dataclass(frozen=True)
class EvaluationProtocol:
    folds: int = 5
    seed: int = 42
    workers: int = 2
    include_lightgbm: bool = False

    def __post_init__(self):
        v2.EvaluationProtocol(folds=self.folds, seed=self.seed, workers=self.workers)
        if not isinstance(self.include_lightgbm, bool):
            raise ValueError("include_lightgbm must be a boolean.")


@dataclass(frozen=True)
class PredictionPolicy:
    workers: int
    binary_threshold: float = .5


def candidate_models(task: str, protocol: EvaluationProtocol) -> dict:
    models = v2.candidate_models(task, protocol)
    if protocol.include_lightgbm:
        try:
            from lightgbm import LGBMClassifier
        except (ImportError, OSError) as error:
            raise RuntimeError("LightGBM was requested but cannot load. Install requirements-upgrade.txt "
                               "and the platform's OpenMP runtime; the study will not silently skip it.") from error
        models["LightGBM"] = clone(models["Prior baseline"]).set_params(model=LGBMClassifier(
            n_estimators=300, learning_rate=.05, num_leaves=31, min_child_samples=20,
            class_weight="balanced", random_state=protocol.seed, n_jobs=protocol.workers,
            deterministic=True, force_col_wise=True, verbosity=-1,
        ))
    return models


def tune_binary_threshold(actual, probability) -> tuple[float, pd.DataFrame]:
    """Use only held-out training predictions; prefer the closest-to-0.5 F1 tie."""
    y, p = np.asarray(actual), np.asarray(probability, dtype=float)
    if (y.ndim != 1 or p.shape != y.shape or set(y) != {0, 1}
            or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()):
        raise ValueError("Threshold tuning requires both binary classes and finite aligned probabilities in [0, 1].")
    precision, recall, thresholds = precision_recall_curve(y, p, drop_intermediate=False)
    scores = np.divide(2 * precision[:-1] * recall[:-1], precision[:-1] + recall[:-1],
                       out=np.zeros_like(thresholds), where=precision[:-1] + recall[:-1] > 0)
    curve = pd.DataFrame({"threshold": thresholds, "precision": precision[:-1],
                          "recall": recall[:-1], "f1": scores})
    default_prediction = p >= .5
    default = pd.DataFrame([{"threshold": .5, "precision": precision_score(y, default_prediction, zero_division=0),
                             "recall": recall_score(y, default_prediction, zero_division=0),
                             "f1": f1_score(y, default_prediction, zero_division=0)}])
    curve = pd.concat([curve, default], ignore_index=True).drop_duplicates("threshold").sort_values("threshold")
    # Floating arithmetic can differ by a few ULPs between the PR formula and sklearn's F1.
    best = curve.loc[np.isclose(curve.f1, curve.f1.max(), atol=1e-14, rtol=0)].copy()
    best["distance"] = (best.threshold - .5).abs()
    chosen = float(best.sort_values(["distance", "threshold"], ascending=[True, False]).iloc[0].threshold)
    curve["selected"] = curve.threshold.eq(chosen)
    return chosen, curve.reset_index(drop=True)


def select_models(training: pd.DataFrame, protocol: EvaluationProtocol) -> tuple[dict, dict, dict, pd.DataFrame]:
    fit, validation, membership = v2.training_partition(training, protocol)
    selected, reports, tables = {}, {}, {}
    for task, (target, _, primary) in v2.TASKS.items():
        print(f"V3 {task}: selecting on training-data validation only...", flush=True)
        candidates, scored, candidate_rows = candidate_models(task, protocol), [], []
        for name, candidate in candidates.items():
            model, notes, converged, seconds = v2._fit(
                candidate, training.iloc[fit][FEATURE_COLUMNS], training.iloc[fit][target], protocol.workers,
            )
            metrics, predictions = v2.score_model(model, training.iloc[validation], task, PredictionPolicy(protocol.workers))
            reference_metrics, threshold = metrics, .5
            if task == "binary":
                threshold, curve = tune_binary_threshold(predictions.actual, predictions.p_1)
                curve.insert(0, "model", name)
                tables.setdefault("binary_threshold_candidates.csv", []).append(curve)
                metrics, predictions = v2.score_model(
                    model, training.iloc[validation], task, PredictionPolicy(protocol.workers, threshold),
                )
                predictions["threshold"] = threshold
            predictions.insert(0, "model", name)
            candidate_rows.append(predictions)
            scored.append({"model": name, "metrics": metrics, "default_threshold_metrics": reference_metrics,
                           "binary_threshold": threshold if task == "binary" else None, "eligible": converged,
                           "fit_seconds": seconds, "warnings": notes})
        eligible = [row for row in scored if row["eligible"]]
        if not eligible:
            raise ValueError(f"No converged candidate is eligible for {task} selection.")
        winner = min(eligible, key=lambda row: (-row["metrics"][primary], row["model"]))
        selected[task] = clone(candidates[winner["model"]])
        reports[task] = {"selected_model": winner["model"], "primary_metric": primary,
                         "binary_threshold": winner["binary_threshold"], "candidates": scored}
        tables[f"validation_{task}_predictions.csv"] = pd.concat(candidate_rows, ignore_index=True)
    tables["binary_threshold_candidates.csv"] = pd.concat(tables["binary_threshold_candidates.csv"], ignore_index=True)
    return selected, reports, tables, membership


def verify_harder_subset(historical_test: pd.DataFrame, harder_test: pd.DataFrame) -> None:
    """Require the exact row multiset left after excluding difficulty 21."""
    expected = historical_test.loc[historical_test.difficulty.lt(21), ALL_COLUMNS]
    if expected.empty or Counter(expected.itertuples(index=False, name=None)) != Counter(
        harder_test[ALL_COLUMNS].itertuples(index=False, name=None)
    ):
        raise ValueError("KDDTest-21 must exactly match the historical test rows with difficulty below 21, including multiplicities.")


def build_evaluation(train_path: Path, test_path: Path, output: Path, project_root: Path,
                     protocol: EvaluationProtocol = EvaluationProtocol(), harder_test_path: Path | None = None) -> Path:
    paths = {"training": train_path.resolve(), "historical_test": test_path.resolve()}
    if harder_test_path is not None:
        paths["harder_test"] = harder_test_path.resolve()
    output = v2.validate_destination(output, project_root, tuple(paths.values()))
    inputs = {name: v2.digest(path) for name, path in paths.items()}
    if len(set(inputs.values())) != len(inputs):
        raise ValueError("Study inputs must not be identical files.")
    sources = {name: (project_root / name).read_bytes() for name in SOURCE_PATHS}
    training = v2.load_evaluation_data(paths["training"])
    if training.attrs["source_sha256"] != inputs["training"]:
        raise ValueError("Training source changed while loading.")
    selected, selection, tables, membership = select_models(training, protocol)
    frozen_threshold = selection["binary"]["binary_threshold"]
    policy = PredictionPolicy(protocol.workers, frozen_threshold)
    models, refits = {}, {}
    for task, (target, _, _) in v2.TASKS.items():
        model, notes, converged, seconds = v2._fit(selected[task], training[FEATURE_COLUMNS], training[target], protocol.workers)
        if not converged:
            raise ValueError(f"Selected {task} refit did not converge; no study was written.")
        models[task], refits[task] = model, {"fit_seconds": seconds, "warnings": notes}
    # Neither benchmark's labels can influence the chosen models, threshold, or refits.
    benchmarks = {name: v2.load_evaluation_data(path) for name, path in paths.items() if name != "training"}
    for name, frame in benchmarks.items():
        if frame.attrs["source_sha256"] != inputs[name]:
            raise ValueError(f"{name} source changed while loading.")
    if "harder_test" in benchmarks:
        verify_harder_subset(benchmarks["historical_test"], benchmarks["harder_test"])
    results, audit = {}, {}
    groups = v2.feature_groups(training)
    audit["training"] = {"rows": len(training), "duplicate_feature_rows": int(groups.duplicated().sum())}
    for name, frame in benchmarks.items():
        results[name] = {}
        audit[name] = {"rows": len(frame), "category_support": frame.attack_category.value_counts().to_dict(),
                       "train_overlapping_feature_rows": int(v2.feature_groups(frame).isin(set(groups)).sum()),
                       "test_only_attack_types": sorted(set(frame.attack_type) - set(training.attack_type))}
        for task, model in models.items():
            metrics, predictions = v2.score_model(model, frame, task, policy)
            reference, _ = v2.score_model(model, frame, task, PredictionPolicy(protocol.workers))
            results[name][task] = {"selected_model": selection[task]["selected_model"], "metrics": metrics,
                                   "binary_threshold": frozen_threshold if task == "binary" else None,
                                   "default_threshold_metrics": reference, "refit": refits[task]}
            tables[f"{name}_{task}_predictions.csv"] = predictions
    if inputs != {name: v2.digest(path) for name, path in paths.items()} or any(
        (project_root / name).read_bytes() != data for name, data in sources.items()
    ):
        raise ValueError("Inputs or implementation changed during evaluation; no study was written.")
    packages = {name: version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "joblib", "threadpoolctl")}
    if protocol.include_lightgbm:
        packages["lightgbm"] = version("lightgbm")
    output = v2.validate_destination(output, project_root, tuple(paths.values()))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    (output / "models").mkdir()
    for task, model in models.items():
        joblib.dump(model, output / "models" / f"{task}_estimator.joblib")
    for name, table in {"split_membership.csv": membership, **tables}.items():
        table.to_csv(output / name, index=False, float_format="%.17g")
    decision = {"attack_label": 1, "binary_threshold": frozen_threshold,
                "binary_rule": "predict_proba column identified by classes_ == 1 >= binary_threshold",
                "warning": "The saved bare estimator's predict() uses its own default, not this selected threshold."}
    for name, data in (("selection.json", selection), ("test_metrics.json", results),
                       ("data_audit.json", audit), ("decision_policy.json", decision)):
        (output / name).write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    for name, data in sources.items():
        path = output / "source_code" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    (output / "README.md").write_bytes(sources["docs/evaluation_v3.md"])
    (output / "requirements.txt").write_text("\n".join(f"{name}=={value}" for name, value in packages.items()) + "\n", encoding="utf-8")
    manifest = {"evaluation_version": 3, "generated_at_utc": datetime.now(UTC).isoformat(),
                "protocol": asdict(protocol), "validation_fold": 0, "decision_policy": decision,
                "input_sha256": inputs, "input_files": {name: path.name for name, path in paths.items()},
                "python": platform.python_version(), "platform": platform.platform(), "packages": packages,
                "source_code_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in sources.items()},
                "interpretation": "Post-hoc NSL-KDD benchmark; validation jointly selects thresholds and models. "
                                  "KDDTest-21 is a dependent harder subset, not external validation. No automatic promotion.",
                "files_sha256": {str(path.relative_to(output)): v2.digest(path) for path in sorted(output.rglob("*")) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output
