"""Export a separate, trusted release bundle from existing v3/SHAP artifacts; never refit."""

import argparse
from datetime import UTC, datetime
import hashlib
from io import BytesIO
import json
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ATTACK_CATEGORY_ORDER, FEATURE_COLUMNS, TEST_FILE, TRAIN_FILE  # noqa: E402
from src.explainability import read_study  # noqa: E402
from src import evaluation_v2 as v2  # noqa: E402


def export_bundle(study: Path, explanations: Path, train: Path, test: Path, output: Path,
                  project_root: Path = PROJECT_ROOT, trust_models: bool = False) -> Path:
    if trust_models is not True: raise ValueError("Explicit trust is required for local executable model files.")
    study, explanations, train, test = (p.resolve() for p in (study, explanations, train, test))
    output = v2.validate_destination(output, project_root, (study, explanations, train, test))
    manifest, original = read_study(study)
    explanation_bytes = (explanations / "manifest.json").read_bytes()
    explanation_manifest = json.loads(explanation_bytes)
    if explanation_manifest.get("study_manifest_sha256") != hashlib.sha256(original).hexdigest():
        raise ValueError("SHAP artifacts do not refer to this exact v3 study.")
    for name, expected in explanation_manifest["files_sha256"].items():
        target = (explanations / name).resolve()
        if explanations not in target.parents or not target.is_file() or v2.digest(target) != expected:
            raise ValueError("SHAP artifact integrity check failed.")
    if v2.digest(train) != manifest["input_sha256"]["training"] or v2.digest(test) != manifest["input_sha256"]["historical_test"]:
        raise ValueError("Release input sources do not match the frozen study.")
    test_frame = v2.load_evaluation_data(test)
    examples = pd.concat([test_frame.loc[test_frame.attack_category.eq(label)].head(10) for label in ATTACK_CATEGORY_ORDER])
    examples = examples[FEATURE_COLUMNS + ["attack_category", "attack_type", "binary_label"]].assign(source_row=examples.index).reset_index(drop=True)
    background = pd.read_csv(explanations / "background_rows.csv", float_precision="round_trip")
    training = v2.load_evaluation_data(train)
    positions = explanation_manifest["background_source_rows"]
    pd.testing.assert_frame_equal(background[FEATURE_COLUMNS].reset_index(drop=True),
                                  training.iloc[positions][FEATURE_COLUMNS].reset_index(drop=True), check_dtype=False)
    source_files = {"binary.joblib": (study / "models/binary_estimator.joblib").read_bytes(),
                    "multiclass.joblib": (study / "models/multiclass_estimator.joblib").read_bytes()}
    for name in ("test_metrics.json", "selection.json", "data_audit.json"):
        source_files[name] = (study / name).read_bytes()
    for task in v2.TASKS:
        name = f"{task}_global_importance.csv"
        source_files[name] = (explanations / name).read_bytes()
    for name, content in source_files.items():
        origin_name = f"models/{name.removesuffix('.joblib')}_estimator.joblib" if name.endswith(".joblib") else name
        expected = (explanation_manifest if "global_importance" in name else manifest)["files_sha256"][origin_name]
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("Source artifacts changed during release export.")
    classes = {task: joblib.load(BytesIO(source_files[f"{task}.joblib"])).classes_.tolist() for task in v2.TASKS}
    source_model_hashes = {task: hashlib.sha256(source_files[f"{task}.joblib"]).hexdigest() for task in v2.TASKS}
    for task in v2.TASKS:
        compressed = BytesIO()
        joblib.dump(joblib.load(BytesIO(source_files[f"{task}.joblib"])), compressed, compress=("xz", 3))
        source_files[f"{task}.joblib"] = compressed.getvalue()
    selection = json.loads(source_files["selection.json"])
    curve = pd.read_csv(study / "binary_threshold_candidates.csv", float_precision="round_trip")
    curve = curve.loc[curve.model.eq(selection["binary"]["selected_model"])].reset_index(drop=True)
    indices = sorted(set(np.linspace(0, len(curve) - 1, min(len(curve), 400), dtype=int)) | set(curve.index[curve.selected]))
    sampled_curve = curve.iloc[indices]
    if ((study / "manifest.json").read_bytes() != original or (explanations / "manifest.json").read_bytes() != explanation_bytes
            or v2.digest(train) != manifest["input_sha256"]["training"] or v2.digest(test) != manifest["input_sha256"]["historical_test"]):
        raise ValueError("Source manifests changed during release export.")
    output = v2.validate_destination(output, project_root, (study, explanations, train, test))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.mkdir(exist_ok=False)
    for name, content in source_files.items(): (output / name).write_bytes(content)
    examples.to_csv(output / "examples.csv", index=False, float_format="%.17g")
    background.to_csv(output / "background.csv", index=False, float_format="%.17g")
    sampled_curve.to_csv(output / "threshold_curve.csv", index=False, float_format="%.17g")
    release = {"bundle_version": 1, "generated_at_utc": datetime.now(UTC).isoformat(), "dataset": "NSL-KDD reference",
               "evaluation_version": 3, "feature_columns": FEATURE_COLUMNS, "python": manifest["python"],
               "binary_threshold": manifest["decision_policy"]["binary_threshold"], "model_classes": classes,
               "model_packages": manifest["packages"], "study_manifest_sha256": hashlib.sha256(original).hexdigest(),
               "source_model_sha256": source_model_hashes, "model_serialization": "joblib, lossless xz level 3; no refit or quantization",
               "explanation_manifest_sha256": hashlib.sha256(explanation_bytes).hexdigest(),
               "input_sha256": manifest["input_sha256"], "example_policy": "First ten available test rows per category; demonstration only, not an accuracy sample.",
               "threshold_curve": {"original_rows": len(curve), "display_rows": len(sampled_curve), "selection_unchanged": True},
               "shap_background_rows": positions, "interpretation": manifest["interpretation"],
               "files_sha256": {p.name: v2.digest(p) for p in output.iterdir() if p.is_file()}}
    (output / "manifest.json").write_text(json.dumps(release, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=PROJECT_ROOT / "runs/evaluation_v3")
    parser.add_argument("--explanations", type=Path, default=PROJECT_ROOT / "runs/explanations_v1_verified")
    parser.add_argument("--train", type=Path, default=TRAIN_FILE)
    parser.add_argument("--test", type=Path, default=TEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "dashboard_assets/reference_v1")
    parser.add_argument("--trust-local-models", action="store_true")
    args = parser.parse_args()
    print(export_bundle(args.study, args.explanations, args.train, args.test, args.output_dir, trust_models=args.trust_local_models))


if __name__ == "__main__": main()
