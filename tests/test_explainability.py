from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler

pytest.importorskip("shap")
pytest.importorskip("lightgbm")

from src.config import FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES
from src import evaluation_v2 as v2
from src import evaluation_v3 as v3
from src import explainability as explanations


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def trusted_study(monkeypatch, source_paths, tmp_path):
    original = v3.candidate_models
    def candidates(task, protocol):
        model = original(task, protocol)["LightGBM"].set_params(model__n_estimators=4, model__min_child_samples=2)
        return {"LightGBM": model}
    monkeypatch.setattr(v3, "candidate_models", candidates)
    study = v3.build_evaluation(*source_paths, tmp_path / "study", ROOT,
                               v3.EvaluationProtocol(workers=1, include_lightgbm=True))
    return study, *source_paths


@pytest.fixture
def small_protocol():
    return explanations.ExplanationProtocol(background_rows=8, explain_rows=4, workers=1)


def cheap_model(source_paths, task, name):
    frame = v2.load_evaluation_data(source_paths[0])
    model = v3.candidate_models(task, v3.EvaluationProtocol(workers=1, include_lightgbm=True))[name]
    if "model__n_estimators" in model.get_params():
        model.set_params(model__n_estimators=4)
    if name == "LightGBM": model.set_params(model__min_child_samples=2)
    if name == "HistGradientBoosting": model.set_params(model__max_iter=4, model__min_samples_leaf=2)
    fitted, _, _, _ = v2._fit(model, frame[FEATURE_COLUMNS], frame[v2.TASKS[task][0]], 1)
    return fitted, frame


@pytest.mark.parametrize("changes", [{"background_rows": 0}, {"background_rows": 101}, {"explain_rows": 129},
    {"explain_rows": True}, {"workers": 0}, {"workers": 17}, {"seed": True}, {"seed": -1}, {"seed": 2**32}])
def test_protocol_bounds(changes):
    with pytest.raises(ValueError):
        replace(explanations.ExplanationProtocol(), **changes)


def test_samples_are_reproducible_distinct_sorted_and_bounded():
    a = explanations.sample_positions(1000, 64, 42)
    np.testing.assert_array_equal(a, explanations.sample_positions(1000, 64, 42))
    assert len(set(a)) == 64 and np.all(np.diff(a) > 0)
    assert not np.array_equal(a, explanations.sample_positions(1000, 64, 43))
    np.testing.assert_array_equal(explanations.sample_positions(3, 64, 42), [0, 1, 2])
    with pytest.raises(ValueError, match="empty"):
        explanations.sample_positions(0, 8, 42)


@pytest.mark.parametrize("task,name,units", [("binary", "LightGBM", "log_odds"),
    ("multiclass", "LightGBM", "class_margin"), ("binary", "Random Forest", "probability"),
    ("multiclass", "Random Forest", "probability"), ("binary", "HistGradientBoosting", "log_odds"),
    ("multiclass", "HistGradientBoosting", "class_margin"), ("binary", "Logistic Regression", "log_odds")])
def test_real_explainers_reconstruct_supported_models_without_refitting(monkeypatch, source_paths, task, name, units):
    model, frame = cheap_model(source_paths, task, name)
    original = model.named_steps["preprocessor"].named_transformers_["numeric"].mean_.copy()
    original_jobs = model.named_steps["model"].get_params().get("n_jobs")
    def fail(*args, **kwargs): pytest.fail("Explanations must not refit models or preprocessing")
    monkeypatch.setattr(type(model.named_steps["model"]), "fit", fail)
    monkeypatch.setattr(StandardScaler, "fit", fail)
    monkeypatch.setattr(OneHotEncoder, "fit", fail)
    result = explanations.explain_pipeline(model, frame.iloc[:8], frame.iloc[20:24], task, 1)
    assert result["units"] == units
    assert result["grouped_values"].shape[1] == 41
    np.testing.assert_allclose(result["base_values"] + result["values"].sum(axis=1), result["model_outputs"], atol=1e-5, rtol=1e-5)
    np.testing.assert_allclose(result["base_values"] + result["grouped_values"].sum(axis=1), result["model_outputs"], atol=1e-5, rtol=1e-5)
    np.testing.assert_array_equal(original, model.named_steps["preprocessor"].named_transformers_["numeric"].mean_)
    assert result["max_reconstruction_error"] < 1e-5
    assert model.named_steps["model"].get_params().get("n_jobs") == original_jobs


def test_one_hot_mapping_uses_transformer_structure_not_underscore_splitting(source_paths):
    frame = v2.load_evaluation_data(source_paths[0])
    frame.loc[frame.index % 2 == 0, "service"] = "service_with__underscores"
    model = v2.candidate_models("binary", v2.EvaluationProtocol())["Random Forest"].set_params(model__n_estimators=3)
    model, *_ = v2._fit(model, frame[FEATURE_COLUMNS], frame.binary_label, 1)
    names, owners = explanations.feature_mapping(model)
    assert list(owners[:len(NUMERIC_FEATURES)]) == NUMERIC_FEATURES
    assert any("service_with__underscores" in name for name in names)
    for col, cats in zip(CATEGORICAL_FEATURES, model.named_steps["preprocessor"].named_transformers_["categorical"].categories_):
        assert (owners == col).sum() == len(cats)


@pytest.mark.parametrize("name", ["LightGBM", "Random Forest"])
def test_prediction_parallelism_is_bounded_without_changing_original_model(monkeypatch, source_paths, name):
    model, frame = cheap_model(source_paths, "binary", name)
    estimator = model.named_steps["model"]
    estimator.set_params(n_jobs=4)
    original = type(estimator).predict_proba
    jobs = []
    def predict(self, *args, **kwargs):
        jobs.append(self.n_jobs)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(type(estimator), "predict_proba", predict)
    explanations.explain_pipeline(model, frame.iloc[:8], frame.iloc[10:13], "binary", 1)
    assert jobs and set(jobs) == {1}
    assert estimator.n_jobs == 4


@pytest.mark.parametrize("parameter,value", [("drop", "first"), ("min_frequency", 2), ("max_categories", 2)])
def test_changed_one_hot_layout_requires_an_explicit_adapter(source_paths, parameter, value):
    frame = v2.load_evaluation_data(source_paths[0])
    model = v2.candidate_models("binary", v2.EvaluationProtocol())["Random Forest"].set_params(
        model__n_estimators=2, **{f"preprocessor__categorical__{parameter}": value})
    model, *_ = v2._fit(model, frame[FEATURE_COLUMNS], frame.binary_label, 1)
    with pytest.raises(ValueError, match="non-grouped one-hot"):
        explanations.feature_mapping(model)


def test_unseen_category_does_not_refit_encoder(source_paths):
    model, frame = cheap_model(source_paths, "binary", "LightGBM")
    examples = frame.iloc[10:13].copy()
    examples["service"] = "UNSEEN_TEST_CATEGORY"
    result = explanations.explain_pipeline(model, frame.iloc[:8], examples, "binary", 1)
    assert result["grouped_values"].shape == (3, 41, 1)
    assert "UNSEEN_TEST_CATEGORY" not in model.named_steps["preprocessor"].named_transformers_["categorical"].categories_[1]


def test_probability_class_axes_follow_estimator_class_order(source_paths):
    model, frame = cheap_model(source_paths, "multiclass", "Random Forest")
    model.named_steps["model"].classes_ = model.named_steps["model"].classes_[::-1]
    result = explanations.explain_pipeline(model, frame.iloc[:8], frame.iloc[10:13], "multiclass", 1)
    assert result["output_labels"] == model.classes_.tolist()
    np.testing.assert_allclose(result["model_outputs"], model.predict_proba(frame.iloc[10:13][FEATURE_COLUMNS]))


@pytest.mark.parametrize("fault", ["value_shape", "base_shape", "nan_values", "nan_base", "reconstruction"])
def test_additivity_and_axis_checks_reject_corrupt_explanations(fault):
    values, bases = np.zeros((3, 4, 2)), np.zeros((3, 2))
    if fault == "value_shape": values = values.transpose(0, 2, 1)
    elif fault == "base_shape": bases = np.zeros((2, 3))
    elif fault == "nan_values": values[0, 0, 0] = np.nan
    elif fault == "nan_base": bases[0, 0] = np.nan
    else: values[0, 0, 0] = 1
    with pytest.raises(ValueError, match="SHAP"):
        explanations.checked_values(SimpleNamespace(values=values, base_values=bases), np.zeros((3, 2)), 4)


def test_unsupported_model_fails_instead_of_explaining_a_surrogate(source_paths):
    model, frame = cheap_model(source_paths, "binary", "Prior baseline")
    with pytest.raises(ValueError, match="Unsupported selected model"):
        explanations.explain_pipeline(model, frame.iloc[:8], frame.iloc[10:13], "binary", 1)


def test_joblib_requires_explicit_trust_before_any_load(monkeypatch, source_paths, tmp_path):
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("Untrusted model must not load"))
    with pytest.raises(ValueError, match="Explicitly trust"):
        explanations.build_explanations(tmp_path / "study", *source_paths, tmp_path / "out", ROOT)


def test_model_hash_failure_stops_before_unpickle(monkeypatch, trusted_study, tmp_path, small_protocol):
    study, train, test = trusted_study
    (study / "models/binary_estimator.joblib").write_bytes(b"not a trusted model")
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("Corrupted model must not load"))
    with pytest.raises(ValueError, match="hash mismatch"):
        explanations.build_explanations(study, train, test, tmp_path / "out", ROOT, small_protocol, trust_models=True)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("name", ["../outside", "/private/tmp/outside", "models\\outside"])
def test_manifest_paths_cannot_escape_study(trusted_study, tmp_path, name):
    study, _, _ = trusted_study
    manifest = json.loads((study / "manifest.json").read_text())
    manifest["files_sha256"][name] = "0" * 64
    (study / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside the study"):
        explanations.read_study(study)


def test_manifest_symlinks_cannot_escape_study(trusted_study, tmp_path):
    study, _, _ = trusted_study
    external = tmp_path / "external"
    external.write_bytes(b"external")
    (study / "alias").symlink_to(external)
    manifest = json.loads((study / "manifest.json").read_text())
    manifest["files_sha256"]["alias"] = v2.digest(external)
    (study / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="inside the study"):
        explanations.read_study(study)


def test_dependency_mismatch_is_rejected(trusted_study):
    study, _, _ = trusted_study
    manifest = json.loads((study / "manifest.json").read_text())
    manifest["packages"]["scikit-learn"] = "0.0.0"
    (study / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="dependency mismatch"):
        explanations.read_study(study)


def test_only_recorded_training_can_be_used_as_background(trusted_study, tmp_path, small_protocol):
    study, _, test = trusted_study
    with pytest.raises(ValueError, match="recorded study hashes"):
        explanations.build_explanations(study, test, test, tmp_path / "out", ROOT, small_protocol, trust_models=True)
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("destination", ["study", "nested", "existing", "models"])
def test_protected_destinations_fail_without_mutation(trusted_study, tmp_path, small_protocol, destination):
    study, train, test = trusted_study
    output = {"study": study, "nested": study / "nested", "existing": tmp_path / "existing",
              "models": ROOT / "Models/explanations"}[destination]
    if destination == "existing": output.mkdir()
    with pytest.raises((ValueError, FileExistsError)):
        explanations.build_explanations(study, train, test, output, ROOT, small_protocol, trust_models=True)


def test_full_export_preserves_study_and_reconstructs_grouped_outputs(trusted_study, tmp_path, small_protocol):
    study, train, test = trusted_study
    before = {str(path.relative_to(study)): v2.digest(path) for path in study.rglob("*") if path.is_file()}
    output = explanations.build_explanations(study, train, test, tmp_path / "out", ROOT, small_protocol, trust_models=True)
    assert before == {str(path.relative_to(study)): v2.digest(path) for path in study.rglob("*") if path.is_file()}
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["explanation_version"] == 1
    assert len(manifest["background_source_rows"]) == 8
    assert len(manifest["explained_source_rows"]) == 4
    assert len(list(output.glob("*.png"))) == 12
    for name, expected in manifest["files_sha256"].items():
        assert v2.digest(output / name) == expected
    diagnostics = json.loads((output / "diagnostics.json").read_text())
    for task in v2.TASKS:
        with np.load(output / f"{task}_shap.npz", allow_pickle=False) as arrays:
            assert arrays["grouped_values"].shape[1] == 41
            np.testing.assert_allclose(arrays["base_values"] + arrays["grouped_values"].sum(axis=1), arrays["model_outputs"], atol=1e-5, rtol=1e-5)
            magnitude = np.abs(arrays["grouped_values"]).mean(axis=0)
            summary = pd.read_csv(output / f"{task}_global_importance.csv", float_precision="round_trip")
            for index, label in enumerate(diagnostics[task]["output_labels"]):
                np.testing.assert_allclose(summary.loc[summary["class"].astype(str).eq(str(label)), "mean_abs_grouped_shap"], magnitude[:, index])
        for kind in ("global", "local"):
            assert (output / f"{task}_output_0_{kind}.png").read_bytes().startswith(b"\x89PNG")
    predictions = pd.read_csv(output / "binary_predictions.csv", float_precision="round_trip")
    np.testing.assert_array_equal(predictions.prediction, predictions.p_1 >= manifest["decision_policy"]["binary_threshold"])
    with pytest.raises(FileExistsError):
        explanations.build_explanations(study, train, test, output, ROOT, small_protocol, trust_models=True)


def test_input_change_during_explanation_leaves_no_completed_export(monkeypatch, trusted_study, tmp_path, small_protocol):
    study, train, test = trusted_study
    original = explanations.explain_pipeline
    def mutate(*args, **kwargs):
        result = original(*args, **kwargs)
        train.write_bytes(train.read_bytes() + b"\n")
        return result
    monkeypatch.setattr(explanations, "explain_pipeline", mutate)
    with pytest.raises(ValueError, match="changed; no export"):
        explanations.build_explanations(study, train, test, tmp_path / "out", ROOT, small_protocol, trust_models=True)
    assert not (tmp_path / "out").exists()


def test_second_model_failure_leaves_no_export(monkeypatch, trusted_study, tmp_path, small_protocol):
    study, train, test = trusted_study
    original = explanations.explain_pipeline
    def fail(model, bg, examples, task, workers):
        if task == "multiclass": raise ValueError("fixture unsupported model")
        return original(model, bg, examples, task, workers)
    monkeypatch.setattr(explanations, "explain_pipeline", fail)
    with pytest.raises(ValueError, match="fixture"):
        explanations.build_explanations(study, train, test, tmp_path / "out", ROOT, small_protocol, trust_models=True)
    assert not (tmp_path / "out").exists()


def test_model_loader_uses_verified_memory_not_reopened_filename(monkeypatch, trusted_study, tmp_path, small_protocol):
    from io import BytesIO
    study, train, test = trusted_study
    original = joblib.load
    loads = []
    def load(value, *args, **kwargs):
        assert isinstance(value, BytesIO)
        loads.append(value.getbuffer().nbytes)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(joblib, "load", load)
    monkeypatch.setattr(explanations, "_save_plots", lambda *args: None)
    explanations.build_explanations(study, train, test, tmp_path / "out", ROOT, small_protocol, trust_models=True)
    assert len(loads) == 2 and all(size > 0 for size in loads)


def test_policy_disagreement_is_rejected_even_with_matching_artifact_hash(trusted_study):
    study, _, _ = trusted_study
    path = study / "decision_policy.json"
    policy = json.loads(path.read_text())
    policy["binary_threshold"] = .123
    path.write_text(json.dumps(policy))
    manifest = json.loads((study / "manifest.json").read_text())
    manifest["files_sha256"]["decision_policy.json"] = v2.digest(path)
    (study / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="records disagree"):
        explanations.read_study(study)


def test_plot_export_keeps_existing_caller_figures(source_paths, tmp_path):
    from matplotlib import pyplot as plt
    model, frame = cheap_model(source_paths, "binary", "LightGBM")
    result = explanations.explain_pipeline(model, frame.iloc[:8], frame.iloc[10:13], "binary", 1)
    existing = plt.figure()
    existing.set_size_inches(4, 3)
    caller_axis = existing.add_subplot()
    caller_axis.plot([1, 2], [3, 4])
    caller_axis.set_title("Caller-owned figure")
    original_size = existing.get_size_inches().copy()
    try:
        explanations._save_plots(tmp_path, "binary", result, frame.iloc[10:13])
        assert existing.number in plt.get_fignums()
        assert len(plt.get_fignums()) == 1
        np.testing.assert_array_equal(existing.get_size_inches(), original_size)
        assert existing.axes == [caller_axis]
        assert caller_axis.get_title() == "Caller-owned figure"
        assert len(caller_axis.lines) == 1
    finally:
        plt.close(existing)


def test_render_failure_does_not_write_a_completion_manifest(monkeypatch, trusted_study, tmp_path, small_protocol):
    study, train, test = trusted_study
    def fail(*args): raise RuntimeError("fixture rendering failure")
    monkeypatch.setattr(explanations, "_save_plots", fail)
    output = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="fixture rendering"):
        explanations.build_explanations(study, train, test, output, ROOT, small_protocol, trust_models=True)
    assert output.exists() and not (output / "manifest.json").exists()


def test_cli_help_is_read_only():
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/explain_models.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--trust-local-models" in result.stdout and "--background-rows" in result.stdout
