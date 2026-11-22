from dataclasses import replace
import builtins
import json
from pathlib import Path
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score

from src.config import ALL_COLUMNS, FEATURE_COLUMNS
from src import evaluation_v2 as v2
from src import evaluation_v3 as evaluation


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cheap_candidates(monkeypatch):
    def candidates(task, protocol):
        template = v2.candidate_models(task, protocol)["Prior baseline"]
        return {"A prior": clone(template), "B uniform": clone(template).set_params(
            model=DummyClassifier(strategy="uniform", random_state=42))}
    monkeypatch.setattr(evaluation, "candidate_models", candidates)


@pytest.fixture
def benchmark_paths(source_paths):
    train, test = source_paths
    frame = pd.read_csv(test, header=None, names=ALL_COLUMNS)
    frame.loc[frame.index % 2 == 0, "difficulty"] = 20
    frame.to_csv(test, header=False, index=False)
    harder = test.with_name("harder.txt")
    frame.loc[frame.difficulty.lt(21)].iloc[::-1].to_csv(harder, header=False, index=False)
    return train, test, harder


@pytest.mark.parametrize("changes", [{"folds": 1}, {"workers": True}, {"seed": -1}, {"include_lightgbm": 1}])
def test_invalid_protocol(changes):
    with pytest.raises(ValueError):
        replace(evaluation.EvaluationProtocol(), **changes)


@pytest.mark.parametrize("actual,probability", [([0, 0], [.1, .2]), ([1, 1], [.1, .2]),
    ([0, 1], [.1]), ([0, 1], [np.nan, .2]), ([0, 1], [-.1, .2]),
    ([0, 1], [.1, 1.1]), ([[0, 1]], [[.1, .2]]), ([0, 2], [.1, .2])])
def test_threshold_rejects_invalid_or_single_class_inputs(actual, probability):
    with pytest.raises(ValueError, match="Threshold tuning"):
        evaluation.tune_binary_threshold(actual, probability)


def test_threshold_changes_only_decision_policy_and_uses_inclusive_boundary():
    y, p = np.asarray([0, 0, 1, 1]), np.asarray([.1, .2, .3, .4])
    before = p.copy()
    threshold, curve = evaluation.tune_binary_threshold(y, p)
    assert threshold == .3
    np.testing.assert_array_equal(p, before)
    np.testing.assert_array_equal(p >= threshold, y)
    assert curve.selected.sum() == 1
    assert curve.loc[curve.selected, "f1"].item() == 1
    assert .5 in set(curve.threshold)


def test_default_threshold_wins_equivalent_decision_ties():
    threshold, curve = evaluation.tune_binary_threshold([0, 1, 1], [.1, .7, .9])
    assert threshold == .5
    assert curve.loc[curve.selected, "f1"].item() == 1


def test_floating_boundary_and_serialized_threshold_remain_exact(tmp_path):
    p = np.asarray([np.nextafter(.5, 0), .5, np.nextafter(.5, 1)])
    threshold, curve = evaluation.tune_binary_threshold([0, 1, 1], p)
    assert threshold == .5
    path = tmp_path / "curve.csv"
    curve.to_csv(path, index=False, float_format="%.17g")
    restored = pd.read_csv(path, float_precision="round_trip")
    np.testing.assert_array_equal(restored.threshold, curve.threshold)


def test_precision_recall_search_agrees_with_exhaustive_f1_decisions():
    for seed in range(20):
        rng = np.random.default_rng(seed)
        y = np.r_[0, 1, rng.integers(0, 2, 28)]
        p = rng.choice(np.linspace(0, 1, 11), len(y))
        threshold, curve = evaluation.tune_binary_threshold(y, p)
        scores = np.asarray([f1_score(y, p >= value) for value in curve.threshold])
        np.testing.assert_allclose(scores, curve.f1, atol=1e-14, rtol=0)
        assert f1_score(y, p >= threshold) >= scores.max() - 1e-14


def test_threshold_policy_respects_reversed_class_order(source_paths):
    frame = v2.load_evaluation_data(source_paths[1])
    class Reversed:
        classes_ = np.asarray([1, 0])
        def predict_proba(self, X):
            return np.tile([.3, .7], (len(X), 1))
    _, rows = v2.score_model(Reversed(), frame, "binary", evaluation.PredictionPolicy(1, .3))
    assert rows.prediction.eq(1).all()
    _, rows = v2.score_model(Reversed(), frame, "binary", evaluation.PredictionPolicy(1, .5))
    assert rows.prediction.eq(0).all()


def test_optional_lightgbm_is_not_imported_by_default(monkeypatch):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "lightgbm":
            raise AssertionError("Unexpected optional import")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    assert "LightGBM" not in evaluation.candidate_models("binary", evaluation.EvaluationProtocol())


@pytest.mark.parametrize("error", [ImportError, OSError])
def test_requested_but_unavailable_lightgbm_is_not_silently_skipped(monkeypatch, error):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name == "lightgbm":
            raise error("fixture missing dependency/runtime")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(RuntimeError, match="LightGBM was requested"):
        evaluation.candidate_models("binary", evaluation.EvaluationProtocol(include_lightgbm=True))


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_real_lightgbm_pipeline_fits_and_scores_unseen_categories(source_paths, task):
    pytest.importorskip("lightgbm")
    protocol = evaluation.EvaluationProtocol(workers=1, include_lightgbm=True)
    training, test = (v2.load_evaluation_data(path) for path in source_paths)
    test["service"] = "UNSEEN_SERVICE"
    model = evaluation.candidate_models(task, protocol)["LightGBM"].set_params(model__n_estimators=5)
    fitted, _, converged, _ = v2._fit(model, training[FEATURE_COLUMNS], training[v2.TASKS[task][0]], 1)
    assert converged
    metrics, rows = v2.score_model(fitted, test, task, evaluation.PredictionPolicy(1))
    assert len(rows) == len(test)
    assert len(metrics["confusion_matrix"]) == len(v2.TASKS[task][1])


def test_thresholds_are_selected_only_on_validation_rows(source_paths, cheap_candidates):
    training = v2.load_evaluation_data(source_paths[0])
    protocol = evaluation.EvaluationProtocol()
    _, validation, _ = v2.training_partition(training, protocol)
    _, reports, tables, membership = evaluation.select_models(training, protocol)
    for task, report in reports.items():
        predictions = tables[f"validation_{task}_predictions.csv"]
        assert set(predictions.source_row) == set(validation)
        for candidate in report["candidates"]:
            if task == "binary":
                rows = predictions.loc[predictions.model.eq(candidate["model"])]
                np.testing.assert_array_equal(rows.prediction, rows.p_1 >= candidate["binary_threshold"])
                assert candidate["metrics"]["f1"] >= candidate["default_threshold_metrics"]["f1"] - 1e-14
    assert membership.partition.eq("validation").sum() == len(validation)


@pytest.mark.parametrize("mutation", ["missing", "extra", "feature", "label", "difficulty", "multiplicity"])
def test_harder_subset_requires_exact_rows_and_multiplicities(benchmark_paths, mutation):
    _, test, harder = benchmark_paths
    full, subset = v2.load_evaluation_data(test), v2.load_evaluation_data(harder)
    evaluation.verify_harder_subset(full, subset)
    if mutation == "missing": subset = subset.iloc[1:]
    elif mutation == "extra": subset = pd.concat([subset, full.loc[full.difficulty.eq(21)].iloc[[0]]])
    elif mutation == "feature": subset.loc[subset.index[0], "src_bytes"] += 1
    elif mutation == "label": subset.loc[subset.index[0], "attack_type"] = "neptune"
    elif mutation == "difficulty": subset.loc[subset.index[0], "difficulty"] = 21
    else: subset = pd.concat([subset.iloc[1:], subset.iloc[[1]]])
    with pytest.raises(ValueError, match="exactly match"):
        evaluation.verify_harder_subset(full, subset)


def test_selection_threshold_and_refits_are_fixed_before_either_test_loads(monkeypatch, tmp_path, benchmark_paths, cheap_candidates):
    train, test, harder = benchmark_paths
    original_load, original_fit, original_tune = v2.load_evaluation_data, v2._fit, evaluation.tune_binary_threshold
    train_count = len(original_load(train))
    state = {"refits": 0, "test_loaded": False}
    def fit(model, X, y, workers):
        assert not state["test_loaded"]
        if len(X) == train_count:
            state["refits"] += 1
        return original_fit(model, X, y, workers)
    def tune(y, p):
        assert not state["test_loaded"]
        return original_tune(y, p)
    def load(path):
        if path in (test, harder):
            assert state["refits"] == 2
            state["test_loaded"] = True
        return original_load(path)
    monkeypatch.setattr(v2, "_fit", fit)
    monkeypatch.setattr(v2, "load_evaluation_data", load)
    monkeypatch.setattr(evaluation, "tune_binary_threshold", tune)
    output = evaluation.build_evaluation(train, test, tmp_path / "study", ROOT, harder_test_path=harder)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["evaluation_version"] == 3
    assert set(manifest["input_sha256"]) == {"training", "historical_test", "harder_test"}
    for name, expected in manifest["files_sha256"].items():
        assert v2.digest(output / name) == expected
    selection = json.loads((output / "selection.json").read_text())
    policy = json.loads((output / "decision_policy.json").read_text())
    assert policy["binary_threshold"] == selection["binary"]["binary_threshold"]
    model = joblib.load(output / "models/binary_estimator.joblib")
    for name, path in (("historical_test", test), ("harder_test", harder)):
        frame = original_load(path)
        _, expected = v2.score_model(model, frame, "binary", evaluation.PredictionPolicy(2, policy["binary_threshold"]))
        saved = pd.read_csv(output / f"{name}_binary_predictions.csv", float_precision="round_trip")
        pd.testing.assert_frame_equal(saved, expected, check_dtype=False)
    assert set(manifest["source_code_sha256"]) == set(evaluation.SOURCE_PATHS)


def test_optional_harder_benchmark_is_explicitly_absent(tmp_path, source_paths, cheap_candidates):
    output = evaluation.build_evaluation(*source_paths, tmp_path / "study", ROOT)
    manifest = json.loads((output / "manifest.json").read_text())
    assert "harder_test" not in manifest["input_sha256"]
    assert not list(output.glob("harder_test*"))


def test_failed_subset_validation_writes_no_output(tmp_path, benchmark_paths, cheap_candidates):
    train, test, harder = benchmark_paths
    data = pd.read_csv(harder, header=None)
    data.iloc[1:].to_csv(harder, header=False, index=False)
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="exactly match"):
        evaluation.build_evaluation(train, test, output, ROOT, harder_test_path=harder)
    assert not output.exists()


def test_existing_destination_is_not_overwritten(tmp_path, source_paths, cheap_candidates):
    output = tmp_path / "study"
    output.mkdir()
    (output / "sentinel").write_text("preserve")
    with pytest.raises(FileExistsError):
        evaluation.build_evaluation(*source_paths, output, ROOT)
    assert (output / "sentinel").read_text() == "preserve"


def test_changing_test_labels_cannot_change_threshold_or_model_selection(tmp_path, source_paths, cheap_candidates):
    train, test = source_paths
    first = evaluation.build_evaluation(train, test, tmp_path / "first", ROOT)
    frame = pd.read_csv(test, header=None, names=ALL_COLUMNS)
    frame["attack_type"] = "normal"
    frame.to_csv(test, header=False, index=False)
    second = evaluation.build_evaluation(train, test, tmp_path / "second", ROOT)
    a, b = (json.loads((output / "selection.json").read_text()) for output in (first, second))
    for task in v2.TASKS:
        assert a[task]["selected_model"] == b[task]["selected_model"]
        assert a[task]["binary_threshold"] == b[task]["binary_threshold"]
        assert [row["metrics"] for row in a[task]["candidates"]] == [row["metrics"] for row in b[task]["candidates"]]


def test_nonconverged_refit_stops_before_test_loads(monkeypatch, tmp_path, source_paths, cheap_candidates):
    original_fit, original_load = v2._fit, v2.load_evaluation_data
    train, test = source_paths
    training_rows = len(original_load(train))
    def fit(model, X, y, workers):
        fitted, notes, converged, seconds = original_fit(model, X, y, workers)
        return fitted, notes, converged and len(X) != training_rows, seconds
    def load(path):
        if path == test:
            pytest.fail("Unsuccessful refit must not reach the test labels")
        return original_load(path)
    monkeypatch.setattr(v2, "_fit", fit)
    monkeypatch.setattr(v2, "load_evaluation_data", load)
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="refit did not converge"):
        evaluation.build_evaluation(train, test, output, ROOT)
    assert not output.exists()


@pytest.mark.parametrize("relative", ["Models/new", "reports/new", "notebooks/new", "Dataset/new", "src/new", "app/new"])
def test_protected_outputs_are_rejected_before_input_loading(monkeypatch, relative):
    monkeypatch.setattr(v2, "load_evaluation_data", lambda *args: pytest.fail("Must reject before source loading"))
    with pytest.raises(ValueError, match="separate"):
        evaluation.build_evaluation(ROOT / "Dataset/raw/KDDTrain+.txt", ROOT / "Dataset/raw/KDDTest+.txt", ROOT / relative, ROOT)


def test_data_change_after_selection_prevents_any_study(monkeypatch, tmp_path, source_paths, cheap_candidates):
    original = evaluation.select_models
    def select(*args):
        selected = original(*args)
        source_paths[0].write_bytes(source_paths[0].read_bytes() + b"\n")
        return selected
    monkeypatch.setattr(evaluation, "select_models", select)
    output = tmp_path / "study"
    with pytest.raises(ValueError, match="changed during"):
        evaluation.build_evaluation(*source_paths, output, ROOT)
    assert not output.exists()


def test_v3_cli_help_does_not_fit_or_require_optional_packages():
    result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts/evaluate_models_v3.py"), "--help"],
                            capture_output=True, text=True)
    assert result.returncode == 0
    assert "--hard-test" in result.stdout and "--lightgbm" in result.stdout
