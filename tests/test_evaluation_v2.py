from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning

from src.config import ALL_COLUMNS, ATTACK_CATEGORY_ORDER, FEATURE_COLUMNS, NUMERIC_FEATURES
import src.evaluation_v2 as evaluation


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cheap_fits(monkeypatch):
    original = evaluation.candidate_models
    def candidates(task, protocol):
        template = original(task, protocol)["Prior baseline"]
        return {"A prior": clone(template),
                "B uniform": clone(template).set_params(model=DummyClassifier(strategy="uniform", random_state=42))}
    monkeypatch.setattr(evaluation, "candidate_models", candidates)


@pytest.mark.parametrize("changes", [{"folds": 1}, {"folds": True}, {"seed": -1}, {"seed": 1.5},
                                      {"workers": 0}, {"workers": True}, {"binary_threshold": .4}, {"seed": 2**32}])
def test_invalid_protocols_cannot_silently_change_the_study(changes):
    with pytest.raises(ValueError):
        replace(evaluation.EvaluationProtocol(), **changes)


def test_source_loader_preserves_source_bytes_and_uses_only_connection_features(source_paths):
    train, _ = source_paths
    before = train.read_bytes()
    frame = evaluation.load_evaluation_data(train)
    assert len(frame) == 75 and len(FEATURE_COLUMNS) == 41
    assert set(frame.attack_category) == set(ATTACK_CATEGORY_ORDER)
    assert frame.binary_label.equals(frame.attack_type.ne("normal").astype(int).rename("binary_label"))
    assert train.read_bytes() == before


@pytest.mark.parametrize("column,value", [("duration", -1), ("src_bytes", np.inf), ("serror_rate", 1.1),
    ("service", ""), ("service", " http"), ("attack_type", "unknown"), ("difficulty", 22),
    ("difficulty", 2.5), ("duration", np.nan)])
def test_malformed_source_values_are_rejected(tmp_path, raw_frame, column, value):
    raw_frame[column] = raw_frame[column].astype(object)
    raw_frame.loc[0, column] = value
    path = tmp_path / "bad.txt"
    raw_frame.to_csv(path, header=False, index=False)
    with pytest.raises(ValueError):
        evaluation.load_evaluation_data(path)


def test_incorrect_column_count_fails_instead_of_shifting_labels(tmp_path, raw_frame):
    path = tmp_path / "bad.txt"
    raw_frame.drop(columns="difficulty").to_csv(path, header=False, index=False)
    with pytest.raises(ValueError, match="43-column"):
        evaluation.load_evaluation_data(path)


def test_feature_duplicates_are_grouped_even_with_conflicting_labels(source_paths):
    training = evaluation.load_evaluation_data(source_paths[0])
    duplicate = training.iloc[[0]].copy()
    duplicate["attack_type"], duplicate["attack_category"], duplicate["binary_label"] = "neptune", "dos", 1
    training = pd.concat([training, duplicate], ignore_index=True)
    before = training.copy(deep=True)
    fit, validation, membership = evaluation.training_partition(training, evaluation.EvaluationProtocol())
    assert set(fit).isdisjoint(validation)
    assert len(fit) + len(validation) == len(training)
    assert membership.loc[0, "partition"] == membership.loc[len(training) - 1, "partition"]
    assert not set(membership.loc[fit, "feature_group"]) & set(membership.loc[validation, "feature_group"])
    for subset in (fit, validation):
        assert set(training.iloc[subset].attack_category) == set(ATTACK_CATEGORY_ORDER)
    pd.testing.assert_frame_equal(training, before)
    again = evaluation.training_partition(training, evaluation.EvaluationProtocol())
    np.testing.assert_array_equal(fit, again[0])
    np.testing.assert_array_equal(validation, again[1])


def test_feature_grouping_excludes_target_and_difficulty_metadata(source_paths):
    frame = evaluation.load_evaluation_data(source_paths[0])
    before = evaluation.feature_groups(frame)
    frame["difficulty"] = 0
    frame["attack_type"] = "normal"
    frame["attack_category"] = "normal"
    frame["binary_label"] = 0
    pd.testing.assert_series_equal(before, evaluation.feature_groups(frame))


def test_equivalent_numeric_representations_do_not_split_feature_groups(source_paths):
    frame = evaluation.load_evaluation_data(source_paths[0]).iloc[[0]].copy()
    duplicate = frame.copy()
    duplicate["num_outbound_cmds"] = -0.0
    combined = pd.concat([frame, duplicate], ignore_index=True)
    assert evaluation.feature_groups(combined).nunique() == 1
    integer = frame.copy()
    integer["src_bytes"] = integer.src_bytes.astype(int)
    assert evaluation.feature_groups(integer).item() == evaluation.feature_groups(frame).item()


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_real_classifier_families_fit_dense_features_and_accept_new_categories(source_paths, task):
    training = evaluation.load_evaluation_data(source_paths[0])
    test = evaluation.load_evaluation_data(source_paths[1])
    test["service"] = "UNSEEN_TEST_SERVICE"
    protocol = evaluation.EvaluationProtocol(workers=1)
    for model in evaluation.candidate_models(task, protocol).values():
        params = model.get_params()
        if "model__n_estimators" in params:
            model.set_params(model__n_estimators=5)
        if "model__max_iter" in params and model.named_steps["model"].__class__.__name__ == "HistGradientBoostingClassifier":
            model.set_params(model__max_iter=5)
        fitted, _, converged, _ = evaluation._fit(model, training[FEATURE_COLUMNS], training[evaluation.TASKS[task][0]], 1)
        assert converged
        assert isinstance(fitted.named_steps["preprocessor"].transform(training[FEATURE_COLUMNS]), np.ndarray)
        _, predictions = evaluation.score_model(fitted, test, task, protocol)
        assert len(predictions) == len(test)


def test_rare_group_support_is_checked_before_splitting(source_paths):
    training = evaluation.load_evaluation_data(source_paths[0])
    small = training.loc[training.attack_category.ne("u2r") | training.index.isin(training.loc[training.attack_category.eq("u2r")].index[:2])]
    with pytest.raises(ValueError, match="distinct feature group"):
        evaluation.training_partition(small, evaluation.EvaluationProtocol())


def test_preprocessing_fits_only_the_training_partition(monkeypatch, source_paths, cheap_fits):
    training = evaluation.load_evaluation_data(source_paths[0])
    protocol = evaluation.EvaluationProtocol()
    fit, validation, membership = evaluation.training_partition(training, protocol)
    training.loc[training.index[validation], "service"] = "VALIDATION_ONLY"
    monkeypatch.setattr(evaluation, "training_partition", lambda frame, config: (fit, validation, membership))
    original = evaluation._fit
    checked = []
    def spy(model, X, y, workers):
        result = original(model, X, y, workers)
        preprocessor = result[0].named_steps["preprocessor"]
        np.testing.assert_allclose(preprocessor.named_transformers_["numeric"].mean_, training.iloc[fit][NUMERIC_FEATURES].mean())
        assert "VALIDATION_ONLY" not in preprocessor.named_transformers_["categorical"].categories_[1]
        assert list(X.columns) == FEATURE_COLUMNS
        checked.append(len(X))
        return result
    monkeypatch.setattr(evaluation, "_fit", spy)
    evaluation.select_models(training, protocol)
    assert checked == [len(fit)] * 4


@pytest.mark.parametrize("task", ["binary", "multiclass"])
def test_probability_columns_follow_classes_not_position(source_paths, task):
    frame = evaluation.load_evaluation_data(source_paths[1])
    labels = evaluation.TASKS[task][1]
    class ReversedClasses:
        classes_ = np.asarray(labels[::-1])
        def predict_proba(self, X):
            values = np.full(len(labels), .2 / (len(labels) - 1))
            values[0] = .8
            return np.tile(values, (len(X), 1))
    metrics, predictions = evaluation.score_model(ReversedClasses(), frame, task, evaluation.EvaluationProtocol())
    assert predictions.prediction.eq(labels[-1]).all()
    np.testing.assert_allclose(predictions[f"p_{labels[-1]}"], .8)
    assert metrics["labels"] == labels
    assert np.asarray(metrics["confusion_matrix"]).shape == (len(labels), len(labels))


def test_multiclass_ties_follow_estimator_order_not_reporting_axis(source_paths):
    frame = evaluation.load_evaluation_data(source_paths[1])
    class TiedClassifier:
        classes_ = np.asarray(["dos", "normal", "probe", "r2l", "u2r"])
        def predict_proba(self, X):
            return np.tile([.5, .5, 0., 0., 0.], (len(X), 1))
    metrics, predictions = evaluation.score_model(TiedClassifier(), frame, "multiclass", evaluation.EvaluationProtocol())
    assert metrics["labels"][0] == "normal"
    assert predictions.prediction.eq("dos").all()


def test_csv_export_can_preserve_decision_boundary_probabilities_exactly(tmp_path):
    values = np.asarray([np.nextafter(.5, 0), .5, np.nextafter(.5, 1)])
    path = tmp_path / "probabilities.csv"
    pd.DataFrame({"p": values}).to_csv(path, index=False, float_format="%.17g")
    restored = pd.read_csv(path, float_precision="round_trip").p.to_numpy()
    np.testing.assert_array_equal(restored, values)
    np.testing.assert_array_equal(restored >= .5, [False, True, True])


@pytest.mark.parametrize("fault", ["nan", "negative", "sum", "shape", "classes"])
def test_invalid_probability_outputs_fail_explicitly(source_paths, fault):
    frame = evaluation.load_evaluation_data(source_paths[1])
    class BadClassifier:
        classes_ = [0] if fault == "classes" else [0, 1]
        def predict_proba(self, X):
            values = np.tile([.4, .6], (len(X), 1))
            if fault == "nan": values[0, 0] = np.nan
            elif fault == "negative": values[0] = [-.1, 1.1]
            elif fault == "sum": values[0] = [.4, .4]
            elif fault == "shape": values = values[:-1]
            return values
    with pytest.raises(ValueError, match="classes|probabilities"):
        evaluation.score_model(BadClassifier(), frame, "binary", evaluation.EvaluationProtocol())


def test_missing_test_classes_keep_axes_and_unscored_auc(source_paths):
    training = evaluation.load_evaluation_data(source_paths[0])
    normal = training.loc[training.binary_label.eq(0)]
    model = evaluation.candidate_models("binary", evaluation.EvaluationProtocol())["Prior baseline"]
    model.fit(training[FEATURE_COLUMNS], training.binary_label)
    metrics, _ = evaluation.score_model(model, normal, "binary", evaluation.EvaluationProtocol())
    assert metrics["roc_auc"] is None and metrics["average_precision"] is None
    assert len(metrics["confusion_matrix"]) == 2
    model = evaluation.candidate_models("multiclass", evaluation.EvaluationProtocol())["Prior baseline"]
    model.fit(training[FEATURE_COLUMNS], training.attack_category)
    metrics, _ = evaluation.score_model(model, normal, "multiclass", evaluation.EvaluationProtocol())
    assert len(metrics["confusion_matrix"]) == 5
    assert metrics["classification_report"]["u2r"]["support"] == 0


def test_convergence_warnings_are_recorded_by_the_fitter(source_paths):
    frame = evaluation.load_evaluation_data(source_paths[0])
    class Warns(DummyClassifier):
        def fit(self, X, y, sample_weight=None):
            warnings.warn("fixture convergence failure", ConvergenceWarning)
            return super().fit(X, y, sample_weight=sample_weight)
    _, notes, converged, _ = evaluation._fit(Warns(), frame[FEATURE_COLUMNS], frame.binary_label, 1)
    assert not converged
    assert notes[0]["category"] == "ConvergenceWarning"


def test_nonconverged_candidate_cannot_win_selection(monkeypatch, source_paths, cheap_fits):
    original = evaluation._fit
    def fit(model, *args):
        fitted, notes, _, seconds = original(model, *args)
        return fitted, notes, model.named_steps["model"].strategy != "prior", seconds
    monkeypatch.setattr(evaluation, "_fit", fit)
    _, reports, _, _ = evaluation.select_models(evaluation.load_evaluation_data(source_paths[0]), evaluation.EvaluationProtocol())
    assert {report["selected_model"] for report in reports.values()} == {"B uniform"}


def test_selection_ties_use_names_not_factory_order(monkeypatch, source_paths):
    original = evaluation.candidate_models
    def candidates(task, protocol):
        template = original(task, protocol)["Prior baseline"]
        return {"Z prior": clone(template), "A prior": clone(template)}
    monkeypatch.setattr(evaluation, "candidate_models", candidates)
    _, reports, _, _ = evaluation.select_models(evaluation.load_evaluation_data(source_paths[0]), evaluation.EvaluationProtocol())
    assert {report["selected_model"] for report in reports.values()} == {"A prior"}


def test_selection_and_refits_finish_before_official_test_is_loaded(monkeypatch, tmp_path, source_paths, cheap_fits):
    train, test = source_paths
    original_load, original_fit, original_select = evaluation.load_evaluation_data, evaluation._fit, evaluation.select_models
    state = {"selected": False, "refits": 0}
    train_rows = len(original_load(train))
    def select(*args):
        result = original_select(*args)
        state["selected"] = True
        return result
    def fit(model, X, y, workers):
        if len(X) == train_rows:
            assert state["selected"]
            state["refits"] += 1
        return original_fit(model, X, y, workers)
    def load(path):
        if path == test:
            assert state == {"selected": True, "refits": 2}
        return original_load(path)
    monkeypatch.setattr(evaluation, "select_models", select)
    monkeypatch.setattr(evaluation, "_fit", fit)
    monkeypatch.setattr(evaluation, "load_evaluation_data", load)
    evaluation.build_evaluation(train, test, tmp_path / "study", ROOT)


def test_changing_test_labels_cannot_change_selection(monkeypatch, tmp_path, source_paths, cheap_fits):
    train, test = source_paths
    first, second = tmp_path / "first", tmp_path / "second"
    evaluation.build_evaluation(train, test, first, ROOT)
    frame = pd.read_csv(test, header=None, names=ALL_COLUMNS)
    frame["attack_type"] = "normal"
    frame.to_csv(test, header=False, index=False)
    evaluation.build_evaluation(train, test, second, ROOT)
    a, b = (json.loads((path / "selection.json").read_text()) for path in (first, second))
    for task in evaluation.TASKS:
        assert a[task]["selected_model"] == b[task]["selected_model"]
        assert [candidate["metrics"] for candidate in a[task]["candidates"]] == [candidate["metrics"] for candidate in b[task]["candidates"]]


def test_write_once_package_has_valid_hashes_and_saved_models_match_metrics(tmp_path, source_paths, cheap_fits):
    output = tmp_path / "study"
    training = pd.read_csv(source_paths[0], header=None, names=ALL_COLUMNS)
    extra = training.iloc[[15]].copy()
    extra["src_bytes"] += 100000
    pd.concat([training, extra], ignore_index=True).to_csv(source_paths[0], header=False, index=False)
    evaluation.build_evaluation(*source_paths, output, ROOT)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["evaluation_version"] == 2
    assert manifest["selection"]["binary"]["primary_metric"] == "f1"
    assert manifest["selection"]["multiclass"]["primary_metric"] == "macro_f1"
    for name, expected in manifest["files_sha256"].items():
        assert evaluation.digest(output / name) == expected
    for name, expected in manifest["source_code_sha256"].items():
        assert evaluation.digest(output / "source_code" / name) == expected
    test = evaluation.load_evaluation_data(source_paths[1])
    saved = json.loads((output / "historical_test_metrics.json").read_text())
    for task in evaluation.TASKS:
        model = joblib.load(output / "models" / f"{task}.joblib")
        metrics, predictions = evaluation.score_model(model, test, task, evaluation.EvaluationProtocol())
        assert metrics == saved[task]["metrics"]
        restored = pd.read_csv(output / f"historical_test_{task}_predictions.csv", float_precision="round_trip")
        for column in (name for name in predictions if name.startswith("p_")):
            np.testing.assert_array_equal(restored[column], predictions[column])
    before = (output / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        evaluation.build_evaluation(*source_paths, output, ROOT)
    assert (output / "manifest.json").read_bytes() == before
    result = subprocess.run([sys.executable, "-B", str(output / "source_code/scripts/evaluate_models_v2.py"), "--help"],
                            cwd=ROOT, capture_output=True, text=True, check=True)
    assert "--output-dir" in result.stdout


@pytest.mark.parametrize("relative", ["Models/new", "reports/new", "notebooks/new", "Dataset/new", "src/new", "app/new", ".github/new", ".git/new"])
def test_protected_output_paths_fail_before_fitting(monkeypatch, relative):
    monkeypatch.setattr(evaluation, "load_evaluation_data", lambda *args: pytest.fail("Must refuse before loading inputs"))
    with pytest.raises(ValueError, match="separate"):
        evaluation.build_evaluation(ROOT / "Dataset/raw/KDDTrain+.txt", ROOT / "Dataset/raw/KDDTest+.txt", ROOT / relative, ROOT)


def test_symlink_and_existing_study_destinations_are_refused(tmp_path):
    legacy = tmp_path / "repo/Models"
    legacy.mkdir(parents=True)
    alias = tmp_path / "alias"
    alias.symlink_to(legacy, target_is_directory=True)
    with pytest.raises(ValueError, match="separate"):
        evaluation.validate_destination(alias / "new", legacy.parent, ())
    study = tmp_path / "study"
    study.mkdir()
    (study / "manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="existing evaluation"):
        evaluation.validate_destination(study / "nested", ROOT, ())


def test_fit_failure_writes_no_study(monkeypatch, tmp_path, source_paths):
    def fail(*args):
        raise RuntimeError("fixture fit failure")
    monkeypatch.setattr(evaluation, "select_models", fail)
    with pytest.raises(RuntimeError, match="fixture"):
        evaluation.build_evaluation(*source_paths, tmp_path / "never-created", ROOT)
    assert not (tmp_path / "never-created").exists()


def test_nonconverged_refit_stops_before_test_data_or_output(monkeypatch, tmp_path, source_paths, cheap_fits):
    original_fit, original_load = evaluation._fit, evaluation.load_evaluation_data
    size = len(original_load(source_paths[0]))
    def fit(model, X, *args):
        fitted, notes, converged, seconds = original_fit(model, X, *args)
        return fitted, notes, converged and len(X) != size, seconds
    def load(path):
        if path == source_paths[1]:
            pytest.fail("An unsuccessful refit must not reach test evaluation")
        return original_load(path)
    monkeypatch.setattr(evaluation, "_fit", fit)
    monkeypatch.setattr(evaluation, "load_evaluation_data", load)
    with pytest.raises(ValueError, match="did not converge on refit"):
        evaluation.build_evaluation(*source_paths, tmp_path / "never-created", ROOT)
    assert not (tmp_path / "never-created").exists()


def test_identical_input_files_are_not_treated_as_a_holdout(tmp_path, source_paths):
    with pytest.raises(ValueError, match="must not be identical"):
        evaluation.build_evaluation(source_paths[0], source_paths[0], tmp_path / "never-created", ROOT)


def test_source_changing_between_hash_and_parse_is_rejected(monkeypatch, tmp_path, source_paths):
    original = evaluation.load_evaluation_data
    def changed(path):
        path.write_bytes(path.read_bytes() + b"\n")
        return original(path)
    monkeypatch.setattr(evaluation, "load_evaluation_data", changed)
    with pytest.raises(ValueError, match="changed while loading"):
        evaluation.build_evaluation(*source_paths, tmp_path / "never-created", ROOT)
    assert not (tmp_path / "never-created").exists()


def test_changed_input_prevents_a_completed_study(monkeypatch, tmp_path, source_paths, cheap_fits):
    original = evaluation.select_models
    def revise(*args):
        result = original(*args)
        source_paths[0].write_bytes(source_paths[0].read_bytes() + b"\n")
        return result
    monkeypatch.setattr(evaluation, "select_models", revise)
    with pytest.raises(ValueError, match="changed during"):
        evaluation.build_evaluation(*source_paths, tmp_path / "never-created", ROOT)
    assert not (tmp_path / "never-created").exists()
