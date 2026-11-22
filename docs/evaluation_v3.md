# Evaluation v3: thresholds, LightGBM, and the harder test subset

This is an opt-in, write-once experiment. Original models, reports, v2 code,
notebooks and outputs, and the Streamlit default are not replaced.

## Protocol

V3 reuses v2's validated loader, 41-feature preprocessing, first stratified-group
validation fold, seed 42, convergence handling, class alignment, and output-path
protection. Identical predictor vectors stay together regardless of their labels.

Each candidate is fitted only on the training partition. For binary candidates,
scikit-learn's precision-recall curve supplies thresholds from validation
probabilities. The fixed 0.5 policy is included. Selection maximizes attack F1;
numerical ties within 1e-14 prefer the threshold nearest 0.5, then the higher
threshold. The best model uses that validation F1, with alphabetical model-name
ties. Multiclass selection remains macro-F1 over the fixed five categories.

Threshold and model selection share the same held-out training partition. Its
scores are selection diagnostics, not unbiased post-selection performance.
No probability calibration, false-positive-rate constraint, cost optimization,
or claim of an operationally optimal security threshold is made. A single split
has limited rare-class support. Thresholds may transfer poorly after full-training
refit or under distribution shift. Grouped nested validation is future work.

Both winners are refitted on all official training data, with the chosen binary
threshold frozen, before either test dataset is loaded. The same fitted models
and threshold score both benchmarks. The saved default-0.5 comparison does not
change the choice after viewing test results.

`--lightgbm` adds a fixed LightGBM candidate for both tasks: 300 estimators,
learning rate 0.05, 31 leaves, minimum 20 child samples, balanced class weights,
two workers by default, seed 42, deterministic CPU fitting and column-wise
histograms. No test-driven early stopping or parameter search is used. LightGBM
uses the existing fitted-on-training dense preprocessing pipeline. If requested
but unavailable, it fails explicitly instead of silently omitting the candidate.
No version is assumed to outperform simpler candidates before evaluation.

## KDDTest-21

`--hard-test` verifies the exact multiset of all 43 source columns against full
test rows with difficulty below 21, including duplicate multiplicities. Rows may
be reordered, but modified, missing, or extra rows fail. Difficulty is never a
predictor. This is a harder, dependent subset of KDDTest+, not a new dataset or
an independent holdout. No thresholds or model choices are updated using it.
Its rows must not be pooled with full-test rows to manufacture extra sample size.

Historical test data were inspected during the original notebook work. V3 is
retrospective methodological improvement, not newly independent evidence or a
validated modern-traffic IDS. The existing attack taxonomy is retained unchanged.

## Run

From the project root, using Python 3.12 in an isolated environment:

```bash
python -m pip install -r requirements-upgrade.txt
python scripts/evaluate_models_v3.py --lightgbm \
  --hard-test Dataset/raw/KDDTest-21.txt
```

On macOS LightGBM may also need an OpenMP runtime; consult its official
installation guide. Omitting `--lightgbm` uses the original candidate families.
Omitting `--hard-test` evaluates only the full benchmark and is recorded as such.
For another run, choose a fresh `--output-dir runs/evaluation_v3-repeat`.

Models, selection scores, threshold curves, validation/test predictions,
confusion matrices, per-class metrics, overlap audits, source/input hashes,
decision metadata, exact dependencies, and archived implementation are saved in
the new ignored run directory. A completion manifest is written last. Partial
directories without it are not completed studies. Existing paths are refused.

The files named `*_estimator.joblib` are bare fitted pipelines. For binary
predictions use `predict_proba`, identify class 1 through `classes_`, and apply
`decision_policy.json`'s threshold using `>=`. Calling the bare estimator's
`predict()` does not apply the saved custom threshold. Only load trusted model
files: joblib/pickle artifacts can execute code during deserialization.

## Tests and next stages

```bash
python -B -m pytest tests -p no:cacheprovider
```

Synthetic regression tests cover threshold ties and boundaries, invalid
probabilities, class order, test-load timing, exact harder-subset membership,
artifact hashes, and preservation. Real LightGBM smoke tests use tiny data, not
a full benchmark in CI.

Separate future stages: model-appropriate SHAP explanations with explicit output
units/background data; a richer dashboard and deployment-safe log uploads;
and a distinct modern-dataset protocol. UNSW-NB15 or CIC-IDS2017 require their
own feature schema, provenance, split/leakage audit, model fits, and evaluation.
They cannot be passed directly to an NSL-KDD-trained model.

References: [threshold tuning](https://scikit-learn.org/stable/modules/classification_threshold.html),
[LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html),
[LightGBM installation](https://lightgbm.readthedocs.io/en/stable/Installation-Guide.html),
[NSL-KDD dataset description](https://www.unb.ca/cic/datasets/nsl.html).
