# SHAP Explainability v1

This opt-in workflow explains the exact selected models in a completed v3 study.
It never refits models, changes thresholds, promotes a demo model, modifies a
notebook, or writes inside the source study. It currently supports the NSL-KDD
reference workflow on the modern-upgrade branch, not a completed newer-data model.

## Meaning

SHAP assigns a contribution to each encoded model input relative to a declared
background. The baseline plus signed contributions must reconstruct the model's
output on each explained row. Supported families and output scales are explicit:

| Selected model | Explainer | Scale |
|---|---|---|
| LightGBM binary / histogram boosting binary | TreeExplainer | Log-odds for `classes_[1]`, not probability points |
| LightGBM / histogram boosting multiclass | TreeExplainer | Raw class margins in the estimator's class order |
| Random Forest | TreeExplainer | Class probabilities, including both binary classes |
| Binary logistic regression | LinearExplainer | Log-odds for `classes_[1]` |

Unsupported winners, including the prior dummy baseline, stop explicitly. A
different model or generic approximate fallback is not substituted. Binary
log-odds are checked through the sigmoid link; multiclass margins through softmax.
Probability outputs are checked against `predict_proba`. A multiclass margin is
not the one-versus-rest log-odds of that class. Probabilities from class-weighted
models are not automatically calibrated security-risk estimates.

Tree SHAP uses explicit interventional perturbation with a fixed training
background. Linear SHAP uses an explicit independent masker. These break feature
dependencies for attribution; they do not identify causal interventions in network
traffic. Masking one-hot columns independently may create impossible category
combinations. Grouping their contributions afterward does not undo that assumption.

## Sampling and Feature Alignment

The background is a uniform sample without replacement from the original
training input whose SHA-256 matches the study. Default: 64 rows, seed 42.
Examples are independently sampled from a named recorded input. Default: 32
historical test rows, seed 43. Positions are sorted and stored. Samples are not
chosen by their labels, interesting explanations, or test scores. Rows need not
include rare attacks; this sample is not a new performance evaluation.

Bounds are 1-100 background rows and 1-128 explanation rows, with at most 16
workers. Execution parallelism is overridden on a shallow estimator wrapper;
the caller's fitted object, learned weights, and saved file are not modified.
Small datasets use all available rows when fewer than requested.
The fitted preprocessor is transformed only, never fitted again. Labels and
difficulty never become explanation features or background predictors.

Feature mapping comes from the fitted numeric/categorical transformer structure
and one-hot category counts, not fragile string-prefix matching. It currently
requires the existing full one-hot NSL-KDD pipeline, with no category dropping or
frequency grouping. Unknown categories retain the pipeline's existing behavior.
Future dataset schemas need an explicit new adapter, not guessed column mappings.

Signed contributions for all encodings of a categorical input are summed back
to its original feature. This yields 41 readable grouped contributions per row
and output. These sums preserve reconstruction, but are not newly computed
group/Owen SHAP values. Global importance is the mean absolute value of these
**signed grouped sums**, not the sum of individual encoding magnitudes. It is
conditional on the selected rows and background, not a universal feature ranking.

## Independent Checks

- Model class order determines SHAP output labels and probability columns.
- SHAP output axes, transformed feature axes, and base values are checked.
- Baseline plus both encoded and grouped contributions reconstruct actual model
  outputs with `atol=1e-5` and `rtol=1e-5`. Values are not rescaled to make them fit.
- Raw-score links and full-pipeline versus transformed-estimator predictions agree.
- All outputs must be finite. Unsupported or mismatched layouts stop the export.
- The explicitly selected background must not be silently downsampled by SHAP.

The explanation does not modify the frozen binary threshold. Its predictions
use the study's saved `>=` rule, which can differ from the bare estimator's
default `predict()` result. The waterfall explains a score, not the threshold
choice or the truth of an attack label.

## Safe Use

In the study's recorded Python major/minor and numerical-package environment:

```bash
python -m pip install -r requirements-explainability.txt
python -B scripts/explain_models.py --trust-local-models
```

Only use the trust flag for your own trusted study. Joblib/pickle files can
execute code on load. File hashes detect changes; they do not make an untrusted
model author safe. Do not expose this loader to public model uploads.

The study's completion manifest and every listed artifact are checked before
model loading, including resolved-path confinement. Verified model bytes are
loaded from memory rather than reopening a path after verification. Recorded
training/example hashes and numerical dependency versions must match. Source and
study integrity are rechecked before writing. Hashes are not signatures or proof
of a scientifically correct model.

SHAP 0.54.1 requires NumPy 2 and a compatible Python environment. If installing it
would change a frozen study's numerical versions, do not silently upgrade that
study's environment or ignore the version check. Use a separately reviewed
compatible explainer environment/protocol instead. This implementation is tested
on the recorded Python 3.12 v3 environment.

To explain the harder subset in a fresh destination:

```bash
python -B scripts/explain_models.py --trust-local-models \
  --dataset harder_test --examples Dataset/raw/KDDTest-21.txt \
  --output-dir runs/explanations_v1_harder
```

## Artifacts

A new ignored `runs/explanations_v1/` directory contains:

- Selected raw background/example feature rows and their original source positions.
- Per-task compressed NPZ arrays: encoded/grouped contributions, base values,
  model outputs, aligned names/owners, and explained output labels. Load with
  `numpy.load(..., allow_pickle=False)`; no explainer pickle is exported.
- Per-class global importance CSVs and sampled predictions using the frozen policy.
- Native SHAP global bar plots and a waterfall for the first sampled row per
  output, with explicit units. Plot filenames use output-axis numbers; metadata
  identifies the corresponding actual class.
- Diagnostics with output units, class order, reconstruction errors, and warnings.
- Archived implementation, original study manifest, exact top-level dependencies,
  input/model hashes, selected positions, and a completion manifest written last.

Existing/protected destinations are refused, including nested study directories
and symlink aliases. A rendering or write failure can leave a partial directory;
without the completion manifest it is not a valid export. Use a new path instead
of overwriting an old package. No original assets or default app behavior change.

Dashboard integration is a later stage. Generating these artifacts does not add
an explainer page to the currently deployed or legacy Streamlit app.

References: [TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html),
[LinearExplainer](https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html),
[Explanation arrays](https://shap.readthedocs.io/en/latest/generated/shap.Explanation.html).
