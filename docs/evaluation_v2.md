# Evaluation v2: training-only selection

V2 is a separate, retrospective NSL-KDD benchmark. It does not replace the saved
demo models, historical reports, or notebook outputs. The original notebooks
compared models using official test scores and selected their saved models from
those results. Their existing results remain historical; v2 cannot make previously
inspected data newly unseen or provide independent external validation.

## Fixed protocol

- Input files are the existing 43-column `KDDTrain+.txt` and `KDDTest+.txt`.
- Features are exactly the 41 connection attributes. Attack labels, attack
  categories, binary labels, and difficulty metadata are excluded from predictors.
- The first fold of shuffled five-fold `StratifiedGroupKFold`, seed 42, supplies
  the validation partition, approximately 20% of the official training data.
  The remaining approximately 80% fits candidates. No fold is chosen by its scores.
- Stratification uses the five attack categories. Identical feature vectors are
  grouped using pandas' 64-bit row hashes, regardless of their labels. Hash
  collisions could conservatively merge groups; identical rows cannot be split.
  Every category must appear in both partitions or the run stops explicitly.
- Preprocessing is fitted only inside each model's training pipeline. Numeric
  features are standardized and categoricals are one-hot encoded with unknown
  categories ignored. Dense output is explicit for histogram gradient boosting.
- Binary candidates are a prior baseline, balanced logistic regression, balanced
  random forest (300 trees), and histogram gradient boosting (220 iterations).
  Multiclass candidates are a prior baseline, random forest (350 trees, balanced
  subsampling), and histogram gradient boosting (260 iterations). Boosting uses
  learning rate 0.08. Remaining estimator defaults belong to the recorded sklearn
  version; the implementation is archived for reproducibility.
- Binary selection maximizes validation attack-class F1 at a fixed 0.5 probability
  threshold. Multiclass selection maximizes macro-F1 over the fixed five labels,
  giving rare categories equal weight. Ties use alphabetical model names.
- Fit warnings are recorded. Convergence warnings make a candidate ineligible;
  unexpected failures stop the study. A nonconverged final refit also stops it.
- Both selections are fixed and both winners are refitted on all official training
  data before the official test labels are loaded. Only those refitted winners
  receive historical test scores, and those exact fitted models are saved in v2.
- There is no test-driven threshold tuning, candidate search, data imputation,
  calibration, or automatic model promotion. Parallelism defaults to two workers.

## Artifacts and safety

The write-once study contains split membership and feature-group IDs, every
candidate's validation predictions/metrics/warnings, historical test predictions
and per-class reports for the two winners, newly fitted model files, source/input
hashes, software versions, and an archived implementation. Raw datasets are not
duplicated; supply files matching the recorded hashes for reproduction.

Probability columns are aligned using `classes_`, not an assumed column position.
Multiclass probability ties follow the estimator's original class order; fixed
reporting axes do not change its predictions. Binary ties follow the explicitly
declared `probability >= 0.5` rule. CSV probabilities use 17 significant digits;
use `float_precision="round_trip"` when reading them for exact reconstruction.
Missing historical test classes retain fixed confusion-matrix axes. Undefined
binary AUC/AP scores are explicit null values, not fabricated scores. The data
audit discloses rare-class support, duplicated training feature rows, train/test
feature overlaps, and attack types seen only in the historical test file. Those
post-selection audits do not change model choices or remove test rows.

Only a new output directory is accepted. Existing destinations, prior-study
subdirectories, and legacy models, reports, notebooks, data, or source directories
are refused, including through resolved symlinks. All fits and integrity checks
complete before a destination is reserved. The completion manifest is written
last; a directory without it is an incomplete run and must not be reported as a
completed study. Existing files are never overwritten.

## Run locally

Use an isolated Python 3.12 environment, or the existing project environment:

```bash
python -m pip install -r requirements-evaluation.txt
python scripts/evaluate_models_v2.py
```

Outputs default to `runs/evaluation_v2/`, which is ignored by Git. Subsequent
studies require a new path:

```bash
python scripts/evaluate_models_v2.py --output-dir runs/evaluation_v2-repeat
```

Use the exact interpreter/package versions in a completed manifest for numerical
reproduction. Fit durations are diagnostics, not reproducible results. To run the
archived implementation, pass explicit input files and a new output directory:

```bash
python runs/evaluation_v2/source_code/scripts/evaluate_models_v2.py \
  --train Dataset/raw/KDDTrain+.txt --test Dataset/raw/KDDTest+.txt \
  --output-dir runs/evaluation_v2-archived
```

## Tests

```bash
python -B -m pytest tests -p no:cacheprovider
```

Tests use small synthetic NSL-KDD-shaped inputs rather than retraining the full
benchmark. They cover preprocessing leakage, duplicate grouping, validation-only
selection, fixed labels, probability alignment, source validation, artifact
integrity, and output protection. Full benchmark runs are separate local checks.

## Interpretation

V2 is post-hoc methodological repair, not evidence that the historical test set
became independent again. Rare U2R validation support is small, so model rankings
can be unstable. A training/validation split does not eliminate the benchmark's
age or distribution shift. There are no uncertainty or clinical/operational
guarantees. `KDDTest-21` overlaps the official test benchmark and must not be
presented as a new independent dataset. The access-log monitor remains a separate,
rule-based demonstration, not a live application of the NSL-KDD model.

References: [model selection and test-set reuse](https://scikit-learn.org/stable/modules/cross_validation.html),
[stratified group splitting](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).
