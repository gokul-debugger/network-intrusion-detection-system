# V3 Local Benchmark Check

This records one completed local run reviewed on 22 November 2026. It is not
external validation, a promoted demo model, or proof of prospective performance.
The original notebook models/results and the v2 implementation were not replaced.

Command, from the project root:

```bash
python -B scripts/evaluate_models_v3.py --lightgbm --hard-test Dataset/raw/KDDTest-21.txt
```

## Inputs and Protocol

- Training: 125,973 rows; full historical test: 22,544 rows.
- Harder dependent subset: 11,850 rows, verified against the exact full-test
  row multiset with difficulty below 21.
- First stratified-group validation fold, seed 42; two workers.
- Models and binary threshold selected from training-data validation, then both
  models refitted before either benchmark was loaded.
- LightGBM selected for both binary and multiclass tasks.
- Binary threshold: `0.16900582783750193`, frozen before test evaluation.

## Observed Results

All values below are percentages except the threshold. Binary metrics refer to
the attack class; multiclass macro-F1 gives the five fixed categories equal weight.

| Benchmark | Binary accuracy | Binary precision | Binary recall | Binary F1 | Binary F1 at 0.5 | Multiclass accuracy | Multiclass macro-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Full historical test | 80.91 | 96.87 | 68.68 | 80.38 | 77.07 | 76.74 | 57.47 |
| KDDTest-21 subset | 63.73 | 95.30 | 58.57 | 72.55 | 67.53 | 55.76 | 48.18 |

The 0.5 comparison uses the same fitted binary model, not a separately selected
model. These figures do not isolate the causal effect of adding LightGBM or
establish superiority to an independently selected v2 result. The selected binary
model's validation F1 was 99.91%, much higher than its test F1. Validation tuning
and distribution shift must remain visible rather than presenting validation
scores as real-world accuracy. Minority-class and harder-subset performance
remain important limitations.

The audit found 16 duplicated training predictor rows and 664 full-test predictor
rows overlapping training (529 within KDDTest-21). Those observations were disclosed,
not removed after selection. The original attack-category mapping was retained.

There were no selected-model refit warnings. A captured warning in the
HistGradientBoosting candidate reported that the sandbox could not inspect the
physical CPU count; it fell back to logical cores. This was not a model convergence
failure, and the warning is retained in the run's selection diagnostics.

## Reproduction Evidence

Input SHA-256:

```text
KDDTrain+.txt   1b86d2f957b33082081bba410fe129b475efebcc13c9014c3f447c8271aadf95
KDDTest+.txt    fa46b0935342616aa83b7c2578db355b6a7aaabbc492248172c7a1e8b7ab8f84
KDDTest-21.txt  746993ac9e25868827cacf09eab450050a2a1056e1ce48a1ad39f5dc801d531d
```

Environment: Python 3.12.13, NumPy 2.4.6, pandas 3.0.3, SciPy 1.17.1,
scikit-learn 1.9.0, joblib 1.5.3, threadpoolctl 3.6.0, LightGBM 4.7.0.

The write-once local `runs/evaluation_v3/` package contains the archived protocol
implementation, split membership, threshold curves, candidate and test scores,
predictions, trusted fitted estimators, decision policy, and completion manifest.
All 26 recorded artifact hashes were verified after the run. The package is
ignored by Git; this note does not make its model files available to a clean
checkout or a hosted app. A tested deployment bundle is separate future work.

Local manifest SHA-256:

```text
63a122eb80137291ab4a56605f022ea1165af6509cc8d5d866b6ca2d066a95c6
```

The run's timestamp is machine-generated and archived; timings are not numerical
reproduction targets. A repeat must use a fresh output path, matching input
hashes and dependencies, and the archived source. Do not overwrite this run or
relabel its NSL-KDD evidence as newer-dataset performance.

The combined software suite passed 99 local tests, including real LightGBM smoke
fits and regression checks. CI is a separate verification, not scientific approval.
