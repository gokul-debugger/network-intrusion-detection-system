# Network Intrusion Detection System

An interactive NSL-KDD reference dashboard with validation-selected LightGBM
models, connection-level SHAP explanations, and a separate rule-based access-log
workspace. This branch is the modern-upgrade preview; newer-dataset training is
still pending, and this is not a production intrusion detection system.

<p>
  <a href="#modern-dashboard"><img src="https://img.shields.io/badge/Dashboard-Run_Locally-167D8D?style=for-the-badge&logo=streamlit&logoColor=white" alt="Run the modern reference dashboard locally"></a>
  <a href="#versioned-evaluation"><img src="https://img.shields.io/badge/Evaluation-V2_Workflow-2457A7?style=for-the-badge&logo=scikitlearn&logoColor=white" alt="Read the versioned evaluation workflow"></a>
  <a href="#how-to-run"><img src="https://img.shields.io/badge/Setup-Getting_Started-455A64?style=for-the-badge&logo=python&logoColor=white" alt="Set up the project"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-8B6508?style=for-the-badge" alt="Read the MIT license"></a>
</p>

The original dashboard remains available as the legacy edition. The modern
dashboard's launch and deployment details are below; a hosted URL is added only
after verification.

## Reference Evidence

| Frozen v3 benchmark | Binary precision | Binary recall | Binary F1 | Multiclass macro-F1 |
|---|---:|---:|---:|---:|
| KDDTest+ | 96.87% | 68.68% | 80.38% | 57.47% |
| KDDTest-21 | 95.30% | 58.57% | 72.55% | 48.18% |

Both tasks selected LightGBM using training-data validation. The binary threshold
was frozen before benchmark scoring. KDDTest-21 is a dependent harder subset,
not external validation. Minority-category performance, prior test inspection,
benchmark age, and train/test feature overlaps remain limitations.
[Protocol and verification](docs/evaluation_v3_smoke_study.md).

## Modern Dashboard

- **Connections:** labelled examples, a complete manual form, bounded CSV batches,
  independent binary/category predictions, and a session-only threshold override.
- **Traffic logs:** immediate synthetic traffic or bounded in-memory uploads,
  masked IP display, request/error summaries, filters, and transparent rule scores.
- **Model evidence:** balanced metrics, per-class support, confusion matrices,
  validation tradeoffs, and frozen source provenance.
- **Explanations:** global importance and on-demand local SHAP waterfalls with
  explicit output units and checked reconstruction.

```bash
python -m pip install -r app/requirements.txt
streamlit run app/modern_app.py
```

The checked deployment bundle is included; visitors do not need to train models,
run notebooks, or start a simulator. No uploaded log is written to a shared
server file, and no server-path or model-upload controls are exposed.
[Deployment, privacy, and attribution](docs/dashboard.md).

## Editions

The [original NSL-KDD edition](https://github.com/gokul-debugger/network-intrusion-detection-system/tree/legacy/nsl-kdd)
is maintained separately from the
[modern upgrade under development](https://github.com/gokul-debugger/network-intrusion-detection-system/tree/feature/modern-upgrade).
The default `main` branch will become the complete upgraded edition only after
the newer-data workflow, explainability, dashboard, and deployment checks pass.
Until then, the upgrade branch is work in progress, not a released modern IDS.

## Project Goals

- Detect whether a network connection is normal or malicious
- Classify attack traffic into high-level categories: DoS, Probe, R2L, and U2R
- Compare classical machine learning models on the official NSL-KDD train/test split
- Explain important network traffic features
- Build a lightweight Streamlit interface for model demonstration
- Monitor Apache/Nginx-style web access logs for suspicious live traffic patterns

## Dataset

Dataset: NSL-KDD  
Recommended files:

```text
KDDTrain+.txt
KDDTest+.txt
KDDTrain+_20Percent.txt
KDDTest-21.txt
```

NSL-KDD is an improved version of KDD Cup 1999. It reduces many duplicate records and is widely used as a learning benchmark for intrusion detection.

The dataset has 41 network connection features plus:

- attack label
- difficulty score

This project maps detailed attack labels into five categories:

| Category | Meaning |
|---|---|
| normal | Legitimate network traffic |
| dos | Denial-of-service attacks |
| probe | Scanning and probing attacks |
| r2l | Remote-to-local attacks |
| u2r | User-to-root attacks |

## Project Structure

```text
network-intrusion-detection-system/
├── Dataset/
│   ├── raw/
│   └── cleaned/
├── Models/
├── app/
│   └── app.py
├── notebooks/
│   ├── 01_data_understanding_eda.ipynb
│   ├── 02_preprocessing_feature_engineering.ipynb
│   ├── 03_binary_intrusion_detection.ipynb
│   ├── 04_multiclass_attack_classification.ipynb
│   └── 05_explainability_and_app_prep.ipynb
├── reports/
├── scripts/
├── src/
├── requirements.txt
└── README.md
```

## Notebook Workflow

Run notebooks in order:

1. `01_data_understanding_eda.ipynb`
   - loads NSL-KDD files
   - checks train/test shapes
   - analyzes binary and multiclass attack distributions
   - identifies attack types that appear only in the test set

2. `02_preprocessing_feature_engineering.ipynb`
   - validates missing values and duplicate rows
   - saves cleaned train/test CSV files
   - documents numeric and categorical feature groups

3. `03_binary_intrusion_detection.ipynb`
   - trains binary classifiers
   - compares Logistic Regression, Random Forest, and HistGradientBoosting
   - saves the best binary intrusion model

4. `04_multiclass_attack_classification.ipynb`
   - trains attack category classifiers
   - compares multiclass Random Forest and HistGradientBoosting models
   - saves the best multiclass attack model

5. `05_explainability_and_app_prep.ipynb`
   - exports feature importance
   - saves example network connections for demo testing
   - prepares deployment summary files

## Why This Project

This project adds a cybersecurity and anomaly detection dimension to a data science portfolio. It also highlights important applied ML concerns:

- imbalanced labels
- categorical network protocol features
- official train/test split generalization
- unseen attack types in testing
- binary detection vs multiclass attack diagnosis
- model explainability for security monitoring

## Model Results

These tables preserve the historical notebook results. They are not results from
the separately versioned v2 workflow described below.

### Binary Intrusion Detection

The binary task predicts whether a connection is normal or malicious.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| HistGradientBoosting | 79.73% | 96.87% | 66.54% | 78.89% | 96.07% |
| Random Forest | 77.75% | 96.84% | 62.97% | 76.32% | 96.23% |
| Logistic Regression | 75.47% | 91.70% | 62.58% | 74.39% | 79.09% |

HistGradientBoosting achieved the best F1 score, while Random Forest produced a slightly higher ROC-AUC. The chosen binary model prioritizes a strong balance between precision and recall.

### Multiclass Attack Classification

The multiclass task predicts the attack category: `normal`, `dos`, `probe`, `r2l`, or `u2r`.

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| HistGradientBoosting | 78.14% | 55.98% | 75.29% |
| Random Forest | 73.44% | 47.51% | 68.70% |

The lower macro F1 reflects the difficulty of rare categories such as R2L and U2R. This is an important security-learning point: overall accuracy can hide weak performance on minority attack types.

## Versioned Evaluation

The original notebooks and their saved outputs remain available as the historical
workflow. They compared candidates on the official test file and selected their
saved models from those scores; those results should not be described as an
independent model-selection holdout.

The separate [v2 evaluation workflow](docs/evaluation_v2.md) selects models using
group-aware validation inside the official training data, refits the selected
models, then reports retrospective official-test results. It never replaces the
existing demo models, reports, or notebook outputs.

V2 retains the project's existing attack-category mapping in `src/config.py` for
comparability. Taxonomy changes require a separate versioned review; improving
the selection protocol does not establish independent validation or better test performance.

```bash
python -m pip install -r requirements-evaluation.txt
python scripts/evaluate_models_v2.py
python -B -m pytest tests -p no:cacheprovider
```

New studies are written once to ignored `runs/` directories. The Streamlit app
continues to use its existing models until a separately reviewed promotion.

### Upgrade Study

The opt-in [v3 workflow](docs/evaluation_v3.md) adds LightGBM candidates,
validation-selected binary thresholds, and exact-membership checks before
evaluating `KDDTest-21`. It saves a separate study and never changes the original
demo or its default models. The harder test file is a dependent subset, not
independent validation. See the [upgrade and deployment roadmap](docs/upgrade_roadmap.md)
for staged explainability, dashboard, packaging, and modern-dataset work.

```bash
python -m pip install -r requirements-upgrade.txt
python scripts/evaluate_models_v3.py --lightgbm --hard-test Dataset/raw/KDDTest-21.txt
```

### Explain the Selected Models

The separate [SHAP workflow](docs/explainability.md) explains the exact selected
models in a trusted completed v3 study. It exports per-class feature importance,
local waterfall plots, contribution arrays, and reconstruction diagnostics.
Background rows come only from the recorded training input. Output units and
class order are explicit; explanations are not causal evidence or new performance
scores. No original model, evaluation package, notebook, or dashboard default changes.

```bash
python -m pip install -r requirements-explainability.txt
python -B scripts/explain_models.py --trust-local-models
```

Use that flag only for model files you trust: joblib loading can execute code.
Exports go to a new ignored run directory, not into the original study or demo.
Dashboard integration follows as a separate upgrade stage.
See the [verified example figures and check results](docs/explainability_check.md).

## How to Run

Create and activate a virtual environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Start Jupyter:

```bash
jupyter notebook
```

Run notebooks in order from `01` to `05`.

## Legacy Streamlit App

After running the notebooks and saving models:

```bash
streamlit run app/app.py
```

The app has two tabs:

1. `NSL-KDD Model Demo`
   - loads the saved binary and multiclass models
   - lets you select example network connections
   - shows normal/attack and attack-category predictions

<img width="1512" height="982" alt="Legacy NSL-KDD dashboard showing binary and attack-category predictions" src="https://github.com/user-attachments/assets/41b18b36-14cc-4eee-b7db-e9009e70d0c1" />

2. `Live Web Traffic Monitor`
   - reads Apache/Nginx-style access logs
   - shows request volume, top IPs, error rate, suspicious paths, and high-risk requests
   - uses transparent rule-based anomaly scoring for live traffic monitoring

<img width="1512" height="982" alt="Legacy web-log monitor showing request volume, status patterns, and rule-based risk summaries" src="https://github.com/user-attachments/assets/f2fbe271-5406-4eb9-86c8-0b18ac4fc770" />

To generate demo traffic for the live dashboard, run this in a second terminal:

```bash
python scripts/simulate_web_traffic.py
```

This appends simulated access-log rows to:

```text
reports/live_access.log
```

The dashboard can also read a real access log if you provide the file path.

## Limitations

NSL-KDD is useful for learning, but it is not modern production network traffic. A real intrusion detection system would require newer datasets, live flow extraction, security-domain validation, and ongoing monitoring for concept drift.

The web traffic monitor is intentionally log-based rather than packet-capture-based. This keeps the project safe, reproducible, and portfolio-friendly. It is not a replacement for enterprise security tools.

## Upgrade Roadmap

- [x] Add optional LightGBM candidates to the v3 reference workflow
- [x] Add separate, verified SHAP explainability exports
- [x] Evaluate the frozen choices on the dependent `KDDTest-21` subset
- [x] Add validation-only threshold tuning for binary detection
- [x] Build a richer, deployment-safe reference Streamlit dashboard
- [ ] Train/evaluate a separate newer-data workflow, such as UNSW-NB15 or CIC-IDS2017
- [ ] Package the local dashboard and publish tested Streamlit deployments

Completed entries concern the NSL-KDD reference workflow on the upgrade branch.
They do not mean the modern-data edition is finished or the original demo has
been replaced. See the [staged roadmap](docs/upgrade_roadmap.md).

## Author

Gokul  
GitHub: [gokul-debugger](https://github.com/gokul-debugger)
