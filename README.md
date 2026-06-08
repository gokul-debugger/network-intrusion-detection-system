# Network Intrusion Detection System

A machine learning project for detecting malicious network connections using the NSL-KDD intrusion detection benchmark. The project builds both binary intrusion detection and multiclass attack classification workflows, then prepares saved models for a Streamlit demo.

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

## Streamlit App

After running the notebooks and saving models:

```bash
streamlit run app/app.py
```

The app has two tabs:

1. `NSL-KDD Model Demo`
   - loads the saved binary and multiclass models
   - lets you select example network connections
   - shows normal/attack and attack-category predictions

<img width="1512" height="982" alt="Screenshot 2026-06-09 at 2 07 27 AM" src="https://github.com/user-attachments/assets/41b18b36-14cc-4eee-b7db-e9009e70d0c1" />

2. `Live Web Traffic Monitor`
   - reads Apache/Nginx-style access logs
   - shows request volume, top IPs, error rate, suspicious paths, and high-risk requests
   - uses transparent rule-based anomaly scoring for live traffic monitoring

<img width="1512" height="982" alt="Screenshot 2026-06-09 at 2 08 23 AM" src="https://github.com/user-attachments/assets/f2fbe271-5406-4eb9-86c8-0b18ac4fc770" />

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

## Future Improvements

- Add XGBoost or LightGBM
- Add SHAP explainability
- Evaluate on `KDDTest-21`
- Add threshold tuning for binary detection
- Build a richer Streamlit dashboard
- Try a modern dataset such as CIC-IDS2017 or UNSW-NB15

## Author

Gokul  
GitHub: [gokul-debugger](https://github.com/gokul-debugger)
