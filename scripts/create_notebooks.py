import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def markdown(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source.strip() + "\n"}


def code(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.strip() + "\n",
    }


def write_notebook(path, cells):
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2), encoding="utf-8")


setup = """
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print(PROJECT_ROOT)
"""


write_notebook(
    NOTEBOOKS / "01_data_understanding_eda.ipynb",
    [
        markdown(
            """
# Network Intrusion Detection: Data Understanding and EDA

This notebook loads the NSL-KDD train and test files, validates label mappings, and explores attack distributions for binary and multiclass intrusion detection.
"""
        ),
        code(setup),
        code(
            """
import pandas as pd

from src.config import TRAIN_FILE, TEST_FILE, TEST_21_FILE, TRAIN_20_FILE, FIGURES_DIR, ensure_project_dirs
from src.data import load_nsl_kdd, summarize_labels, validate_raw_files
from src.visualize import save_countplot

ensure_project_dirs()
validate_raw_files([TRAIN_FILE, TEST_FILE, TRAIN_20_FILE, TEST_21_FILE])
"""
        ),
        code(
            """
train_df = load_nsl_kdd(TRAIN_FILE)
test_df = load_nsl_kdd(TEST_FILE)
test_21_df = load_nsl_kdd(TEST_21_FILE)

print("Train:", train_df.shape)
print("Test:", test_df.shape)
print("Hard test:", test_21_df.shape)
display(train_df.head())
"""
        ),
        markdown("## Binary Class Distribution"),
        code(
            """
binary_train = summarize_labels(train_df, "binary_label")
binary_test = summarize_labels(test_df, "binary_label")

display(binary_train)
display(binary_test)

save_countplot(train_df, "binary_label", FIGURES_DIR / "binary_distribution_train.png", "Binary Label Distribution - Train")
save_countplot(test_df, "binary_label", FIGURES_DIR / "binary_distribution_test.png", "Binary Label Distribution - Test")
"""
        ),
        markdown("## Attack Category Distribution"),
        code(
            """
category_train = summarize_labels(train_df, "attack_category")
category_test = summarize_labels(test_df, "attack_category")

display(category_train)
display(category_test)

save_countplot(train_df, "attack_category", FIGURES_DIR / "attack_category_train.png", "Attack Categories - Train")
save_countplot(test_df, "attack_category", FIGURES_DIR / "attack_category_test.png", "Attack Categories - Test")
"""
        ),
        markdown("## Attack Type Coverage"),
        code(
            """
train_attacks = set(train_df["attack_type"].unique())
test_attacks = set(test_df["attack_type"].unique())

print("Train attack types:", len(train_attacks))
print("Test attack types:", len(test_attacks))
print("Attack types only in test:")
print(sorted(test_attacks - train_attacks))
"""
        ),
        markdown("## Key EDA Notes"),
        code(
            """
eda_summary = {
    "train_rows": len(train_df),
    "test_rows": len(test_df),
    "hard_test_rows": len(test_21_df),
    "train_attack_rate": float(train_df["binary_label"].mean()),
    "test_attack_rate": float(test_df["binary_label"].mean()),
    "train_attack_types": len(train_attacks),
    "test_attack_types": len(test_attacks),
    "test_only_attack_types": sorted(test_attacks - train_attacks),
}

with open(PROJECT_ROOT / "reports" / "eda_summary.json", "w", encoding="utf-8") as file:
    json.dump(eda_summary, file, indent=2)

eda_summary
"""
        ),
    ],
)


write_notebook(
    NOTEBOOKS / "02_preprocessing_feature_engineering.ipynb",
    [
        markdown(
            """
# Preprocessing and Feature Engineering

This notebook prepares clean train/test datasets for modeling while preserving the official NSL-KDD train-test split.
"""
        ),
        code(setup),
        code(
            """
from src.config import TRAIN_FILE, TEST_FILE, CLEANED_DIR, FEATURE_COLUMNS, CATEGORICAL_FEATURES, NUMERIC_FEATURES, ensure_project_dirs
from src.data import load_nsl_kdd, save_cleaned

ensure_project_dirs()
train_df = load_nsl_kdd(TRAIN_FILE)
test_df = load_nsl_kdd(TEST_FILE)
"""
        ),
        markdown("## Data Quality Checks"),
        code(
            """
print("Missing values in train:", int(train_df.isna().sum().sum()))
print("Missing values in test:", int(test_df.isna().sum().sum()))
print("Duplicate rows in train:", int(train_df.duplicated().sum()))
print("Duplicate rows in test:", int(test_df.duplicated().sum()))

print("Categorical features:", CATEGORICAL_FEATURES)
print("Numeric feature count:", len(NUMERIC_FEATURES))
"""
        ),
        markdown("## Save Cleaned Data"),
        code(
            """
save_cleaned(train_df, test_df, CLEANED_DIR)

print(CLEANED_DIR / "train_cleaned.csv")
print(CLEANED_DIR / "test_cleaned.csv")
"""
        ),
        markdown("## Modeling Columns"),
        code(
            """
modeling_summary = {
    "feature_count": len(FEATURE_COLUMNS),
    "numeric_features": len(NUMERIC_FEATURES),
    "categorical_features": CATEGORICAL_FEATURES,
    "binary_target": "binary_label",
    "multiclass_target": "attack_category",
}

modeling_summary
"""
        ),
    ],
)


write_notebook(
    NOTEBOOKS / "03_binary_intrusion_detection.ipynb",
    [
        markdown(
            """
# Binary Intrusion Detection

This notebook trains models to classify network connections as `normal` or `attack`. The official NSL-KDD test split is used for evaluation to measure generalization to unseen attack patterns.
"""
        ),
        code(setup),
        code(
            """
import json

import pandas as pd

from src.config import CLEANED_DIR, FEATURE_COLUMNS, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, ensure_project_dirs
from src.modeling import binary_models, evaluate_binary, metrics_to_frame, save_model
from src.visualize import save_metric_barplot

ensure_project_dirs()
train_df = pd.read_csv(CLEANED_DIR / "train_cleaned.csv")
test_df = pd.read_csv(CLEANED_DIR / "test_cleaned.csv")

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["binary_label"]
X_test = test_df[FEATURE_COLUMNS]
y_test = test_df["binary_label"]
"""
        ),
        markdown("## Train and Evaluate Models"),
        code(
            """
models = binary_models()
results = {}
fitted_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    fitted_models[name] = model
    results[name] = evaluate_binary(model, X_test, y_test)

binary_results = metrics_to_frame(results).sort_values("f1", ascending=False)
display(binary_results)
binary_results.to_csv(REPORTS_DIR / "binary_model_comparison.csv", index=False)
save_metric_barplot(binary_results, "f1", FIGURES_DIR / "binary_f1_comparison.png", "Binary Intrusion Detection - F1 Score")
"""
        ),
        markdown("## Save Best Binary Model"),
        code(
            """
best_binary_name = binary_results.iloc[0]["model"]
best_binary_model = fitted_models[best_binary_name]
save_model(best_binary_model, MODELS_DIR / "binary_intrusion_model.pkl")

with open(REPORTS_DIR / "binary_metrics.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("Best binary model:", best_binary_name)
"""
        ),
        markdown("## Business Interpretation"),
        code(
            """
binary_results[["model", "precision", "recall", "f1", "roc_auc"]]
"""
        ),
    ],
)


write_notebook(
    NOTEBOOKS / "04_multiclass_attack_classification.ipynb",
    [
        markdown(
            """
# Multiclass Attack Classification

This notebook predicts the attack category: `normal`, `dos`, `probe`, `r2l`, or `u2r`. This is harder than binary detection because minority attack categories are rare.
"""
        ),
        code(setup),
        code(
            """
import json

import pandas as pd

from src.config import CLEANED_DIR, FEATURE_COLUMNS, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, ensure_project_dirs
from src.modeling import evaluate_multiclass, metrics_to_frame, multiclass_models, save_model
from src.visualize import save_metric_barplot

ensure_project_dirs()
train_df = pd.read_csv(CLEANED_DIR / "train_cleaned.csv")
test_df = pd.read_csv(CLEANED_DIR / "test_cleaned.csv")

X_train = train_df[FEATURE_COLUMNS]
y_train = train_df["attack_category"]
X_test = test_df[FEATURE_COLUMNS]
y_test = test_df["attack_category"]
"""
        ),
        markdown("## Train and Evaluate Multiclass Models"),
        code(
            """
models = multiclass_models()
results = {}
fitted_models = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    fitted_models[name] = model
    results[name] = evaluate_multiclass(model, X_test, y_test)

multiclass_results = metrics_to_frame(results).sort_values("weighted_f1", ascending=False)
display(multiclass_results)
multiclass_results.to_csv(REPORTS_DIR / "multiclass_model_comparison.csv", index=False)
save_metric_barplot(multiclass_results, "weighted_f1", FIGURES_DIR / "multiclass_weighted_f1_comparison.png", "Multiclass Attack Classification - Weighted F1")
"""
        ),
        markdown("## Save Best Multiclass Model"),
        code(
            """
best_multiclass_name = multiclass_results.iloc[0]["model"]
best_multiclass_model = fitted_models[best_multiclass_name]
save_model(best_multiclass_model, MODELS_DIR / "multiclass_attack_model.pkl")

with open(REPORTS_DIR / "multiclass_metrics.json", "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2)

print("Best multiclass model:", best_multiclass_name)
"""
        ),
        markdown("## Per-Class Notes"),
        code(
            """
best_report = results[best_multiclass_name]["classification_report"]
pd.DataFrame(best_report).T
"""
        ),
    ],
)


write_notebook(
    NOTEBOOKS / "05_explainability_and_app_prep.ipynb",
    [
        markdown(
            """
# Explainability and App Preparation

This notebook prepares deployment-facing artifacts: feature importance, model summaries, and example rows for the Streamlit app.
"""
        ),
        code(setup),
        code(
            """
import json

import joblib
import pandas as pd
from sklearn.inspection import permutation_importance

from src.config import CLEANED_DIR, FEATURE_COLUMNS, MODELS_DIR, REPORTS_DIR, ensure_project_dirs

ensure_project_dirs()
train_df = pd.read_csv(CLEANED_DIR / "train_cleaned.csv")
test_df = pd.read_csv(CLEANED_DIR / "test_cleaned.csv")
binary_model = joblib.load(MODELS_DIR / "binary_intrusion_model.pkl")
multiclass_model = joblib.load(MODELS_DIR / "multiclass_attack_model.pkl")
"""
        ),
        markdown("## Feature Importance"),
        code(
            """
sample = test_df.sample(n=min(2500, len(test_df)), random_state=42)
X_sample = sample[FEATURE_COLUMNS]
y_binary_sample = sample["binary_label"]
y_multiclass_sample = sample["attack_category"]


def permutation_importance_frame(model, X, y, scoring):
    result = permutation_importance(
        model,
        X,
        y,
        scoring=scoring,
        n_repeats=5,
        random_state=42,
        n_jobs=-1,
    )
    importance = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    return importance

binary_importance = permutation_importance_frame(binary_model, X_sample, y_binary_sample, scoring="f1")
multiclass_importance = permutation_importance_frame(multiclass_model, X_sample, y_multiclass_sample, scoring="f1_weighted")

display(binary_importance.head(20))
display(multiclass_importance.head(20))

binary_importance.to_csv(REPORTS_DIR / "binary_feature_importance.csv", index=False)
multiclass_importance.to_csv(REPORTS_DIR / "multiclass_feature_importance.csv", index=False)
"""
        ),
        markdown("## Example App Inputs"),
        code(
            """
examples = pd.concat(
    [
        test_df[test_df["binary_label"] == 0].head(5),
        test_df[test_df["binary_label"] == 1].head(5),
    ],
    ignore_index=True,
)

examples[FEATURE_COLUMNS + ["binary_label", "attack_category", "attack_type"]].to_csv(
    REPORTS_DIR / "example_connections.csv",
    index=False,
)

display(examples[FEATURE_COLUMNS + ["binary_label", "attack_category", "attack_type"]].head(10))
"""
        ),
        markdown("## Deployment Notes"),
        code(
            """
deployment_summary = {
    "binary_model": "Models/binary_intrusion_model.pkl",
    "multiclass_model": "Models/multiclass_attack_model.pkl",
    "feature_count": len(FEATURE_COLUMNS),
    "supported_outputs": {
        "binary": ["normal", "attack"],
        "multiclass": ["normal", "dos", "probe", "r2l", "u2r"],
    },
}

with open(REPORTS_DIR / "deployment_summary.json", "w", encoding="utf-8") as file:
    json.dump(deployment_summary, file, indent=2)

deployment_summary
"""
        ),
    ],
)


print(f"Created notebooks in {NOTEBOOKS}")
