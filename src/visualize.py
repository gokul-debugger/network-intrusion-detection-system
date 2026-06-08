from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def save_countplot(df: pd.DataFrame, column: str, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    order = df[column].value_counts().index
    sns.countplot(data=df, x=column, order=order, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(column.replace("_", " ").title())
    ax.set_ylabel("Records")
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_metric_barplot(results: pd.DataFrame, metric: str, output_path: Path, title: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    sorted_results = results.sort_values(metric, ascending=False)
    sns.barplot(data=sorted_results, x=metric, y="model", ax=ax)
    ax.set_title(title)
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

