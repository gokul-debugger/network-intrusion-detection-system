from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ALL_COLUMNS, ATTACK_CATEGORY_MAP


def validate_raw_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        formatted = "\n".join(f"- {path}" for path in missing)
        raise FileNotFoundError(f"Missing NSL-KDD files:\n{formatted}")


def load_nsl_kdd(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, names=ALL_COLUMNS)
    df["binary_label"] = (df["attack_type"] != "normal").astype(int)
    df["attack_category"] = df["attack_type"].map(ATTACK_CATEGORY_MAP)

    unknown = sorted(df.loc[df["attack_category"].isna(), "attack_type"].unique())
    if unknown:
        raise ValueError(f"Unknown attack labels found in {path.name}: {unknown}")

    return df


def summarize_labels(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    counts = df[label_column].value_counts().rename_axis(label_column).reset_index(name="count")
    counts["percentage"] = counts["count"] / len(df) * 100
    return counts


def save_cleaned(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(output_dir / "train_cleaned.csv", index=False)
    test_df.to_csv(output_dir / "test_cleaned.csv", index=False)

