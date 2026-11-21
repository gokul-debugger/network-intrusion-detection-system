from pathlib import Path
import sys

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import ALL_COLUMNS, FEATURE_COLUMNS  # noqa: E402


def synthetic_source(per_class=15, offset=0):
    rows = []
    for category, attack in enumerate(("normal", "neptune", "ipsweep", "guess_passwd", "buffer_overflow")):
        for number in range(per_class):
            row = dict.fromkeys(FEATURE_COLUMNS, 0)
            row.update(duration=category + number % 3, protocol_type="tcp", service="http", flag="SF",
                       src_bytes=offset + len(rows) + 1, dst_bytes=number * 2,
                       attack_type=attack, difficulty=21)
            rows.append(row)
    return pd.DataFrame(rows)[ALL_COLUMNS]


@pytest.fixture
def raw_frame():
    return synthetic_source()


@pytest.fixture
def source_paths(tmp_path):
    directory = tmp_path / "input"
    directory.mkdir()
    train, test = directory / "train.txt", directory / "test.txt"
    synthetic_source().to_csv(train, header=False, index=False)
    synthetic_source(7, offset=1000).to_csv(test, header=False, index=False)
    return train, test
