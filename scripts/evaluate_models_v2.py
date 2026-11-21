"""Build a separate training-selected NSL-KDD evaluation without replacing the demo."""

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TEST_FILE, TRAIN_FILE  # noqa: E402
from src.evaluation_v2 import EvaluationProtocol, build_evaluation  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=TRAIN_FILE)
    parser.add_argument("--test", type=Path, default=TEST_FILE)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/evaluation_v2")
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    output = build_evaluation(args.train, args.test, args.output_dir, PROJECT_ROOT,
                              EvaluationProtocol(workers=args.workers))
    print(f"New v2 evaluation saved to {output}")


if __name__ == "__main__":
    main()
