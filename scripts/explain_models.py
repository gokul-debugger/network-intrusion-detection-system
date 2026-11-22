"""Export bounded SHAP explanations from a trusted completed v3 study without refitting."""

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import TEST_FILE, TRAIN_FILE  # noqa: E402
from src.explainability import ExplanationProtocol, build_explanations  # noqa: E402


def main():
    os.environ.setdefault("MPLBACKEND", "Agg")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", type=Path, default=PROJECT_ROOT / "runs/evaluation_v3")
    parser.add_argument("--train", type=Path, default=TRAIN_FILE)
    parser.add_argument("--examples", type=Path, default=TEST_FILE)
    parser.add_argument("--dataset", choices=("training", "historical_test", "harder_test"), default="historical_test")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "runs/explanations_v1")
    parser.add_argument("--background-rows", type=int, default=64)
    parser.add_argument("--explain-rows", type=int, default=32)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--trust-local-models", action="store_true", help="Only for a study whose executable model files you trust")
    args = parser.parse_args()
    output = build_explanations(args.study, args.train, args.examples, args.output_dir, PROJECT_ROOT,
                                ExplanationProtocol(args.background_rows, args.explain_rows, workers=args.workers),
                                args.dataset, trust_models=args.trust_local_models)
    print(f"Separate SHAP export saved to {output}")


if __name__ == "__main__":
    main()
