import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.web_traffic import append_simulated_traffic


def main():
    parser = argparse.ArgumentParser(description="Append simulated web access logs for the dashboard.")
    parser.add_argument("--output", default="reports/live_access.log")
    parser.add_argument("--interval", type=float, default=2.0)
    parser.add_argument("--rows", type=int, default=8)
    parser.add_argument("--risky-rate", type=float, default=0.18)
    args = parser.parse_args()

    output_path = Path(args.output)
    print(f"Writing simulated traffic to {output_path.resolve()}")
    print("Press Ctrl+C to stop.")

    try:
        while True:
            append_simulated_traffic(output_path, rows=args.rows, risky_rate=args.risky_rate)
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped traffic simulation.")


if __name__ == "__main__":
    main()
