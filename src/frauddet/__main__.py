from __future__ import annotations

import argparse
import json
from pathlib import Path

from frauddet.config import METRICS_PATH, RAW_CSV
from frauddet.data import download_kaggle_dataset, load_transactions
from frauddet.predict import score_csv
from frauddet.train import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m frauddet",
        description="Credit-card fraud detection with anomaly detection.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("download", help="Download the ULB / Kaggle transaction CSV")
    sub.add_parser("train", help="Fit anomaly detectors and write reports/")
    sub.add_parser("metrics", help="Print the last training metrics JSON")

    predict = sub.add_parser("predict", help="Score a CSV of transactions")
    predict.add_argument("--input", required=True, type=Path)
    predict.add_argument("--output", type=Path, default=None)

    ui = sub.add_parser("ui", help="Open a local web UI")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=7860)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "download":
        path = download_kaggle_dataset(RAW_CSV)
        df = load_transactions(path if path.exists() else None)
        print(f"{len(df):,} rows, fraud rate {df['Class'].mean():.4%}")
    elif args.command == "train":
        metrics = train()
        print(json.dumps({k: metrics[k] for k in ("n_rows", "fraud_rate", "best_anomaly_model")}, indent=2))
    elif args.command == "metrics":
        print(METRICS_PATH.read_text())
    elif args.command == "predict":
        dest = score_csv(args.input, args.output)
        print(f"Wrote {dest}")
    elif args.command == "ui":
        from frauddet.app import run_ui

        run_ui(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
