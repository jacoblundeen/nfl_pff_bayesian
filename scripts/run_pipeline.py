"""
Runs the full pipeline end to end: fetch -> build -> fit -> validate.

Usage:
    python scripts/run_pipeline.py             # full run
    python scripts/run_pipeline.py --skip-fetch  # reuse existing raw files
"""

import argparse

from src.config import JOINED_FILE, SEASONS
from src.data.build_dataset import build
from src.data.fetch_nflfastr import fetch_qb_weekly_context
from src.data.fetch_pff import fetch_pff_weekly_grades
from src.validation.backtest import run as run_backtest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fetch", action="store_true", help="Reuse existing raw CSVs.")
    args = parser.parse_args()

    if not args.skip_fetch:
        print("Fetching PFF weekly grades...")
        fetch_pff_weekly_grades()
        print("Fetching nflfastR weekly context...")
        fetch_qb_weekly_context()

    print("Building joined dataset...")
    df = build()

    print("Running train/holdout backtest (this fits PyMC models per season)...")
    results = run_backtest(df, SEASONS)
    print(results.to_string(index=False))

    print(f"\nJoined dataset available at {JOINED_FILE} for the Streamlit app.")


if __name__ == "__main__":
    main()
