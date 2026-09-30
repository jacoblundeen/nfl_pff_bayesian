"""
Runs the full pipeline end to end: fetch -> build -> fit -> validate.

Usage:
    python scripts/run_pipeline.py             # full run
    python scripts/run_pipeline.py --skip-fetch  # reuse existing raw files
"""

import argparse
import sys
from pathlib import Path

# Running this as `python scripts/run_pipeline.py` only puts scripts/ on
# sys.path, not the project root that `src` lives under — unlike
# `python -m src.data.fetch_pff`, which uses the current working
# directory instead. Without this, `from src...` below fails with
# ModuleNotFoundError regardless of which directory you launch it from.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import JOINED_FILE, SEASON_SUMMARY_FILE, SEASONS
from src.data.build_dataset import build
from src.data.fetch_nflfastr import fetch_qb_weekly_context
from src.data.fetch_pff import fetch_pff_weekly_grades
from src.models.shrinkage_model import season_player_summary
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

    print("\nFitting full-season shrinkage summaries for the dashboard...")
    summaries = pd.concat(
        [season_player_summary(df, season) for season in SEASONS], ignore_index=True
    )
    summaries.to_csv(SEASON_SUMMARY_FILE, index=False)
    print(f"Wrote {len(summaries)} player-season rows to {SEASON_SUMMARY_FILE}")

    print(f"\nJoined dataset available at {JOINED_FILE} for the Streamlit app.")


if __name__ == "__main__":
    main()
