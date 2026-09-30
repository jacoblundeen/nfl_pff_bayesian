"""
Fits the shrinkage model once per season and writes the per-player
results to data/processed/season_summaries.csv.

Run this (after build_dataset has produced the joined parquet) before
deploying the dashboard anywhere with limited CPU, such as Streamlit
Community Cloud: it lets Player Explorer and League Overview read a
static file instead of running PyMC live for every visitor and every
season they pick. Locally, the app still falls back to a live fit if
this file doesn't exist yet, so this step is optional for local dev.

Usage:
    python scripts/precompute_season_summaries.py
"""

import sys
from pathlib import Path

# See scripts/run_pipeline.py for why this is needed: `python
# scripts/foo.py` only puts scripts/ on sys.path, not the project root
# that `src` lives under.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import JOINED_FILE, SEASON_SUMMARY_FILE, SEASONS
from src.models.shrinkage_model import season_player_summary


def main() -> None:
    df = pd.read_parquet(JOINED_FILE)

    summaries = []
    for season in SEASONS:
        print(f"Fitting {season}...")
        summaries.append(season_player_summary(df, season))

    result = pd.concat(summaries, ignore_index=True)
    result.to_csv(SEASON_SUMMARY_FILE, index=False)
    print(f"\nWrote {len(result)} player-season rows to {SEASON_SUMMARY_FILE}")


if __name__ == "__main__":
    main()
