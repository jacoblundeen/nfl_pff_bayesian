"""
Pull weekly QB context data (dropbacks, EPA, rookie/vet status) via
nfl_data_py, which wraps the nflverse/nflfastR data releases.

This is the "how much should we trust this grade" signal — dropback
count is the exposure variable the Bayesian model uses to decide how
hard to shrink a given week's grade toward the group mean.
"""

import nfl_data_py as nfl
import pandas as pd

from src.config import NFLFASTR_RAW_FILE, SEASONS


def fetch_qb_weekly_context(seasons: list[int] = SEASONS) -> pd.DataFrame:
    """Weekly QB rows: dropbacks, EPA, and basic roster context."""
    weekly = nfl.import_weekly_data(seasons, downcast=True)
    weekly = weekly[weekly["position"] == "QB"].copy()

    keep = [
        "player_id",
        "player_display_name",
        "recent_team",
        "season",
        "week",
        "attempts",
        "passing_epa",
        "sacks",
    ]
    weekly = weekly[[c for c in keep if c in weekly.columns]]
    weekly["dropbacks"] = weekly["attempts"] + weekly.get("sacks", 0)

    # Draft year / draft capital, to identify rookies and build a
    # rookie-vs-veteran grouping level in the model later.
    rosters = nfl.import_seasonal_rosters(seasons)
    rookie_info = rosters[["player_id", "season", "entry_year", "draft_number"]].drop_duplicates()

    merged = weekly.merge(rookie_info, on=["player_id", "season"], how="left")
    merged["is_rookie_season"] = merged["season"] == merged["entry_year"]

    return merged.rename(columns={"recent_team": "team", "player_display_name": "player_name"})


def main() -> None:
    df = fetch_qb_weekly_context()
    df.to_csv(NFLFASTR_RAW_FILE, index=False)
    print(f"Wrote {len(df)} rows to {NFLFASTR_RAW_FILE}")


if __name__ == "__main__":
    main()
