"""
The step that turns this from "I built a Bayesian model" into "I proved
it beats the naive approach."

Procedure, per season:
  1. Fit the shrinkage model on weeks 1-8 only.
  2. For each QB, compute:
       - their shrunk theta estimate from weeks 1-8
       - their raw (unweighted) average grade from weeks 1-8
  3. Compare both against their ACTUAL average grade in weeks 9-17.
  4. Whichever estimate has lower RMSE against that holdout is the winner.

If shrinkage doesn't win this comparison, that's a real, reportable
finding too — it just means the writeup's claim changes from "shrinkage
improves prediction" to "here's what I learned about when it doesn't."
Either outcome is a legitimate result; don't massage the split to force
a particular answer.
"""

import numpy as np
import pandas as pd

from src.models.shrinkage_model import fit, player_shrinkage_summary

TRAIN_WEEKS = range(1, 9)
HOLDOUT_WEEKS = range(9, 18)


def rmse(a: pd.Series, b: pd.Series) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def backtest_season(df: pd.DataFrame, season: int) -> dict:
    season_df = df[df["season"] == season]
    train = season_df[season_df["week"].isin(TRAIN_WEEKS)]
    holdout = season_df[season_df["week"].isin(HOLDOUT_WEEKS)]

    if train.empty or holdout.empty:
        raise ValueError(f"Season {season} doesn't have both train and holdout weeks present.")

    # Naive baseline: unweighted mean of each player's train-window grades.
    raw_avg = train.groupby("player_id")["grade_offense"].mean().rename("raw_avg")

    # Shrunk estimate from the Bayesian model fit on the same train window.
    idata = fit(train)
    shrunk = player_shrinkage_summary(idata).set_index("player_id")["theta_mean"].rename("shrunk")

    actual_holdout = holdout.groupby("player_id")["grade_offense"].mean().rename("actual_holdout")

    compare = pd.concat([raw_avg, shrunk, actual_holdout], axis=1).dropna()

    return {
        "season": season,
        "n_players": len(compare),
        "rmse_raw_avg": rmse(compare["raw_avg"], compare["actual_holdout"]),
        "rmse_shrunk": rmse(compare["shrunk"], compare["actual_holdout"]),
        "detail": compare,
    }


def run(df: pd.DataFrame, seasons: list[int]) -> pd.DataFrame:
    rows = []
    for season in seasons:
        result = backtest_season(df, season)
        rows.append(
            {
                "season": result["season"],
                "n_players": result["n_players"],
                "rmse_raw_avg": result["rmse_raw_avg"],
                "rmse_shrunk": result["rmse_shrunk"],
                "shrinkage_wins": result["rmse_shrunk"] < result["rmse_raw_avg"],
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    from src.config import JOINED_FILE, SEASONS

    df = pd.read_parquet(JOINED_FILE)
    results = run(df, SEASONS)
    print(results.to_string(index=False))
