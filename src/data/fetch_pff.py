"""
Pull weekly QB offense grades from the PFF API via the Restish CLI.

Confirmed against https://developer.pff.com/reference/ (2026-09-18) — this
replaces an earlier guess at a single bulk "/v1/grades/weekly" endpoint,
which doesn't exist. PFF's real shape is two endpoints used together:

  1. GET /v1/facet/passing        (restish command: `passing`, real name
     `facet-passing-summary` — confirmed via `restish pff passing --help`)
     League-wide leaderboard. Takes `--league` and `--season`; there is
     NO server-side position filter (an earlier version of this file
     guessed `--position` — that flag doesn't exist and restish rejects
     it outright). We fetch the full leaderboard and filter to QBs
     client-side on the `position` column in the response instead.

  2. GET /v1/player/offense/summary   (restish command: `player-offense-summary`)
     Per-player, per-season. Passing `season` without `week` returns one
     row per game that player played that season (a `weeks[]` array),
     each with `grades_offense` and snap counts — the week-by-week series
     the shrinkage model needs. NOT yet confirmed via --help like the
     command above was.

So pulling one season costs 1 (leaderboard) + (# of QBs found) API calls.
Watch your PFF rate limit if you widen SEASONS or POSITION later;
REQUEST_DELAY_SECONDS below adds a small pause between per-player calls.
"""

import json
import subprocess
import time

import pandas as pd

from src.config import (
    LEAGUE,
    PFF_RAW_FILE,
    PFF_RESTISH_ALIAS,
    PFF_RESTISH_PROFILE,
    POSITION,
    SEASONS,
)

REQUEST_DELAY_SECONDS = 0.25

# Columns the rest of the pipeline expects, regardless of PFF's raw naming.
REQUIRED_COLUMNS = [
    "player_id",
    "player_name",
    "team",
    "season",
    "week",
    "position",
    "grade_offense",
    "snap_counts_offense",
]


def _restish_json(args: list[str]) -> dict:
    """Run a restish command against the PFF API and parse its JSON body.

    `-o json` gives clean JSON with no HTTP headers (per PFF's exporting
    guide), and `-p ci` selects the API-key auth profile set up in
    guide/authentication — required outside an interactive browser login.
    """
    cmd = ["restish", PFF_RESTISH_ALIAS, *args, "-o", "json", "-p", PFF_RESTISH_PROFILE]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _list_qb_ids(season: int, position: str) -> pd.DataFrame:
    """player_id + player_name for every `position` player in `season`.

    Confirmed against `restish pff passing --help`: the leaderboard has
    no server-side position filter, so we pull everyone and filter here.
    Response envelope is `passing_summary`, each row's name field is
    `player` (not `name`) and its id field is already `player_id`.
    """
    raw = _restish_json(["passing", "--league", LEAGUE, "--season", str(season)])
    rows = raw.get("passing_summary", [])
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["player_id", "player_name"])

    df = df.rename(columns={"player": "player_name"})
    df = df[df["position"] == position]
    return df[["player_id", "player_name"]].drop_duplicates()


def _weekly_grades_for_player(player_id: str, season: int) -> pd.DataFrame:
    """One row per game `player_id` played in `season`, with grade + snaps + team."""
    raw = _restish_json(["player-offense-summary", LEAGUE, str(player_id), "--season", str(season)])
    weeks = raw.get("offense_summary", {}).get("weeks", [])
    df = pd.DataFrame(weeks)
    if df.empty:
        return df

    df["snap_counts_offense"] = df["snap_counts_total"]
    df["team"] = df.apply(
        lambda r: r["home_team_name"]
        if r.get("player_franchise_id") == r.get("home_franchise_id")
        else r["away_team_name"],
        axis=1,
    )
    df = df.rename(columns={"grades_offense": "grade_offense"})
    return df[["player_id", "week", "position", "grade_offense", "snap_counts_offense", "team"]]


def fetch_pff_weekly_grades(
    seasons: list[int] = SEASONS, position: str = POSITION
) -> pd.DataFrame:
    """Fetch weekly grades for every season in `seasons`, concatenated."""
    frames = []
    for season in seasons:
        qbs = _list_qb_ids(season, position)
        print(f"{season}: found {len(qbs)} {position}s on the passing leaderboard")

        for _, qb in qbs.iterrows():
            weekly = _weekly_grades_for_player(qb["player_id"], season)
            if weekly.empty:
                continue
            weekly["season"] = season
            weekly["player_name"] = qb["player_name"]
            frames.append(weekly)
            time.sleep(REQUEST_DELAY_SECONDS)

    if not frames:
        raise RuntimeError(
            "No PFF data returned for any season/player — check season/position "
            "values and that `restish pff whoami -p ci` still succeeds."
        )

    df = pd.concat(frames, ignore_index=True)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns after fetch: {missing}")
    return df[REQUIRED_COLUMNS]


def main() -> None:
    df = fetch_pff_weekly_grades()
    df.to_csv(PFF_RAW_FILE, index=False)
    print(f"Wrote {len(df)} rows to {PFF_RAW_FILE}")


if __name__ == "__main__":
    main()
