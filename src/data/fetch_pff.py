"""
Pull weekly QB grades from the PFF API via the Restish CLI.

Why Restish and not `requests` directly: you've already got PFF auth
configured as a Restish API alias, so shelling out to the CLI reuses
that instead of re-implementing auth here. If you'd rather call the API
directly with `requests`/`httpx` later, only this file needs to change —
everything downstream just expects a CSV at config.PFF_RAW_FILE with the
columns listed below.

TODO before this runs for real:
  1. Confirm your Restish alias name matches config.PFF_RESTISH_ALIAS
     (check with `restish api list`).
  2. Replace ENDPOINT_PATH below with the actual PFF endpoint for weekly
     player grades (this varies by PFF contract/tier — check your API
     docs or `restish <alias> --help` for the available routes).
  3. Confirm the response field names in `_normalize()` match what PFF
     actually returns — adjust the rename map accordingly.
"""

import json
import subprocess

import pandas as pd

from src.config import PFF_RAW_FILE, PFF_RESTISH_ALIAS, POSITION, SEASONS

# TODO: replace with the real PFF endpoint path for weekly grades.
ENDPOINT_PATH = "/v1/grades/weekly"

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


def _restish_get(path: str, params: dict) -> dict:
    """Run a single `restish <alias> GET <path>` call and parse the JSON."""
    query = "&".join(f"{k}={v}" for k, v in params.items())
    cmd = ["restish", PFF_RESTISH_ALIAS, "get", f"{path}?{query}"]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def _normalize(raw: list[dict]) -> pd.DataFrame:
    """Map PFF's raw field names onto our internal schema."""
    df = pd.DataFrame(raw)

    # TODO: adjust this rename map once you've confirmed PFF's actual
    # field names from a real response.
    rename_map = {
        "playerId": "player_id",
        "playerName": "player_name",
        "teamName": "team",
        "grade": "grade_offense",
        "snapCounts": "snap_counts_offense",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"PFF response is missing expected columns after normalization: {missing}. "
            "Update the rename_map in _normalize() to match PFF's actual field names."
        )
    return df[REQUIRED_COLUMNS]


def fetch_pff_weekly_grades(
    seasons: list[int] = SEASONS, position: str = POSITION
) -> pd.DataFrame:
    """Fetch weekly grades for every season in `seasons`, concatenated."""
    frames = []
    for season in seasons:
        raw = _restish_get(ENDPOINT_PATH, {"season": season, "position": position})
        # PFF's paginated/list responses commonly nest results under a key
        # like "players" or "data" — adjust if `raw` isn't already a list.
        records = raw["data"] if isinstance(raw, dict) and "data" in raw else raw
        frames.append(_normalize(records))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    df = fetch_pff_weekly_grades()
    df.to_csv(PFF_RAW_FILE, index=False)
    print(f"Wrote {len(df)} rows to {PFF_RAW_FILE}")


if __name__ == "__main__":
    main()
