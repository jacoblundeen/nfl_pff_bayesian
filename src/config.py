"""
Central config for the project: paths, seasons in scope, and the Restish
API alias PFF pulls go through. Keeping this in one place means every
script (fetch, build, model, app) agrees on where things live.
"""

from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

# --- Scope -------------------------------------------------------------
# Start narrow (one position, a handful of seasons) and widen once the
# pipeline + model are validated end-to-end.
POSITION = "QB"
SEASONS = [2022, 2023, 2024]

# --- Restish / PFF -----------------------------------------------------
# The alias you configured with `restish api configure <alias> <base-url>`
# when you set up PFF API access. Restish stores the auth for this alias
# itself (~/.restish/apis.json) — nothing sensitive needs to live here.
PFF_RESTISH_ALIAS = "pff"

# Raw filenames, so every script reads/writes the same paths.
PFF_RAW_FILE = DATA_RAW / "pff_qb_weekly_grades.csv"
NFLFASTR_RAW_FILE = DATA_RAW / "nflfastr_qb_weekly.csv"
JOINED_FILE = DATA_PROCESSED / "qb_weekly_joined.parquet"
