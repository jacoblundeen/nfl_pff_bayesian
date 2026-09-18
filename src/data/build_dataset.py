"""
Join PFF weekly grades to nflfastR weekly context on (player, season, week),
clean the result, and write a single analysis-ready parquet file.

This is the one script the model and the app both read from — neither of
them should ever touch the raw PFF/nflfastR files directly. Keeping that
boundary is what makes the pipeline "a pipeline" rather than a notebook
with steps in it: each stage has one job and a clear file on either side.
"""

import pandas as pd

from src.config import JOINED_FILE, NFLFASTR_RAW_FILE, PFF_RAW_FILE

# Below this many dropbacks in a week, a grade is mostly noise — drop
# rather than model it. Revisit this threshold once you've looked at the
# real distribution; it's a guess to start with, not a finding.
MIN_DROPBACKS = 5


def load_raw() -> tuple[pd.DataFrame, pd.DataFrame]:
    pff = pd.read_csv(PFF_RAW_FILE)
    fastr = pd.read_csv(NFLFASTR_RAW_FILE)
    return pff, fastr


def join_sources(pff: pd.DataFrame, fastr: pd.DataFrame) -> pd.DataFrame:
    # player_id namespaces differ between PFF and nflverse — this join key
    # almost certainly needs a crosswalk. nfl_data_py exposes one via
    # nfl.import_ids(), which maps pff_id -> gsis_id among others. Wire
    # that in here once you've confirmed the column names on a real pull;
    # joining on (player_name, season, week) is a stopgap that will
    # silently drop or mis-merge suffix/Jr. name mismatches.
    merged = pff.merge(
        fastr,
        left_on=["player_name", "season", "week"],
        right_on=["player_name", "season", "week"],
        how="inner",
        suffixes=("_pff", "_fastr"),
    )
    return merged


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df[df["dropbacks"] >= MIN_DROPBACKS].copy()
    df = df.dropna(subset=["grade_offense", "dropbacks"])
    dropped = before - len(df)
    if dropped:
        print(f"Dropped {dropped} rows below {MIN_DROPBACKS} dropbacks or with missing grades.")

    df["is_rookie_season"] = df["is_rookie_season"].fillna(False)
    return df.reset_index(drop=True)


def build() -> pd.DataFrame:
    pff, fastr = load_raw()
    joined = join_sources(pff, fastr)
    cleaned = clean(joined)
    cleaned.to_parquet(JOINED_FILE, index=False)
    print(f"Wrote {len(cleaned)} rows to {JOINED_FILE}")
    return cleaned


if __name__ == "__main__":
    build()
