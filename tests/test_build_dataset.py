import pandas as pd

from src.data.build_dataset import clean, join_sources


def test_join_sources_inner_merges_on_name_season_week():
    pff = pd.DataFrame(
        {
            "player_name": ["Test QB"],
            "season": [2023],
            "week": [1],
            "grade_offense": [85.0],
        }
    )
    fastr = pd.DataFrame(
        {
            "player_name": ["Test QB"],
            "season": [2023],
            "week": [1],
            "dropbacks": [30],
            "is_rookie_season": [False],
        }
    )
    result = join_sources(pff, fastr)
    assert len(result) == 1
    assert result.loc[0, "grade_offense"] == 85.0
    assert result.loc[0, "dropbacks"] == 30


def test_clean_drops_low_dropback_rows():
    df = pd.DataFrame(
        {
            "grade_offense": [85.0, 60.0],
            "dropbacks": [30, 2],  # second row below MIN_DROPBACKS
            "is_rookie_season": [False, True],
        }
    )
    cleaned = clean(df)
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["dropbacks"] == 30


def test_clean_fills_missing_rookie_flag():
    df = pd.DataFrame(
        {
            "grade_offense": [85.0],
            "dropbacks": [30],
            "is_rookie_season": [None],
        }
    )
    cleaned = clean(df)
    assert cleaned.iloc[0]["is_rookie_season"] is False
