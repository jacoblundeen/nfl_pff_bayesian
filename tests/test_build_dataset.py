import pandas as pd

from src.data.build_dataset import clean, join_sources


def test_join_sources_inner_merges_on_name_season_week():
    # player_id and team are deliberately present on BOTH sides here,
    # matching the real PFF/nflfastR schemas — this is what triggers
    # pandas' suffixing behavior on merge, and it's exactly what broke
    # in production before join_sources() started resolving it.
    pff = pd.DataFrame(
        {
            "player_id": ["pff_1"],
            "player_name": ["Test QB"],
            "team": ["KC (per PFF)"],
            "season": [2023],
            "week": [1],
            "grade_offense": [85.0],
        }
    )
    fastr = pd.DataFrame(
        {
            "player_id": ["00-0001"],
            "player_name": ["Test QB"],
            "team": ["KC"],
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

    # The canonical columns must exist and resolve to the source each is
    # documented to prefer: player_id from PFF (what's being modeled),
    # team from nflfastR (the standard abbreviation, not PFF's derived
    # home/away guess).
    assert result.loc[0, "player_id"] == "pff_1"
    assert result.loc[0, "team"] == "KC"
    # The raw suffixed columns should still be there too, for anyone
    # building a real id crosswalk later.
    assert result.loc[0, "player_id_fastr"] == "00-0001"
    assert result.loc[0, "team_pff"] == "KC (per PFF)"


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
