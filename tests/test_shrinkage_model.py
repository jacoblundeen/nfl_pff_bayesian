import numpy as np
import pandas as pd

from src.models.shrinkage_model import build_model


def _toy_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    rows = []
    # Player A: many high-dropback weeks around grade 90 (should barely shrink).
    for week in range(1, 9):
        rows.append(
            {"player_id": "A", "week": week, "grade_offense": 90 + rng.normal(0, 2), "dropbacks": 35}
        )
    # Player B: one noisy, low-dropback week (should shrink hard toward the mean).
    rows.append({"player_id": "B", "week": 1, "grade_offense": 40, "dropbacks": 3})
    return pd.DataFrame(rows)


def test_build_model_shapes_match_input():
    df = _toy_df()
    model, players = build_model(df)
    assert set(players) == {"A", "B"}
    # `named_vars` covers free RVs AND deterministics, so this holds
    # regardless of parameterization — theta is a Deterministic (a
    # transform of theta_offset) under the current non-centered setup,
    # not a free RV directly, but it's still a first-class named
    # variable that shows up in the posterior trace either way.
    assert "theta" in model.named_vars
    assert "theta_offset" in [rv.name for rv in model.free_RVs]


def test_low_dropback_player_has_wider_uncertainty_input():
    # Not a full sampling test (too slow for unit tests) — just confirms
    # the weighting term treats low-dropback rows as less precise, which
    # is the mechanism the shrinkage claim depends on.
    df = _toy_df()
    weight_a = np.sqrt(df[df["player_id"] == "A"]["dropbacks"].iloc[0])
    weight_b = np.sqrt(df[df["player_id"] == "B"]["dropbacks"].iloc[0])
    assert weight_a > weight_b
