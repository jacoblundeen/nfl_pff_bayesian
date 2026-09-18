"""
Hierarchical Bayesian shrinkage model for weekly PFF QB grades.

Core idea: a QB's true underlying skill (`theta`) is a stable per-player
quantity. Each week's observed grade is a noisy draw around that theta,
where the noise shrinks as dropback count grows. Partial pooling means a
QB with few career dropbacks gets pulled hard toward the league mean,
while a QB with a long track record barely moves — the model estimates
how much to trust each player's own data rather than that being a fixed
rule we hand-code.

model:
    mu            ~ Normal(league mean grade, wide prior)
    sigma_player  ~ HalfNormal (spread of true skill across players)
    theta[player] ~ Normal(mu, sigma_player)          # true skill per QB
    sigma_obs     ~ HalfNormal (week-to-week noise around true skill)
    grade[i]      ~ Normal(theta[player[i]], sigma_obs / sqrt(dropbacks[i]))

The 1/sqrt(dropbacks) term is what ties shrinkage strength to sample
size: a 50-dropback week is treated as a far more precise read on skill
than a 10-dropback week.
"""

import arviz as az
import numpy as np
import pandas as pd
import pymc as pm


def build_model(df: pd.DataFrame) -> tuple[pm.Model, np.ndarray]:
    """
    df must have columns: player_id, grade_offense, dropbacks.
    Returns the PyMC model plus the integer player-index array used to
    map posterior draws back to player_id.
    """
    players, player_idx = np.unique(df["player_id"], return_inverse=True)
    n_players = len(players)

    grades = df["grade_offense"].to_numpy()
    dropbacks = df["dropbacks"].to_numpy()
    # Guard against zero/negative dropbacks blowing up the precision term.
    weight = np.sqrt(np.clip(dropbacks, 1, None))

    with pm.Model(coords={"player": players}) as model:
        mu = pm.Normal("mu", mu=grades.mean(), sigma=15)
        sigma_player = pm.HalfNormal("sigma_player", sigma=10)
        theta = pm.Normal("theta", mu=mu, sigma=sigma_player, dims="player")

        sigma_obs = pm.HalfNormal("sigma_obs", sigma=15)
        obs_sigma = sigma_obs / weight

        pm.Normal(
            "grade_obs",
            mu=theta[player_idx],
            sigma=obs_sigma,
            observed=grades,
        )

    return model, players


def fit(df: pd.DataFrame, draws: int = 2000, tune: int = 1000, chains: int = 4) -> az.InferenceData:
    model, players = build_model(df)
    with model:
        idata = pm.sample(draws=draws, tune=tune, chains=chains, target_accept=0.9)
    idata.attrs["players"] = list(players)
    return idata


def player_shrinkage_summary(idata: az.InferenceData) -> pd.DataFrame:
    """One row per player: posterior mean/credible interval for theta."""
    summary = az.summary(idata, var_names=["theta"], hdi_prob=0.94)
    summary = summary.reset_index().rename(columns={"index": "theta_index"})
    players = idata.attrs["players"]
    summary["player_id"] = [players[i] for i in range(len(players))]
    return summary[["player_id", "mean", "sd", "hdi_3%", "hdi_97%"]].rename(
        columns={"mean": "theta_mean", "sd": "theta_sd"}
    )


if __name__ == "__main__":
    from src.config import JOINED_FILE

    df = pd.read_parquet(JOINED_FILE)
    idata = fit(df)
    print(player_shrinkage_summary(idata).sort_values("theta_mean", ascending=False).head(10))
