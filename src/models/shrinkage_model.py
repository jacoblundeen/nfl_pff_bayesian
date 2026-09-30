"""
Hierarchical Bayesian shrinkage model for weekly PFF QB grades.

Core idea: a QB's true underlying skill (`theta`) is a stable per-player
quantity. Each week's observed grade is a noisy draw around that theta,
where the noise shrinks as dropback count grows. Partial pooling means a
QB with few career dropbacks gets pulled hard toward the league mean,
while a QB with a long track record barely moves — the model estimates
how much to trust each player's own data rather than that being a fixed
rule we hand-code.

model (non-centered parameterization — see note below):
    mu              ~ Normal(league mean grade, wide prior)
    sigma_player    ~ HalfNormal (spread of true skill across players)
    theta_offset[p] ~ Normal(0, 1)                       # unit-scale per QB
    theta[player]   = mu + theta_offset[player] * sigma_player   # true skill
    sigma_obs       ~ HalfNormal (week-to-week noise around true skill)
    grade[i]        ~ Normal(theta[player[i]], sigma_obs / sqrt(dropbacks[i]))

The 1/sqrt(dropbacks) term is what ties shrinkage strength to sample
size: a 50-dropback week is treated as a far more precise read on skill
than a 10-dropback week.

Why non-centered: a direct `theta ~ Normal(mu, sigma_player)` (the
"centered" form) creates a funnel-shaped posterior between theta and
sigma_player that NUTS struggles to sample efficiently — worst exactly
for players with few weeks of data, which is the case this whole
project cares about getting right. Sampling `theta_offset ~ Normal(0,1)`
instead and computing theta as a deterministic transform is
mathematically equivalent but reparameterizes away the funnel; it's the
standard fix recommended in the PyMC/Stan literature for this failure
mode (see Betancourt's "A Conceptual Introduction to HMC" on
divergences). `theta` stays a first-class variable in the trace either
way — it's just a Deterministic instead of a free RV now — so nothing
downstream (player_shrinkage_summary, backtest, the app) needs to know
the difference.
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
        theta_offset = pm.Normal("theta_offset", mu=0, sigma=1, dims="player")
        theta = pm.Deterministic("theta", mu + theta_offset * sigma_player, dims="player")

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


def _az_summary_compat(idata: az.InferenceData, prob: float) -> pd.DataFrame:
    """Call az.summary() with whichever credible-interval kwarg this
    installed arviz accepts.

    arviz is mid-refactor and has renamed `hdi_prob` to `ci_prob` in at
    least one release path — pinning an exact arviz version felt more
    fragile long-term (this project will outlive today's pinned version)
    than just handling both spellings here.
    """
    for kwarg in ("hdi_prob", "ci_prob"):
        try:
            return az.summary(idata, var_names=["theta"], **{kwarg: prob})
        except TypeError:
            continue
    return az.summary(idata, var_names=["theta"])  # last resort: library default


def player_shrinkage_summary(idata: az.InferenceData, prob: float = 0.94) -> pd.DataFrame:
    """One row per player: posterior mean/sd and credible interval for theta."""
    summary = _az_summary_compat(idata, prob)
    summary = summary.reset_index().rename(columns={"index": "theta_index"})
    players = idata.attrs["players"]
    summary["player_id"] = [players[i] for i in range(len(players))]

    # The credible-interval columns' NAMES have also moved around across
    # arviz versions (e.g. "hdi_3%"/"hdi_97%" vs some future "ci_..."
    # equivalent), so find them by elimination rather than hardcoding a
    # name: everything that isn't mean/sd/diagnostics/our own added
    # columns is one of the two interval bounds, and whichever has the
    # smaller average value is the lower bound.
    known = {
        "theta_index", "player_id", "mean", "sd",
        "mcse_mean", "mcse_sd", "ess_bulk", "ess_tail", "r_hat",
    }
    interval_cols = [c for c in summary.columns if c not in known]
    if len(interval_cols) != 2:
        raise RuntimeError(
            f"Expected exactly 2 credible-interval columns from az.summary(), got "
            f"{interval_cols} instead. arviz's output shape has likely changed again — "
            f"run `print(summary.columns)` to see what's actually there and adjust "
            f"the `known` set above."
        )
    a, b = interval_cols
    lower_col, upper_col = (a, b) if summary[a].mean() <= summary[b].mean() else (b, a)

    return summary[["player_id", "mean", "sd", lower_col, upper_col]].rename(
        columns={"mean": "theta_mean", "sd": "theta_sd", lower_col: "hdi_3%", upper_col: "hdi_97%"}
    )


if __name__ == "__main__":
    from src.config import JOINED_FILE

    df = pd.read_parquet(JOINED_FILE)
    idata = fit(df)
    print(player_shrinkage_summary(idata).sort_values("theta_mean", ascending=False).head(10))
