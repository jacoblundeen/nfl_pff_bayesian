"""
Streamlit dashboard for the QB grade shrinkage project, in four tabs:

  How It Works     — plain-language explanation of hierarchical Bayesian
                      shrinkage, why it fits this problem, the actual
                      model equations, and what a shrinkage number is
                      (and isn't) claiming. Read this first if the model
                      is new to you; everything else assumes it.
  Player Explorer  — pick a QB and season, see raw weekly grades vs. the
                      shrunk (Bayesian) estimate with a credible interval
                      band, and the raw-vs-shrunk numbers side by side.
  League Overview  — every QB in a season at once: shrinkage magnitude
                      vs. sample size, the single chart that shows the
                      model's whole point — low-snap players get pulled
                      hard, high-snap players barely move.
  Model Validation — the backtest's raw-average-vs-shrunk RMSE comparison
                      (src/validation/backtest.py), read from a
                      precomputed file rather than refit live (see note
                      on fit_season_model below).

Run with:
    streamlit run src/app/streamlit_app.py
"""

import sys
from pathlib import Path

# `streamlit run` (like `python script.py`) only puts this script's own
# folder on sys.path, not the project root — unlike `python -m
# src.data.fetch_pff`, which uses the current working directory instead.
# Without this, `from src...` below fails with ModuleNotFoundError no
# matter which directory you launch streamlit from.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import BACKTEST_RESULTS_FILE, JOINED_FILE
from src.models.shrinkage_model import fit, player_shrinkage_summary

st.set_page_config(page_title="QB Grade Shrinkage", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(JOINED_FILE)


@st.cache_resource
def fit_season_model(df: pd.DataFrame, season: int) -> pd.DataFrame:
    """Fit the shrinkage model on ONE season only, and return a summary
    with raw average, shrunk estimate, and total dropbacks per player.

    Fitting per season (rather than once on all seasons pooled) matters:
    a player's true skill can genuinely shift year to year, and the
    chart below shows one season's weekly dots at a time — a shrunk line
    blended across seasons the viewer isn't looking at would be
    misleading, not just imprecise. This also matches how the backtest
    in src/validation/backtest.py already works (one fit per season), so
    the whole project is now consistent about what "the model" means.

    Streamlit reruns this once per season the first time it's selected,
    then serves it from cache — expect a ~15-20s pause the first time
    you pick a given season, instant after that.
    """
    season_df = df[df["season"] == season]
    idata = fit(season_df)
    summary = player_shrinkage_summary(idata)

    name_lookup = season_df[["player_id", "player_name"]].drop_duplicates().set_index(
        "player_id"
    )["player_name"]
    raw_avg = season_df.groupby("player_id")["grade_offense"].mean().rename("raw_avg")
    total_dropbacks = season_df.groupby("player_id")["dropbacks"].sum().rename("total_dropbacks")

    summary = summary.set_index("player_id")
    summary = summary.join(name_lookup).join(raw_avg).join(total_dropbacks)
    summary["shrinkage_magnitude"] = (summary["theta_mean"] - summary["raw_avg"]).abs()
    return summary.reset_index()


@st.cache_data
def load_backtest_results() -> pd.DataFrame | None:
    """Read the precomputed backtest table (src/validation/backtest.py).

    This is NOT refit live here on purpose: the backtest fits a separate
    model per season on top of what fit_season_model() above already
    fits per season, and doing both live would roughly double an
    already-slow first load. It's also a genuinely different computation
    (weeks 1-8 train / weeks 9-17 holdout) from the full-season fits
    elsewhere on this page, so reusing those wouldn't be correct anyway.
    Run `python -m src.validation.backtest` to (re)generate this file.
    """
    if not BACKTEST_RESULTS_FILE.exists():
        return None
    return pd.read_csv(BACKTEST_RESULTS_FILE)


def render_methodology() -> None:
    st.header("Why shrinkage, and what it's actually doing")

    st.markdown(
        """
A single PFF weekly grade doesn't say how many snaps it's built on. A
QB who mopped up 4 dropbacks in a blowout and a QB who started and
threw 40 can both walk away with a grade of, say, 68 — but one of
those numbers is a much shakier read on how that player actually
plays than the other. Treating them as equally trustworthy is where
a lot of "my model says X" claims about weekly grades quietly go
wrong.

**Shrinkage is the fix**: pull every player's grade toward the
league-wide pattern, but pull harder on the players with less data
to back their number up, and barely at all on the players with a lot
of it. Nobody hand-codes *how much* to pull — the model estimates
that from the data itself.

This is the same idea behind the classic baseball example: a rookie
batting .400 in April isn't actually a .400 hitter, and a hierarchical
model figures out how much of that .400 to believe without anyone
telling it the rookie's sample is small — it infers that from how
much all *other* early-season samples have bounced around before
settling.
        """
    )

    st.subheader("The model")
    st.markdown(
        "Fit separately per season (see the caption on **Player Explorer** "
        "for why pooling seasons together would be misleading). For every "
        "player *p* and week *i*:"
    )
    st.latex(
        r"""
        \begin{aligned}
        \mu &\sim \text{Normal}(\bar{y}, 15) &&\text{league-wide mean grade} \\
        \sigma_{\text{player}} &\sim \text{HalfNormal}(10) &&\text{spread of true skill across players} \\
        \tilde{\theta}_p &\sim \text{Normal}(0, 1) &&\text{unit-scale offset per player} \\
        \theta_p &= \mu + \tilde{\theta}_p \cdot \sigma_{\text{player}} &&\text{player } p\text{'s true skill} \\
        \sigma_{\text{obs}} &\sim \text{HalfNormal}(15) &&\text{week-to-week noise} \\
        y_i &\sim \text{Normal}\!\left(\theta_{p(i)},\ \sigma_{\text{obs}} / \sqrt{\text{dropbacks}_i}\right) &&\text{observed weekly grade}
        \end{aligned}
        """
    )
    st.markdown(
        """
The line that does the actual work is the last one: dividing the
observation noise by `sqrt(dropbacks)` means a high-snap week is
treated as a far more precise reading of `theta_p` than a low-snap
week. Everything upstream of that — `theta_p` built from a shared
`mu` and `sigma_player` rather than estimated independently for each
player — is what lets the model borrow strength across players: a QB
with three career starts still gets a sensible estimate, because the
model has *600 other players' worth* of information about how far a
player's true skill typically sits from the league mean.

(`theta` is written as `mu + offset * sigma_player` — the
"non-centered" form — rather than `theta ~ Normal(mu, sigma_player)`
directly. Mathematically identical, but it's the difference between
the sampler converging cleanly and it not; see the README for that
story.)
        """
    )

    st.subheader("What a shrinkage number is actually telling you")
    n = np.linspace(1, 600, 300)
    k_examples = [30, 120, 300]
    fig = go.Figure()
    for k in k_examples:
        weight = n / (n + k)
        fig.add_trace(
            go.Scatter(
                x=n,
                y=weight,
                mode="lines",
                name=f"k = {k}",
            )
        )
    fig.update_layout(
        xaxis_title="A player's dropbacks this season",
        yaxis_title="Weight on the player's own data (vs. the league mean)",
        height=400,
        legend_title="noise-to-signal ratio (k)",
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Schematic, not this model's actual fitted values — shown to build "
        "intuition for the shape, not to be read off precisely. For any "
        "hierarchical normal model like this one, the optimal weight on a "
        "player's own average works out to `n / (n + k)`, where `n` is "
        "their effective sample size (dropbacks) and `k` is the ratio of "
        "week-to-week noise to across-player spread (`sigma_obs² / "
        "sigma_player²`) that this model fits directly from the data — a "
        "noisier league (bigger k) means the curve needs more dropbacks "
        "before it starts trusting a player's own number. For this "
        "model's actual fitted shrinkage on real players, see **League "
        "Overview**."
    )

    st.subheader("What it isn't saying")
    st.markdown(
        """
- **It isn't claiming a low-snap player is worse than their grade.**
  It's saying their grade alone isn't strong evidence yet, so the
  best single-number guess leans on the league pattern until more
  evidence comes in.
- **It isn't a claim about any one player being "due for regression."**
  The model has no opinion on any individual QB — `theta_p` is just
  the number that minimizes expected squared error *on average*
  across every player in the pool, given what it knows.
- **It's only as good as its assumptions** — one league-wide `mu`
  every QB gets pooled toward, grades that are roughly normal, and
  noise that scales with `sqrt(dropbacks)`. All three are
  simplifications (a rookie and a 10-year veteran probably shouldn't
  share one prior mean, for instance — see *What I'd extend next* in
  the README).
- **The only thing that actually justifies using it** is the
  backtest: does the shrunk estimate predict a player's *future*
  grades better than their raw average does? See **Model
  Validation** for that result — it's the real evidence, everything
  above is just what the model is doing to produce it.
        """
    )


def render_player_explorer(df: pd.DataFrame, seasons: list[int]) -> None:
    col1, col2 = st.columns([1, 3])
    with col1:
        season = st.selectbox("Season", seasons, key="explorer_season")
        summary = fit_season_model(df, season)
        player_options = sorted(summary["player_name"].dropna().unique())
        player_name = st.selectbox("Quarterback", player_options, key="explorer_player")

    player_df = df[(df["season"] == season) & (df["player_name"] == player_name)].sort_values(
        "week"
    )
    theta_row = summary[summary["player_name"] == player_name].iloc[0]

    with col2:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=player_df["week"],
                y=player_df["grade_offense"],
                mode="markers+lines",
                name="Raw weekly grade",
                marker=dict(size=8),
            )
        )
        fig.add_hline(
            y=theta_row["theta_mean"],
            line_dash="dash",
            annotation_text="Shrunk season estimate",
        )
        fig.add_hrect(
            y0=theta_row["hdi_3%"],
            y1=theta_row["hdi_97%"],
            fillcolor="LightSkyBlue",
            opacity=0.25,
            line_width=0,
            annotation_text="94% credible interval",
        )
        fig.update_layout(xaxis_title="Week", yaxis_title="PFF Offense Grade", height=450)
        st.plotly_chart(fig, width="stretch")

    m1, m2, m3 = st.columns(3)
    m1.metric("Raw season average", f"{theta_row['raw_avg']:.1f}")
    m2.metric(
        "Shrunk estimate",
        f"{theta_row['theta_mean']:.1f}",
        delta=f"{theta_row['theta_mean'] - theta_row['raw_avg']:+.1f}",
        delta_color="off",
    )
    m3.metric("Total dropbacks", int(theta_row["total_dropbacks"]))
    st.caption(
        "Fewer dropbacks -> the shrunk estimate leans harder on the league-wide pattern "
        "rather than this player's own small sample. The delta above is how far shrinkage "
        "moved this player's number from their raw average."
    )


def render_league_overview(df: pd.DataFrame, seasons: list[int]) -> None:
    season = st.selectbox("Season", seasons, key="league_season")
    summary = fit_season_model(df, season)

    fig = px.scatter(
        summary,
        x="total_dropbacks",
        y="shrinkage_magnitude",
        hover_name="player_name",
        labels={
            "total_dropbacks": "Total dropbacks this season",
            "shrinkage_magnitude": "How far shrinkage moved the grade",
        },
        title=f"Shrinkage magnitude vs. sample size — {season}",
    )
    fig.update_traces(marker=dict(size=10, opacity=0.75))
    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "Each point is one QB. The left side (fewer dropbacks) is where shrinkage does the "
        "most work, pulling noisy small-sample grades toward the league pattern; the right "
        "side (a full season of snaps) is where the model trusts the player's own data almost "
        "as-is. Hover a point to see who it is."
    )


def render_validation(results: pd.DataFrame | None) -> None:
    if results is None:
        st.info(
            "No backtest results found yet. Run `python -m src.validation.backtest` "
            "from the project root, then reload this page."
        )
        return

    st.caption(
        "Per season: fit the model on weeks 1-8 only, then check whether the shrunk "
        "estimate or the raw weeks 1-8 average better predicts each QB's actual weeks "
        "9-17 performance (lower RMSE wins). This is the real test of whether shrinkage "
        "helps, not just whether the model runs."
    )

    display = results.rename(
        columns={
            "season": "Season",
            "n_players": "QBs",
            "rmse_raw_avg": "RMSE (raw average)",
            "rmse_shrunk": "RMSE (shrunk)",
            "shrinkage_wins": "Shrinkage wins",
        }
    )
    st.dataframe(display, hide_index=True, width="stretch")

    melted = results.melt(
        id_vars="season",
        value_vars=["rmse_raw_avg", "rmse_shrunk"],
        var_name="method",
        value_name="rmse",
    )
    melted["method"] = melted["method"].map(
        {"rmse_raw_avg": "Raw average", "rmse_shrunk": "Shrunk estimate"}
    )
    fig = px.bar(
        melted,
        x="season",
        y="rmse",
        color="method",
        barmode="group",
        labels={"season": "Season", "rmse": "RMSE (lower is better)", "method": ""},
        title="Predicting weeks 9-17: raw average vs. shrunk estimate",
    )
    fig.update_layout(height=450)
    st.plotly_chart(fig, width="stretch")


def main() -> None:
    st.title("How much should you trust a PFF grade?")
    st.caption(
        "Hierarchical Bayesian shrinkage of weekly QB grades, weighted by dropback count — "
        "pulling low-snap, noisy grades toward the league pattern while trusting high-snap "
        "grades close to face value."
    )

    df = load_data()
    seasons = sorted(df["season"].unique(), reverse=True)

    tab_how, tab_explorer, tab_league, tab_validation = st.tabs(
        ["How It Works", "Player Explorer", "League Overview", "Model Validation"]
    )
    with tab_how:
        render_methodology()
    with tab_explorer:
        render_player_explorer(df, seasons)
    with tab_league:
        render_league_overview(df, seasons)
    with tab_validation:
        render_validation(load_backtest_results())


if __name__ == "__main__":
    main()
