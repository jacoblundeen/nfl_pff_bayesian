"""
Streamlit dashboard: pick a QB and season, see raw weekly grades vs. the
shrunk (Bayesian) estimate with a credible interval band, plus a marker
for roughly how many dropbacks it took before the estimate stabilized.

Run with:
    streamlit run src/app/streamlit_app.py
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config import JOINED_FILE
from src.models.shrinkage_model import fit, player_shrinkage_summary

st.set_page_config(page_title="QB Grade Shrinkage", layout="wide")


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_parquet(JOINED_FILE)


@st.cache_resource
def fit_model(df: pd.DataFrame):
    return fit(df)


def main() -> None:
    st.title("How much should you trust a PFF grade?")
    st.caption(
        "Hierarchical Bayesian shrinkage of weekly QB grades, weighted by dropback count. "
        "A low-snap week gets pulled hard toward the group mean; a full season of "
        "dropbacks barely moves."
    )

    df = load_data()
    idata = fit_model(df)
    summary = player_shrinkage_summary(idata)

    name_lookup = df[["player_id", "player_name"]].drop_duplicates().set_index("player_id")
    summary = summary.join(name_lookup, on="player_id")

    col1, col2 = st.columns([1, 3])
    with col1:
        season = st.selectbox("Season", sorted(df["season"].unique(), reverse=True))
        player_options = sorted(df[df["season"] == season]["player_name"].dropna().unique())
        player_name = st.selectbox("Quarterback", player_options)

    player_df = df[(df["season"] == season) & (df["player_name"] == player_name)].sort_values(
        "week"
    )
    player_id = player_df["player_id"].iloc[0]
    theta_row = summary[summary["player_id"] == player_id].iloc[0]

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
        fig.update_layout(
            xaxis_title="Week",
            yaxis_title="PFF Offense Grade",
            height=450,
        )
        st.plotly_chart(fig, use_container_width=True)

    total_dropbacks = int(player_df["dropbacks"].sum())
    st.metric("Total dropbacks this season", total_dropbacks)
    st.caption(
        "Fewer dropbacks -> the shrunk estimate leans harder on the league-wide pattern "
        "rather than this player's own small sample. Compare across players with very "
        "different snap counts to see the effect."
    )


if __name__ == "__main__":
    main()
