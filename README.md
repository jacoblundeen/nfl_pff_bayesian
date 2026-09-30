# How much should you trust a PFF grade?

A hierarchical Bayesian shrinkage model for weekly PFF quarterback grades,
using dropback counts from nflfastR to decide how much to trust each grade.

## The problem

A single PFF grade treats a QB's 4-dropback relief appearance and a
14-year veteran's 40-dropback start the same way: one number, no sense
of how much data it's built on. A hot or cold small-sample week can look
identical to a real, stable level of play.

## The approach

A hierarchical Bayesian model treats each QB's *true* skill level as a
stable, unobserved quantity (`theta`), and each week's observed grade as
a noisy draw around it — where the noise shrinks as dropback count
grows. The result: low-snap weeks get pulled hard toward the
position-wide mean; high-snap weeks are trusted close to face value.
This is the same family of technique as the classic baseball
batting-average shrinkage examples — well-established, and a good
demonstration of *why* Bayesian methods matter, not just that they can
be invoked.

## Data sources

- **[Pro Football Focus (PFF)](https://www.pff.com/)** — weekly offense
  grades, pulled via their API through the [Restish](https://rest.sh/) CLI.
- **[nflfastR](https://www.nflfastr.com/)** (via
  [`nfl_data_py`](https://github.com/nflverse/nfl_data_py)) — dropback
  counts, EPA, and roster context (rookie/veteran status) used as the
  exposure variable that drives shrinkage strength.

## Validation

The real test: does the shrunk estimate predict a QB's *future*
performance better than their raw average grade does? The backtest
fits the model on weeks 1-8 of a season, then compares both the shrunk
estimate and the raw average against actual weeks 9-17 performance via
RMSE. See `src/validation/backtest.py`.

**Result:** shrinkage wins in every season tested, by a real margin —
15-35% lower RMSE than just averaging a QB's own weeks 1-8 grades:

| Season | QBs | RMSE (raw average) | RMSE (shrunk) | Shrinkage wins |
|---|---|---|---|---|
| 2022 | 44 | 12.19 | 10.16 | Yes |
| 2023 | 43 | 11.46 | 8.48  | Yes |
| 2024 | 44 | 12.48 | 8.28  | Yes |

Pooling a QB's weeks 1-8 grades toward the rest of the league — with the
pull strength set by how many dropbacks back that grade — predicts their
next nine weeks of play better than trusting their own small sample at
face value. That's the core claim of the project, holding up on three
separate seasons rather than one favorable split.

### A convergence issue worth documenting

The first version of this model used a "centered" parameterization
(`theta ~ Normal(mu, sigma_player)`), which produced sampler divergences
and a failed rhat check on real data — worst for exactly the low-snap
players this project is built to handle carefully, since a hierarchical
model's posterior geometry forms a funnel that's hardest to sample near
weakly-informed groups. Switching to a non-centered form
(`theta = mu + theta_offset * sigma_player`, with `theta_offset ~
Normal(0, 1)`) is the standard fix for this failure mode and eliminated
the divergences entirely (0 across all three season fits, clean rhat)
without materially changing the RMSE results above — a useful reminder
that a model can look like it's "working" (reasonable point estimates)
while its underlying posterior samples aren't fully trustworthy yet. See
the docstring in `src/models/shrinkage_model.py` for the full writeup.

## Project structure

```
src/
  config.py              # paths, seasons in scope, Restish alias
  data/
    fetch_pff.py          # PFF weekly grades via Restish
    fetch_nflfastr.py     # nflfastR weekly context via nfl_data_py
    build_dataset.py      # join, clean, write analysis-ready parquet
  models/
    shrinkage_model.py     # PyMC hierarchical model
  validation/
    backtest.py            # train/holdout comparison vs. naive baseline
  app/
    streamlit_app.py       # interactive dashboard
scripts/
  run_pipeline.py           # fetch -> build -> fit -> validate, end to end
tests/                       # unit tests for the join/clean logic and model shape
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Confirm your PFF Restish alias is configured:
restish api list
```

`src/data/fetch_pff.py` calls two real PFF endpoints, confirmed against
live data: `passing` to list QBs for a season (no server-side position
filter — it pulls the full leaderboard and filters to QBs locally), then
`player-offense-summary` per QB to get that player's week-by-week
grades. Confirm `PFF_RESTISH_PROFILE` in `src/config.py` matches the
Restish profile name you set up in PFF's authentication guide (`ci` by
default, used for non-interactive/API-key auth).

## Running the pipeline

```bash
python scripts/run_pipeline.py          # full run: fetch + build + validate + precompute
python scripts/run_pipeline.py --skip-fetch   # reuse existing raw CSVs
streamlit run src/app/streamlit_app.py   # launch the dashboard
```

`run_pipeline.py` writes three files the dashboard reads directly:
`qb_weekly_joined.parquet`, `backtest_results.csv`, and
`season_summaries.csv` (the last from
`scripts/precompute_season_summaries.py`, which you can also run on
its own to refresh just that file). Without `season_summaries.csv`,
Player Explorer and League Overview still work locally — they just
fall back to fitting PyMC live the first time each season is picked.

## Deploying to Streamlit Community Cloud

These three processed files are deliberately *not* gitignored (see the
comment in `.gitignore`), specifically so a deployment doesn't need to
re-run the PFF fetch or fit PyMC live on limited cloud CPU. `data/raw/`
(the untouched PFF API pulls) stays ignored either way.

> **Before doing this**, know that PFF's terms of service
> ([pff.com/terms](https://pff.com/terms)) restrict public distribution
> and republishing of API Data, including on third-party websites —
> which a Streamlit Community Cloud deployment is. That restriction is
> why `data/` was gitignored by default in the first place. This repo
> proceeds with real data deployed publicly as a deliberate, informed
> choice; read the terms yourself before doing the same with your own
> API access.

1. Run `python scripts/run_pipeline.py` locally so all three processed
   files exist, then commit and push them along with the rest of the repo:
   ```bash
   git add data/processed/qb_weekly_joined.parquet data/processed/backtest_results.csv data/processed/season_summaries.csv
   git add -A
   git commit -m "Add processed data for deployment"
   git push
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click **New app**.
3. Pick this repo and branch, and set the main file path to
   `src/app/streamlit_app.py`.
4. Under **Advanced settings**, set the Python version to match
   `requires-python` in `pyproject.toml` (3.11+). No secrets are
   needed — the deployed app only reads the committed static files,
   it never calls the PFF API or PyMC.
5. Deploy. First build takes a few minutes (installing `pymc`/`arviz`
   even though the live-fit path they support isn't exercised by
   default); page loads after that should be near-instant, since
   `season_summaries.csv` is what actually gets read.

## Tests

```bash
pytest
```

## What I'd extend next

- A second grouping level (rookie vs. veteran, or draft capital) instead
  of pooling every QB toward one league-wide mean.
- Feeding the shrunk estimates in as informative priors on team strength
  for a win-probability model, compared against nflfastR's own baseline
  win-probability model as a calibration check.
