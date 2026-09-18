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
RMSE. See `src/validation/backtest.py` and the results section below
once it's been run.

**Result:** _TBD — fill in after running `scripts/run_pipeline.py` on
real data._

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

`src/config.py` has two TODOs to fill in against your actual PFF API
access: the real endpoint path for weekly grades, and confirming the
response field names match what `fetch_pff.py` expects. See the
comments in `src/data/fetch_pff.py` for specifics.

## Running the pipeline

```bash
python scripts/run_pipeline.py          # full run: fetch + build + validate
python scripts/run_pipeline.py --skip-fetch   # reuse existing raw CSVs
streamlit run src/app/streamlit_app.py   # launch the dashboard
```

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
