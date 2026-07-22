# NOTINO / Interview Assignment — Quantity Forecast

Interview assignment: forecast total `Quantity` (`QuantityApp + QuantityWeb`)
for 30 products over the 7 days following the training window. The complete
brief is in `task.md`. Because the brief explicitly requests a **non-tree-based**
primary solution, the canonical submission is a PyTorch feed-forward neural
network with product and campaign embeddings. XGBoost and LightGBM are retained
as fully walk-forward-validated reference models rather than hidden comparisons.

## Final state

- **Canonical submission:** NeuralNet using the direct multi-horizon strategy.
- **Primary metric:** WAPE for observed sales conditional on availability on the
  common evaluation population. This is a less-censored demand proxy, not latent demand.
- **Selection protocol:** frozen test-aligned weighting of development strata;
  the recent benchmark is reporting-only and cannot change eligibility.
- **Estimator policy:** the same three-seed, 30-epoch NeuralNet is used in
  validation and final deployment.
- **Confirmed NeuralNet objective:** MSE on a baseline-relative log residual.
- **Confirmed structured-model targets:** XGBoost `residual`, LightGBM `log1p`.
- **Frozen cross-model ensemble:** 0.36 NeuralNet + 0.25 XGBoost + 0.39 LightGBM.
- **Historical spent audit:** NeuralNet test-aligned WAPE `27.83%`; ensemble
  `27.97%`. Its source/architecture provenance was incomplete, so it remains a
  transparent historical artifact and is excluded from the current dashboard.
  Three origins do not establish superiority; NeuralNet is canonical because
  the brief predeclares a non-tree primary model.
- **Submission artifact:** `outputs/submission.csv`; the transparent secondary
  blend is `outputs/submission_ensemble.csv`.

## Presentation quick start

```bash
uv sync --frozen --group dev
uv run python webapp/server.py
```

The submission runs locally at `http://127.0.0.1:8999`. `webapp/static` is the
authored source and `docs/` is deterministic generated output prepared for
GitHub Pages. The repository does not assume that Pages is already enabled;
the public deployment should be treated as pending until the repository
coordinator enables it.
See `RETRAINING_AND_CLI_GUIDE.md` for exact retraining commands and every
supported pipeline flag, `outputs/README.md` for the retained artifacts, and
`PRESENTATION_CLEANUP.md` for the scope of the repository cleanup.

## Approach

- **Features:** cyclic calendar encodings; campaign, discount, list-price and
  effective-price state; price relative to product history; separate
  first-row and first-available lifecycle clocks; leakage-safe market state;
  event proximity; and lagged/rolling demand summaries. Calendar gaps,
  observed stockouts and valid available observations remain distinct states.
- **Categorical handling:** `CampaignSubTypeWeb/App` are nominal category codes
  (`-1, 0, 1, 2, 3, 4, 5, 16, 18, 19`), so the NN uses embeddings rather than
  treating the raw IDs as an ordinal numeric scale.
- **Neural model:** an MLP (`256 -> 128 -> 64`) with BatchNorm, GELU and Dropout,
  plus product, campaign-web, campaign-app and horizon embeddings. It predicts
  `log1p(Quantity) - log1p(target_baseline)` and the confirmed final run uses
  MSE on that residual.
- **Missing seasonal history:** annual references remain nullable for young
  products and early history. The NN uses a train-fitted median imputer plus
  missingness indicators; trees use native missing-value handling. Rows are not
  silently discarded because a 364/365/371-day reference is unavailable.
- **Forecast strategies:** `direct` predicts all seven target days from a
  stacked `(ForecastOrigin, Horizon, ProductId)` panel. `recursive` trains a
  genuine one-step model and feeds each generated prediction into the next
  synthetic state. Both use the same end-of-origin information cutoff and are
  trained independently. The final project decision is `direct`.
- **Availability contract:** unavailable rows are excluded from supervised
  targets and their quantities are censored from sales lags rather than
  interpreted as genuine zero sales. The primary target is observed sales
  conditional on availability, a less-censored demand proxy; realized-sales
  reporting remains diagnostic.
- **Validation:** rolling-origin walk-forward evaluation has a seasonally
  distributed `development` set for decisions, a disjoint `recent_benchmark`
  for reporting, and three now-spent historical audit origins. Every fold trains
  strictly before its evaluation block; the evaluation fold is never used for
  early stopping or feature construction.
- **Common population:** comparisons use rows on which every candidate being
  compared produced a valid prediction. Model-specific coverage is retained as
  a separate diagnostic so apparent gains cannot come from silently scoring an
  easier subset.
- **Reference models:** XGBoost and LightGBM support direct and recursive
  inference. Dynamic Ridge is direct-only because recursive feedback was
  empirically unstable. Seasonal-naive and a 28-day moving average provide
  transparent non-learned baselines.
- **Final model policy:** the submitted NeuralNet is the predeclared assignment
  model and averages seeds `42`, `123` and `777` for 30 epochs each. The
  cross-model convex ensemble is retained as an auditable alternative; the
  spent audit did not select or reject either candidate.

## Repository layout

```text
data/                    train and test Parquet inputs
ml/
  framework.py           configuration, feature engineering, panel builders,
                         recursive state transition, metadata and metrics
  models/                NeuralNet, XGBoost, LightGBM, Dynamic Ridge and naive baselines
  tree_worker.py         isolated native-tree subprocess dispatcher for macOS
  pipeline.py            CV, selection, final training and artifact export
  benchmark_nn_batch_size.py
                         quality-aware MPS/CUDA batch-size benchmark
  experiments/           optional C0.1/C1/C2/C3/C4 research and ablation runners
  run_final_audit.py     one-shot evaluation on untouched origins
  export_results.py      rebuilds dashboard JSON from persisted artifacts only
outputs/                 canonical predictions, OOF results, screening summaries,
                         frozen ensemble and final-audit artifacts
webapp/                  local FastAPI dashboard and static frontend
docs/                    generated-only static dashboard for GitHub Pages
tests/                   Python contracts and JavaScript smoke tests
task.md                  original assignment brief in Czech
DATASET_PROFILE_MODELING_AUDIT.md
                         dataset story and modeling decisions
RETRAINING_AND_CLI_GUIDE.md
                         exact retraining recipes and complete flag reference
PRESENTATION_CLEANUP.md  files removed from the delivery package and why
```

## Running

Install the locked environment and run the fast verification suite:

```bash
uv sync --frozen --group dev
uv run pytest tests/ -m "not integration" -q
node tests/webapp_smoke_test.js
```

Reproduce the confirmed final submission:

```bash
caffeinate -i uv run python ml/pipeline.py \
  --forecast-strategy direct \
  --primary-strategy direct \
  --submission-model NeuralNet \
  --selection-metric WAPE \
  --selection-protocol test-aligned \
  --training-window-days all \
  --recency-half-life-days none \
  --baseline-variant weighted_4321 \
  --trend-features off \
  --c2-feature-groups price,campaign,lifecycle,market,event \
  --c34-config outputs/c34_screening/recommendation.json \
  --ensemble on \
  --ensemble-models NeuralNet,XGBoost,LightGBM \
  --nn-batch-size 512 \
  --nn-lr-scaling fixed \
  --reset-checkpoints \
  2>&1 | tee pipeline_final_retrain.log
```

The pipeline rewrites the submission, OOF diagnostics and both dashboard data
copies. Use `--resume` instead of `--reset-checkpoints` when continuing the
same interrupted configuration. The complete precedence rules and all CLI
alternatives are documented in `RETRAINING_AND_CLI_GUIDE.md`.

To regenerate final forecasts without retraining or revisiting validation
origins, use the same command with
`--reuse-oof outputs/oof_predictions.parquet` and omit checkpoint flags. The
pipeline validates the retained OOF contract and rejects final-audit rows.

### Apple Silicon performance profile

The NN keeps a complete fold's tensors on MPS and shuffles/slices them on the
device, avoiding a CPU-to-MPS copy for every mini-batch. This is a
throughput-only implementation optimization: it does not change the feature
set, target, epoch budget, or number of optimizer updates.

A larger batch can improve GPU utilisation, but it also reduces optimizer
updates per epoch and can change validation quality. Do not select it from GPU
usage alone. Run the real-fold benchmark first:

```bash
caffeinate -i uv run python ml/benchmark_nn_batch_size.py \
  --batch-sizes 512 1024 2048 4096 \
  --lr-scalings fixed sqrt \
  --epochs 10 \
  --quality-tolerance 0.02
```

The benchmark measures held-out WAPE/MAE and throughput, writes
`outputs/nn_batch_benchmark.json`, and recommends the fastest policy within
2% relative WAPE of the historical `512/fixed` reference. The pipeline's
default `--nn-batch-size auto --nn-lr-scaling auto` consumes that recommendation
only when it was measured on the same accelerator type **and the exact current
feature/preprocessing signature**; without that match, the safe 512/fixed
policy is preserved. Pre-C0 benchmark files are therefore ignored
automatically.

Typical M4 Pro candidate order is 1024 -> 2048 -> 4096. With ~30% GPU usage at
batch 512 and ample unified memory, 2048 is the most plausible first winner,
but the repository deliberately requires the quality benchmark rather than
hard-coding that guess.

Manual override example:

```bash
caffeinate -i uv run python ml/pipeline.py \
  --forecast-strategy both \
  --primary-strategy auto \
  --submission-model NeuralNet \
  --selection-metric WAPE \
  --nn-batch-size 2048 \
  --nn-lr-scaling sqrt \
  --resume
```

Changing batch/LR policy invalidates checkpoints trained with a different
model policy. C0 changes the feature schema and numeric preprocessing, so pre-C0
checkpoints are intentionally incompatible. Use `--reset-checkpoints` for the
first C0 regression run; later C0 checkpoints remain reusable with `--resume`.

**macOS setup note:** XGBoost/LightGBM's macOS wheels need Homebrew's OpenMP
runtime: `brew install libomp`. Separately, PyTorch bundles its *own* copy of
that same runtime -- loading both copies in one process crashes the
interpreter the moment either trains. That's why `tree_worker.py` runs
XGBoost/LightGBM (`models/xgboost_model.py`, `models/lightgbm_model.py`) in a
dedicated subprocess (via `run_tree_baselines` in `pipeline.py`) instead of
importing them alongside torch directly; `ml/framework.py` holds the
torch-free code every model shares, and `ml/models/__init__.py` documents why
it must never eagerly import both a torch-based and a tree-based model
module together.

## Interactive results dashboard

A standalone FastAPI + vanilla-JS/Chart.js dashboard presents the original
forecasting assignment: data constraints, baseline and model development,
leakage-safe walk-forward evidence, direct-vs-recursive engineering history,
the canonical NeuralNet, limitations, and the final seven-day forecast.

```bash
uv run python webapp/server.py       # http://127.0.0.1:8999 (port set in webapp/server.py)
```

It reads `outputs/results.json` fresh on every request, so after editing
`ml/pipeline.py` (or a model under `ml/models/`) and rerunning it (or just
`uv run python ml/export_results.py` to skip retraining), restart the command
and reload the page to see updated numbers. The frontend is static HTML/CSS/JS
with no build step.

**Pages and navigation:**
- `/` — assignment objective, canonical NeuralNet, development progression,
  baselines, evaluation evidence, forecast explorer and final submission grid.
- `/dataset` — supplied-data profile, finding-to-decision mapping,
  retained/rejected experiments and honest limitations.
- `/evaluation` — the complete rolling-origin evaluation contract: the
  distinction between walk-forward validation and direct/recursive inference,
  development/recent-benchmark/final-audit roles, common-population scoring,
  test-aligned WAPE, leakage controls and metric definitions.
- `/model/<slug>` — one page per model (`neuralnet`, `xgboost`, `lightgbm`,
  `dynamicridge`, `seasonalnaive`, `movingavg28`) with strategy-aware metrics,
  plus links to the dataset rationale and complete evaluation contract,
  folds and product forecasts. `model.html` is shared and `model.js` reads the
  slug from the URL.

Each model's color is its own project's real brand color (PyTorch orange for
the NN, XGBoost's brandfetch.com purple, the `sphinx_rtd_theme` blue LightGBM's
own readthedocs page uses), defined once in `ml/framework.py::MODEL_META` and
served to the frontend via `results.json["models"]` -- not hand-picked or
duplicated in CSS/JS.

## Forecast strategy modes

The pipeline now supports two separately trained multi-step strategies and a
comparison mode:

```bash
uv run python ml/pipeline.py --forecast-strategy direct
uv run python ml/pipeline.py --forecast-strategy recursive
uv run python ml/pipeline.py --forecast-strategy both \
  --primary-strategy auto \
  --submission-model NeuralNet \
  --selection-metric WAPE \
  --selection-protocol global
```

- **Direct** trains on the stacked `(ForecastOrigin, Horizon, ProductId)`
  panel and predicts the complete seven-day horizon in one batch.
- **Recursive** trains genuine one-step models and then predicts one day at a
  time, appending each prediction to history before building the next step.
- **Both** evaluates and exports both strategies independently. Automatic
  strategy selection defaults to development OOF metrics from the
  `conditional/common/global` population. `--selection-protocol test-aligned`
  instead uses frozen winter/regular/event stratum weights. The recent
  benchmark is reporting only and never changes selection or ensemble eligibility.

NeuralNet, XGBoost, and LightGBM are trained separately for direct and
recursive use. Dynamic Ridge remains a direct-only structured benchmark.
Recursive inference receives only an explicit allowlist of future-known
covariates and treats generated future rows as available, matching the
observed-sales-conditional-on-availability forecast contract.

The unrounded model-strategy forecasts are stored in
`outputs/final_forecasts.parquet`. Strategy-specific submissions and paired
strategy diagnostics are written alongside the canonical submission.

## Interrupted-run recovery and recursive stability

Every completed CV fold is now written atomically to:

```text
outputs/checkpoints/<strategy>/<origin_type>/<origin>.pkl
```

Resume an interrupted run with:

```bash
caffeinate -i uv run python ml/pipeline.py \
  --forecast-strategy both \
  --primary-strategy auto \
  --submission-model NeuralNet \
  --selection-metric WAPE \
  --resume \
  2>&1 | tee pipeline_both.log
```

Use `--reset-checkpoints` after changing model or feature semantics. Each
checkpoint includes a schema/config signature, so incompatible checkpoints
are ignored rather than silently mixed into a new experiment.

Dynamic Ridge is direct-only. The generic recursive engine still replaces
non-finite or catastrophically large NeuralNet/tree outputs with the recorded
same-weekday baseline before they can contaminate later lag features. This is
a numerical stability guard, not the prediction-cap optimization left for
Tier C3.
