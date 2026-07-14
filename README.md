# Notino Quantity Forecast

Interview assignment: forecast total `Quantity` (`QuantityApp + QuantityWeb`)
for 30 products over the 7 days following the training window. See
`task.md` for the full brief (Czech). The brief explicitly asks for a
**non-tree-based** approach as the primary solution, so the submission is a
PyTorch feed-forward network with product/campaign embeddings -- XGBoost and
LightGBM are still included as walk-forward-validated baselines, since the
brief itself frames them as the standard comparison point.

## Approach

- **Features**: cyclic calendar encodings (day-of-week/month/day-of-year/
  week-of-year), campaign/discount/price info, price relative to a
  product's own historical median, days-since-launch, and rolling
  mean/std/median demand lags (7/14/28 days).
- **Categorical handling**: `CampaignSubTypeWeb/App` are category codes
  (-1, 0, 1, 2, 3, 4, 5, 16, 18, 19), not an ordinal scale, so they're fed
  through embedding layers instead of as raw numeric features.
- **Model**: an MLP (256→128→64) with BatchNorm/GELU/Dropout, taking the
  numeric features plus product, campaign-web and campaign-app embeddings.
  Target is a **baseline-relative log residual**: `log1p(Quantity) - log1p(target_baseline)`.
  Trained with Huber loss.
- **Seeds**: random seeds for both the NN ensemble and the tree-based baselines are
  fixed (`Config.seeds`) to ensure reproducibility.
- **Two forecast strategies**: **direct** predicts all seven target days
  from a stacked `(ForecastOrigin, Horizon, ProductId)` panel, while
  **recursive** trains genuine one-step models and feeds each generated
  prediction into the next step's history. Both strategies share the same
  end-of-origin information cutoff and are trained independently. `both`
  mode evaluates them on paired development OOF keys and selects the
  canonical strategy using development-only `conditional/common/global`
  metrics.
- **Training data**: rows where `ProductAvailable == False` are excluded
  from supervised targets. Their quantities are censored from lag and
  rolling-demand features rather than being treated as genuine zero demand.
  Recursive synthetic future rows are marked available, matching the
  conditional-demand forecast contract.
- **Validation**: walk-forward (rolling-origin) cross-validation over two
  labeled sets of origins -- a broader, seasonally-scattered `development`
  set used to make modeling decisions, and a `recent_benchmark` set (the last
  `n_cv_folds` non-overlapping 7-day blocks, a pseudo-test check) as
  a final benchmark of recent performance. Each fold trains only on data
  strictly before its evaluation block, so the reported metrics mirror the
  real deployment scenario (no early-stopping on the eval fold, no leakage).
- **Baselines**: XGBoost, LightGBM, and Dynamic Ridge use the same
  strategy-specific feature contracts as the NN (native categorical support
  or one-hot encoding as appropriate). 
  **NeuralNet** and **Dynamic Ridge** predict a baseline-relative log residual, 
  while **XGBoost** and **LightGBM** predict `log1p(Quantity)` directly. 
  Two naive baselines are also included: seasonal-naive (value from 7 days prior) 
  and a 28-day moving average. All are evaluated on the same folds and the same 
  **common population** of rows where every model produced a valid prediction.
- **Evaluation Regimes**: **Conditional Demand** (only days where the product
  was available in stock) is the primary evaluation regime, as it measures the
  true demand the model is meant to capture. **Realized Sales** (all days,
  including stockouts) is available as a diagnostic toggle.
- **Final submission**: by default, an ensemble of three NN seeds trained
  under the development-selected direct or recursive strategy. XGBoost,
  LightGBM and Dynamic Ridge remain comparison models unless
  `--submission-model` explicitly selects otherwise.

## Results (walk-forward CV, 4 folds x 7 days, Conditional Demand)

The results below are from the `recent_benchmark` origins:

| model         |   MAE |  RMSE |   MAPE |
|---------------|------:|------:|-------:|
| NeuralNet     |  9.60 | 13.89 |  74.7% |
| XGBoost       |  7.85 | 12.26 |  55.0% |
| LightGBM      |  7.79 | 11.86 |  55.7% |
| SeasonalNaive | 23.27 | 34.62 | 214.6% |
| MovingAvg28   | 37.17 | 49.97 | 509.3% |

Honest result: the tree baselines actually edge out the neural net here on
raw error (though all three comfortably beat the naive baselines -- **NN is
+58.8%** MAE better than seasonal-naive). This is unsurprising on a small,
tabular, ~50k-row dataset -- exactly the regime the task brief itself says
trees are the standard choice for. The NN remains the submission because the
brief explicitly asked for a non-tree approach; the tree numbers are here so
that trade-off is transparent rather than hidden. Exact numbers regenerate into `cv_results.csv` each run. Seeds are fixed;
small differences can still arise across hardware/library backends. These
benchmarks use the most recent history available at training time.

## Repo layout

```
data/                    train_data.parquet, test_data.parquet (inputs)
ml/
  framework.py           torch-free: config, feature engineering, direct and
                          one-step panel builders, recursive state transition,
                          model registry/metadata, metrics. Shared by every
                          model under models/ and by pipeline.py (see "macOS
                          note" below for why the torch-free split exists).
  models/
    neural_net.py          NN model: QuantityNet (w/ horizon embedding),
                          shared training and direct/recursive prediction
    xgboost_model.py        XGBoost: train/predict directly on the panel
    lightgbm_model.py        LightGBM: train/predict directly on the panel
    dynamic_ridge.py        Dynamic Ridge: sklearn linear baseline with scaling/imputation
    naive_baselines.py        seasonal-naive + 28-day moving average
  tree_worker.py         isolated structured-model dispatcher for direct and
                          recursive XGBoost, LightGBM and Dynamic Ridge jobs
  pipeline.py            strategy-aware CV/training/export orchestrator,
                          development-only selection and artifact generation
  benchmark_nn_batch_size.py
                          real-fold MPS/CUDA batch throughput + WAPE benchmark
  export_results.py      rebuilds results.json exclusively from persisted
                          artifacts; it never retrains models
outputs/                 canonical and strategy-specific submissions, OOF and
                          final forecasts, strategy summaries, horizon metrics,
                          timings and results.json
tests/
  test_pipeline.py       feature engineering, baselines and metric tests
  test_direct_recursive_strategies.py
                          strategy alignment, feedback and leakage tests
  test_webapp_strategy_sync.py
                          JSON contract and frontend smoke checks
  test_nn_performance.py  batch/LR/backend and recommendation tests
webapp/
  server.py              FastAPI app serving the dashboard + /api/results,
                          plus /model/{slug} for the per-model pages
  static/                index.html + app.js   (overview / comparison page)
                          model.html + model.js (shared per-model page template)
                          common.js              (shared nav + fetch/format helpers)
                          styles.css             (Chart.js loaded from CDN)
archive/
  solution_draft_v1.py   earlier draft, kept for reference only
task.md                  original assignment brief (Czech)
```

## Running

```bash
uv run python ml/pipeline.py         # runs CV, trains final ensemble, writes outputs/ + results.json
uv run pytest tests/ -v              # unit tests
```

Uses `mps` automatically on Apple Silicon if available, else CPU.

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
only when it was measured on the same accelerator type; without it, the safe
512/fixed policy is preserved.

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
model policy. Existing 512/fixed stability-v3 checkpoints remain reusable
with the automatic safe fallback.

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

A local FastAPI + vanilla-JS/Chart.js dashboard presents strategy-aware
walk-forward comparisons, per-fold tables, paired direct-vs-recursive results,
horizon curves, per-product forecasts, and the canonical submission grid.

```bash
uv run python webapp/server.py       # http://127.0.0.1:8999 (port set in webapp/server.py)
```

It reads `outputs/results.json` fresh on every request, so after editing
`ml/pipeline.py` (or a model under `ml/models/`) and rerunning it (or just
`uv run python ml/export_results.py` to skip retraining), reload the page to
see updated numbers. The server runs with `--reload`, and the frontend is
static HTML/CSS/JS with no build step — edit `webapp/static/*` and refresh
the browser to iterate on the presentation.

**Pages:**
- `/` — overview with direct/recursive and conditional/realized selectors,
  all six models, paired strategy comparison, development horizon curves,
  benchmark fold metrics, product explorer and canonical submission.
- `/model/<slug>` — one page per model (`neuralnet`, `xgboost`, `lightgbm`,
  `dynamicridge`, `seasonalnaive`, `movingavg28`) with strategy-aware metrics,
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
  --selection-metric WAPE
```

- **Direct** trains on the stacked `(ForecastOrigin, Horizon, ProductId)`
  panel and predicts the complete seven-day horizon in one batch.
- **Recursive** trains genuine one-step models and then predicts one day at a
  time, appending each prediction to history before building the next step.
- **Both** evaluates and exports both strategies independently. Automatic
  strategy selection uses development OOF metrics from the
  `conditional/common/global` population; the recent benchmark is reporting
  only and never changes the selection.

All model families—NeuralNet, XGBoost, LightGBM, and Dynamic Ridge—are trained
separately for direct and recursive use. Recursive inference receives only an
explicit allowlist of future-known covariates and treats generated future rows
as available, matching the conditional-demand forecast contract.

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

Before committing to the full run, the regression test for the previously
failing recursive Dynamic Ridge fold can be executed directly:

```bash
uv run pytest -q \
  tests/test_recursive_dynamic_ridge_real_fold.py::test_recursive_dynamic_ridge_real_2024_11_29_fold_is_finite
```

Recursive Dynamic Ridge constrains only its recursive residual extrapolation
to the residual support observed during training. The generic recursive engine
also replaces non-finite or catastrophically large numerical outputs with the
recorded same-weekday baseline before they can contaminate later lag features.
This is a numerical stability guard, not the prediction-cap optimization left
for Tier C3.
