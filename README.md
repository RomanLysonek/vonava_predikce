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
  Target is `log1p`-transformed; trained with Huber loss.
- **Recursive multi-step forecasting**: the 7-day horizon is predicted one
  day at a time. Each day's (ensemble-averaged) prediction is appended to
  that product's history before computing the next day's lag features.
  This matters a lot here — freezing the lag features at the last known
  training value (a common shortcut) makes every day of the horizon look
  identical to the model, which defeats the purpose of a 7-day forecast.
- **Training data**: rows where `ProductAvailable == False` (~2.9% of rows,
  with anomalous demand) are excluded from the supervised examples, since
  the forecast period is implicitly "available" days — but they're kept
  when computing rolling lag history, since that history should reflect
  what actually happened.
- **Validation**: walk-forward (rolling-origin) cross-validation over the
  last 4 non-overlapping 7-day blocks. Each fold trains only on data
  strictly before its evaluation block and forecasts recursively, so the
  reported metrics mirror the real deployment scenario (no early-stopping
  on the eval fold, no leakage).
- **Baselines**: XGBoost and LightGBM (native categorical support, same
  feature set, log1p target, same recursive forecasting -- an apples-to-apples
  comparison) plus two naive baselines: seasonal-naive (value from 7 days
  prior) and a 28-day moving average. All are evaluated on the same folds.
- **Final submission**: an ensemble of 3 NN seeds trained on all available
  history (XGBoost/LightGBM are comparison baselines only, not the
  submission, per the task brief).

## Results (walk-forward CV, 4 folds x 7 days)

| model         |   MAE |  RMSE |   MAPE |
|---------------|------:|------:|-------:|
| NeuralNet     | 12.14 | 18.13 |  91.7% |
| XGBoost       |  8.34 | 12.56 |  52.8% |
| LightGBM      |  8.59 | 13.88 |  52.1% |
| SeasonalNaive | 23.27 | 34.62 | 214.6% |
| MovingAvg28   | 37.17 | 49.97 | 509.3% |

Honest result: the tree baselines actually edge out the neural net here on
raw error (though all three comfortably beat the naive baselines -- **NN is
+47.8%** MAE better than seasonal-naive). This is unsurprising on a small,
tabular, ~50k-row dataset -- exactly the regime the task brief itself says
trees are the standard choice for. The NN remains the submission because the
brief explicitly asked for a non-tree approach; the tree numbers are here so
that trade-off is transparent rather than hidden. Exact numbers regenerate
into `cv_results.csv` each run and will vary slightly run-to-run (no fixed
seed across folds' data-dependent init) but the ranking is stable.

## Repo layout

```
data/                    train_data.parquet, test_data.parquet (inputs)
ml/
  features.py            torch-free: config, feature engineering, tree feature
                          framing, model-agnostic recursive forecasting, naive
                          baselines, metrics. Shared by solution_final.py and
                          tree_worker.py (see "macOS note" below for why the
                          split exists).
  tree_worker.py         XGBoost/LightGBM training + recursive forecasting,
                          run as a subprocess (never imports torch)
  solution_final.py      main pipeline: NN model, walk-forward CV (incl. the
                          tree baselines via tree_worker.py), final ensemble
                          training, recursive forecasting, submission +
                          results.json export
  export_results.py      rebuild outputs/results.json without the CV/NN retrain
                          (still retrains XGBoost/LightGBM's final forecast --
                          cheap relative to the full walk-forward CV)
outputs/                 submission.csv/.parquet, cv_results.csv, forecast_plot.png,
                          results.json (generated by ml/solution_final.py)
tests/
  test_pipeline.py       unit tests: feature engineering, baselines, metrics,
                          recursive-forecast lag update mechanic, tree_worker
                          subprocess smoke test
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
uv run python ml/solution_final.py   # runs CV, trains final ensemble, writes outputs/ + results.json
uv run pytest tests/ -v              # unit tests
```

Uses `mps` automatically on Apple Silicon if available, else CPU.

**macOS setup note:** XGBoost/LightGBM's macOS wheels need Homebrew's OpenMP
runtime: `brew install libomp`. Separately, PyTorch bundles its *own* copy of
that same runtime -- loading both copies in one process crashes the
interpreter the moment either trains. That's why `tree_worker.py` runs
XGBoost/LightGBM in a dedicated subprocess (via `run_tree_baselines` in
`solution_final.py`) instead of importing them alongside torch directly;
`ml/features.py` holds the torch-free code both sides share.

## Interactive results dashboard

A local FastAPI + vanilla-JS/Chart.js dashboard presents the walk-forward CV
comparison, per-fold table, a per-product history-vs-forecast chart, and the
full submission grid.

```bash
uv run python webapp/server.py       # http://127.0.0.1:8999 (port set in webapp/server.py)
```

It reads `outputs/results.json` fresh on every request, so after editing
`ml/solution_final.py` and rerunning it (or just `uv run python
ml/export_results.py` to skip retraining), reload the page to see updated
numbers. The server runs with `--reload`, and the frontend is static
HTML/CSS/JS with no build step — edit `webapp/static/*` and refresh the
browser to iterate on the presentation.

**Pages:**
- `/` — overview: all 5 models side by side (one column each), MAE/RMSE bar
  chart, the full CV fold table, and a multi-model forecast-vs-history chart
  per product.
- `/model/<slug>` — one page per model (`neuralnet`, `xgboost`, `lightgbm`,
  `seasonalnaive`, `movingavg28`) with its own metrics, per-fold chart, and
  product explorer. `webapp/static/model.html` is one shared template;
  `model.js` reads the slug from the URL.

Each model's color is its own project's real brand color (PyTorch orange for
the NN, XGBoost's brandfetch.com purple, the `sphinx_rtd_theme` blue LightGBM's
own readthedocs page uses), defined once in `ml/features.py::MODEL_META` and
served to the frontend via `results.json["models"]` -- not hand-picked or
duplicated in CSS/JS.
