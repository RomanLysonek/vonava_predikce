# Notino Quantity Forecast

Interview assignment: forecast total `Quantity` (`QuantityApp + QuantityWeb`)
for 30 products over the 7 days following the training window. The complete
brief is in `task.md`. Because the brief explicitly requests a **non-tree-based**
primary solution, the canonical submission is a PyTorch feed-forward neural
network with product and campaign embeddings. XGBoost and LightGBM are retained
as fully walk-forward-validated reference models rather than hidden comparisons.

## Final state

- **Canonical submission:** NeuralNet using the direct multi-horizon strategy.
- **Primary metric:** conditional-demand WAPE on the common evaluation population.
- **Selection protocol:** frozen test-aligned weighting of development strata;
  the recent benchmark is confirmation-only.
- **Confirmed NeuralNet objective:** MSE on a baseline-relative log residual.
- **Confirmed structured-model targets:** XGBoost `residual`, LightGBM `log1p`.
- **Frozen cross-model ensemble:** 0.36 NeuralNet + 0.25 XGBoost + 0.39 LightGBM.
- **Untouched final audit:** NeuralNet test-aligned WAPE `27.83%`; ensemble
  `27.97%`. NeuralNet therefore remains canonical even though the ensemble was
  slightly better on global WAPE (`29.93%` versus `30.11%`).
- **Submission artifact:** `outputs/submission.csv`; the transparent secondary
  blend is `outputs/submission_ensemble.csv`.

## Presentation quick start

```bash
uv sync --frozen --group dev
uv run python webapp/server.py
```

Open `http://127.0.0.1:8999`. The static GitHub Pages build is in `docs/`.
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
  targets and their quantities are censored from demand lags rather than
  interpreted as genuine zero demand. The primary target is therefore demand
  conditional on availability; realized-sales reporting remains diagnostic.
- **Validation:** rolling-origin walk-forward evaluation has a seasonally
  distributed `development` set for decisions, a disjoint `recent_benchmark`
  for confirmation, and three untouched final-audit origins. Every fold trains
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
- **Final model policy:** the submitted NeuralNet is an ensemble of seeds
  `42`, `123` and `777`. The cross-model convex ensemble is retained as an
  auditable alternative but was not promoted after the frozen final audit.

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
docs/                    byte-equivalent static dashboard for GitHub Pages
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
- `/dataset` — concise dataset profile, finding-to-decision mapping, retained/rejected experiments and known limitations,
  all seven models, paired strategy comparison, development horizon curves,
  benchmark fold metrics, product explorer and canonical submission.
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
  benchmark is reporting only and never changes the selection.

NeuralNet, XGBoost, and LightGBM are trained separately for direct and
recursive use. Dynamic Ridge remains a direct-only structured benchmark.
Recursive inference receives only an explicit allowlist of future-known
covariates and treats generated future rows as available, matching the
conditional-demand forecast contract.

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

## Tier C0.1 recursive stability and Tier C1 nonstationarity

Tier C0 fixed direct-model coverage, but the broader early-history training
population exposed a finite recursive NeuralNet extrapolation above 100,000
units. Tier C0.1 contains that failure in two layers:

1. each NN seed stores robust 0.1%/99.9% training residual bounds, widened by
   one log unit, and applies them **only during recursive inference**;
2. the generic recursive engine uses a broad last-resort numerical limit of
   `max(10_000, 50 × observed pre-origin product maximum)` and never lets a
   generated prediction inflate its own next-step limit.

These are numerical/extrapolation guards, not an ordinary demand cap. Direct
predictions remain uncapped. Recursive artifacts now expose residual-guard,
raw-residual, safety-limit, fallback, non-finite and catastrophic-guard
statistics, including per-origin summaries.

Run the reduced real-data C0.1 check before the C1 screen:

```bash
caffeinate -i uv run python ml/experiments/c01_recursive_stability.py \
  --strict \
  2>&1 | tee pipeline_c01_recursive_check.log
```

The check trains one seed for four representative recursive folds. It fails
with a non-zero exit status if predictions are non-finite, exceed the broad
safety envelope, or remain dominated by an explosion.

### C1 controls

The pipeline now supports four orthogonal nonstationarity controls:

```text
--training-window-days all|730|365|...
--recency-half-life-days none|365|180|90|...
--baseline-variant lag7|weighted_4321|weighted_8421|weekday_median
--trend-features on|off
```

History windows filter supervised target rows while leaving earlier history
available for leakage-safe lag construction. Exponential half-life weights
are normalised to mean one and are passed consistently to the NN, XGBoost,
LightGBM and Dynamic Ridge training objectives.

The optional trend group contains:

- absolute target calendar time;
- 7-day/28-day and 14-day/28-day log-level ratios;
- latest-demand/28-day log-level ratio;
- 7-day and 28-day log-demand slopes;
- a robust annual reference from lags 364/365/371;
- baseline-versus-annual log ratio and annual-reference missingness.

### Staged C1 screen

Do not launch a full Cartesian search. The dedicated runner evaluates a
controlled direct-only screen using one seed, 12 epochs, batch 2048/fixed and
four stratified origins. It executes:

1. recency window/half-life candidates;
2. baseline variants around the recency winner;
3. trend features off/on around the preceding winner.

Start a fresh screen with:

```bash
caffeinate -i uv run python ml/experiments/c1_recency_screening.py \
  --reset \
  2>&1 | tee pipeline_c1_screening.log
```

If interrupted, preserve completed candidate-fold checkpoints:

```bash
caffeinate -i uv run python ml/experiments/c1_recency_screening.py \
  --resume \
  2>&1 | tee -a pipeline_c1_screening.log
```

Outputs:

```text
outputs/c1_screening/c1_screening_results.csv
outputs/c1_screening/recommendation.json
outputs/c1_screening/candidate_oof/*.csv
outputs/c1_screening/checkpoints/
```

The recommendation is selected by test-aligned NeuralNet WAPE, subject to a
3% broad-development WAPE quality guard against the identical-runtime control.
The screen is a ranking experiment, not a final reported benchmark.

The runner prints two full-confirmation commands. The first intentionally
resets the old full-pipeline checkpoints; after an interruption, use the
separate resume command and do **not** reset them again. The confirmation uses
all direct development and recent-benchmark origins, three seeds, 30 CV epochs
and the statistically controlled `512/fixed` policy.

A recommendation can also be applied manually:

```bash
caffeinate -i uv run python ml/pipeline.py \
  --forecast-strategy direct \
  --primary-strategy direct \
  --submission-model NeuralNet \
  --selection-metric WAPE \
  --selection-protocol test-aligned \
  --c1-config outputs/c1_screening/recommendation.json \
  --nn-batch-size 512 \
  --nn-lr-scaling fixed \
  --reset-checkpoints \
  2>&1 | tee pipeline_c1_direct_512_fixed.log
```

After an interrupted confirmation run:

```bash
caffeinate -i uv run python ml/pipeline.py \
  --forecast-strategy direct \
  --primary-strategy direct \
  --submission-model NeuralNet \
  --selection-metric WAPE \
  --selection-protocol test-aligned \
  --c1-config outputs/c1_screening/recommendation.json \
  --nn-batch-size 512 \
  --nn-lr-scaling fixed \
  --resume \
  2>&1 | tee -a pipeline_c1_direct_512_fixed.log
```

The final C1 candidate should be compared with the frozen C0 direct baseline
before C2 feature-group work begins. Recursive strategy robustness is checked
later for the winning data-aware configuration rather than doubling every C1
experiment.

## Tier C2 semantic feature-group ablations

The confirmed C1 half-life-365 run improved recent-benchmark NeuralNet WAPE,
but its broad-development WAPE was slightly worse and its test-aligned score
was effectively tied with the C0 baseline. C2 therefore does not silently
assume that 365 days is universally optimal: after selecting semantic groups,
the screening runner rechecks the winning representation under no recency
decay and a 90-day half-life.

C2 features are disabled by default. Enable named groups with:

```text
--c2-feature-groups none|all|price,campaign,lifecycle,market,event
```

The groups are:

- **price**: target list/effective price relative to the observed origin,
  lag-7 price and recent 28-day product median, plus app-vs-web effective-price
  advantage;
- **campaign**: web/app campaign-active flags, app-only incentives, subtype
  agreement, positive discount with subtype `-1`, and app discount advantage;
- **lifecycle**: current availability/gap state, consecutive unavailability,
  days since the last observed row, cumulative observed/available history and
  reavailability;
- **market**: leakage-safe aggregate demand known at the forecast origin and
  future-known cross-sectional campaign/discount intensity for the target day;
- **event**: deterministic distance/proximity to Black Friday, Christmas,
  Valentine's Day and Mother's Day, plus Black-Friday/Christmas/New-Year
  windows.

Market demand features use quantities only through the origin. Target-date
market features use campaign, discount and price covariates only; they never
use target quantities.

### Staged C2 screen

Run the direct-first screen from the confirmed C1 policy:

```bash
caffeinate -i uv run python ml/experiments/c2_feature_screening.py \
  --reset \
  2>&1 | tee pipeline_c2_screening.log
```

The runner evaluates every group individually, then performs forward selection
around the best eligible candidate. A candidate must retain full coverage and
stay within the broad-development WAPE guard. It finally evaluates the selected
semantic representation under the C1 half-life sensitivity policies.

Resume an interrupted screen without deleting completed folds:

```bash
caffeinate -i uv run python ml/experiments/c2_feature_screening.py \
  --resume \
  2>&1 | tee -a pipeline_c2_screening.log
```

Artifacts:

```text
outputs/c2_screening/c2_screening_results.csv
outputs/c2_screening/recommendation.json
outputs/c2_screening/candidate_oof/*.csv
outputs/c2_screening/checkpoints/
```

The recommendation JSON contains separate fresh and resume commands for the
full `512/fixed`, three-seed direct confirmation. The winning data-aware
configuration is checked under `both` strategies only after C2/C3 choices are
frozen; recursive execution is not duplicated across every semantic ablation.

## Combined Tier C3 objectives and Tier C4 channel model

The C2 screen selected all five semantic groups. No recency decay and the
365-day half-life were effectively tied for the NeuralNet, so the combined
C3/C4 runner carries both policies into one final sensitivity gate instead of
requiring a separate full C2 confirmation first.

C3 evaluates:

- NeuralNet total-demand loss: Huber, MSE, Huber/MSE mixture, and Log-Cosh;
- NeuralNet target: baseline-relative `log1p` residual or raw `log1p` demand;
- XGBoost/LightGBM target: raw `log1p`, baseline-relative residual, or Tweedie,
  selected independently for each tree family.

C4 adds an optional leakage-safe channel state:

- current and lag-7 app share;
- volume-weighted 7/28-day app share;
- recent-versus-long app-share movement;
- recent app/web quantity levels;
- an auxiliary app-share head sharing the total-demand representation.

Total quantity remains the submitted target. The auxiliary head is selected
only when it materially improves total-demand validation within the broad-WAPE
guard. App-share error is a diagnostic/tiebreaker, never a substitute for total
forecast quality. Recursive inference feeds the predicted share back into
synthetic app/web history; models without a share head use the observed recent
28-day mix rather than an all-app placeholder.

### Fast C3/C4 screen

```bash
caffeinate -i uv run python ml/experiments/c34_objective_channel_screening.py \
  --reset \
  2>&1 | tee pipeline_c34_screening.log
```

The runner is substantially faster than C2 because NN candidates skip all
structured models. Statistically identical NN configurations are reused in
memory, and XGBoost/LightGBM are trained only for the three tree target
formulations. Screening uses four stratified origins, one seed, 12
epochs, and batch `2048/fixed`. Every stage retains its control unless the
candidate improves test-aligned WAPE by at least 0.2% relative while remaining
inside the 3% broad-development guard.

Resume without deleting completed candidate folds:

```bash
caffeinate -i uv run python ml/experiments/c34_objective_channel_screening.py \
  --resume \
  2>&1 | tee -a pipeline_c34_screening.log
```

Artifacts:

```text
outputs/c34_screening/c34_screening_results.csv
outputs/c34_screening/recommendation.json
outputs/c34_screening/candidate_oof/*.csv
outputs/c34_screening/checkpoints/
```

The recommendation contains the one full direct `512/fixed`, three-seed
confirmation command. That single confirmation jointly validates the selected
C2 features, C3 objective, and C4 channel formulation before Tier C5 ensemble
fitting.

## Tier C5 constrained OOF ensemble and Tier C6 delivery

Tier C5 fits a convex blend only after the member models have produced full
walk-forward predictions. The default members are NeuralNet, XGBoost, and
LightGBM. For each available strategy, the ensemble:

- uses development OOF rows only for fitting;
- requires the common conditional-demand population across all members;
- minimizes the frozen test-aligned stratum-weighted WAPE;
- constrains every weight to be nonnegative and all weights to sum to one;
- performs a deterministic exhaustive simplex search (1% grid by default);
- applies the frozen weights unchanged to the recent benchmark and final test;
- records recent-benchmark confirmation but never refits from that benchmark.

Enable it with:

```text
--ensemble on
--ensemble-models NeuralNet,XGBoost,LightGBM
--ensemble-grid-step 0.01
```

The pipeline writes:

```text
outputs/ensemble_weights.json
outputs/ensemble_weights.csv
outputs/ensemble_comparison.csv
outputs/submission_ensemble.csv
outputs/per_product_summary.csv
outputs/top_decile_summary.csv
outputs/top_error_rows.csv
outputs/ablation_showcase.csv
```

The canonical task submission may remain NeuralNet while the unrounded and
rounded ensemble forecasts are preserved as separate artifacts. With
`--submission-model auto`, an ensemble is eligible only after it clears the
minimum development gain and recent-benchmark tolerance.

### Combined C3/C4 confirmation and C5 fit

Run the selected C3/C4 configuration once and fit C5 from that same full OOF
run rather than performing another duplicate training pass:

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
  2>&1 | tee pipeline_c34_c5_c6_direct_512_fixed.log
```

Resume after interruption by replacing `--reset-checkpoints` with `--resume`
and appending to the same log.

### Frozen final audit

After the full run has frozen C5 weights, execute the disjoint audit origins
once using the exact same modeling arguments:

```bash
caffeinate -i uv run python ml/run_final_audit.py \
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
  2>&1 | tee pipeline_final_audit.log
```

The script refuses a second audit unless `--force` is supplied. It never
refits ensemble weights and refreshes `results.json` from persisted artifacts.

### Tier C6 dashboard and GitHub Pages

Every normal pipeline/export now adds:

- per-product WAPE, MAE, volume and bias;
- winter/test-like, regular, and event-regime diagnostics;
- highest-demand-decile performance;
- the largest recent row-level errors for business interpretation;
- channel-share diagnostics when a C4 auxiliary head is present;
- C1/C2/C3/C4 ablation recommendations;
- C5 weights and benchmark confirmation;
- the one-shot final-audit table once available.

`outputs/results.json` is copied to `webapp/static/results.json`, and a static
GitHub Pages site is generated in `docs/`. Configure Pages to serve the `/docs`
directory. The static model pages use query-string navigation and no FastAPI
server is required.

### Final C5/C6 confirmation result

The completed direct confirmation run froze the convex weights at 0.36
NeuralNet, 0.25 XGBoost, and 0.39 LightGBM. The ensemble improved the
development test-aligned WAPE by 4.97% and passed the recent-benchmark guard.
On the three untouched final-audit origins, the ensemble was slightly better
on global WAPE (0.299304 vs 0.301079 for NeuralNet) but slightly worse on the
frozen test-aligned objective (0.279737 vs 0.278328). Therefore the canonical
submission remains NeuralNet, while the ensemble is retained as a transparent
secondary artifact rather than being refitted after the audit. The dashboard
shows both audit metrics explicitly.
