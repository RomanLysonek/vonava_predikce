# Retraining the submission: commands and complete CLI reference

This guide applies to the presentation-ready English repository. Run all
commands from the repository root. The primary entry point is:

```bash
uv run python ml/pipeline.py [OPTIONS]
```

## 1. Confirmed final configuration

The checked-in canonical submission uses:

- forecast strategy: `direct`;
- canonical model: `NeuralNet`;
- selection metric: `WAPE`;
- selection protocol: `test-aligned`;
- training history: all available supervised targets, without recency decay;
- baseline: `weighted_4321`;
- C2 groups: `price,campaign,lifecycle,market,event`;
- NeuralNet objective: `mse` on a baseline-relative `residual` target;
- XGBoost target: `residual`;
- LightGBM target: `log1p`;
- channel-history features and auxiliary channel head: disabled;
- NeuralNet seeds: `42`, `123`, `777`;
- batch/LR policy: `512/fixed`;
- optional C5 ensemble: enabled for comparison, while
  `outputs/submission.csv` remains NeuralNet/direct.

Important output files after a successful run:

```text
outputs/submission.csv                 canonical rounded submission
outputs/submission.parquet             canonical submission in Parquet
outputs/submission_ensemble.csv         secondary frozen ensemble submission
outputs/final_forecasts.parquet         unrounded final model forecasts
outputs/oof_predictions.parquet         development and benchmark OOF predictions
outputs/results.json                    dashboard data contract
webapp/static/results.json              local dashboard copy
docs/results.json                       GitHub Pages copy
```

`--submission-model NeuralNet` controls the canonical `submission.csv`.
`--ensemble on` additionally fits and exports the cross-model ensemble; it does
not automatically replace the canonical model.

## 2. Environment preparation

```bash
cd /path/to/vonava_predikce-en-final
uv sync --frozen --group dev
```

On macOS, native XGBoost and LightGBM wheels require OpenMP:

```bash
brew install libomp
```

Inspect the parser directly at any time:

```bash
uv run python ml/pipeline.py --help
```

The examples use `caffeinate -i` to prevent a Mac from sleeping. `tee` displays
the log and writes it to disk simultaneously.

# 3. Recommended retraining commands

## 3.1 Exact final retraining using the retained recommendation JSON

This is the recommended command for reproducing the confirmed final project
configuration. The C3/C4 settings are loaded from the versioned
`outputs/c34_screening/recommendation.json` file.

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

This command:

- deletes stale CV checkpoints;
- reruns the complete walk-forward evaluation;
- retrains final models on all eligible training history;
- refits the C5 convex ensemble using development OOF only;
- regenerates submissions, diagnostics and dashboard artifacts;
- keeps NeuralNet/direct as the canonical submission.

## 3.2 Fully explicit reproduction without a recommendation file

This version writes every selected C3/C4 option directly on the command line.
It is the most self-contained reproducibility recipe.

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
  --nn-loss mse \
  --nn-target-mode residual \
  --nn-combined-mse-weight 0.25 \
  --tree-target-mode log1p \
  --xgboost-target-mode residual \
  --lightgbm-target-mode log1p \
  --channel-history-features off \
  --channel-aux-weight 0.0 \
  --channel-share-smoothing 0.5 \
  --ensemble on \
  --ensemble-models NeuralNet,XGBoost,LightGBM \
  --ensemble-grid-step 0.01 \
  --ensemble-min-relative-improvement 0.002 \
  --ensemble-benchmark-tolerance 0.02 \
  --nn-batch-size 512 \
  --nn-lr-scaling fixed \
  --nn-training-backend auto \
  --reset-checkpoints \
  2>&1 | tee pipeline_final_retrain_explicit.log
```

On Apple Silicon, `--nn-training-backend auto` normally resolves to
`device_resident`.

## 3.3 Resume the same interrupted run

Use exactly the same modeling options, replace `--reset-checkpoints` with
`--resume`, and append to the existing log:

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
  --resume \
  2>&1 | tee -a pipeline_final_retrain.log
```

Only checkpoints whose strategy, feature schema and model signature match the
current run are reused. An incompatible checkpoint is not silently accepted.

## 3.4 NeuralNet/direct without fitting the cross-model ensemble

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
  --ensemble off \
  --nn-batch-size 512 \
  --nn-lr-scaling fixed \
  --reset-checkpoints \
  2>&1 | tee pipeline_neuralnet_direct_only.log
```

The pipeline still trains the reference models required by its standard
comparison tables, but it does not fit or export a new C5 blend.

## 3.5 Evaluate both strategies but keep direct as canonical

This is useful for a presentation comparison. It is substantially more
expensive because the supported learned models are trained independently for
both direct and recursive inference.

```bash
caffeinate -i uv run python ml/pipeline.py \
  --forecast-strategy both \
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
  2>&1 | tee pipeline_both_strategies_direct_submission.log
```

## 3.6 Automatic model and strategy selection

This is a new experiment, not an exact reproduction of the manually frozen
submission decision. The pipeline selects the eligible candidate with the
lowest development `test_aligned_score`; an accepted ensemble can become
canonical.

```bash
caffeinate -i uv run python ml/pipeline.py \
  --forecast-strategy both \
  --primary-strategy auto \
  --submission-model auto \
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
  2>&1 | tee pipeline_auto_selection.log
```

## 3.7 Force another model as the canonical submission

For XGBoost, use the final configuration but change:

```text
--submission-model XGBoost
```

For LightGBM:

```text
--submission-model LightGBM
```

Dynamic Ridge is direct-only:

```text
--forecast-strategy direct --primary-strategy direct --submission-model DynamicRidge
```

To make the accepted cross-model blend canonical:

```text
--submission-model Ensemble --ensemble on
```

That changes the meaning of `outputs/submission.csv`: it will contain the
accepted blend instead of the standalone NeuralNet.

## 3.8 Recursive NeuralNet experiment

```bash
caffeinate -i uv run python ml/pipeline.py \
  --forecast-strategy recursive \
  --primary-strategy recursive \
  --submission-model NeuralNet \
  --selection-metric WAPE \
  --selection-protocol test-aligned \
  --training-window-days all \
  --recency-half-life-days none \
  --baseline-variant weighted_4321 \
  --trend-features off \
  --c2-feature-groups price,campaign,lifecycle,market,event \
  --c34-config outputs/c34_screening/recommendation.json \
  --nn-batch-size 512 \
  --nn-lr-scaling fixed \
  --reset-checkpoints \
  2>&1 | tee pipeline_recursive_experiment.log
```

Recursive inference feeds previous predictions back into later lag state, so
error can accumulate across the seven steps. This is not the confirmed final
configuration.

# 4. Complete `ml/pipeline.py` flag reference

The parser exposes 34 project-specific options, plus standard `-h/--help`.

## 4.1 Strategy, selection and canonical output

### `--forecast-strategy`

```text
direct | recursive | both
parser default: direct
confirmed final value: direct
```

- `direct`: trains multi-horizon models on the stacked
  `(ForecastOrigin, Horizon, ProductId)` panel. All seven days are predicted
  without feeding predictions back into the feature history.
- `recursive`: trains a one-step contract and inserts every generated value
  into the synthetic state before the next step.
- `both`: evaluates both independently and creates a larger set of OOF and
  final-forecast artifacts.

**Effect:** changes the training panel, inference mechanism, metrics and final
predictions. This is a major modeling choice.

### `--primary-strategy`

```text
auto | direct | recursive
parser default: auto
confirmed final value: direct
```

Used when both strategies were trained and the selected model supports both.
`auto` chooses from development OOF according to the selection protocol;
`direct` or `recursive` freezes the canonical strategy manually.

**Effect:** normally changes which already-trained strategy is written to the
canonical submission. With a single `--forecast-strategy`, it has no practical
choice to make. Dynamic Ridge always resolves to direct.

### `--submission-model`

```text
NeuralNet | Ensemble | DynamicRidge | XGBoost | LightGBM | auto
parser default: NeuralNet
confirmed final value: NeuralNet
```

A concrete model name freezes the canonical output. `auto` lets development OOF
select an eligible learned model. `Ensemble` requires `--ensemble on` and a
successful ensemble fit. Dynamic Ridge supports direct only.

**Effect:** primarily changes which forecast becomes `submission.csv`; it does
not by itself alter the hyperparameters of the member estimators.

### `--selection-metric`

```text
WAPE | MAE | RMSE
parser default: WAPE
confirmed final value: WAPE
```

- `WAPE`: absolute error normalized by total actual volume; the project's main
  business objective.
- `MAE`: equal absolute weight per scored product-day.
- `RMSE`: disproportionately penalizes large individual misses.

**Effect:** changes automatic model/strategy ranking and the metric combined by
the test-aligned stratum score. It does not automatically change the training
loss of every estimator. C5 ensemble fitting remains explicitly based on its
frozen test-aligned WAPE objective.

### `--selection-protocol`

```text
global | test-aligned
parser default: global
confirmed final value: test-aligned
```

- `global`: ranks the common conditional-demand development population using
  one global metric.
- `test-aligned`: combines development validation strata using frozen weights:
  `winter_test_like=0.60`, `regular=0.25`, `holiday_event=0.15`.

**Effect:** changes automatic canonical model/strategy selection. It does not
change the raw training rows or an estimator's architecture.

## 4.2 Checkpoint control

### `--resume`

Boolean switch, off by default. Reuses completed fold checkpoints whose stored
signature exactly matches the current run.

**Effect:** intended to reduce runtime only. Use it to continue the same
interrupted experiment.

### `--reset-checkpoints`

Boolean switch, off by default. Deletes the configured checkpoint directory
before training.

**Effect:** forces fresh CV fits. Use after changing strategy, features, target,
loss, batch policy or other model-defining settings.

### `--checkpoint-dir`

```text
path
parser default: outputs/checkpoints
```

Changes where atomic per-fold checkpoints are read and written.

**Effect:** no model change; useful for isolating experiments, for example:

```text
--checkpoint-dir outputs/checkpoints_final
```

## 4.3 NeuralNet runtime and optimization policy

### `--nn-batch-size`

```text
auto | integer >= 2
parser default: auto
confirmed final value: 512
```

`auto` consumes a compatible recommendation from `--nn-benchmark-file`; if the
device type or feature/preprocessing signature does not match, it preserves the
safe batch size 512. An integer forces the value.

**Effect:** changes optimizer steps per epoch, memory use, throughput and
potentially model quality. It is not merely a performance switch.

### `--nn-lr-scaling`

```text
auto | fixed | sqrt | linear
parser default: auto
confirmed final value: fixed
```

Relative to reference batch 512 and base LR 0.001:

- `fixed`: keep LR at 0.001;
- `sqrt`: multiply by `sqrt(batch_size / 512)`;
- `linear`: multiply by `batch_size / 512`;
- `auto`: use the compatible benchmark policy; without one, keep `fixed` for
  batch 512 and use the runtime fallback for other batch sizes.

**Effect:** changes the optimization trajectory and predictions. Use
`512/fixed` for final reproduction.

### `--nn-training-backend`

```text
auto | device_resident | dataloader
parser default: auto
confirmed Apple Silicon resolution: device_resident
```

- `device_resident`: keeps complete fold tensors on MPS/CUDA and slices batches
  on-device.
- `dataloader`: standard CPU-side DataLoader batching.
- `auto`: device-resident on MPS/CUDA, DataLoader on CPU.

**Effect:** designed to change execution speed and memory behavior only. It is
excluded from the checkpoint signature, although floating-point execution may
not be bit-identical across hardware backends.

### `--nn-benchmark-file`

```text
path
parser default: outputs/nn_batch_benchmark.json
```

Used only by automatic batch/LR resolution. The file is produced by
`ml/benchmark_nn_batch_size.py` and accepted only when its device and model
signature match the current run.

**Effect:** can indirectly change batch size and LR scaling. It has no effect
when both are explicitly fixed.

## 4.4 C1: history, recency, baseline and trend

### `--c1-config`

Path to a recommendation JSON produced by `ml/experiments/c1_recency_screening.py`.

**Precedence:** explicit C1 CLI option > JSON value > `Config` default.

**Effect:** depending on its content, can change training population, sample
weights, baseline formulation and trend features.

### `--training-window-days`

```text
all | positive integer
code default: all
confirmed final value: all
```

`all` retains every eligible supervised target. A value such as `730` or `365`
keeps only targets within that many days before each forecast origin. Older
rows may still be used to construct leakage-safe lags even when they are no
longer supervised targets.

**Effect:** changes training-set size and temporal composition. A shorter window
adapts faster to regime change but increases variance and discards older
seasonal examples.

### `--recency-half-life-days`

```text
none | positive number
code default: none
confirmed final value: none
```

`none` gives eligible supervised rows equal temporal weight. With `365`, sample
weight halves for every 365 days of age. Weights are normalized to mean one and
are used by NeuralNet, XGBoost, LightGBM and Dynamic Ridge.

**Effect:** changes the optimization objective without removing old rows. A
shorter half-life emphasizes recent demand regimes more strongly.

### `--baseline-variant`

```text
lag7 | weighted_4321 | weighted_8421 | weekday_median
code default: weighted_4321
confirmed final value: weighted_4321
```

The baseline uses same-weekday lags 7, 14, 21 and 28:

- `lag7`: most recent same weekday only;
- `weighted_4321`: recency weights 4:3:2:1;
- `weighted_8421`: stronger newest-week emphasis, 8:4:2:1;
- `weekday_median`: robust median of available values.

Weights are renormalized when some lags are missing.

**Effect:** changes naive baseline values, residual training targets and related
features/fallbacks. It can materially affect every residual-target model.

### `--trend-features`

```text
on | off
code default: off
confirmed final value: off
```

Enables absolute calendar time, short/long demand ratios, log-demand slopes and
annual references from 364/365/371-day lookups.

**Effect:** changes the feature schema and checkpoint signature. The final C1
screen did not retain this group.

## 4.5 C2 semantic feature groups

### `--c2-config`

Path to a recommendation JSON produced by `ml/experiments/c2_feature_screening.py`.

**Precedence:** `--c2-feature-groups` > JSON value > empty default group set.

**Effect:** can replace the feature-group configuration according to the file.

### `--c2-feature-groups`

```text
none | all | comma-separated subset
available groups: price,campaign,lifecycle,market,event
code default: none
confirmed final value: price,campaign,lifecycle,market,event
```

- `price`: list/effective-price ratios, changes versus origin/lag-7/rolling
  history, and app price advantage.
- `campaign`: active campaign state, app-only incentives, subtype agreement,
  subtype/discount inconsistencies and app discount advantage.
- `lifecycle`: current availability, calendar-gap state, unavailable streaks,
  observation age, reavailability and product-history support.
- `market`: leakage-safe aggregate demand history and future-known cross-section
  campaign/price intensity.
- `event`: distances and windows around Black Friday, Christmas, New Year,
  Valentine's Day and Mother's Day.

Examples:

```text
--c2-feature-groups none
--c2-feature-groups all
--c2-feature-groups price,campaign,event
```

**Effect:** changes the feature matrix, model fits, checkpoint signature and
predictions.

## 4.6 C3 objective and target representation

### `--c34-config`

Path to a recommendation JSON produced by `ml/experiments/c34_objective_channel_screening.py`.

**Precedence:** explicit C3/C4 CLI option > JSON value > `Config` default.

The retained final recommendation resolves to:

```text
nn_loss=mse
nn_target_mode=residual
nn_combined_mse_weight=0.25
tree_target_mode=log1p
xgboost_target_mode=residual
lightgbm_target_mode=log1p
enable_channel_history_features=false
channel_aux_weight=0.0
channel_share_smoothing=0.5
```

### `--nn-loss`

```text
huber | mse | combined | logcosh
code default: huber
confirmed final value: mse
```

- `huber`: quadratic for small residuals and linear for large residuals;
- `mse`: quadratically penalizes large residual misses;
- `combined`: weighted mixture of Huber and MSE;
- `logcosh`: smooth robust loss, MSE-like near zero and MAE-like in the tails.

**Effect:** directly changes NeuralNet optimization and predictions.

### `--nn-target-mode`

```text
residual | log1p
code default: residual
confirmed final value: residual
```

- `residual`: predict `log1p(Quantity) - log1p(target_baseline)`;
- `log1p`: predict `log1p(Quantity)` directly.

**Effect:** fundamentally changes target semantics and inverse transformation.

### `--nn-combined-mse-weight`

```text
number in [0, 1]
code default: 0.25
confirmed final value: 0.25
```

Controls the MSE share of the `combined` loss. Zero is pure Huber within that
formula; one is pure MSE.

**Effect:** only changes training when `--nn-loss combined`; it is inert for
`mse`, `huber` and `logcosh`.

### `--tree-target-mode`

```text
log1p | residual | tweedie
code default: log1p
confirmed shared fallback: log1p
```

Shared fallback for XGBoost and LightGBM unless a model-specific override is
provided.

- `log1p`: direct log-demand target;
- `residual`: log residual relative to the seasonal baseline;
- `tweedie`: positive heteroskedastic count-like objective.

**Effect:** changes both tree targets when they are not overridden.

### `--xgboost-target-mode`

```text
log1p | residual | tweedie
fallback: --tree-target-mode
confirmed final value: residual
```

**Effect:** overrides target representation for XGBoost only.

### `--lightgbm-target-mode`

```text
log1p | residual | tweedie
fallback: --tree-target-mode
confirmed final value: log1p
```

**Effect:** overrides target representation for LightGBM only.

## 4.7 C4 channel-history features and auxiliary app/web task

### `--channel-history-features`

```text
on | off
code default: off
confirmed final value: off
```

Enables leakage-safe historical app-share and channel-quantity features, such
as lag-0/7 app share and 7/28-day app/web rolling state.

**Effect:** changes the feature matrix and model signature. The final screen
selected the control configuration with this group disabled.

### `--channel-aux-weight`

```text
nonnegative number
code default: 0.0
confirmed final value: 0.0
```

A positive value adds an auxiliary app-share prediction loss to the shared
NeuralNet representation:

```text
total_demand_loss + channel_aux_weight * app_share_loss
```

**Effect:** a positive value changes NeuralNet architecture, objective and
checkpoint signature. The submitted target remains total `Quantity`.

### `--channel-share-smoothing`

```text
nonnegative number
code default: 0.5
confirmed final value: 0.5
```

Stabilizes the app-share auxiliary target, particularly for low-volume rows
that would otherwise have extreme 0/1 shares.

**Effect:** mainly matters when the auxiliary head is enabled. With auxiliary
weight zero, it does not change the main prediction loss.

## 4.8 C5 convex OOF ensemble

### `--ensemble`

```text
on | off
parser default: off
confirmed final value: on
```

`on` fits nonnegative weights summing to one on development OOF only, freezes
them, and applies them unchanged to recent benchmark and final forecasts.

**Effect:** does not retrain member estimators. It changes post-processing,
ensemble artifacts and whether `Ensemble` is eligible as a submission model.

### `--ensemble-models`

```text
comma-separated model list
code default: NeuralNet,XGBoost,LightGBM
confirmed final value: NeuralNet,XGBoost,LightGBM
```

Every member must have OOF predictions on the common evaluation population and
support the requested strategy.

**Effect:** changes the blend search space and resulting weights. More members
can add diversity but enlarge the simplex grid.

### `--ensemble-grid-step`

```text
number in (0, 0.5] that divides 1.0 exactly
code default: 0.01
confirmed final value: 0.01
```

A step of 0.01 searches weights in one-percentage-point increments; 0.05 is
coarser and faster.

**Effect:** changes search resolution and can change selected weights. It does
not alter member checkpoints.

### `--ensemble-min-relative-improvement`

```text
nonnegative number
code default: 0.002
confirmed final value: 0.002
```

Minimum relative development improvement over the best individual member on
the frozen test-aligned WAPE objective. `0.002` means at least 0.2% relative.

**Effect:** changes the development acceptance gate, not the fitted member
models or the candidate weights themselves.

### `--ensemble-benchmark-tolerance`

```text
nonnegative number
code default: 0.02
confirmed final value: 0.02
```

Maximum allowed relative regression against the best individual member on the
recent benchmark. `0.02` permits at most 2% relative deterioration.

**Effect:** changes confirmation/eligibility. The benchmark never refits the
weights.

## 4.9 Built-in help

### `-h`, `--help`

Prints the current parser syntax and exits without training:

```bash
uv run python ml/pipeline.py --help
```

# 5. Configuration precedence

C1, C2 and C3/C4 settings follow:

```text
explicit CLI override
    > value loaded from recommendation JSON
        > default in ml/framework.py::Config
```

Example:

```bash
uv run python ml/pipeline.py \
  --c34-config outputs/c34_screening/recommendation.json \
  --nn-loss huber
```

Even if the JSON recommends `mse`, the explicit `huber` override wins.

# 6. Which changes invalidate checkpoints

The fold-checkpoint signature includes model-defining configuration. Changing
any of these areas causes an incompatible checkpoint to be rejected:

- direct/recursive strategy;
- batch size and LR scaling;
- training window, recency weighting, baseline or trend features;
- C2 feature groups;
- NeuralNet loss or target mode;
- XGBoost/LightGBM target representation;
- channel-history features or auxiliary head;
- other model parameters included in `Config`.

Intentionally excluded from the signature:

- `nn_training_backend`, because it is an execution backend rather than a model
  definition;
- C5 ensemble settings, because the ensemble is fitted after member OOF
  predictions already exist.

Practical rule:

- model/feature change: use `--reset-checkpoints`;
- same interrupted run: use `--resume`;
- only ensemble grid or acceptance-gate change: member checkpoints can be
  reused with `--resume`.

# 7. Important parameters without a main-pipeline CLI flag

These values are defined in `ml/framework.py::Config` and cannot be changed by
adding a nonexistent `ml/pipeline.py` option:

```text
train_path = data/train_data.parquet
test_path = data/test_data.parquet
output_dir = outputs
horizon = 7
lag_windows = (7, 14, 28)
cv_epochs = 30
final_epochs = 60
seeds = (42, 123, 777)
n_cv_folds = 4 recent-benchmark origins
hidden_dims = (256, 128, 64)
dropout = (0.20, 0.15, 0.10)
base learning rate = 0.001
weight_decay = 0.0001
```

The project additionally uses 12 fixed, seasonally distributed development
origins in `ml/pipeline.py::DEVELOPMENT_ORIGINS`. Changing these values requires
a code/config edit rather than a pipeline flag.

# 8. Related commands that do not retrain the submission

## 8.1 Rebuild dashboard artifacts only

```bash
uv run python ml/export_results.py
```

This reads persisted results and rebuilds the JSON/static dashboard copies. It
does not train any model.

## 8.2 Run the local dashboard

```bash
uv run python webapp/server.py
```

Open `http://127.0.0.1:8999`.

## 8.3 Verify the repository after retraining

```bash
uv run pytest tests/ -m "not integration" -q
node tests/webapp_smoke_test.js
python -m compileall -q ml webapp tests
```

## 8.4 Frozen final audit

`ml/run_final_audit.py` accepts the pipeline modeling arguments and evaluates
three separate audit origins. It is not a replacement for normal submission
retraining.

Additional audit-only flags:

- `--force`: permit an audit rerun that is otherwise deliberately blocked;
- `--no-refresh-dashboard`: do not rebuild dashboard artifacts after the audit.

Original audit configuration:

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

The audit never refits ensemble weights; it evaluates the frozen decision on
disjoint origins.

# 9. Recommended final-presentation workflow

1. Preserve the checked-in output directory as the reference snapshot.
2. Use command 3.1, or 3.2 for a completely explicit reproduction.
3. If interrupted, continue only with command 3.3 and identical modeling flags.
4. Verify `outputs/submission.csv`, `outputs/results.json` and the terminal line
   reporting the selected model and strategy.
5. Run the Python and JavaScript checks.
6. Launch `uv run python webapp/server.py` for the presentation.

For exact reproduction, do not use `--submission-model auto`, do not change the
`512/fixed` batch/LR policy, and do not switch the forecast strategy to
`recursive`.
