# Tier B Completion — Remaining Corrections

## Verdict

B3 and B4 are wired into the repository, but Tier B is not yet methodologically complete. Apply the corrections below before starting the direct/recursive/both implementation.

## 1. Correct Dynamic Ridge

### Target formulation

Train Dynamic Ridge on the same baseline-relative log residual used by the neural network:

```python
y = (
    np.log1p(train_panel["target"].to_numpy(dtype=float))
    - np.log1p(train_panel["target_baseline"].to_numpy(dtype=float))
)
```

Reconstruct on prediction:

```python
residual = model.predict(panel)

pred = np.expm1(
    residual
    + np.log1p(panel["target_baseline"].to_numpy(dtype=float))
)

pred = np.clip(pred, 0.0, None)
```

### Configuration

Add to `Config`:

```python
ridge_alpha: float = 10.0
ridge_prediction_cap: float | None = None
```

Use:

```python
Ridge(alpha=cfg.ridge_alpha)
```

Do not hard-code `alpha=1.0`.

### Prediction cap

Remove the unconditional hard-coded cap at `500`.

Tier C3 is the cap ablation. Until C3 is executed, the default must be uncapped except for the non-negativity floor:

```python
pred = np.clip(pred, 0.0, None)

if cfg.ridge_prediction_cap is not None:
    pred = np.minimum(pred, cfg.ridge_prediction_cap)
```

### Preprocessing

Use training-fitted median imputation for numeric variables:

```python
SimpleImputer(strategy="median")
```

For categorical variables, use:

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
```

## 2. Complete B4 with common-population evaluation

The current realized/conditional masks are useful, but each model is still scored after independently dropping its own missing predictions.

Refactor summaries to produce both:

```text
comparison_population = common
comparison_population = model_specific
```

### Regimes

```python
regime_masks = {
    "realized": oof["actual"].notna(),
    "conditional": (
        oof["actual"].notna()
        & oof["ProductAvailable"].fillna(False)
    ),
}
```

### Common comparison population

For each regime:

```python
pred_cols = list(pred_columns.values())

common_mask = (
    regime_mask
    & oof[pred_cols].apply(np.isfinite).all(axis=1)
)
```

All headline model comparisons must use this exact same row mask.

### Model-specific diagnostics

For each model:

```python
model_mask = regime_mask & np.isfinite(oof[pred_col])
```

Use this only for coverage diagnostics.

### Required summary columns

```text
model
evaluation_regime
comparison_population
aggregation
n_folds
n_expected
n_actual
n_predicted
n_scored
coverage
MAE
RMSE
WAPE
sMAPE
RMSLE
Bias
BiasRatio
MAPE
```

Do not encode the regime only through aggregation strings such as:

```text
global_conditional
mean_fold_conditional
```

Use explicit columns instead.

### Primary regime

Use as the default headline/dashboard regime:

```text
evaluation_regime = conditional
comparison_population = common
aggregation = global
```

Keep realized sales available as a diagnostic toggle.

## 3. Add direct-panel safety validation

At the beginning of `build_direct_panel`:

```python
horizons = tuple(int(h) for h in horizons)

if not horizons:
    raise ValueError("At least one forecast horizon is required")

if min(horizons) < 1:
    raise ValueError("Forecast horizons must be positive")

if max(horizons) > min(SEASONAL_LAG_DAYS):
    raise ValueError(
        "Target-relative seasonal lags would require future observations"
    )
```

Validate unique keys:

```python
for name, frame in [
    ("train_feat", train_feat),
    ("future_covariates", future_covariates),
]:
    if frame is not None and frame.duplicated(["ProductId", "DateKey"]).any():
        raise ValueError(f"{name} contains duplicate ProductId/DateKey keys")
```

## 4. Add missing tests

Add dedicated tests for:

1. Dynamic Ridge zero residual reconstructs `target_baseline`.
2. Dynamic Ridge predictions are finite and non-negative.
3. Dynamic Ridge handles unseen categories.
4. Dynamic Ridge cap is disabled by default.
5. Configured cap affects only values above the cap.
6. Conditional regime excludes unavailable rows.
7. Realized regime includes unavailable rows.
8. Common-population summaries use identical keys and `n_scored` across models.
9. Model-specific coverage reports missing predictions correctly.
10. Unsafe direct horizons raise `ValueError`.
11. Duplicate panel keys raise `ValueError`.
12. Full tree-worker smoke test still returns XGBoost, LightGBM, and DynamicRidge.

Add an explicit timeout to subprocess tests and production calls.

## 5. Repair exports and documentation

### `export_results.py`

Include Dynamic Ridge in `final_forecasts`.

Load and preserve:

```text
outputs/dev_summary.csv
outputs/holdout_summary.csv
```

Do not rewrite them to `null` in `results.json`.

### README

Correct the NN description to:

```text
The neural network predicts a baseline-relative log residual:
log1p(Quantity) - log1p(target_baseline).
```

Do not claim all models use the same target unless they actually do.

State that seeds are fixed.

Explain that conditional demand is the primary evaluation regime and realized sales is diagnostic.

### `progress.txt`

Remove stale statements including:

```text
dynamic_ridge.py - NEW, Tier B3 (not yet created)
```

Update the worker description to include Dynamic Ridge.

## Acceptance criteria

Tier B is complete when:

- Dynamic Ridge uses residual reconstruction and configurable alpha.
- The hard-coded cap is removed or disabled by default.
- Conditional and realized metrics are both available.
- Headline comparisons use a shared common population.
- Coverage is explicitly reported.
- Conditional/common/global is the default headline view.
- Direct-panel leakage guards and duplicate-key checks exist.
- Dedicated B3/B4 tests pass.
- Lightweight export preserves Dynamic Ridge and both summary files.
- README and progress notes match the source code.
