# Final Tier B Fix Plan

## Objective

Resolve the remaining Tier B issues before launching another full pipeline run or starting the direct/recursive/both strategy implementation.

## 1. Fix the blocking primary-summary selection bug

The expanded holdout summary now contains multiple global rows per model:

```text
conditional × common
conditional × model_specific
realized × common
realized × model_specific
```

The current code filters only:

```python
holdout_summary["aggregation"] == "global"
```

and then indexes by model. This returns multiple rows per model, so metric lookups produce pandas `Series` rather than scalar values. Formatting the resulting skill value will crash the pipeline before final model training and export.

### Required replacement

Add a helper:

```python
def select_primary_summary(
    summary: pd.DataFrame,
    *,
    evaluation_regime: str = "conditional",
    comparison_population: str = "common",
    aggregation: str = "global",
) -> pd.DataFrame:
    selected = summary[
        (summary["evaluation_regime"] == evaluation_regime)
        & (
            summary["comparison_population"]
            == comparison_population
        )
        & (summary["aggregation"] == aggregation)
    ].copy()

    if selected.empty:
        raise RuntimeError("Primary evaluation summary is empty")

    if selected["model"].duplicated().any():
        raise RuntimeError(
            "Primary evaluation summary contains duplicate model rows"
        )

    return selected
```

Use it in `main()`:

```python
primary_holdout = select_primary_summary(holdout_summary).set_index("model")

nn_mae = float(primary_holdout.loc["NeuralNet", "MAE"])
naive_mae = float(primary_holdout.loc["SeasonalNaive", "MAE"])

skill = 1.0 - nn_mae / naive_mae

print(
    "\nSkill vs seasonal-naive baseline "
    f"(holdout, conditional/common/global MAE): {skill:+.1%} "
    "(positive = model beats naive)"
)
```

Also use the same helper anywhere else that selects a canonical summary slice for:

```text
dashboard headline metrics
results.json primary summary
model ranking
submission model selection
README result generation
```

## 2. Make legacy fold-level comparisons population-consistent

`oof_to_legacy_cv_results()` still removes missing predictions independently for each model.

That means the modern summary is fair, but the legacy per-fold dashboard table may compare models on different rows.

### Recommended correction

For each fold and evaluation regime:

```python
available_pred_cols = [
    col
    for col in pred_columns.values()
    if col in fold_df.columns
]

common_mask = (
    regime_mask
    & fold_df[available_pred_cols]
        .apply(np.isfinite)
        .all(axis=1)
)
```

Use the same `common_mask` for every model shown in the comparative fold table.

If model-specific fold metrics are retained, label them explicitly:

```text
comparison_population = model_specific
```

Do not present them as a direct comparative ranking.

## 3. Replace the empty skipped worker test with a real integration test

The current Tier B correction test contains a permanently skipped placeholder for the full tree-worker smoke test.

Implement an actual test marked as integration:

```python
@pytest.mark.integration
def test_tree_worker_full_smoke(tmp_path):
    ...
```

The test should verify that the worker returns:

```text
XGBoost
LightGBM
DynamicRidge
```

and that each prediction vector:

```text
has the expected length
contains only finite values
contains no negative values
aligns to the expected ProductId + DateKey keys
```

Register the marker in `pyproject.toml` or `pytest.ini`:

```ini
[pytest]
markers =
    integration: slower subprocess and native-library integration tests
```

Run fast tests normally:

```bash
uv run pytest -m "not integration"
```

Run the worker integration explicitly:

```bash
uv run pytest -m integration
```

## 4. Add subprocess timeout protection

Apply an explicit timeout to production and test subprocess calls:

```python
subprocess.run(
    command,
    check=True,
    capture_output=True,
    text=True,
    timeout=180,
)
```

Handle timeout failures with useful context:

```python
except subprocess.TimeoutExpired as exc:
    raise RuntimeError(
        "Tree worker timed out after 180 seconds"
    ) from exc
```

Include the requested model names and captured stdout/stderr where available.

## 5. Add regression tests for the blocking bug

Add tests for `select_primary_summary()`:

1. Returns exactly one row per model for `conditional/common/global`.
2. Raises when the selected slice is empty.
3. Raises when duplicate model rows exist.
4. Produces scalar numeric values for metric lookups.
5. Skill calculation completes without a pandas `Series`.
6. Realized or model-specific rows cannot accidentally enter the primary summary.

Example:

```python
def test_select_primary_summary_returns_unique_models():
    selected = select_primary_summary(summary)

    assert not selected["model"].duplicated().any()
    assert set(selected["evaluation_regime"]) == {"conditional"}
    assert set(selected["comparison_population"]) == {"common"}
    assert set(selected["aggregation"]) == {"global"}
```

## 6. Run sequence before declaring Tier B complete

Execute:

```bash
uv run pytest -m "not integration"
uv run pytest -m integration
uv run python ml/pipeline.py
```

After the full run, verify:

```text
final submission files exist
results.json is regenerated
dev_summary.csv is populated
holdout_summary.csv is populated
DynamicRidge appears in OOF and final diagnostics
the primary summary uses conditional/common/global
the final pipeline does not crash after holdout reporting
```

## Acceptance criteria

Tier B is complete when:

- The primary holdout slice contains exactly one row per model.
- Skill calculation uses scalar values and no longer crashes.
- Legacy comparative fold metrics use a common population.
- The tree-worker integration test is real and executable.
- Subprocess calls have explicit timeout protection.
- Fast tests, integration tests and the complete pipeline run all pass.
- Final artifacts are regenerated successfully from the corrected source.
