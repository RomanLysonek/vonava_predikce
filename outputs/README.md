# Output artifacts

This directory contains only artifacts with presentation, audit or
reproducibility value. Large fold checkpoints and per-candidate OOF files from
screening runs were removed from the delivery package because the corresponding
scripts can regenerate them.

## Canonical submission

- `submission.csv` and `submission.parquet` — final NeuralNet submission using
  the direct multi-horizon strategy.
- `final_forecasts.parquet` — unrounded final forecasts from the trained models.
- `submission_ensemble.csv` and `submission_ensemble.parquet` — transparent
  secondary output from the frozen cross-model ensemble; this is not the
  canonical submission.

## Dashboard and evaluation summaries

- `results.json` — complete dashboard data contract. Copies are published to
  `webapp/static/results.json` and `docs/results.json`.
- `cv_results.csv`, `cv_results_all.csv`, `benchmark_summary.csv` and
  `dev_summary.csv` — walk-forward fold results and aggregate summaries.
- `oof_predictions.parquet` — complete retained OOF predictions for the main
  confirmed run.
- `per_product_summary.csv`, `strategy_by_horizon.csv`,
  `validation_strata_summary.csv`, `top_decile_summary.csv` and
  `top_error_rows.csv` — diagnostics by product, horizon, regime and demand
  concentration.
- `prediction_diagnostics.csv` and
  `prediction_diagnostics_by_origin.csv` — coverage, fallback and numerical
  stability diagnostics.

## Ensemble and final audit

- `ensemble_weights.json` and `ensemble_weights.csv` — frozen weights:
  0.36 NeuralNet, 0.25 XGBoost and 0.39 LightGBM.
- `ensemble_comparison.csv` — development fit and recent-benchmark confirmation.
- `final_audit_manifest.json` — immutable description of the disjoint final
  audit, including file hashes and audit origins.
- `final_audit_*.csv` and `final_audit_oof.parquet` — results from the three
  previously untouched audit origins.

## Development decisions

The directories `c01_recursive_check/`, `c1_screening/`, `c2_screening/` and
`c34_screening/` retain their result tables and recommendation JSON. Raw
candidate checkpoints and candidate-level OOF predictions are intentionally not
part of the presentation package.
