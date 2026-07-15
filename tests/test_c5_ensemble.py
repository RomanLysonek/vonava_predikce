import numpy as np
import pandas as pd

from ensemble import (
    apply_ensemble_prediction,
    combine_forecasts,
    evaluate_fit,
    fit_convex_ensemble,
    simplex_weights,
)


def _oof(strategy="direct"):
    actual = np.array([10, 20, 30, 40, 50, 60], dtype=float)
    return pd.DataFrame({
        "origin_type": "development",
        "strategy": strategy,
        "origin": pd.to_datetime(["2024-01-01"] * 3 + ["2024-06-01"] * 3),
        "validation_stratum": ["winter_test_like"] * 3 + ["regular"] * 3,
        "ProductId": [1, 2, 3, 1, 2, 3],
        "DateKey": pd.date_range("2024-01-02", periods=6),
        "horizon": [1, 2, 3, 1, 2, 3],
        "ProductAvailable": True,
        "actual": actual,
        "pred_NeuralNet": actual + 4,
        "pred_XGBoost": actual - 4,
        "pred_LightGBM": actual + 8,
    })


def test_simplex_grid_is_nonnegative_and_sums_to_one():
    grid = simplex_weights(3, 0.1)
    assert len(grid) == 66
    assert np.all(grid >= 0)
    assert np.allclose(grid.sum(axis=1), 1.0)
    assert any(np.allclose(row, [1, 0, 0]) for row in grid)


def test_dev_only_convex_fit_recovers_useful_blend():
    fit = fit_convex_ensemble(
        _oof(),
        strategy="direct",
        models=("NeuralNet", "XGBoost", "LightGBM"),
        stratum_weights={"winter_test_like": 0.6, "regular": 0.4},
        grid_step=0.1,
        min_relative_improvement=0.001,
    )
    assert fit.accepted_on_development
    assert np.isclose(sum(fit.weights.values()), 1.0)
    assert fit.ensemble_test_aligned_wape < fit.best_single_test_aligned_wape
    assert all(weight >= 0.0 for weight in fit.weights.values())
    blended_offset = (
        4 * fit.weights["NeuralNet"]
        - 4 * fit.weights["XGBoost"]
        + 8 * fit.weights["LightGBM"]
    )
    assert np.isclose(blended_offset, 0.0)


def test_weights_are_applied_without_refit_to_benchmark_and_final_forecast():
    fit = fit_convex_ensemble(
        _oof(), strategy="direct",
        models=("NeuralNet", "XGBoost", "LightGBM"),
        stratum_weights={"winter_test_like": 0.6, "regular": 0.4},
        grid_step=0.1,
    )
    benchmark = _oof().assign(origin_type="recent_benchmark")
    benchmark["pred_NeuralNet"] += 1
    evaluated = evaluate_fit(
        benchmark, fit,
        stratum_weights={"winter_test_like": 0.6, "regular": 0.4},
    )
    assert evaluated["n_rows"] == 6

    with_ensemble = apply_ensemble_prediction(benchmark, {"direct": fit})
    expected = sum(
        benchmark[f"pred_{model}"].to_numpy() * weight
        for model, weight in fit.weights.items()
    )
    assert np.allclose(with_ensemble["pred_Ensemble"], expected)

    final = combine_forecasts(
        {model: benchmark[f"pred_{model}"].to_numpy() for model in fit.weights},
        fit.weights,
    )
    assert np.allclose(final, expected)


def test_ensemble_uses_common_conditional_rows_only():
    frame = _oof()
    frame.loc[0, "ProductAvailable"] = False
    frame.loc[1, "pred_XGBoost"] = np.nan
    fit = fit_convex_ensemble(
        frame, strategy="direct",
        models=("NeuralNet", "XGBoost", "LightGBM"),
        stratum_weights={"winter_test_like": 0.6, "regular": 0.4},
        grid_step=0.1,
    )
    assert fit.n_rows == 4


def test_duplicate_oof_forecast_keys_are_rejected():
    frame = pd.concat([_oof(), _oof().iloc[[0]]], ignore_index=True)
    import pytest
    with pytest.raises(ValueError, match="duplicate forecast keys"):
        fit_convex_ensemble(
            frame, strategy="direct",
            models=("NeuralNet", "XGBoost", "LightGBM"),
            stratum_weights={"winter_test_like": 0.6, "regular": 0.4},
            grid_step=0.1,
        )


def test_stale_final_audit_is_not_loaded(tmp_path):
    import hashlib
    import json

    from pipeline import load_current_final_audit_artifacts

    weights = tmp_path / "ensemble_weights.json"
    weights.write_text('{"weights": 1}', encoding="utf-8")
    stale_hash = hashlib.sha256(b'different').hexdigest()
    (tmp_path / "final_audit_manifest.json").write_text(
        json.dumps({"ensemble_weights_sha256": stale_hash}), encoding="utf-8"
    )
    pd.DataFrame([{"model": "Ensemble", "WAPE": 0.1}]).to_csv(
        tmp_path / "final_audit_summary.csv", index=False
    )
    pd.DataFrame([{"model": "Ensemble", "test_aligned_score": 0.1}]).to_csv(
        tmp_path / "final_audit_test_aligned_scores.csv", index=False
    )

    summary, aligned = load_current_final_audit_artifacts(str(tmp_path))
    assert summary.empty
    assert aligned.empty

    valid_hash = hashlib.sha256(weights.read_bytes()).hexdigest()
    (tmp_path / "final_audit_manifest.json").write_text(
        json.dumps({"ensemble_weights_sha256": valid_hash}), encoding="utf-8"
    )
    summary, aligned = load_current_final_audit_artifacts(str(tmp_path))
    assert summary.iloc[0]["model"] == "Ensemble"
    assert aligned.iloc[0]["test_aligned_score"] == 0.1
