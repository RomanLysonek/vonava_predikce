import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from dataclasses import asdict

from ensemble import (
    apply_ensemble_prediction,
    combine_forecasts,
    evaluate_fit,
    fit_convex_ensemble,
    simplex_weights,
)
from framework import Config
from pipeline import (
    ForecastStrategy,
    PrimaryStrategy,
    RuntimeOptions,
    SubmissionModel,
    OOF_MODEL_COLUMNS,
    _choose_canonical_model_strategy,
    fit_c5_ensembles,
    load_reusable_oof,
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


def test_benchmark_regression_is_reporting_only_for_ensemble_eligibility():
    development = _oof()
    benchmark = _oof().assign(origin_type="recent_benchmark")
    benchmark["pred_NeuralNet"] = benchmark["actual"]
    benchmark["pred_XGBoost"] = benchmark["actual"]
    benchmark["pred_LightGBM"] = benchmark["actual"] + 100
    cfg = Config(
        enable_ensemble=True,
        ensemble_grid_step=0.1,
        ensemble_min_relative_improvement=0.001,
        ensemble_benchmark_max_relative_regression=0.0,
    )

    _, _, _, payload, comparison = fit_c5_ensembles(
        development,
        benchmark,
        (ForecastStrategy.DIRECT,),
        cfg,
    )

    details = payload["strategies"]["direct"]
    assert details["accepted_on_development"]
    assert not details["benchmark_confirmed"]
    assert details["eligible_on_development"]
    assert details["accepted"]
    assert comparison.iloc[0]["eligible_on_development"]


def test_auto_selection_uses_development_eligibility_not_benchmark_status():
    scores = pd.DataFrame([
        {"strategy": "direct", "model": "Ensemble", "test_aligned_score": 0.1},
        {"strategy": "direct", "model": "NeuralNet", "test_aligned_score": 0.2},
    ])
    summary = pd.DataFrame([
        {
            "strategy": "direct",
            "model": "Ensemble",
            "evaluation_regime": "conditional",
            "comparison_population": "common",
            "aggregation": "global",
            "WAPE": 0.1,
        },
        {
            "strategy": "direct",
            "model": "NeuralNet",
            "evaluation_regime": "conditional",
            "comparison_population": "common",
            "aggregation": "global",
            "WAPE": 0.2,
        },
    ])
    options = RuntimeOptions(
        submission_model=SubmissionModel.AUTO,
        primary_strategy=PrimaryStrategy.AUTO,
        selection_protocol="test-aligned",
    )
    payload = {
        "strategies": {
            "direct": {
                "accepted_on_development": True,
                "benchmark_confirmed": False,
            }
        }
    }

    assert _choose_canonical_model_strategy(
        options, summary, scores, payload
    ) == ("Ensemble", "direct")


def test_reusable_oof_rejects_audit_rows_and_policy_mismatches(tmp_path):
    cfg = Config(
        num_products=3,
        c2_feature_groups=(),
        nn_loss="huber",
        xgboost_target_mode=None,
        lightgbm_target_mode=None,
    )
    options = RuntimeOptions()
    frame = _oof()
    for column in OOF_MODEL_COLUMNS.values():
        if column not in frame:
            frame[column] = frame["actual"]
    frame["origin_type"] = [
        "development", "development", "development",
        "recent_benchmark", "recent_benchmark", "recent_benchmark",
    ]
    path = tmp_path / "oof_predictions.parquet"
    frame.to_parquet(path, index=False)
    from pipeline import _effective_oof_config
    from pipeline import _json_safe
    from provenance import sha256_file
    (tmp_path / "results.json").write_text(
        __import__("json").dumps({"config": _effective_oof_config(cfg, options)}),
        encoding="utf-8",
    )
    root = Path.cwd()
    (tmp_path / "run_manifest.json").write_text(__import__("json").dumps({
        "configuration": {"values": _json_safe(asdict(cfg))},
        "inputs": {
            "train": {"sha256": sha256_file(root / cfg.train_path)},
            "test": {"sha256": sha256_file(root / cfg.test_path)},
        },
        "outputs": {
            "outputs/oof_predictions.parquet": sha256_file(path),
            "outputs/results.json": sha256_file(tmp_path / "results.json"),
        },
    }), encoding="utf-8")

    loaded = load_reusable_oof(
        str(path), cfg, options, (ForecastStrategy.DIRECT,)
    )
    assert len(loaded) == len(frame)

    frame.loc[0, "origin_type"] = "final_audit"
    frame.to_parquet(path, index=False)
    (tmp_path / "run_manifest.json").write_text(__import__("json").dumps({
        "configuration": {"values": _json_safe(asdict(cfg))},
        "inputs": {
            "train": {"sha256": sha256_file(root / cfg.train_path)},
            "test": {"sha256": sha256_file(root / cfg.test_path)},
        },
        "outputs": {
            "outputs/oof_predictions.parquet": sha256_file(path),
            "outputs/results.json": sha256_file(tmp_path / "results.json"),
        },
    }), encoding="utf-8")
    with pytest.raises(RuntimeError, match="only development and recent_benchmark"):
        load_reusable_oof(str(path), cfg, options, (ForecastStrategy.DIRECT,))

    frame.loc[0, "origin_type"] = "development"
    frame.to_parquet(path, index=False)
    (tmp_path / "run_manifest.json").write_text(__import__("json").dumps({
        "configuration": {"values": _json_safe(asdict(cfg))},
        "inputs": {
            "train": {"sha256": sha256_file(root / cfg.train_path)},
            "test": {"sha256": sha256_file(root / cfg.test_path)},
        },
        "outputs": {
            "outputs/oof_predictions.parquet": sha256_file(path),
            "outputs/results.json": sha256_file(tmp_path / "results.json"),
        },
    }), encoding="utf-8")
    incompatible = Config(
        num_products=3,
        cv_epochs=20,
        final_epochs=20,
        c2_feature_groups=(),
        nn_loss="huber",
        xgboost_target_mode=None,
        lightgbm_target_mode=None,
    )
    with pytest.raises(RuntimeError, match="cv_epochs"):
        load_reusable_oof(
            str(path), incompatible, options, (ForecastStrategy.DIRECT,)
        )


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
    current_config = json.loads(
        (Path.cwd() / "outputs" / "results.json").read_text()
    )["config"]
    current_full_config = json.loads(
        (Path.cwd() / "outputs" / "run_manifest.json").read_text()
    )["configuration"]["values"]
    audit_names = (
        "final_audit_oof.parquet",
        "final_audit_summary.csv",
        "final_audit_validation_strata.csv",
        "final_audit_test_aligned_scores.csv",
        "final_audit_prediction_diagnostics.csv",
        "final_audit_prediction_diagnostics_by_origin.csv",
    )
    for name in audit_names:
        target = tmp_path / name
        if not target.exists():
            target.write_text("placeholder", encoding="utf-8")
    (tmp_path / "results.json").write_text(
        json.dumps({"config": current_config}), encoding="utf-8"
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "configuration": {"values": current_full_config}
    }), encoding="utf-8")
    (tmp_path / "final_audit_manifest.json").write_text(
        json.dumps({
            "ensemble_weights_sha256": valid_hash,
            "configuration": current_config,
            "evaluated_configuration": current_full_config,
            "audit_output_hashes": {
                f"outputs/{name}": hashlib.sha256(
                    (tmp_path / name).read_bytes()
                ).hexdigest()
                for name in audit_names
            },
        }), encoding="utf-8"
    )
    summary, aligned = load_current_final_audit_artifacts(str(tmp_path))
    assert summary.iloc[0]["model"] == "Ensemble"
    assert aligned.iloc[0]["test_aligned_score"] == 0.1


def test_audit_survives_metadata_only_ensemble_artifact_change(tmp_path):
    import hashlib
    import json

    from pipeline import load_current_final_audit_artifacts

    weights = tmp_path / "ensemble_weights.json"
    weights.write_text(json.dumps({
        "benchmark_role": "reporting_only",
        "strategies": {
            "direct": {"weights": {"NeuralNet": 0.36, "XGBoost": 0.64}}
        },
    }), encoding="utf-8")
    old_hash = hashlib.sha256(b"old metadata wrapper").hexdigest()
    pd.DataFrame([{"model": "NeuralNet", "WAPE": 0.2}]).to_csv(
        tmp_path / "final_audit_summary.csv", index=False
    )
    pd.DataFrame([
        {"model": "NeuralNet", "test_aligned_score": 0.2}
    ]).to_csv(tmp_path / "final_audit_test_aligned_scores.csv", index=False)
    from pipeline import _file_sha256
    current_config = json.loads(
        (Path.cwd() / "outputs" / "results.json").read_text()
    )["config"]
    current_full_config = json.loads(
        (Path.cwd() / "outputs" / "run_manifest.json").read_text()
    )["configuration"]["values"]
    (tmp_path / "results.json").write_text(
        json.dumps({"config": current_config}), encoding="utf-8"
    )
    (tmp_path / "run_manifest.json").write_text(json.dumps({
        "configuration": {"values": current_full_config}
    }), encoding="utf-8")
    audit_names = (
        "final_audit_oof.parquet",
        "final_audit_summary.csv",
        "final_audit_validation_strata.csv",
        "final_audit_test_aligned_scores.csv",
        "final_audit_prediction_diagnostics.csv",
        "final_audit_prediction_diagnostics_by_origin.csv",
    )
    for name in audit_names:
        target = tmp_path / name
        if not target.exists():
            target.write_text("placeholder", encoding="utf-8")
    (tmp_path / "final_audit_manifest.json").write_text(json.dumps({
        "ensemble_weights_sha256": old_hash,
        "ensemble_weights": {"NeuralNet": 0.36, "XGBoost": 0.64},
        "configuration": current_config,
        "evaluated_configuration": current_full_config,
        "provenance_status": "partial_historical",
        "audit_output_hashes": {
            f"outputs/{name}": _file_sha256(str(tmp_path / name))
            for name in audit_names
        },
    }), encoding="utf-8")

    summary, aligned = load_current_final_audit_artifacts(str(tmp_path))

    assert summary.iloc[0]["model"] == "NeuralNet"
    assert aligned.iloc[0]["test_aligned_score"] == 0.2
