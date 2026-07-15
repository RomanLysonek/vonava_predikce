import json
from pathlib import Path

import numpy as np
import pandas as pd

from framework import Config
from pipeline import (
    ForecastStrategy,
    PrimaryStrategy,
    RuntimeOptions,
    SubmissionModel,
    export_results_json,
)


ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / "webapp" / "static"


def _summary_rows():
    rows = []
    for strategy, offset in (("direct", 0.0), ("recursive", 1.0)):
        for model, mae in (("NeuralNet", 2.0 + offset), ("SeasonalNaive", 4.0)):
            rows.append({
                "model": model,
                "evaluation_regime": "conditional",
                "comparison_population": "common",
                "aggregation": "global",
                "n_folds": 1,
                "n_expected": 2,
                "n_actual": 2,
                "n_predicted": 2,
                "n_scored": 2,
                "coverage": 1.0,
                "MAE": mae,
                "RMSE": mae + 0.5,
                "WAPE": mae / 10.0,
                "sMAPE": mae / 10.0,
                "RMSLE": 0.1,
                "Bias": 0.0,
                "BiasRatio": 0.0,
                "MAPE": 10.0,
                "strategy": strategy,
            })
    return pd.DataFrame(rows)


def test_export_results_json_exposes_complete_strategy_payload(tmp_path):
    train = pd.DataFrame({
        "ProductId": [1, 1],
        "DateKey": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "Quantity": [10.0, 11.0],
    })
    test = pd.DataFrame({
        "ProductId": [1, 1],
        "DateKey": pd.to_datetime(["2026-01-03", "2026-01-04"]),
    })
    submission = test.copy()
    submission["Quantity"] = [12, 13]
    final_forecasts = {
        "NeuralNet": np.array([12.0, 13.0]),
        "DynamicRidge": np.array([12.2, 12.8]),
        "SeasonalNaive": np.array([11.0, 11.0]),
    }
    cv_results = pd.DataFrame([
        {"fold": 0, "model": "NeuralNet", "regime": "conditional", "comparison_population": "common", "MAE": 2.0, "RMSE": 2.5, "WAPE": 0.2, "sMAPE": 0.2, "RMSLE": 0.1, "Bias": 0.0, "BiasRatio": 0.0, "MAPE": 10.0},
        {"fold": 0, "model": "SeasonalNaive", "regime": "conditional", "comparison_population": "common", "MAE": 4.0, "RMSE": 4.5, "WAPE": 0.4, "sMAPE": 0.4, "RMSLE": 0.2, "Bias": 0.0, "BiasRatio": 0.0, "MAPE": 20.0},
    ])
    cv_results_all = pd.concat([
        cv_results.assign(strategy="direct"),
        cv_results.assign(strategy="recursive", MAE=cv_results["MAE"] + 1.0),
    ], ignore_index=True)
    summaries = _summary_rows()
    pair_summary = pd.DataFrame([{
        "model": "NeuralNet",
        "evaluation_regime": "conditional",
        "direct_n": 2,
        "recursive_n": 2,
        "paired_n": 2,
        "metric": "WAPE",
        "direct_value": 0.2,
        "recursive_value": 0.3,
        "absolute_delta": 0.1,
        "relative_delta": 0.5,
        "winner": "direct",
    }])
    by_horizon = pd.DataFrame([
        {
            **summaries.iloc[0].to_dict(),
            "horizon": 1,
            "origin_type": "development",
        },
        {
            **summaries[summaries["strategy"].eq("recursive")].iloc[0].to_dict(),
            "horizon": 1,
            "origin_type": "development",
        },
        {
            **summaries.iloc[0].to_dict(),
            "horizon": 1,
            "origin_type": "recent_benchmark",
        },
    ])
    forecasts_by_strategy = {
        "direct": {
            "NeuralNet": {"1": {"dates": ["2026-01-03", "2026-01-04"], "quantity": [12.0, 13.0]}},
            "DynamicRidge": {"1": {"dates": ["2026-01-03", "2026-01-04"], "quantity": [12.2, 12.8]}},
            "SeasonalNaive": {"1": {"dates": ["2026-01-03", "2026-01-04"], "quantity": [11.0, 11.0]}},
        },
        "recursive": {
            "NeuralNet": {"1": {"dates": ["2026-01-03", "2026-01-04"], "quantity": [12.5, 13.5]}},
            "SeasonalNaive": {"1": {"dates": ["2026-01-03", "2026-01-04"], "quantity": [11.0, 11.0]}},
        },
    }
    validation_strata = summaries.assign(
        origin_type="development", validation_stratum="winter_test_like"
    )
    test_aligned_scores = pd.DataFrame([{
        "strategy": "direct", "model": "NeuralNet", "metric": "WAPE",
        "test_aligned_score": 0.2, "weight_sum": 1.0,
        "strata_present": "winter_test_like",
    }])
    prediction_diagnostics = pd.DataFrame([{
        "origin_type": "development", "strategy": "direct",
        "model": "NeuralNet", "n_rows": 2, "n_finite": 2,
        "coverage": 1.0, "fallback_count": 0, "fallback_rate": 0.0,
        "nonfinite_raw_count": 0, "catastrophic_guard_count": 0,
        "prediction_max": 13.0, "prediction_p99": 12.99,
        "observed_max": 11.0, "prediction_to_observed_max_ratio": 13 / 11,
    }])
    final_audit_summary = summaries[summaries["strategy"].eq("direct")].copy()
    final_audit_test_aligned_scores = pd.DataFrame([{
        "strategy": "direct", "model": "NeuralNet", "metric": "WAPE",
        "test_aligned_score": 0.19, "weight_sum": 1.0,
        "strata_present": "winter_test_like",
    }])

    options = RuntimeOptions(
        forecast_strategy=ForecastStrategy.BOTH,
        primary_strategy=PrimaryStrategy.AUTO,
        submission_model=SubmissionModel.NEURAL_NET,
        selection_metric="WAPE",
    )

    payload = export_results_json(
        train,
        test,
        submission,
        final_forecasts,
        cv_results,
        Config(num_products=1, horizon=2, n_cv_folds=1),
        path=str(tmp_path / "results.json"),
        dev_summary=summaries,
        benchmark_summary=summaries,
        runtime_options=options,
        forecasts_by_strategy=forecasts_by_strategy,
        strategy_comparison=pair_summary,
        canonical_strategy="direct",
        canonical_model="NeuralNet",
        cv_results_all=cv_results_all,
        strategy_by_horizon=by_horizon,
        validation_strata_summary=validation_strata,
        test_aligned_scores=test_aligned_scores,
        prediction_diagnostics=prediction_diagnostics,
        final_audit_summary=final_audit_summary,
        final_audit_test_aligned_scores=final_audit_test_aligned_scores,
    )

    assert set(payload["forecasts_by_strategy"]) == {"direct", "recursive"}
    assert {row["strategy"] for row in payload["benchmark_summary_all"]} == {"direct", "recursive"}
    assert {row["strategy"] for row in payload["dev_summary_all"]} == {"direct", "recursive"}
    assert {row["strategy"] for row in payload["cv_results_all"]} == {"direct", "recursive"}
    assert payload["strategy_comparison"][0]["winner"] == "direct"
    assert payload["strategy_by_horizon"]
    assert {row["origin_type"] for row in payload["strategy_by_horizon"]} == {
        "development", "recent_benchmark"
    }
    assert payload["validation_strata_summary"]
    assert payload["test_aligned_scores"]
    assert payload["prediction_diagnostics"]
    assert payload["final_audit_summary"]
    assert payload["final_audit_test_aligned_scores"][0]["test_aligned_score"] == 0.19
    ridge = next(model for model in payload["models"] if model["key"] == "DynamicRidge")
    assert ridge["strategies"] == ["direct"]
    assert payload["selection"]["canonical_model"] == "NeuralNet"
    assert payload["selection"]["canonical_strategy"] == "direct"
    assert payload["selection"]["benchmark_winner"] == "direct"
    assert payload["selection"]["recent_benchmark_confirmation"] is True
    assert all(row.get("strategy") == "direct" for row in payload["benchmark_summary"])
    json.loads((tmp_path / "results.json").read_text())


def test_strategy_controls_exist_on_overview_and_model_pages():
    for name in ("index.html", "model.html"):
        text = (STATIC / name).read_text()
        assert 'rel="icon" href="/static/favicon.svg"' in text
        assert 'id="strategy-select"' in text
        assert 'id="regime-select"' in text
        assert 'id="promo-strategy"' in text

    overview = (STATIC / "index.html").read_text()
    assert 'id="strategy-comparison-table"' in overview
    assert 'id="chart-horizon"' in overview
    assert "Aligned WAPE" in overview
    assert "Δ WAPE (pp)" in overview
    assert "Relative change" in overview
    assert 'id="regime-explanation"' in overview
    assert 'id="regime-definitions"' in overview
    assert 'id="top-decile-explanation"' in overview
    assert 'id="top-error-insight"' in overview
    assert 'id="product-history-toggle"' in overview
    assert 'id="product-models-select-all"' in overview
    assert 'id="product-models-deselect-all"' in overview
    assert 'class="panel model-comparison-panel"' in overview

    app_js = (STATIC / "app.js").read_text()
    assert "(ensembleValue - singleValue) * 100" in app_js
    assert '`${absoluteDeltaPp >= 0 ? "+" : ""}${fmt(absoluteDeltaPp, 2)} pp`' in app_js
    assert "function renderRegimeExplanation" in app_js
    assert "function renderRegimeDefinitions" in app_js
    assert "function renderTopErrorInsight" in app_js
    assert "function setAllProductModels" in app_js
    assert "const labels = productHistoryVisible" in app_js
    assert 'row.stage === "channel_aux"' in app_js



def test_api_results_serves_strategy_payload(tmp_path, monkeypatch):
    from webapp import server

    payload = {
        "forecasts_by_strategy": {"direct": {}, "recursive": {}},
        "benchmark_summary_all": [],
        "dev_summary_all": [],
        "cv_results_all": [],
        "strategy_comparison": [],
        "strategy_by_horizon": [],
        "selection": {"canonical_model": "NeuralNet", "canonical_strategy": "direct"},
    }
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(payload))
    monkeypatch.setattr(server, "RESULTS_PATH", results_path)

    response = server.get_results()
    assert response.status_code == 200
    served = json.loads(response.body)
    assert set(served["forecasts_by_strategy"]) == {"direct", "recursive"}
    assert served["selection"]["canonical_strategy"] == "direct"


def test_favicon_route_serves_static_icon():
    from webapp import server

    response = server.favicon()
    assert Path(response.path) == STATIC / "favicon.svg"
    assert response.media_type == "image/svg+xml"

def test_model_comparison_uses_wide_seven_column_desktop_layout():
    styles = (STATIC / "styles.css").read_text()
    assert "width: min(1480px, calc(100vw - 32px));" in styles
    assert "grid-template-columns: repeat(7, minmax(0, 1fr));" in styles
    assert ".model-comparison-panel .model-column-header h3" in styles
    assert "white-space: nowrap;" in styles
