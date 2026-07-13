import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

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
    ])
    forecasts_by_strategy = {
        "direct": {
            "NeuralNet": {"1": {"dates": ["2026-01-03", "2026-01-04"], "quantity": [12.0, 13.0]}},
            "SeasonalNaive": {"1": {"dates": ["2026-01-03", "2026-01-04"], "quantity": [11.0, 11.0]}},
        },
        "recursive": {
            "NeuralNet": {"1": {"dates": ["2026-01-03", "2026-01-04"], "quantity": [12.5, 13.5]}},
            "SeasonalNaive": {"1": {"dates": ["2026-01-03", "2026-01-04"], "quantity": [11.0, 11.0]}},
        },
    }
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
    )

    assert set(payload["forecasts_by_strategy"]) == {"direct", "recursive"}
    assert {row["strategy"] for row in payload["benchmark_summary_all"]} == {"direct", "recursive"}
    assert {row["strategy"] for row in payload["dev_summary_all"]} == {"direct", "recursive"}
    assert {row["strategy"] for row in payload["cv_results_all"]} == {"direct", "recursive"}
    assert payload["strategy_comparison"][0]["winner"] == "direct"
    assert payload["strategy_by_horizon"]
    assert payload["selection"]["canonical_model"] == "NeuralNet"
    assert payload["selection"]["canonical_strategy"] == "direct"
    assert all(row.get("strategy") == "direct" for row in payload["benchmark_summary"])
    json.loads((tmp_path / "results.json").read_text())


def test_strategy_controls_exist_on_overview_and_model_pages():
    for name in ("index.html", "model.html"):
        text = (STATIC / name).read_text()
        assert 'id="strategy-select"' in text
        assert 'id="regime-select"' in text
        assert 'id="promo-strategy"' in text

    overview = (STATIC / "index.html").read_text()
    assert 'id="strategy-comparison-table"' in overview
    assert 'id="chart-horizon"' in overview


def test_frontend_javascript_is_syntax_valid():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    for name in ("common.js", "app.js", "model.js"):
        subprocess.run(
            [node, "--check", str(STATIC / name)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )


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


def test_common_js_strategy_helpers_support_single_and_both_modes():
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    script = f"""
      const fs = require('fs');
      const vm = require('vm');
      const context = {{ window: {{}}, console }};
      vm.createContext(context);
      vm.runInContext(fs.readFileSync({json.dumps(str(STATIC / 'common.js'))}, 'utf8'), context);

      const directOnly = {{
        forecasts_by_strategy: {{ direct: {{}} }},
        selection: {{ canonical_strategy: 'direct' }},
      }};
      if (context.availableStrategies(directOnly).join(',') !== 'direct') process.exit(2);
      if (context.canonicalStrategy(directOnly) !== 'direct') process.exit(3);

      const recursiveOnly = {{
        forecasts_by_strategy: {{ recursive: {{}} }},
        selection: {{ canonical_strategy: 'recursive' }},
      }};
      if (context.canonicalStrategy(recursiveOnly) !== 'recursive') process.exit(4);

      const both = {{
        forecasts_by_strategy: {{ direct: {{}}, recursive: {{}} }},
        selection: {{ canonical_strategy: 'recursive' }},
        benchmark_summary_all: [
          {{ model: 'NeuralNet', strategy: 'direct', evaluation_regime: 'conditional', comparison_population: 'common', aggregation: 'global', MAE: 1 }},
          {{ model: 'NeuralNet', strategy: 'recursive', evaluation_regime: 'conditional', comparison_population: 'common', aggregation: 'global', MAE: 2 }},
        ],
      }};
      if (context.availableStrategies(both).length !== 2) process.exit(5);
      const rows = context.summaryRows(both, {{ strategy: 'recursive', regime: 'conditional' }});
      if (rows.length !== 1 || rows[0].MAE !== 2) process.exit(6);
    """
    subprocess.run(
        [node, "-e", script],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
