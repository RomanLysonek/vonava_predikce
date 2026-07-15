import json
from pathlib import Path

import numpy as np
import pandas as pd

from dashboard_artifacts import (
    collect_ablation_showcase,
    publish_static_dashboard,
    summarize_per_product_oof,
    summarize_top_deciles,
)


PREDICTIONS = {"NeuralNet": "pred_NeuralNet", "XGBoost": "pred_XGBoost"}


def _oof():
    actual = np.array([10, 20, 80, 100, 15, 25], dtype=float)
    return pd.DataFrame({
        "origin_type": ["development"] * 3 + ["recent_benchmark"] * 3,
        "strategy": "direct",
        "origin": pd.to_datetime(["2024-01-01"] * 3 + ["2025-12-01"] * 3),
        "validation_stratum": ["winter_test_like"] * 3 + ["holiday_event"] * 3,
        "ProductId": [1, 1, 2, 1, 2, 2],
        "DateKey": pd.date_range("2024-01-02", periods=6),
        "horizon": [1, 2, 3, 1, 2, 3],
        "ProductAvailable": True,
        "actual": actual,
        "pred_NeuralNet": actual + np.array([1, -2, 10, -10, 1, 2]),
        "pred_XGBoost": actual + np.array([2, -1, 5, -5, -1, 1]),
    })


def test_product_and_top_decile_diagnostics_are_common_population():
    oof = _oof()
    product = summarize_per_product_oof(oof, PREDICTIONS)
    assert set(product["model"]) == {"NeuralNet", "XGBoost"}
    assert set(product["origin_type"]) == {"development", "recent_benchmark"}
    assert product["WAPE"].notna().all()

    decile, errors = summarize_top_deciles(oof, PREDICTIONS, quantile=0.8, max_error_rows=2)
    assert set(decile["model"]) == {"NeuralNet", "XGBoost"}
    assert (decile["n"] >= 1).all()
    assert len(errors) <= 8
    assert errors["absolute_error"].notna().all()


def test_ablation_showcase_marks_recommendations(tmp_path):
    c1 = tmp_path / "c1_screening"
    c1.mkdir()
    pd.DataFrame([
        {"candidate": "control", "stage": "recency", "model": "NeuralNet", "WAPE": 0.3, "test_aligned_WAPE": 0.31, "BiasRatio": 0.1, "Coverage": 1.0},
        {"candidate": "winner", "stage": "recency", "model": "NeuralNet", "WAPE": 0.28, "test_aligned_WAPE": 0.29, "BiasRatio": 0.05, "Coverage": 1.0},
    ]).to_csv(c1 / "c1_screening_results.csv", index=False)
    (c1 / "recommendation.json").write_text(json.dumps({"recommendation": {"candidate": "winner"}}))
    result = collect_ablation_showcase(tmp_path)
    assert result.loc[result["candidate"].eq("winner"), "selected"].all()


def test_static_dashboard_is_self_contained(tmp_path):
    static = tmp_path / "webapp" / "static"
    outputs = tmp_path / "outputs"
    static.mkdir(parents=True)
    outputs.mkdir()
    (static / "index.html").write_text('<link href="/static/styles.css"><script src="/static/common.js?v=1"></script>')
    (static / "model.html").write_text('<script src="/static/common.js?v=1"></script>')
    (static / "evaluation.html").write_text('<script src="/static/common.js?v=1"></script><script src="/static/evaluation.js?v=1"></script>')
    (static / "dataset.html").write_text('<script src="/static/common.js?v=1"></script><script src="/static/dataset.js?v=1"></script>')
    (static / "common.js").write_text("function ok() {}")
    (static / "evaluation.js").write_text("function evaluation() {}")
    (static / "dataset.js").write_text("function dataset() {}")
    (static / "styles.css").write_text("body{}")
    results = outputs / "results.json"
    results.write_text(json.dumps({"selection": {"canonical_model": "NeuralNet"}}))

    manifest = publish_static_dashboard(tmp_path, results)
    assert (tmp_path / "docs" / "index.html").exists()
    assert (tmp_path / "docs" / "evaluation.html").exists()
    assert (tmp_path / "docs" / "evaluation.js").exists()
    assert (tmp_path / "docs" / "dataset.html").exists()
    assert (tmp_path / "docs" / "dataset.js").exists()
    html = (tmp_path / "docs" / "index.html").read_text()
    assert 'window.STATIC_DASHBOARD = true' in html
    assert './styles.css' in html
    assert (tmp_path / "docs" / "results.json").exists()
    assert manifest["entrypoint"] == "docs/index.html"
