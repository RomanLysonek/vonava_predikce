import pytest
import numpy as np
import pandas as pd

from ml.framework import (
    Config,
    add_train_lags,
    build_direct_panel,
    build_one_step_panel,
    forecast_recursive,
    prepare_features,
    sanitize_future_covariates,
)
from ml.pipeline import (
    ForecastStrategy,
    RuntimeOptions,
    SubmissionModel,
    PrimaryStrategy,
    resolve_strategies,
    summarize_oof_by_strategy,
    summarize_strategy_pairs,
    select_primary_strategy,
)


def _raw(periods=50, future=0):
    dates = pd.date_range("2025-01-01", periods=periods + future, freq="D")
    rows = []
    for pid in (1, 2):
        for i, date in enumerate(dates):
            rows.append({
                "ProductId": pid,
                "DateKey": date,
                "Quantity": float(10 + i + pid),
                "QuantityApp": float(10 + i + pid),
                "QuantityWeb": 0.0,
                "ProductAvailable": True,
                "CampaignSubTypeWeb": -1,
                "CampaignSubTypeApp": -1,
                "DiscountValueWebRelative": 0.0,
                "DiscountValueAppRelative": 0.0,
                "IsSaleOrPromo": False,
                "PriceLocalVat": 100.0,
            })
    return pd.DataFrame(rows)


def test_direct_origin_features_include_origin_day():
    raw = _raw(periods=40)
    cfg = Config(num_products=2, horizon=7)
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = raw.groupby("ProductId")["DateKey"].min()
    feat = add_train_lags(prepare_features(raw, price_ref, first_seen), cfg.lag_windows)
    panel = build_direct_panel(feat, [1], cfg)
    row = panel[(panel["ProductId"] == 1) & (panel["OriginDateKey"] == raw["DateKey"].max() - pd.Timedelta(days=1))].iloc[0]
    expected = raw[(raw["ProductId"] == 1) & (raw["DateKey"] == row["OriginDateKey"])]["Quantity"].iloc[0]
    assert row["qty_lag_0"] == expected


def test_one_step_panel_matches_direct_horizon_one():
    raw = _raw(periods=40)
    cfg = Config(num_products=2, horizon=7)
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = raw.groupby("ProductId")["DateKey"].min()
    one = build_one_step_panel(raw, price_ref, first_seen, cfg)
    feat = add_train_lags(prepare_features(raw, price_ref, first_seen), cfg.lag_windows)
    direct = build_direct_panel(feat, [1], cfg)
    keys = ["ProductId", "OriginDateKey", "TargetDateKey"]
    cols = keys + ["qty_lag_0", "qty_lag_1", "seasonal_lag_7", "target_baseline"]
    pd.testing.assert_frame_equal(
        one[cols].sort_values(keys).reset_index(drop=True),
        direct[cols].sort_values(keys).reset_index(drop=True),
    )


def test_recursive_feedback_and_future_target_leakage_guard():
    cfg = Config(num_products=2, horizon=3)
    history = _raw(periods=35)
    future_full = _raw(periods=35, future=3)
    future_full = future_full[future_full["DateKey"] > history["DateKey"].max()].copy()
    price_ref = history.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = history.groupby("ProductId")["DateKey"].min()

    def predictor(panel):
        return panel["qty_lag_0"].to_numpy(dtype=float) + 1.0

    path_a = forecast_recursive(history, future_full, predictor, price_ref, first_seen, cfg)
    poisoned = future_full.copy()
    poisoned["Quantity"] = 999999.0
    poisoned["QuantityApp"] = 999999.0
    poisoned["QuantityWeb"] = 999999.0
    poisoned["ProductAvailable"] = False
    path_b = forecast_recursive(history, poisoned, predictor, price_ref, first_seen, cfg)
    np.testing.assert_allclose(path_a["prediction"], path_b["prediction"])
    for pid, group in path_a.groupby("ProductId"):
        last = history[history["ProductId"] == pid].sort_values("DateKey")["Quantity"].iloc[-1]
        np.testing.assert_allclose(group.sort_values("forecast_horizon")["prediction"], [last + 1, last + 2, last + 3])


def test_strategy_summary_and_selection_are_strategy_aware():
    rows = []
    for strategy, offset in (("direct", 1.0), ("recursive", 2.0)):
        for h in (1, 2):
            rows.append({
                "strategy": strategy, "origin_type": "development", "origin": pd.Timestamp("2025-01-01"),
                "ProductId": 1, "DateKey": pd.Timestamp("2025-01-01") + pd.Timedelta(days=h),
                "horizon": h, "actual": 10.0, "ProductAvailable": True,
                "pred_NeuralNet": 10.0 + offset,
                "pred_XGBoost": 10.0 + offset,
                "pred_LightGBM": 10.0 + offset,
                "pred_DynamicRidge": 10.0 + offset,
                "pred_SeasonalNaive": 15.0,
                "pred_MovingAvg28": 15.0,
            })
    oof = pd.DataFrame(rows)
    summary = summarize_oof_by_strategy(oof)
    assert set(summary["strategy"]) == {"direct", "recursive"}
    assert not (
        summary["strategy"].eq("recursive")
        & summary["model"].eq("DynamicRidge")
    ).any()
    assert select_primary_strategy(summary, model="NeuralNet", metric="WAPE") == "direct"
    paired = summarize_strategy_pairs(oof)
    winner = paired[(paired["model"] == "NeuralNet") & (paired["metric"] == "WAPE")]["winner"].iloc[0]
    assert winner == "direct"


def test_resolve_both_strategy():
    assert resolve_strategies(ForecastStrategy.BOTH) == (
        ForecastStrategy.DIRECT,
        ForecastStrategy.RECURSIVE,
    )


def test_recursive_catastrophic_prediction_uses_recorded_baseline_fallback():
    cfg = Config(num_products=2, horizon=3)
    history = _raw(periods=35)
    future = _raw(periods=35, future=3)
    future = future[future["DateKey"] > history["DateKey"].max()].copy()
    price_ref = history.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = history.groupby("ProductId")["DateKey"].min()

    path = forecast_recursive(
        history,
        future,
        lambda panel: np.full(len(panel), 1e100),
        price_ref,
        first_seen,
        cfg,
    )

    assert np.isfinite(path["prediction"]).all()
    assert path["fallback_used"].all()
    assert (path["prediction"] < cfg.recursive_safety_floor).all()


def test_recursive_dynamic_ridge_is_explicitly_unsupported():
    from dataclasses import asdict

    from ml.framework import select_trainable_panel_rows
    from ml.tree_worker import run_job

    cfg = Config(num_products=2, horizon=3)
    history = _raw(periods=400)
    future = _raw(periods=400, future=3)
    future = future[future["DateKey"] > history["DateKey"].max()].copy()
    price_ref = history.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = history.groupby("ProductId")["DateKey"].min()
    panel = build_one_step_panel(history, price_ref, first_seen, cfg)
    train_panel = select_trainable_panel_rows(
        panel, cutoff=history["DateKey"].max(), available_only=True
    )

    with pytest.raises(ValueError, match="does not support recursive"):
        run_job({
            "cfg": asdict(cfg),
            "strategy": "recursive",
            "models": ["DynamicRidge"],
            "train_panel": train_panel,
            "history_raw": history,
            "future_covariates": sanitize_future_covariates(future),
            "price_ref": price_ref,
            "first_seen": first_seen,
        })
