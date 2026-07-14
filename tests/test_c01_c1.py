import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from ml.framework import (
    Config,
    _weighted_baseline,
    add_train_lags,
    build_direct_panel,
    direct_panel_feature_names,
    forecast_recursive,
    prepare_features,
    product_reference_dates,
    recency_sample_weights,
    select_trainable_panel_rows,
)
from ml.models.neural_net import predict_ensemble, residual_support_bounds
from ml.pipeline import RuntimeOptions, _json_safe, configure_c1_runtime
from ml.run_c1_screening import _config_key, _extract_model_rows, _pick_winner
from ml.export_results import _read_csv_if_present


def _raw(periods=400, future=0):
    dates = pd.date_range("2024-01-01", periods=periods + future, freq="D")
    rows = []
    for pid in (1, 2):
        for i, date in enumerate(dates):
            qty = float(20 + pid + (i % 7))
            rows.append({
                "ProductId": pid,
                "DateKey": date,
                "Quantity": qty,
                "QuantityApp": qty,
                "QuantityWeb": 0.0,
                "ProductAvailable": True,
                "CampaignSubTypeWeb": -1,
                "CampaignSubTypeApp": -1,
                "DiscountValueWebRelative": 0.0,
                "DiscountValueAppRelative": 0.0,
                "IsSaleOrPromo": False,
                "PriceLocalVat": 100.0,
                "is_gap_filled": False,
            })
    return pd.DataFrame(rows)


def test_c1_baseline_variants_have_expected_semantics():
    lags = np.array([
        [10.0, 20.0, 30.0, 40.0],
        [10.0, np.nan, 30.0, np.nan],
    ])
    np.testing.assert_allclose(
        _weighted_baseline(lags, "lag7"), [10.0, 10.0]
    )
    np.testing.assert_allclose(
        _weighted_baseline(lags, "weighted_4321"), [20.0, 100.0 / 6.0]
    )
    np.testing.assert_allclose(
        _weighted_baseline(lags, "weighted_8421"), [260.0 / 15.0, 140.0 / 10.0]
    )
    np.testing.assert_allclose(
        _weighted_baseline(lags, "weekday_median"), [25.0, 20.0]
    )


def test_c1_recency_weights_are_mean_one_and_recent_heavier():
    dates = pd.Series(pd.date_range("2025-01-01", periods=11, freq="D"))
    weights = recency_sample_weights(dates, pd.Timestamp("2025-01-11"), 5.0)
    assert np.isclose(weights.mean(), 1.0)
    assert weights[-1] > weights[0]
    assert np.isclose(weights[-1] / weights[-6], 2.0)


def test_c1_training_window_filters_targets_and_attaches_weights():
    dates = pd.date_range("2025-01-01", periods=30, freq="D")
    panel = pd.DataFrame({
        "TargetDateKey": dates,
        "target": 10.0,
        "target_baseline": 9.0,
        "TargetProductAvailable": True,
    })
    cfg = Config(training_window_days=10, recency_half_life_days=5.0)
    selected = select_trainable_panel_rows(
        panel, cutoff=dates.max(), available_only=True, cfg=cfg
    )
    assert len(selected) == 10
    assert selected["TargetDateKey"].min() == dates.max() - pd.Timedelta(days=9)
    assert np.isclose(selected["sample_weight"].mean(), 1.0)
    assert selected["sample_weight"].iloc[-1] > selected["sample_weight"].iloc[0]


def test_c1_trend_feature_group_is_opt_in_and_finite_after_preprocessing():
    raw = _raw(periods=410)
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(raw)

    off = Config(num_products=2, enable_trend_features=False)
    off_feat = add_train_lags(
        prepare_features(raw, price_ref, first_seen, first_available),
        off.lag_windows,
        baseline_variant=off.baseline_variant,
    )
    off_panel = build_direct_panel(off_feat, [1], off)
    assert "trend_log_ratio_baseline_annual" not in direct_panel_feature_names(off)
    assert "trend_log_ratio_baseline_annual" not in off_panel.columns

    on = Config(num_products=2, enable_trend_features=True)
    on_feat = add_train_lags(
        prepare_features(raw, price_ref, first_seen, first_available),
        on.lag_windows,
        baseline_variant=on.baseline_variant,
    )
    on_panel = build_direct_panel(on_feat, [1], on)
    expected = {
        "calendar_time_years",
        "trend_log_ratio_mean_7_28",
        "trend_log_ratio_mean_14_28",
        "trend_log_ratio_lag0_28",
        "trend_log_slope_7",
        "trend_log_slope_28",
        "annual_reference",
        "annual_reference_missing",
        "trend_log_ratio_baseline_annual",
    }
    assert expected.issubset(set(direct_panel_feature_names(on)))
    assert expected.issubset(set(on_panel.columns))


def test_residual_support_bounds_are_robust_and_widened():
    cfg = Config(
        nn_residual_guard_lower_quantile=0.1,
        nn_residual_guard_upper_quantile=0.9,
        nn_residual_guard_margin=1.0,
    )
    lower, upper = residual_support_bounds(np.arange(11, dtype=float), cfg)
    assert np.isclose(lower, 0.0)
    assert np.isclose(upper, 10.0)


class _ConstantResidual(nn.Module):
    def __init__(self, value, lower=-2.0, upper=2.0):
        super().__init__()
        self.value = float(value)
        self.residual_guard_lower = float(lower)
        self.residual_guard_upper = float(upper)

    def forward(self, x_num, x_prod, x_cw, x_ca, x_horizon):
        return torch.full((len(x_num),), self.value, device=x_num.device)


def _minimal_tensors(n=3, baseline=10.0):
    return {
        "num": torch.zeros((n, 1), dtype=torch.float32),
        "prod": torch.zeros(n, dtype=torch.int64),
        "cw": torch.zeros(n, dtype=torch.int64),
        "ca": torch.zeros(n, dtype=torch.int64),
        "horizon": torch.zeros(n, dtype=torch.int64),
        "baseline_log1p": torch.full(
            (n,), float(np.log1p(baseline)), dtype=torch.float32
        ),
        "sample_weight": torch.ones(n, dtype=torch.float32),
    }


def test_recursive_residual_guard_constrains_finite_extrapolation_only_when_enabled():
    model = _ConstantResidual(10.0, lower=-2.0, upper=2.0)
    tensors = _minimal_tensors()
    direct = predict_ensemble([model], tensors)
    guarded = predict_ensemble(
        [model], tensors, apply_residual_guard=True, return_diagnostics=True
    )
    assert (direct > 100_000).all()
    assert (guarded["prediction"] < 100).all()
    assert guarded["residual_guard"].all()
    assert not guarded["residual_nonfinite"].any()
    assert np.allclose(guarded["residual_raw_max"], 10.0)


def test_nonfinite_seed_residuals_reach_recursive_fallback():
    model = _ConstantResidual(float("inf"), lower=-2.0, upper=2.0)
    guarded = predict_ensemble(
        [model], _minimal_tensors(),
        apply_residual_guard=True,
        return_diagnostics=True,
    )
    assert np.isnan(guarded["prediction"]).all()
    assert guarded["residual_nonfinite"].all()
    assert not guarded["residual_guard"].any()


def test_generic_recursive_guard_catches_100k_but_preserves_retail_scale():
    cfg = Config(num_products=2, horizon=3)
    history = _raw(periods=35)
    full = _raw(periods=35, future=3)
    future = full[full["DateKey"] > history["DateKey"].max()].copy()
    price_ref = history.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(history)

    bad = forecast_recursive(
        history, future, lambda panel: np.full(len(panel), 100_000.0),
        price_ref, first_seen, cfg, first_available,
    )
    assert bad["catastrophic_guard"].all()
    assert bad["fallback_used"].all()
    assert (bad["prediction"] < 100_000).all()

    plausible = forecast_recursive(
        history, future, lambda panel: np.full(len(panel), 600.0),
        price_ref, first_seen, cfg, first_available,
    )
    assert not plausible["catastrophic_guard"].any()
    np.testing.assert_allclose(plausible["prediction"], 600.0)




def test_pipeline_accepts_direct_c1_config_payload(tmp_path):
    path = tmp_path / "direct.json"
    path.write_text(json.dumps({
        "training_window_days": 365,
        "recency_half_life_days": 90.0,
        "baseline_variant": "lag7",
        "enable_trend_features": False,
    }))
    cfg = Config()
    configure_c1_runtime(cfg, RuntimeOptions(c1_config=str(path)))
    assert cfg.training_window_days == 365
    assert cfg.recency_half_life_days == 90.0
    assert cfg.baseline_variant == "lag7"

def test_pipeline_loads_c1_recommendation_and_cli_can_override(tmp_path):
    path = tmp_path / "recommendation.json"
    path.write_text(json.dumps({
        "recommendation": {
            "config": {
                "training_window_days": 730,
                "recency_half_life_days": 180.0,
                "baseline_variant": "weighted_8421",
                "enable_trend_features": True,
            }
        }
    }))
    cfg = Config()
    runtime = configure_c1_runtime(
        cfg,
        RuntimeOptions(c1_config=str(path), training_window_days="365"),
    )
    assert cfg.training_window_days == 365
    assert cfg.recency_half_life_days == 180.0
    assert cfg.baseline_variant == "weighted_8421"
    assert cfg.enable_trend_features is True
    assert runtime["sources"]["training_window_days"] == "CLI override"


def test_recursive_safety_limit_does_not_expand_from_generated_history():
    cfg = Config(num_products=2, horizon=3)
    history = _raw(periods=35)
    full = _raw(periods=35, future=3)
    future = full[full["DateKey"] > history["DateKey"].max()].copy()
    price_ref = history.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(history)
    values = iter((9_000.0, 100_000.0, 600.0))

    def step(panel):
        return np.full(len(panel), next(values))

    result = forecast_recursive(
        history, future, step, price_ref, first_seen, cfg, first_available,
    )
    first = result[result["forecast_horizon"].eq(1)]
    second = result[result["forecast_horizon"].eq(2)]
    third = result[result["forecast_horizon"].eq(3)]
    assert not first["catastrophic_guard"].any()
    assert second["catastrophic_guard"].all()
    assert second["fallback_used"].all()
    assert not third["catastrophic_guard"].any()
    np.testing.assert_allclose(third["prediction"], 600.0)
    assert (second["safety_limit"] == first["safety_limit"].to_numpy()).all()


def test_c1_trend_slopes_capture_direction():
    raw = _raw(periods=410)
    raw.loc[raw["ProductId"].eq(1), "Quantity"] = np.arange(410, dtype=float) + 1.0
    raw.loc[raw["ProductId"].eq(1), "QuantityApp"] = np.arange(410, dtype=float) + 1.0
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(raw)
    cfg = Config(num_products=2, enable_trend_features=True)
    feat = add_train_lags(
        prepare_features(raw, price_ref, first_seen, first_available),
        cfg.lag_windows,
        baseline_variant=cfg.baseline_variant,
    )
    panel = build_direct_panel(feat, [1], cfg)
    latest = panel[panel["ProductId"].eq(1)].sort_values("OriginDateKey").iloc[-1]
    assert latest["trend_log_slope_7"] > 0.0
    assert latest["trend_log_slope_28"] > 0.0


def test_c1_screening_quality_guard_and_checkpoint_key():
    rows = [
        {"candidate": "control", "model": "NeuralNet", "WAPE": 0.30,
         "test_aligned_WAPE": 0.30, "BiasRatio": 0.02},
        {"candidate": "eligible", "model": "NeuralNet", "WAPE": 0.308,
         "test_aligned_WAPE": 0.25, "BiasRatio": 0.01},
        {"candidate": "overfit", "model": "NeuralNet", "WAPE": 0.32,
         "test_aligned_WAPE": 0.20, "BiasRatio": 0.00},
    ]
    assert _pick_winner(
        rows, ["control", "eligible", "overfit"], "control", 0.03
    ) == "eligible"
    key = _config_key({
        "training_window_days": None,
        "recency_half_life_days": 180.0,
        "baseline_variant": "weighted_4321",
        "enable_trend_features": False,
    })
    assert "training_window_days-all" in key
    assert "recency_half_life_days-180p0" in key


def test_c1_screening_extracts_lowercase_summary_coverage():
    oof = pd.DataFrame({
        "strategy": ["direct", "direct", "direct", "direct"],
        "origin_type": ["development"] * 4,
        "origin": pd.to_datetime([
            "2023-01-10", "2023-01-10", "2024-06-20", "2024-06-20",
        ]),
        "ProductId": [1, 2, 1, 2],
        "DateKey": pd.to_datetime([
            "2023-01-11", "2023-01-11", "2024-06-21", "2024-06-21",
        ]),
        "horizon": [1, 1, 1, 1],
        "ProductAvailable": [True, True, True, True],
        "actual": [10.0, 20.0, 12.0, 18.0],
        "pred_NeuralNet": [11.0, 19.0, 13.0, 17.0],
    })
    config = {
        "training_window_days": None,
        "recency_half_life_days": None,
        "baseline_variant": "weighted_4321",
        "enable_trend_features": False,
    }
    rows = _extract_model_rows("control", "recency", config, oof, 1.0)
    assert len(rows) == 1
    assert rows[0]["model"] == "NeuralNet"
    assert rows[0]["Coverage"] == 1.0


def test_json_safe_serializes_timestamp_and_numpy_scalars_strictly():
    payload = {
        "origin": pd.Timestamp("2024-11-29"),
        "generated": np.datetime64("2026-01-04"),
        "count": np.int64(7),
        "ratio": np.float64(0.25),
        "flag": np.bool_(True),
        "missing": np.float64(np.nan),
        "nested": [pd.NaT, np.array([1, 2], dtype=np.int64)],
    }
    safe = _json_safe(payload)
    assert safe["origin"].startswith("2024-11-29")
    assert safe["generated"].startswith("2026-01-04")
    assert safe["count"] == 7
    assert safe["ratio"] == 0.25
    assert safe["flag"] is True
    assert safe["missing"] is None
    assert safe["nested"] == [None, [1, 2]]
    json.dumps(safe, allow_nan=False)


def test_artifact_exporter_accepts_missing_and_empty_optional_csvs(tmp_path):
    missing = tmp_path / "missing.csv"
    assert _read_csv_if_present(str(missing)).empty

    empty = tmp_path / "strategy_pair_summary.csv"
    empty.write_bytes(b"")
    assert _read_csv_if_present(str(empty)).empty

    whitespace = tmp_path / "whitespace.csv"
    whitespace.write_text("  \n")
    assert _read_csv_if_present(str(whitespace)).empty

    populated = tmp_path / "populated.csv"
    populated.write_text("model,WAPE\nNeuralNet,0.3\n")
    loaded = _read_csv_if_present(str(populated))
    assert loaded.to_dict(orient="records") == [
        {"model": "NeuralNet", "WAPE": 0.3}
    ]
