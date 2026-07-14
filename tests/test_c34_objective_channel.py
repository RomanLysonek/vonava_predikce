import json

import numpy as np
import pandas as pd
import torch

from ml.framework import (
    Config,
    add_train_lags,
    build_direct_panel,
    forecast_recursive,
    prepare_features,
    product_reference_dates,
    select_trainable_panel_rows,
)
from ml.models.lightgbm_model import _target as lightgbm_target
from ml.models.neural_net import (
    make_numeric_preprocessor,
    make_tensors,
    neural_training_target,
    predict_direct,
    total_loss_values,
    train_model,
)
from ml.models.xgboost_model import _target as xgboost_target
from ml.pipeline import (
    RuntimeOptions,
    configure_c34_runtime,
    run_walk_forward_cv_direct,
    summarize_channel_share_oof,
)
from ml.run_c34_screening import _best_nn, _best_tree_mode


def _raw(periods=390, products=(1, 2)):
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    rows = []
    for pid in products:
        for i, date in enumerate(dates):
            total = float(12 + pid + i % 7)
            share = 0.2 + 0.01 * (i % 10)
            app = total * share
            rows.append({
                "ProductId": pid,
                "DateKey": date,
                "Quantity": total,
                "QuantityApp": app,
                "QuantityWeb": total - app,
                "ProductAvailable": True,
                "CampaignSubTypeWeb": -1,
                "CampaignSubTypeApp": -1,
                "DiscountValueWebRelative": 0.0,
                "DiscountValueAppRelative": 0.0,
                "IsSaleOrPromo": False,
                "PriceLocalVat": 100.0 + pid,
                "is_gap_filled": False,
            })
    return pd.DataFrame(rows)


def _panel(raw, cfg):
    price_ref = raw.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(raw)
    feat = prepare_features(raw, price_ref, first_seen, first_available, cfg)
    feat = add_train_lags(feat, cfg.lag_windows, baseline_variant=cfg.baseline_variant)
    return build_direct_panel(feat, range(1, cfg.horizon + 1), cfg)


def test_c3_training_target_modes():
    frame = pd.DataFrame({"target": [3.0, 8.0], "target_baseline": [1.0, 4.0]})
    residual = neural_training_target(frame, Config(nn_target_mode="residual"))
    raw = neural_training_target(frame, Config(nn_target_mode="log1p"))
    np.testing.assert_allclose(residual, np.log1p(frame.target) - np.log1p(frame.target_baseline))
    np.testing.assert_allclose(raw, np.log1p(frame.target))


def test_c3_loss_formulations_are_per_row_and_finite():
    pred = torch.tensor([0.0, 2.0])
    target = torch.tensor([1.0, 0.0])
    mse = total_loss_values(pred, target, Config(nn_loss="mse"))
    np.testing.assert_allclose(mse.numpy(), [1.0, 4.0])
    for loss in ("huber", "combined", "logcosh"):
        values = total_loss_values(pred, target, Config(nn_loss=loss))
        assert values.shape == pred.shape
        assert torch.isfinite(values).all()
        assert (values >= 0).all()


def test_direct_panel_carries_channel_targets():
    raw = _raw(periods=45, products=(1,))
    cfg = Config(num_products=1, horizon=2)
    panel = _panel(raw, cfg)
    row = panel[panel["horizon"].eq(1)].iloc[0]
    target_date = row["TargetDateKey"]
    expected = raw[raw["DateKey"].eq(target_date)].iloc[0]
    assert np.isclose(row["target_app"], expected["QuantityApp"])
    assert np.isclose(row["target_web"], expected["QuantityWeb"])
    assert np.isclose(row["target_app"] + row["target_web"], row["target"])


def test_c4_multitask_prediction_splits_total_consistently():
    raw = _raw(periods=80)
    cfg = Config(
        num_products=2,
        horizon=2,
        hidden_dims=(16,),
        dropout=(0.0,),
        batch_size=64,
        seeds=(42,),
        enable_channel_history_features=True,
        channel_aux_weight=0.1,
        nn_training_backend="dataloader",
    )
    panel = _panel(raw, cfg)
    train = select_trainable_panel_rows(panel, cutoff=raw.DateKey.max(), cfg=cfg)
    train = train.iloc[:-4].reset_index(drop=True)
    eval_panel = train.tail(4).reset_index(drop=True)
    scaler = make_numeric_preprocessor()
    tensors = make_tensors(train, scaler, fit=True, cfg=cfg)
    target = neural_training_target(train, cfg)
    model = train_model(tensors, target, cfg, epochs=1, seed=42)
    output = predict_direct([model], scaler, eval_panel, cfg, return_diagnostics=True)
    assert np.isfinite(output["prediction"]).all()
    assert ((output["app_share"] >= 0) & (output["app_share"] <= 1)).all()
    np.testing.assert_allclose(
        output["prediction_app"] + output["prediction_web"],
        output["prediction"],
        rtol=1e-6,
    )


def test_tree_target_modes_have_expected_transforms():
    panel = pd.DataFrame({"target": [0.0, 3.0], "target_baseline": [1.0, 1.0]})
    for target_fn in (xgboost_target, lightgbm_target):
        np.testing.assert_allclose(target_fn(panel, "log1p"), np.log1p(panel.target))
        np.testing.assert_allclose(
            target_fn(panel, "residual"),
            np.log1p(panel.target) - np.log1p(panel.target_baseline),
        )
        np.testing.assert_allclose(target_fn(panel, "tweedie"), panel.target)


def test_pipeline_loads_c34_recommendation_and_cli_override(tmp_path):
    path = tmp_path / "c34.json"
    path.write_text(json.dumps({"recommendation": {"config": {
        "nn_loss": "logcosh",
        "nn_target_mode": "residual",
        "enable_channel_history_features": True,
        "channel_aux_weight": 0.2,
        "tree_target_mode": "log1p",
        "xgboost_target_mode": "tweedie",
        "lightgbm_target_mode": "residual",
    }}}))
    cfg = Config()
    runtime = configure_c34_runtime(cfg, RuntimeOptions(c34_config=str(path), nn_loss="mse"))
    assert cfg.nn_loss == "mse"
    assert cfg.enable_channel_history_features is True
    assert cfg.channel_aux_weight == 0.2
    assert cfg.tree_target_mode == "log1p"
    assert cfg.xgboost_target_mode == "tweedie"
    assert cfg.lightgbm_target_mode == "residual"
    assert runtime["sources"]["nn_loss"] == "CLI override"


def test_fast_direct_cv_can_skip_all_structured_models():
    raw = _raw(periods=390)
    cfg = Config(
        num_products=2,
        horizon=2,
        hidden_dims=(8,),
        dropout=(0.0,),
        seeds=(42,),
        cv_epochs=1,
        batch_size=256,
        nn_training_backend="dataloader",
    )
    origin = raw.DateKey.max() - pd.Timedelta(days=2)
    oof = run_walk_forward_cv_direct(
        raw,
        [origin],
        "development",
        cfg,
        run_neural=True,
        structured_models=(),
    )
    assert "pred_NeuralNet" in oof
    assert "pred_XGBoost" not in oof
    assert len(oof) == 4


def test_c34_selection_helpers_respect_quality_guard():
    rows = []
    for candidate, aligned, broad, bias in [
        ("control", 0.30, 0.30, 0.03),
        ("bad_broad", 0.20, 0.40, 0.01),
        ("winner", 0.28, 0.305, 0.02),
    ]:
        rows.append({
            "candidate": candidate,
            "stage": "nn_loss",
            "model": "NeuralNet",
            "test_aligned_WAPE": aligned,
            "WAPE": broad,
            "BiasRatio": bias,
            "Coverage": 1.0,
            "app_share_weighted_MAE": np.nan,
        })
    assert _best_nn(rows, ["control", "bad_broad", "winner"], "control", 0.03) == "winner"

    tree_rows = []
    for mode, score in [("log1p", 0.30), ("residual", 0.28), ("tweedie", 0.31)]:
        for model in ("XGBoost", "LightGBM"):
            tree_rows.append({
                "stage": "tree_target",
                "tree_target_mode": mode,
                "model": model,
                "test_aligned_WAPE": score,
                "WAPE": score,
                "BiasRatio": 0.0,
                "Coverage": 1.0,
            })
    assert _best_tree_mode(tree_rows, 0.03) == "residual"



def test_c4_channel_history_features_are_origin_safe():
    raw = _raw(periods=80, products=(1,))
    cfg = Config(
        num_products=1,
        horizon=2,
        enable_channel_history_features=True,
    )
    panel = _panel(raw, cfg)
    row = panel[panel["horizon"].eq(1)].iloc[20]
    origin = row["OriginDateKey"]
    hist = raw[raw["DateKey"] <= origin].sort_values("DateKey")
    current = hist.iloc[-1]
    lag7 = hist.iloc[-8]
    current_share = current["QuantityApp"] / (
        current["QuantityApp"] + current["QuantityWeb"]
    )
    lag7_share = lag7["QuantityApp"] / (
        lag7["QuantityApp"] + lag7["QuantityWeb"]
    )
    assert np.isclose(row["app_share_lag_0"], current_share)
    assert np.isclose(row["app_share_lag_7"], lag7_share)
    assert 0.0 <= row["app_share_roll_7"] <= 1.0
    assert 0.0 <= row["app_share_roll_28"] <= 1.0


def test_c34_material_improvement_threshold_keeps_control():
    rows = []
    for candidate, aligned in [("control", 0.3000), ("tiny", 0.2997)]:
        rows.append({
            "candidate": candidate,
            "stage": "nn_loss",
            "model": "NeuralNet",
            "test_aligned_WAPE": aligned,
            "WAPE": 0.30,
            "BiasRatio": 0.0,
            "Coverage": 1.0,
            "app_share_weighted_MAE": np.nan,
        })
    assert _best_nn(
        rows, ["control", "tiny"], "control", 0.03, 0.002
    ) == "control"

    tree_rows = []
    for mode, score in [("log1p", 0.3000), ("residual", 0.2997)]:
        for model in ("XGBoost", "LightGBM"):
            tree_rows.append({
                "stage": "tree_target",
                "tree_target_mode": mode,
                "model": model,
                "test_aligned_WAPE": score,
                "WAPE": 0.30,
                "BiasRatio": 0.0,
                "Coverage": 1.0,
            })
    assert _best_tree_mode(tree_rows, 0.03, 0.002) == "log1p"



def test_recursive_channel_history_uses_observed_share_without_aux_head():
    raw = _raw(periods=80, products=(1,))
    history = raw.iloc[:-2].copy()
    future = raw.iloc[-2:].copy()
    cfg = Config(
        num_products=1,
        horizon=2,
        enable_channel_history_features=True,
    )
    price_ref = history.groupby("ProductId")["PriceLocalVat"].median()
    first_seen, first_available = product_reference_dates(history)
    seen_lag0 = []

    def predict(panel):
        seen_lag0.append(float(panel["app_share_lag_0"].iloc[0]))
        return np.array([20.0])

    path = forecast_recursive(
        history,
        future,
        predict,
        price_ref,
        first_seen,
        cfg,
        first_available,
    )
    assert len(path) == 2
    assert np.isfinite(seen_lag0).all()
    # The second synthetic state should retain a plausible recent channel mix,
    # not the legacy all-app fallback share of 1.0.
    assert 0.0 < seen_lag0[1] < 1.0
    assert path["app_share"].isna().all()
    assert path["feedback_app_share"].between(0.0, 1.0).all()



def test_channel_share_summary_reports_weighted_error():
    oof = pd.DataFrame({
        "origin_type": ["development", "development"],
        "strategy": ["direct", "direct"],
        "actual": [10.0, 30.0],
        "actual_AppShare": [0.2, 0.6],
        "pred_AppShare_NeuralNet": [0.3, 0.5],
    })
    summary = summarize_channel_share_oof(oof)
    assert len(summary) == 1
    row = summary.iloc[0]
    assert np.isclose(row["app_share_MAE"], 0.1)
    assert np.isclose(row["app_share_weighted_MAE"], 0.1)
    assert row["coverage"] == 1.0



def test_tree_target_modes_can_be_selected_per_model():
    rows = []
    scores = {
        "XGBoost": {"log1p": 0.30, "residual": 0.27, "tweedie": 0.31},
        "LightGBM": {"log1p": 0.26, "residual": 0.28, "tweedie": 0.29},
    }
    for model, modes in scores.items():
        for mode, score in modes.items():
            rows.append({
                "stage": "tree_target",
                "tree_target_mode": mode,
                "model": model,
                "test_aligned_WAPE": score,
                "WAPE": score,
                "BiasRatio": 0.0,
                "Coverage": 1.0,
            })
    assert _best_tree_mode(rows, 0.03, 0.002, model="XGBoost") == "residual"
    assert _best_tree_mode(rows, 0.03, 0.002, model="LightGBM") == "log1p"
