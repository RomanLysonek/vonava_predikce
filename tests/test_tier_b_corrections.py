import pytest
import numpy as np
import pandas as pd
from ml.framework import Config, build_direct_panel, compute_metrics
from ml.models.dynamic_ridge import train_dynamic_ridge, predict_dynamic_ridge
from ml.pipeline import summarize_oof, run_tree_baselines, select_primary_summary

def _add_required_cols(df, cfg):
    from ml.framework import STATIC_NUMERIC_FEATURES, lag_feature_names, SEASONAL_LAG_DAYS, RECENT_POINT_LAGS
    for col in STATIC_NUMERIC_FEATURES:
        if col not in df.columns:
            df[col] = 0.0
    for col in lag_feature_names(cfg.lag_windows):
        if col not in df.columns:
            df[col] = 0.0
    for lag in SEASONAL_LAG_DAYS:
        if f"seasonal_lag_{lag}" not in df.columns:
            df[f"seasonal_lag_{lag}"] = 10.0
    for lag in RECENT_POINT_LAGS:
        if f"qty_lag_{lag}" not in df.columns:
            df[f"qty_lag_{lag}"] = 10.0
    return df

def test_dynamic_ridge_reconstructs_baseline():
    """1. Dynamic Ridge zero residual reconstructs target_baseline."""
    cfg = Config(ridge_alpha=10.0)
    df = pd.DataFrame({
        "ProductId": [1, 1],
        "DateKey": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "target": [10.0, 20.0],
        "target_baseline": [10.0, 20.0],
        "product_idx": [0, 0],
        "campaign_idx_web": [0, 0],
        "campaign_idx_app": [0, 0],
        "horizon": [1, 2],
    })
    df = _add_required_cols(df, cfg)

    model = train_dynamic_ridge(df, cfg)
    preds = predict_dynamic_ridge(model, df, cfg)
    np.testing.assert_allclose(preds, df["target_baseline"], atol=1e-2)

def test_dynamic_ridge_finite_non_negative():
    """2. Dynamic Ridge predictions are finite and non-negative."""
    cfg = Config()
    df = pd.DataFrame({
        "ProductId": [1],
        "DateKey": pd.to_datetime(["2024-01-01"]),
        "target": [10.0],
        "target_baseline": [10.0],
        "product_idx": [0],
        "campaign_idx_web": [0],
        "campaign_idx_app": [0],
        "horizon": [1],
    })
    df = _add_required_cols(df, cfg)
        
    model = train_dynamic_ridge(df, cfg)
    
    # Predict on very different data that might cause negative/large values
    test_df = df.copy()
    test_df["qty_lag_1"] = -1000.0 
    preds = predict_dynamic_ridge(model, test_df, cfg)
    assert np.all(np.isfinite(preds))
    assert np.all(preds >= 0.0)

def test_dynamic_ridge_handles_unseen_categories():
    """3. Dynamic Ridge handles unseen categories."""
    cfg = Config()
    train_df = pd.DataFrame({
        "ProductId": [1],
        "DateKey": pd.to_datetime(["2024-01-01"]),
        "target": [10.0],
        "target_baseline": [10.0],
        "product_idx": [0],
        "campaign_idx_web": [0],
        "campaign_idx_app": [0],
        "horizon": [1],
    })
    train_df = _add_required_cols(train_df, cfg)
        
    model = train_dynamic_ridge(train_df, cfg)
    
    test_df = train_df.copy()
    test_df["product_idx"] = 999 # Unseen
    preds = predict_dynamic_ridge(model, test_df, cfg)
    assert len(preds) == 1

def test_dynamic_ridge_cap_behavior():
    """4. Dynamic Ridge cap is disabled by default. 5. Configured cap affects only values above."""
    cfg_no_cap = Config(ridge_prediction_cap=None)
    cfg_cap = Config(ridge_prediction_cap=5.0)
    
    df = pd.DataFrame({
        "ProductId": [1],
        "DateKey": pd.to_datetime(["2024-01-01"]),
        "target": [100.0],
        "target_baseline": [100.0],
        "product_idx": [0],
        "campaign_idx_web": [0],
        "campaign_idx_app": [0],
        "horizon": [1],
    })
    df = _add_required_cols(df, cfg_no_cap)
        
    model = train_dynamic_ridge(df, cfg_no_cap)
    
    preds_no_cap = predict_dynamic_ridge(model, df, cfg_no_cap)
    assert preds_no_cap[0] > 50.0 # Should be near 100
    
    preds_cap = predict_dynamic_ridge(model, df, cfg_cap)
    assert preds_cap[0] == 5.0

def test_evaluation_regimes():
    """6. Conditional regime excludes unavailable rows. 7. Realized regime includes them."""
    oof = pd.DataFrame({
        "origin": [1, 1],
        "actual": [10.0, 20.0],
        "pred_NeuralNet": [11.0, 21.0],
        "ProductAvailable": [True, False],
    })
    # Add dummy cols for other models to satisfy common population logic
    from ml.pipeline import OOF_MODEL_COLUMNS
    for m, col in OOF_MODEL_COLUMNS.items():
        if col != "pred_NeuralNet":
            oof[col] = 1.0

    summary = summarize_oof(oof)
    
    # Realized global
    realized = summary[(summary["evaluation_regime"] == "realized") & (summary["aggregation"] == "global") & (summary["model"] == "NeuralNet")]
    assert realized.iloc[0]["n_scored"] == 2
    
    # Conditional global
    conditional = summary[(summary["evaluation_regime"] == "conditional") & (summary["aggregation"] == "global") & (summary["model"] == "NeuralNet")]
    assert conditional.iloc[0]["n_scored"] == 1

def test_common_population_scoring():
    """8. Common-population summaries use identical keys and n_scored across models."""
    oof = pd.DataFrame({
        "origin": [1, 1, 1],
        "actual": [10.0, 20.0, 30.0],
        "pred_ModelA": [11.0, 21.0, np.nan],
        "pred_ModelB": [12.0, np.nan, 32.0],
        "ProductAvailable": [True, True, True],
    })
    pred_cols = {"ModelA": "pred_ModelA", "ModelB": "pred_ModelB"}
    summary = summarize_oof(oof, pred_columns=pred_cols)
    
    common = summary[summary["comparison_population"] == "common"]
    assert (common["n_scored"] == 1).all()
    
    specific = summary[summary["comparison_population"] == "model_specific"]
    assert specific[specific["model"] == "ModelA"]["n_scored"].iloc[0] == 2
    assert specific[specific["model"] == "ModelB"]["n_scored"].iloc[0] == 2

def test_model_specific_coverage():
    """9. Model-specific coverage reports missing predictions correctly."""
    oof = pd.DataFrame({
        "origin": [1, 1, 1, 1],
        "actual": [10.0, 20.0, 30.0, 40.0],
        "pred_ModelA": [11.0, 21.0, 31.0, np.nan],
        "ProductAvailable": [True, True, True, True],
    })
    pred_cols = {"ModelA": "pred_ModelA"}
    summary = summarize_oof(oof, pred_columns=pred_cols)
    
    row = summary[summary["model"] == "ModelA"].iloc[0]
    assert row["n_expected"] == 4
    assert row["n_predicted"] == 3
    assert row["coverage"] == 0.75

def test_direct_panel_safety():
    """10. Unsafe direct horizons raise ValueError. 11. Duplicate panel keys raise ValueError."""
    cfg = Config()
    train_feat = pd.DataFrame({"ProductId": [1, 1], "DateKey": pd.to_datetime(["2024-01-01", "2024-01-01"])})
    
    # Duplicate keys
    with pytest.raises(ValueError, match="train_feat contains duplicate"):
        build_direct_panel(train_feat, [1], cfg)
        
    train_feat = pd.DataFrame({"ProductId": [1, 2], "DateKey": pd.to_datetime(["2024-01-01", "2024-01-01"])})
    # Non-positive horizon
    with pytest.raises(ValueError, match="positive"):
        build_direct_panel(train_feat, [0], cfg)
        
    # Too large horizon vs seasonal lags
    with pytest.raises(ValueError, match="future observations"):
        build_direct_panel(train_feat, [100], cfg)

    # Too large horizon vs config
    cfg_short = Config(horizon=3)
    with pytest.raises(ValueError, match="Config.horizon"):
        build_direct_panel(train_feat, [4], cfg_short)

@pytest.mark.integration
def test_tree_worker_full_smoke(tmp_path):
    """12. Full tree-worker smoke test still returns XGBoost, LightGBM, and DynamicRidge."""
    from ml.framework import prepare_features, add_train_lags, direct_panel_feature_names
    cfg = Config()
    
    # Tiny data that produces at least one trainable row and one eval row
    df = pd.DataFrame({
        "ProductId": [1]*20,
        "DateKey": pd.to_datetime([f"2024-01-{i:02d}" for i in range(1, 21)]),
        "Quantity": [1.0]*20,
        "PriceLocalVat": [100.0]*20,
        "ProductAvailable": [True]*20,
        "CampaignSubTypeWeb": [None]*20,
        "CampaignSubTypeApp": [None]*20,
        "DiscountValueWebRelative": [0.0]*20,
        "DiscountValueAppRelative": [0.0]*20,
        "IsSaleOrPromo": [False]*20,
    })
    
    price_ref = df.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = df.groupby("ProductId")["DateKey"].min()
    feat = prepare_features(df, price_ref, first_seen)
    feat = add_train_lags(feat, cfg.lag_windows)
    
    # Origin at day 10, predict 2 days ahead
    origin = pd.to_datetime("2024-01-10")
    panel = build_direct_panel(feat, [1, 2], cfg)
    
    train_panel = panel[panel["TargetDateKey"] <= origin].copy()
    eval_panel = panel[panel["OriginDateKey"] == origin].copy()
    
    # Trees need target_baseline
    train_panel["target_baseline"] = 1.0
    eval_panel["target_baseline"] = 1.0
    
    models = ("XGBoost", "LightGBM", "DynamicRidge")
    results = run_tree_baselines(train_panel, eval_panel, cfg, models=models)
    
    assert set(results.keys()) == set(models)
    for name, preds in results.items():
        assert len(preds) == len(eval_panel)
        assert np.all(np.isfinite(preds))
        assert np.all(preds >= 0.0)
        # Check that we didn't just get zeros
        assert np.allclose(preds, 1.0, atol=0.1)


def test_select_primary_summary_logic():
    """13. select_primary_summary filters correctly and handles errors."""
    df = pd.DataFrame([
        {"model": "M1", "evaluation_regime": "conditional", "comparison_population": "common", "aggregation": "global", "MAE": 1.0},
        {"model": "M1", "evaluation_regime": "realized", "comparison_population": "common", "aggregation": "global", "MAE": 1.1},
        {"model": "M2", "evaluation_regime": "conditional", "comparison_population": "common", "aggregation": "global", "MAE": 2.0},
    ])
    
    selected = select_primary_summary(df)
    assert len(selected) == 2
    assert set(selected["model"]) == {"M1", "M2"}
    
    # Raises when empty
    with pytest.raises(RuntimeError, match="empty"):
        select_primary_summary(df, evaluation_regime="nonexistent")
        
    # Raises when duplicates
    df_dup = pd.concat([df, df[df["model"] == "M1"].iloc[:1]])
    with pytest.raises(RuntimeError, match="duplicate"):
        select_primary_summary(df_dup)


def test_skill_calculation_with_primary_summary():
    """14. Skill calculation completes without a pandas Series."""
    df = pd.DataFrame([
        {"model": "NeuralNet", "evaluation_regime": "conditional", "comparison_population": "common", "aggregation": "global", "MAE": 0.8},
        {"model": "SeasonalNaive", "evaluation_regime": "conditional", "comparison_population": "common", "aggregation": "global", "MAE": 1.0},
        {"model": "NeuralNet", "evaluation_regime": "realized", "comparison_population": "common", "aggregation": "global", "MAE": 0.9},
    ])
    
    primary = select_primary_summary(df).set_index("model")
    nn_mae = float(primary.loc["NeuralNet", "MAE"])
    naive_mae = float(primary.loc["SeasonalNaive", "MAE"])
    skill = 1.0 - nn_mae / naive_mae
    assert isinstance(skill, float)
    assert skill == pytest.approx(0.2)


def test_dynamic_ridge_converts_infinite_numeric_features_to_missing():
    cfg = Config()
    df = pd.DataFrame({
        "ProductId": [1, 1],
        "DateKey": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "target": [10.0, 11.0],
        "target_baseline": [10.0, 10.0],
        "product_idx": [0, 0],
        "campaign_idx_web": [0, 0],
        "campaign_idx_app": [0, 0],
        "horizon": [1, 1],
    })
    df = _add_required_cols(df, cfg)
    model = train_dynamic_ridge(df, cfg)

    test_df = df.iloc[[0]].copy()
    test_df["qty_lag_0"] = np.inf
    preds = predict_dynamic_ridge(model, test_df, cfg)
    assert np.isfinite(preds).all()
