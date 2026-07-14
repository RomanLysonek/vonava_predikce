"""LightGBM direct/recursive structured model with C3 target alternatives."""

from __future__ import annotations

import numpy as np
import pandas as pd

from framework import CFG, Config, direct_panel_tree_frame


def _target(train_panel: pd.DataFrame, mode: str) -> np.ndarray:
    quantity = train_panel["target"].to_numpy(dtype=np.float32)
    if mode == "log1p":
        return np.log1p(quantity)
    if mode == "residual":
        baseline = train_panel["target_baseline"].to_numpy(dtype=np.float32)
        return np.log1p(quantity) - np.log1p(baseline)
    if mode == "tweedie":
        return np.clip(quantity, 0.0, None)
    raise ValueError("tree_target_mode must be one of: log1p, residual, tweedie")


def train_lightgbm(train_panel: pd.DataFrame, cfg: Config = CFG):
    from lightgbm import LGBMRegressor

    mode = cfg.lightgbm_target_mode or cfg.tree_target_mode
    X = direct_panel_tree_frame(train_panel, cfg)
    y = _target(train_panel, mode)
    params = dict(
        n_estimators=400,
        num_leaves=31,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_samples=10,
        random_state=cfg.seed,
        verbosity=-1,
    )
    if mode == "tweedie":
        params.update(
            objective="tweedie",
            tweedie_variance_power=float(cfg.tree_tweedie_variance_power),
        )
    else:
        params["objective"] = "regression"
    model = LGBMRegressor(**params)
    sample_weight = train_panel.get(
        "sample_weight", pd.Series(1.0, index=train_panel.index)
    ).to_numpy(dtype=float)
    model.fit(X, y, sample_weight=sample_weight)
    return {"estimator": model, "target_mode": mode}


def predict_lightgbm(model, panel: pd.DataFrame, cfg: Config = CFG) -> np.ndarray:
    X = direct_panel_tree_frame(panel, cfg)
    if isinstance(model, dict):
        estimator = model["estimator"]
        mode = model.get("target_mode", "log1p")
    else:
        estimator = model
        mode = "log1p"
    raw = np.asarray(estimator.predict(X), dtype=float)
    if mode == "residual":
        baseline = panel["target_baseline"].to_numpy(dtype=float)
        return np.clip(np.expm1(raw + np.log1p(baseline)), 0.0, None)
    if mode == "log1p":
        return np.clip(np.expm1(raw), 0.0, None)
    if mode == "tweedie":
        return np.clip(raw, 0.0, None)
    raise ValueError(f"Unsupported LightGBM target mode: {mode}")
