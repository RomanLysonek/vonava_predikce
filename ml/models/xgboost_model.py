"""XGBoost model: gradient-boosted trees (dmlc/xgboost).

The task brief's own "standard approach" baseline -- evaluated for an
honest comparison against the NN, not used for the final submission. Only
ever imported by `tree_worker.py`'s subprocess (never alongside torch --
see that module's docstring for why).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from framework import CFG, Config, recursive_forecast_generic, tree_feature_frame


def train_xgboost(train_examples: pd.DataFrame, cfg: Config = CFG):
    from xgboost import XGBRegressor

    X = tree_feature_frame(train_examples, cfg)
    y = np.log1p(train_examples["Quantity"].to_numpy(dtype=np.float32))
    model = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        tree_method="hist", enable_categorical=True,
        random_state=cfg.seed, verbosity=0,
    )
    model.fit(X, y)
    return model


def predict_xgboost(model, day_df: pd.DataFrame, cfg: Config = CFG) -> np.ndarray:
    X = tree_feature_frame(day_df, cfg)
    pred_log = model.predict(X)
    return np.clip(np.expm1(pred_log), 0, None)


def recursive_forecast_xgboost(model, static_df: pd.DataFrame, history: dict,
                                cfg: Config = CFG) -> np.ndarray:
    def predict_fn(day_df: pd.DataFrame) -> np.ndarray:
        return predict_xgboost(model, day_df, cfg)

    return recursive_forecast_generic(predict_fn, static_df, history, cfg)
