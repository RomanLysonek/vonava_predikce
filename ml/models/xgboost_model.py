"""XGBoost model: gradient-boosted trees (dmlc/xgboost).

Trained directly on the multi-horizon panel (`framework.build_direct_panel`)
-- `horizon` is just another feature, so one `.predict()` call covers every
horizon at once, no recursion.

The task brief's own "standard approach" baseline -- evaluated for an
honest comparison against the NN, not used for the final submission. Only
ever imported by `tree_worker.py`'s subprocess (never alongside torch --
see that module's docstring for why).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from framework import CFG, Config, direct_panel_tree_frame


def train_xgboost(train_panel: pd.DataFrame, cfg: Config = CFG):
    from xgboost import XGBRegressor

    X = direct_panel_tree_frame(train_panel, cfg)
    y = np.log1p(train_panel["target"].to_numpy(dtype=np.float32))
    model = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        tree_method="hist", enable_categorical=True,
        random_state=cfg.seed, verbosity=0,
    )
    model.fit(X, y)
    return model


def predict_xgboost(model, panel: pd.DataFrame, cfg: Config = CFG) -> np.ndarray:
    X = direct_panel_tree_frame(panel, cfg)
    pred_log = model.predict(X)
    return np.clip(np.expm1(pred_log), 0, None)
