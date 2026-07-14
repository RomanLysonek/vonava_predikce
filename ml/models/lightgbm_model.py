"""LightGBM model: gradient-boosted trees with leaf-wise growth (Microsoft).

Trained directly on the multi-horizon panel (`framework.build_direct_panel`)
-- `horizon` is just another feature, so one `.predict()` call covers every
horizon at once, no recursion.

Same role as XGBoost: a second "standard approach" baseline for comparison,
not the submission. Only ever imported by `tree_worker.py`'s subprocess
(never alongside torch -- see that module's docstring for why).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from framework import CFG, Config, direct_panel_tree_frame


def train_lightgbm(train_panel: pd.DataFrame, cfg: Config = CFG):
    from lightgbm import LGBMRegressor

    X = direct_panel_tree_frame(train_panel, cfg)
    y = np.log1p(train_panel["target"].to_numpy(dtype=np.float32))
    model = LGBMRegressor(
        n_estimators=400, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        random_state=cfg.seed, verbosity=-1,
    )
    sample_weight = train_panel.get(
        "sample_weight", pd.Series(1.0, index=train_panel.index)
    ).to_numpy(dtype=float)
    model.fit(X, y, sample_weight=sample_weight)
    return model


def predict_lightgbm(model, panel: pd.DataFrame, cfg: Config = CFG) -> np.ndarray:
    X = direct_panel_tree_frame(panel, cfg)
    pred_log = model.predict(X)
    return np.clip(np.expm1(pred_log), 0, None)
