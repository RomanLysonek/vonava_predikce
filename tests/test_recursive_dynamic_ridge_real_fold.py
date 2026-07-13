"""Regression test for the recursive Dynamic Ridge overflow found in a real CV fold."""

import numpy as np
import pandas as pd
import pytest

from ml.framework import Config, load_raw, sanitize_future_covariates
from ml.pipeline import _recursive_panel_training_data, run_structured_models


@pytest.mark.integration
def test_recursive_dynamic_ridge_real_2024_11_29_fold_is_finite():
    cfg = Config()
    train_raw, _ = load_raw(cfg)
    cfg.num_products = int(train_raw["ProductId"].max())
    origin = pd.Timestamp("2024-11-29")
    fold_train = train_raw[train_raw["DateKey"].le(origin)].copy()
    fold_eval = train_raw[
        train_raw["DateKey"].between(
            origin + pd.Timedelta(days=1),
            origin + pd.Timedelta(days=cfg.horizon),
        )
    ].copy()
    price_ref = fold_train.groupby("ProductId")["PriceLocalVat"].median()
    first_seen = train_raw.groupby("ProductId")["DateKey"].min()
    train_panel = _recursive_panel_training_data(
        fold_train, price_ref, first_seen, cfg
    )

    result = run_structured_models(
        train_panel,
        cfg,
        models=("DynamicRidge",),
        strategy="recursive",
        history_raw=fold_train,
        future_covariates=sanitize_future_covariates(fold_eval),
        price_ref=price_ref,
        first_seen=first_seen,
    )
    path = pd.DataFrame(result["DynamicRidge"])

    assert len(path) == cfg.num_products * cfg.horizon
    assert np.isfinite(path["prediction"]).all()
    assert (path["prediction"] >= 0.0).all()
