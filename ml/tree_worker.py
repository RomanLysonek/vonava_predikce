"""Standalone subprocess worker for the XGBoost/LightGBM baselines.

Why a separate process: on macOS, PyTorch and XGBoost/LightGBM each bundle
their own copy of the LLVM OpenMP runtime (libomp.dylib). Loading both into
the same process crashes the interpreter (segfault) as soon as either one
actually runs its native training code. This module has NO torch import, so
running it as a child process keeps it fully isolated from
`solution_final.py`'s torch usage -- the two runtimes never share a process.

Protocol: the caller pickles a job dict (see `run_job`) to a temp file,
invokes this script with `<job_path> <output_path>`, and reads back a
pickled `{model_name: predictions}` dict from `<output_path>`.

CLI: python ml/tree_worker.py <job_pickle_path> <output_pickle_path>
"""

from __future__ import annotations

import pickle
import sys

import numpy as np
import pandas as pd

from features import Config, recursive_forecast_generic, tree_feature_frame


def train_xgboost(train_examples: pd.DataFrame, cfg: Config):
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


def train_lightgbm(train_examples: pd.DataFrame, cfg: Config):
    from lightgbm import LGBMRegressor

    X = tree_feature_frame(train_examples, cfg)
    y = np.log1p(train_examples["Quantity"].to_numpy(dtype=np.float32))
    model = LGBMRegressor(
        n_estimators=400, num_leaves=31, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        random_state=cfg.seed, verbosity=-1,
    )
    model.fit(X, y)
    return model


def predict_tree(model, day_df: pd.DataFrame, cfg: Config) -> np.ndarray:
    X = tree_feature_frame(day_df, cfg)
    pred_log = model.predict(X)
    return np.clip(np.expm1(pred_log), 0, None)


TRAINERS = {"XGBoost": train_xgboost, "LightGBM": train_lightgbm}


def run_job(job: dict) -> dict:
    """job = {
        "cfg": {... Config field overrides ...},
        "train_examples": pd.DataFrame,
        "static_eval": pd.DataFrame,
        "history": {product_id: [floats]},   # shared *starting* history per model
        "models": ["XGBoost", "LightGBM"],   # optional, defaults to both
    }
    Returns {model_name: [predictions aligned with static_eval row order]}.
    """
    cfg = Config(**job["cfg"])
    train_examples = job["train_examples"]
    static_eval = job["static_eval"]
    history_seed = job["history"]

    results = {}
    for name in job.get("models", list(TRAINERS)):
        trainer = TRAINERS[name]
        model = trainer(train_examples, cfg)
        # Fresh copy per model: recursion mutates history in place with that
        # model's own predictions, so sharing one dict across models would
        # leak one model's forecasts into another's lag features.
        history = {k: list(v) for k, v in history_seed.items()}
        preds = recursive_forecast_generic(lambda d: predict_tree(model, d, cfg), static_eval, history, cfg)
        results[name] = preds.tolist()
    return results


def main() -> None:
    job_path, out_path = sys.argv[1], sys.argv[2]
    with open(job_path, "rb") as f:
        job = pickle.load(f)
    results = run_job(job)
    with open(out_path, "wb") as f:
        pickle.dump(results, f)


if __name__ == "__main__":
    main()
