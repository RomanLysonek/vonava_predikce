"""Standalone subprocess worker/dispatcher for the XGBoost/LightGBM models.

Why a separate process: on macOS, PyTorch and XGBoost/LightGBM each bundle
their own copy of the LLVM OpenMP runtime (libomp.dylib). Loading both into
the same process crashes the interpreter (segfault) as soon as either one
actually runs its native training code. This module has NO torch import (and
must never gain one), so running it as a child process keeps it fully
isolated from `pipeline.py`'s torch usage -- the two runtimes never share a
process. Each model's actual train/predict/recursive-forecast definition
lives in its own file under `models/` (`models/xgboost_model.py`,
`models/lightgbm_model.py`); this module just dispatches to them.

Protocol: the caller pickles a job dict (see `run_job`) to a temp file,
invokes this script with `<job_path> <output_path>`, and reads back a
pickled `{model_name: predictions}` dict from `<output_path>`.

CLI: python ml/tree_worker.py <job_pickle_path> <output_pickle_path>
"""

from __future__ import annotations

import pickle
import sys

from framework import Config
from models.lightgbm_model import recursive_forecast_lightgbm, train_lightgbm
from models.xgboost_model import recursive_forecast_xgboost, train_xgboost

TRAINERS = {"XGBoost": train_xgboost, "LightGBM": train_lightgbm}
RECURSIVE_FORECASTERS = {"XGBoost": recursive_forecast_xgboost, "LightGBM": recursive_forecast_lightgbm}


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
        model = TRAINERS[name](train_examples, cfg)
        # Fresh copy per model: recursion mutates history in place with that
        # model's own predictions, so sharing one dict across models would
        # leak one model's forecasts into another's lag features.
        history = {k: list(v) for k, v in history_seed.items()}
        preds = RECURSIVE_FORECASTERS[name](model, static_eval, history, cfg)
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
