"""Standalone subprocess worker/dispatcher for the XGBoost/LightGBM models.

Why a separate process: on macOS, PyTorch and XGBoost/LightGBM each bundle
their own copy of the LLVM OpenMP runtime (libomp.dylib). Loading both into
the same process crashes the interpreter (segfault) as soon as either one
actually runs its native training code. This module has NO torch import (and
must never gain one), so running it as a child process keeps it fully
isolated from `pipeline.py`'s torch usage -- the two runtimes never share a
process. Each model's actual train/predict definition (direct on the
multi-horizon panel, no recursion) lives in its own file under `models/`
(`models/xgboost_model.py`, `models/lightgbm_model.py`); this module just
dispatches to them.

Protocol: the caller pickles a job dict (see `run_job`) to a temp file,
invokes this script with `<job_path> <output_path>`, and reads back a
pickled `{model_name: predictions}` dict from `<output_path>`.

CLI: python ml/tree_worker.py <job_pickle_path> <output_pickle_path>
"""

from __future__ import annotations

import pickle
import sys

from framework import Config
from models.dynamic_ridge import predict_dynamic_ridge, train_dynamic_ridge
from models.lightgbm_model import predict_lightgbm, train_lightgbm
from models.xgboost_model import predict_xgboost, train_xgboost

TRAINERS = {
    "XGBoost": train_xgboost,
    "LightGBM": train_lightgbm,
    "DynamicRidge": train_dynamic_ridge,
}
PREDICTORS = {
    "XGBoost": predict_xgboost,
    "LightGBM": predict_lightgbm,
    "DynamicRidge": predict_dynamic_ridge,
}


def run_job(job: dict) -> dict:
    """job = {
        "cfg": {... Config field overrides ...},
        "train_panel": pd.DataFrame,   # build_direct_panel() training rows
        "eval_panel": pd.DataFrame,    # build_direct_panel() rows to predict for
        "models": ["XGBoost", "LightGBM"],   # optional, defaults to both
    }
    Returns {model_name: [predictions aligned with eval_panel row order]}.
    No recursive state needed -- every horizon's features in `eval_panel`
    are already lookups into observed data (see `framework.build_direct_panel`),
    so one `.predict()` call per model covers every horizon at once.
    """
    cfg = Config(**job["cfg"])
    train_panel = job["train_panel"]
    eval_panel = job["eval_panel"]

    results = {}
    for name in job.get("models", list(TRAINERS)):
        model = TRAINERS[name](train_panel, cfg)
        preds = PREDICTORS[name](model, eval_panel, cfg)
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
