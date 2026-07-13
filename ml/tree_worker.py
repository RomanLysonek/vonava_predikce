"""Native-model subprocess worker for direct and recursive forecasting.

This module intentionally never imports torch. XGBoost/LightGBM and PyTorch
remain in separate processes to avoid incompatible OpenMP runtimes on macOS.
"""
from __future__ import annotations

import pickle
import sys

from framework import Config, forecast_recursive
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
    cfg = Config(**job["cfg"])
    strategy = job.get("strategy", "direct")
    train_panel = job["train_panel"]
    results = {}
    for name in job.get("models", list(TRAINERS)):
        model = TRAINERS[name](train_panel, cfg)
        if strategy == "direct":
            preds = PREDICTORS[name](model, job["eval_panel"], cfg)
            results[name] = preds.tolist()
        elif strategy == "recursive":
            def predict_recursive_step(panel, *, model=model, name=name):
                if name == "DynamicRidge":
                    return predict_dynamic_ridge(model, panel, cfg, recursive=True)
                return PREDICTORS[name](model, panel, cfg)

            recursive = forecast_recursive(
                history_raw=job["history_raw"].copy(),
                future_covariates=job["future_covariates"],
                predict_step=predict_recursive_step,
                price_ref=job["price_ref"],
                first_seen=job["first_seen"],
                cfg=cfg,
            )
            results[name] = recursive.to_dict(orient="list")
        else:
            raise ValueError(f"Unsupported forecast strategy: {strategy}")
    return results


def main() -> None:
    job_path, out_path = sys.argv[1], sys.argv[2]
    with open(job_path, "rb") as f:
        job = pickle.load(f)
    with open(out_path, "wb") as f:
        pickle.dump(run_job(job), f)


if __name__ == "__main__":
    main()
