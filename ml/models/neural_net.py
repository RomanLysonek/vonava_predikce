"""Neural-network model and accelerator-aware training utilities.

The network predicts a baseline-relative log residual for either the direct
multi-horizon panel or the one-step recursive panel.  On MPS/CUDA, training
uses device-resident tensors by default: the complete fold is copied to the
accelerator once, then shuffled/indexed on device.  This avoids thousands of
small host-to-device copies without changing the model, target, epoch count,
or optimizer-step semantics.

Batch size remains a model hyperparameter.  Larger batches can improve device
utilisation but reduce the number of optimizer updates per epoch and can alter
generalisation.  ``ml/benchmark_nn_batch_size.py`` therefore measures both
throughput and held-out WAPE before a larger batch is adopted.
"""

from __future__ import annotations

import math
import time
import warnings
from collections.abc import Iterator

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from framework import CFG, Config, NUM_CAMPAIGN_CATS, direct_panel_feature_names

torch.manual_seed(CFG.seed)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")


def numeric_feature_columns(cfg: Config = CFG) -> list[str]:
    """Direct-panel numeric columns excluding embedded ``horizon``."""
    return [c for c in direct_panel_feature_names(cfg) if c != "horizon"]


def make_numeric_preprocessor() -> Pipeline:
    """Training-fitted numeric imputation plus missingness indicators.

    Annual seasonal lags are legitimately unavailable for young products and
    early history.  Keeping those rows is preferable to silently deleting
    them; the fitted median and indicator columns make the missing state
    explicit while preserving train/eval consistency.
    """
    return Pipeline([
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        ),
        ("scaler", StandardScaler()),
    ])


class QuantityNet(nn.Module):
    def __init__(self, num_numeric: int, cfg: Config):
        super().__init__()
        self.product_emb = nn.Embedding(cfg.num_products, cfg.embed_dim_product)
        self.campaign_emb_web = nn.Embedding(NUM_CAMPAIGN_CATS, cfg.embed_dim_campaign)
        self.campaign_emb_app = nn.Embedding(NUM_CAMPAIGN_CATS, cfg.embed_dim_campaign)
        self.horizon_emb = nn.Embedding(cfg.horizon, cfg.embed_dim_horizon)

        input_dim = (
            num_numeric
            + cfg.embed_dim_product
            + 2 * cfg.embed_dim_campaign
            + cfg.embed_dim_horizon
        )
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden, p in zip(cfg.hidden_dims, cfg.dropout):
            layers += [
                nn.Linear(prev, hidden),
                nn.BatchNorm1d(hidden),
                nn.GELU(),
                nn.Dropout(p),
            ]
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_prod, x_camp_web, x_camp_app, x_horizon):
        emb = torch.cat([
            self.product_emb(x_prod),
            self.campaign_emb_web(x_camp_web),
            self.campaign_emb_app(x_camp_app),
            self.horizon_emb(x_horizon),
        ], dim=1)
        return self.net(torch.cat([x_num, emb], dim=1)).squeeze(-1)


def make_tensors(
    df: pd.DataFrame,
    scaler,
    fit: bool,
    cfg: Config = CFG,
) -> dict[str, torch.Tensor]:
    num = df[numeric_feature_columns(cfg)].replace(
        [np.inf, -np.inf], np.nan
    ).to_numpy(dtype=np.float32)
    num = scaler.fit_transform(num) if fit else scaler.transform(num)
    if not np.isfinite(num).all():
        raise ValueError("Numeric preprocessing produced non-finite NN inputs")
    return {
        "num": torch.tensor(num, dtype=torch.float32),
        "prod": torch.tensor(df["product_idx"].to_numpy(dtype=np.int64)),
        "cw": torch.tensor(df["campaign_idx_web"].to_numpy(dtype=np.int64)),
        "ca": torch.tensor(df["campaign_idx_app"].to_numpy(dtype=np.int64)),
        "horizon": torch.tensor(df["horizon"].to_numpy(dtype=np.int64) - 1),
        "baseline_log1p": torch.tensor(
            np.log1p(df["target_baseline"].to_numpy(dtype=np.float32)),
            dtype=torch.float32,
        ),
    }


def residual_log1p_target(df: pd.DataFrame) -> np.ndarray:
    """Return ``log1p(actual) - log1p(target_baseline)``."""
    return (
        np.log1p(df["target"].to_numpy(dtype=np.float32))
        - np.log1p(df["target_baseline"].to_numpy(dtype=np.float32))
    )


def nn_performance_signature(cfg: Config) -> dict:
    """Configuration fingerprint for batch-benchmark reuse."""
    return {
        "numeric_features": tuple(numeric_feature_columns(cfg)),
        "hidden_dims": tuple(cfg.hidden_dims),
        "dropout": tuple(cfg.dropout),
        "embed_dim_product": cfg.embed_dim_product,
        "embed_dim_campaign": cfg.embed_dim_campaign,
        "embed_dim_horizon": cfg.embed_dim_horizon,
        "horizon": cfg.horizon,
        "base_learning_rate": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "reference_batch_size": cfg.reference_batch_size,
        "loss": "HuberLoss(delta=1.0)",
        "target": "baseline_relative_log1p_residual",
        "numeric_preprocessing": "median_impute+missing_indicator+standardize",
    }


def effective_learning_rate(cfg: Config) -> float:
    """Scale LR relative to ``reference_batch_size``.

    ``fixed`` preserves the historical optimiser exactly. ``sqrt`` is the
    conservative larger-batch policy used by the benchmark. ``linear`` is
    available for experimentation but is intentionally not the default.
    """
    if cfg.batch_size <= 0 or cfg.reference_batch_size <= 0:
        raise ValueError("batch sizes must be positive")
    ratio = cfg.batch_size / cfg.reference_batch_size
    if cfg.nn_lr_scaling == "fixed":
        factor = 1.0
    elif cfg.nn_lr_scaling == "sqrt":
        factor = math.sqrt(ratio)
    elif cfg.nn_lr_scaling == "linear":
        factor = ratio
    else:
        raise ValueError(
            "nn_lr_scaling must be one of: fixed, sqrt, linear"
        )
    return float(cfg.lr * factor)


def resolve_training_backend(cfg: Config, device: torch.device = DEVICE) -> str:
    if cfg.nn_training_backend == "auto":
        return "device_resident" if device.type in {"mps", "cuda"} else "dataloader"
    if cfg.nn_training_backend not in {"device_resident", "dataloader"}:
        raise ValueError(
            "nn_training_backend must be one of: auto, device_resident, dataloader"
        )
    return cfg.nn_training_backend


def training_tensor_bytes(tensors: dict[str, torch.Tensor], n_targets: int) -> int:
    """Approximate bytes required to keep one training fold on-device."""
    keys = ("num", "prod", "cw", "ca", "horizon")
    total = sum(tensors[k].numel() * tensors[k].element_size() for k in keys)
    return int(total + n_targets * torch.tensor([], dtype=torch.float32).element_size())


def _synchronize(device: torch.device = DEVICE) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def _batch_ranges(n_rows: int, batch_size: int) -> Iterator[tuple[int, int]]:
    """Yield batches while avoiding a final singleton (BatchNorm invalid)."""
    if n_rows < 2:
        raise ValueError("Neural-network training requires at least two rows")
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 for BatchNorm")
    starts = list(range(0, n_rows, batch_size))
    if len(starts) > 1 and n_rows - starts[-1] == 1:
        starts[-1] -= 1
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else n_rows
        yield start, end


def _device_permutation(n_rows: int, device: torch.device) -> torch.Tensor:
    try:
        return torch.randperm(n_rows, device=device)
    except (RuntimeError, NotImplementedError):
        # Some MPS/PyTorch combinations do not implement every random op.
        return torch.randperm(n_rows, device="cpu").to(device)


def _is_oom(exc: RuntimeError) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "mps backend out of memory" in text


def _train_device_resident(
    model: QuantityNet,
    tensors: dict[str, torch.Tensor],
    y_residual: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    cfg: Config,
    epochs: int,
) -> tuple[float, int]:
    x_num = tensors["num"].to(DEVICE)
    x_prod = tensors["prod"].to(DEVICE)
    x_cw = tensors["cw"].to(DEVICE)
    x_ca = tensors["ca"].to(DEVICE)
    x_horizon = tensors["horizon"].to(DEVICE)
    y = torch.as_tensor(y_residual, dtype=torch.float32, device=DEVICE)
    n_rows = len(y)
    optimizer_steps = 0
    final_loss = float("nan")

    for epoch in range(1, epochs + 1):
        total = 0.0
        permutation = _device_permutation(n_rows, DEVICE)
        for start, end in _batch_ranges(n_rows, cfg.batch_size):
            idx = permutation[start:end]
            pred = model(
                x_num[idx], x_prod[idx], x_cw[idx], x_ca[idx], x_horizon[idx]
            )
            loss = criterion(pred, y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total += loss.detach().item() * (end - start)
        scheduler.step()
        final_loss = total / n_rows
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"      epoch {epoch:3d}/{epochs} | train loss {final_loss:.4f}")
    return final_loss, optimizer_steps


def _train_dataloader(
    model: QuantityNet,
    tensors: dict[str, torch.Tensor],
    y_residual: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion: nn.Module,
    cfg: Config,
    epochs: int,
) -> tuple[float, int]:
    ds = TensorDataset(
        tensors["num"], tensors["prod"], tensors["cw"], tensors["ca"],
        tensors["horizon"], torch.tensor(y_residual, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    optimizer_steps = 0
    final_loss = float("nan")
    for epoch in range(1, epochs + 1):
        total = 0.0
        seen = 0
        for xn, xp, xcw, xca, xh, yt in dl:
            # Avoid a singleton BatchNorm batch without globally dropping data.
            if len(yt) == 1:
                continue
            xn, xp, xcw, xca, xh, yt = (
                xn.to(DEVICE), xp.to(DEVICE), xcw.to(DEVICE),
                xca.to(DEVICE), xh.to(DEVICE), yt.to(DEVICE),
            )
            pred = model(xn, xp, xcw, xca, xh)
            loss = criterion(pred, yt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            total += loss.detach().item() * len(yt)
            seen += len(yt)
        scheduler.step()
        final_loss = total / max(seen, 1)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"      epoch {epoch:3d}/{epochs} | train loss {final_loss:.4f}")
    return final_loss, optimizer_steps


def train_model(
    tensors: dict,
    y_residual: np.ndarray,
    cfg: Config,
    epochs: int,
    seed: int,
    *,
    stats_out: dict | None = None,
) -> QuantityNet:
    """Train one seed with a fixed epoch budget.

    ``stats_out`` is an optional mutable dict used by the batch benchmark; it
    does not alter the return type used by the production pipeline.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    elif DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    model = QuantityNet(int(tensors["num"].shape[1]), cfg).to(DEVICE)
    learning_rate = effective_learning_rate(cfg)
    opt = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=learning_rate * 0.01
    )
    crit = nn.HuberLoss(delta=1.0)
    backend = resolve_training_backend(cfg)
    started = time.perf_counter()
    _synchronize()

    model.train()
    try:
        if backend == "device_resident":
            final_loss, optimizer_steps = _train_device_resident(
                model, tensors, y_residual, opt, sched, crit, cfg, epochs
            )
        else:
            final_loss, optimizer_steps = _train_dataloader(
                model, tensors, y_residual, opt, sched, crit, cfg, epochs
            )
    except RuntimeError as exc:
        if backend == "device_resident" and cfg.nn_training_backend == "auto" and _is_oom(exc):
            warnings.warn(
                "Device-resident training ran out of accelerator memory; "
                "falling back to the DataLoader backend.",
                RuntimeWarning,
            )
            if DEVICE.type == "mps":
                torch.mps.empty_cache()
            elif DEVICE.type == "cuda":
                torch.cuda.empty_cache()
            torch.manual_seed(seed)
            np.random.seed(seed)
            model = QuantityNet(int(tensors["num"].shape[1]), cfg).to(DEVICE)
            opt = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay
            )
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=epochs, eta_min=learning_rate * 0.01
            )
            final_loss, optimizer_steps = _train_dataloader(
                model, tensors, y_residual, opt, sched, crit, cfg, epochs
            )
            backend = "dataloader_fallback"
        else:
            raise

    _synchronize()
    elapsed = time.perf_counter() - started
    model.eval()
    if stats_out is not None:
        n_rows = len(y_residual)
        stats_out.update({
            "device": DEVICE.type,
            "backend": backend,
            "batch_size": int(cfg.batch_size),
            "reference_batch_size": int(cfg.reference_batch_size),
            "lr_scaling": cfg.nn_lr_scaling,
            "effective_learning_rate": learning_rate,
            "epochs": int(epochs),
            "rows": int(n_rows),
            "optimizer_steps": int(optimizer_steps),
            "elapsed_seconds": float(elapsed),
            "examples_per_second": float(n_rows * epochs / max(elapsed, 1e-9)),
            "final_train_loss": float(final_loss),
            "estimated_device_tensor_mb": training_tensor_bytes(
                tensors, n_rows
            ) / (1024 ** 2),
        })
    return model


def predict_ensemble(models: list, tensors: dict) -> np.ndarray:
    """Reconstruct quantity and average seed predictions in natural scale."""
    baseline_log1p = tensors["baseline_log1p"].cpu().numpy()
    # Transfer each input once, not once per seed.
    x_num = tensors["num"].to(DEVICE)
    x_prod = tensors["prod"].to(DEVICE)
    x_cw = tensors["cw"].to(DEVICE)
    x_ca = tensors["ca"].to(DEVICE)
    x_horizon = tensors["horizon"].to(DEVICE)
    preds = []
    for model in models:
        model.eval()
        with torch.inference_mode():
            residual = model(x_num, x_prod, x_cw, x_ca, x_horizon).cpu().numpy()
        preds.append(np.expm1(residual + baseline_log1p))
    return np.clip(np.mean(preds, axis=0), 0, None)


def predict_direct(
    models: list,
    scaler,
    panel: pd.DataFrame,
    cfg: Config = CFG,
) -> np.ndarray:
    tensors = make_tensors(panel, scaler, fit=False, cfg=cfg)
    return predict_ensemble(models, tensors)
