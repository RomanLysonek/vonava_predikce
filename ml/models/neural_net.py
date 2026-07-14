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
    """Shared representation with a total-demand head and optional app-share head."""

    def __init__(self, num_numeric: int, cfg: Config):
        super().__init__()
        self.product_emb = nn.Embedding(cfg.num_products, cfg.embed_dim_product)
        self.campaign_emb_web = nn.Embedding(NUM_CAMPAIGN_CATS, cfg.embed_dim_campaign)
        self.campaign_emb_app = nn.Embedding(NUM_CAMPAIGN_CATS, cfg.embed_dim_campaign)
        self.horizon_emb = nn.Embedding(cfg.horizon, cfg.embed_dim_horizon)
        self.target_mode = cfg.nn_target_mode
        self.channel_aux_weight = float(cfg.channel_aux_weight)

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
        self.trunk = nn.Sequential(*layers)
        self.total_head = nn.Linear(prev, 1)
        self.share_head = (
            nn.Linear(prev, 1) if self.channel_aux_weight > 0.0 else None
        )

    def _representation(self, x_num, x_prod, x_camp_web, x_camp_app, x_horizon):
        emb = torch.cat([
            self.product_emb(x_prod),
            self.campaign_emb_web(x_camp_web),
            self.campaign_emb_app(x_camp_app),
            self.horizon_emb(x_horizon),
        ], dim=1)
        return self.trunk(torch.cat([x_num, emb], dim=1))

    def forward(self, x_num, x_prod, x_camp_web, x_camp_app, x_horizon):
        representation = self._representation(
            x_num, x_prod, x_camp_web, x_camp_app, x_horizon
        )
        return self.total_head(representation).squeeze(-1)

    def forward_multitask(self, x_num, x_prod, x_camp_web, x_camp_app, x_horizon):
        representation = self._representation(
            x_num, x_prod, x_camp_web, x_camp_app, x_horizon
        )
        total = self.total_head(representation).squeeze(-1)
        share_logit = (
            self.share_head(representation).squeeze(-1)
            if self.share_head is not None else None
        )
        return total, share_logit


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
    app = pd.to_numeric(
        df.get("target_app", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).to_numpy(dtype=np.float32)
    web = pd.to_numeric(
        df.get("target_web", pd.Series(np.nan, index=df.index)), errors="coerce"
    ).to_numpy(dtype=np.float32)
    total = app + web
    share_valid = np.isfinite(app) & np.isfinite(web) & np.isfinite(total) & (total > 0.0)
    smoothing = float(cfg.channel_share_smoothing)
    if not np.isfinite(smoothing) or smoothing < 0.0:
        raise ValueError("channel_share_smoothing must be finite and nonnegative")
    share_target = np.full(len(df), 0.5, dtype=np.float32)
    denominator = total + 2.0 * smoothing
    valid_denominator = share_valid & (denominator > 0.0)
    share_target[valid_denominator] = (
        (app[valid_denominator] + smoothing) / denominator[valid_denominator]
    )
    share_target = np.clip(share_target, 0.0, 1.0)

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
        "sample_weight": torch.tensor(
            df.get("sample_weight", pd.Series(1.0, index=df.index))
            .to_numpy(dtype=np.float32),
            dtype=torch.float32,
        ),
        "share_target": torch.tensor(share_target, dtype=torch.float32),
        "share_valid": torch.tensor(share_valid.astype(np.float32), dtype=torch.float32),
    }


def residual_log1p_target(df: pd.DataFrame) -> np.ndarray:
    """Return ``log1p(actual) - log1p(target_baseline)``."""
    return (
        np.log1p(df["target"].to_numpy(dtype=np.float32))
        - np.log1p(df["target_baseline"].to_numpy(dtype=np.float32))
    )


def neural_training_target(df: pd.DataFrame, cfg: Config = CFG) -> np.ndarray:
    """Return the configured total-demand target in log space."""
    if cfg.nn_target_mode == "residual":
        return residual_log1p_target(df)
    if cfg.nn_target_mode == "log1p":
        return np.log1p(df["target"].to_numpy(dtype=np.float32))
    raise ValueError("nn_target_mode must be one of: residual, log1p")


def total_loss_values(
    prediction: torch.Tensor,
    target: torch.Tensor,
    cfg: Config = CFG,
) -> torch.Tensor:
    """Per-row C3 total-demand loss without reduction."""
    error = prediction - target
    if cfg.nn_loss == "huber":
        return nn.functional.huber_loss(
            prediction, target, delta=float(cfg.nn_huber_delta), reduction="none"
        )
    if cfg.nn_loss == "mse":
        return error.square()
    if cfg.nn_loss == "combined":
        mse_weight = float(cfg.nn_combined_mse_weight)
        if not 0.0 <= mse_weight <= 1.0:
            raise ValueError("nn_combined_mse_weight must be between 0 and 1")
        huber = nn.functional.huber_loss(
            prediction, target, delta=float(cfg.nn_huber_delta), reduction="none"
        )
        return (1.0 - mse_weight) * huber + mse_weight * error.square()
    if cfg.nn_loss == "logcosh":
        # Stable log(cosh(x)) = x + softplus(-2x) - log(2).
        return error + nn.functional.softplus(-2.0 * error) - math.log(2.0)
    raise ValueError(
        "nn_loss must be one of: huber, mse, combined, logcosh"
    )


def multitask_loss_values(
    model: QuantityNet,
    total_prediction: torch.Tensor,
    total_target: torch.Tensor,
    share_logit: torch.Tensor | None,
    share_target: torch.Tensor,
    share_valid: torch.Tensor,
    cfg: Config = CFG,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return total, auxiliary, and combined per-row C3/C4 losses."""
    total = total_loss_values(total_prediction, total_target, cfg)
    if float(cfg.channel_aux_weight) <= 0.0:
        zero = torch.zeros_like(total)
        return total, zero, total
    if share_logit is None:
        raise ValueError("channel_aux_weight > 0 requires a channel share head")
    auxiliary = nn.functional.binary_cross_entropy_with_logits(
        share_logit, share_target, reduction="none"
    ) * share_valid
    combined = total + float(cfg.channel_aux_weight) * auxiliary
    return total, auxiliary, combined


def residual_support_bounds(
    y_residual: np.ndarray,
    cfg: Config = CFG,
) -> tuple[float, float]:
    """Robust support bounds for recursive NN residual extrapolation.

    Bounds are learned from each fold's training target and widened by a full
    log unit by default.  They are intentionally applied only during recursive
    prediction; direct predictions remain uncapped for an honest comparison.
    """
    values = np.asarray(y_residual, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Cannot derive residual support from no finite targets")
    lower_q = float(cfg.nn_residual_guard_lower_quantile)
    upper_q = float(cfg.nn_residual_guard_upper_quantile)
    if not (0.0 <= lower_q < upper_q <= 1.0):
        raise ValueError("NN residual guard quantiles must satisfy 0 <= lower < upper <= 1")
    margin = float(cfg.nn_residual_guard_margin)
    if not np.isfinite(margin) or margin < 0.0:
        raise ValueError("nn_residual_guard_margin must be finite and nonnegative")
    lower = float(np.quantile(values, lower_q) - margin)
    upper = float(np.quantile(values, upper_q) + margin)
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        raise ValueError("Invalid NN residual support bounds")
    return lower, upper


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
        "loss": cfg.nn_loss,
        "huber_delta": cfg.nn_huber_delta,
        "combined_mse_weight": cfg.nn_combined_mse_weight,
        "target": cfg.nn_target_mode,
        "enable_channel_history_features": cfg.enable_channel_history_features,
        "channel_aux_weight": cfg.channel_aux_weight,
        "channel_share_smoothing": cfg.channel_share_smoothing,
        "numeric_preprocessing": "median_impute+missing_indicator+standardize",
        "training_window_days": cfg.training_window_days,
        "recency_half_life_days": cfg.recency_half_life_days,
        "baseline_variant": cfg.baseline_variant,
        "enable_trend_features": cfg.enable_trend_features,
        "c2_feature_groups": tuple(cfg.c2_feature_groups),
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
    keys = (
        "num", "prod", "cw", "ca", "horizon", "sample_weight",
        "share_target", "share_valid",
    )
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
    y_target: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: Config,
    epochs: int,
) -> tuple[float, int, float, float]:
    x_num = tensors["num"].to(DEVICE)
    x_prod = tensors["prod"].to(DEVICE)
    x_cw = tensors["cw"].to(DEVICE)
    x_ca = tensors["ca"].to(DEVICE)
    x_horizon = tensors["horizon"].to(DEVICE)
    y = torch.as_tensor(y_target, dtype=torch.float32, device=DEVICE)
    sample_weight = tensors["sample_weight"].to(DEVICE)
    share_target = tensors["share_target"].to(DEVICE)
    share_valid = tensors["share_valid"].to(DEVICE)
    n_rows = len(y)
    optimizer_steps = 0
    final_loss = float("nan")
    final_total_loss = float("nan")
    final_aux_loss = 0.0

    for epoch in range(1, epochs + 1):
        weighted_total = 0.0
        weighted_primary = 0.0
        weighted_auxiliary = 0.0
        weight_total = 0.0
        permutation = _device_permutation(n_rows, DEVICE)
        for start, end in _batch_ranges(n_rows, cfg.batch_size):
            idx = permutation[start:end]
            pred, share_logit = model.forward_multitask(
                x_num[idx], x_prod[idx], x_cw[idx], x_ca[idx], x_horizon[idx]
            )
            primary_values, auxiliary_values, loss_values = multitask_loss_values(
                model,
                pred,
                y[idx],
                share_logit,
                share_target[idx],
                share_valid[idx],
                cfg,
            )
            batch_weight = sample_weight[idx]
            loss = (loss_values * batch_weight).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            weighted_total += float(
                (loss_values.detach() * batch_weight).sum().item()
            )
            weighted_primary += float(
                (primary_values.detach() * batch_weight).sum().item()
            )
            weighted_auxiliary += float(
                (auxiliary_values.detach() * batch_weight).sum().item()
            )
            weight_total += float(len(idx))
        scheduler.step()
        final_loss = weighted_total / max(weight_total, 1e-12)
        final_total_loss = weighted_primary / max(weight_total, 1e-12)
        final_aux_loss = weighted_auxiliary / max(weight_total, 1e-12)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            extra = (
                f" | total {final_total_loss:.4f} | share {final_aux_loss:.4f}"
                if float(cfg.channel_aux_weight) > 0.0 else ""
            )
            print(
                f"      epoch {epoch:3d}/{epochs} | train loss "
                f"{final_loss:.4f}{extra}"
            )
    return final_loss, optimizer_steps, final_total_loss, final_aux_loss


def _train_dataloader(
    model: QuantityNet,
    tensors: dict[str, torch.Tensor],
    y_target: np.ndarray,
    optimizer: torch.optim.Optimizer,
    scheduler,
    cfg: Config,
    epochs: int,
) -> tuple[float, int, float, float]:
    ds = TensorDataset(
        tensors["num"], tensors["prod"], tensors["cw"], tensors["ca"],
        tensors["horizon"], torch.tensor(y_target, dtype=torch.float32),
        tensors["sample_weight"], tensors["share_target"],
        tensors["share_valid"],
    )
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)
    optimizer_steps = 0
    final_loss = float("nan")
    final_total_loss = float("nan")
    final_aux_loss = 0.0
    for epoch in range(1, epochs + 1):
        weighted_total = 0.0
        weighted_primary = 0.0
        weighted_auxiliary = 0.0
        weight_total = 0.0
        for xn, xp, xcw, xca, xh, yt, wt, share_y, share_mask in dl:
            if len(yt) == 1:
                continue
            xn, xp, xcw, xca, xh, yt, wt, share_y, share_mask = (
                xn.to(DEVICE), xp.to(DEVICE), xcw.to(DEVICE),
                xca.to(DEVICE), xh.to(DEVICE), yt.to(DEVICE), wt.to(DEVICE),
                share_y.to(DEVICE), share_mask.to(DEVICE),
            )
            pred, share_logit = model.forward_multitask(xn, xp, xcw, xca, xh)
            primary_values, auxiliary_values, loss_values = multitask_loss_values(
                model, pred, yt, share_logit, share_y, share_mask, cfg
            )
            loss = (loss_values * wt).mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            weighted_total += float((loss_values.detach() * wt).sum().item())
            weighted_primary += float((primary_values.detach() * wt).sum().item())
            weighted_auxiliary += float((auxiliary_values.detach() * wt).sum().item())
            weight_total += float(len(yt))
        scheduler.step()
        final_loss = weighted_total / max(weight_total, 1e-12)
        final_total_loss = weighted_primary / max(weight_total, 1e-12)
        final_aux_loss = weighted_auxiliary / max(weight_total, 1e-12)
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            extra = (
                f" | total {final_total_loss:.4f} | share {final_aux_loss:.4f}"
                if float(cfg.channel_aux_weight) > 0.0 else ""
            )
            print(
                f"      epoch {epoch:3d}/{epochs} | train loss "
                f"{final_loss:.4f}{extra}"
            )
    return final_loss, optimizer_steps, final_total_loss, final_aux_loss


def train_model(
    tensors: dict,
    y_target: np.ndarray,
    cfg: Config,
    epochs: int,
    seed: int,
    *,
    stats_out: dict | None = None,
) -> QuantityNet:
    """Train one seed with the configured C3 objective and optional C4 head."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if DEVICE.type == "mps" and hasattr(torch.mps, "manual_seed"):
        torch.mps.manual_seed(seed)
    elif DEVICE.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    if cfg.nn_target_mode not in {"residual", "log1p"}:
        raise ValueError("nn_target_mode must be one of: residual, log1p")
    if cfg.nn_loss not in {"huber", "mse", "combined", "logcosh"}:
        raise ValueError("nn_loss must be one of: huber, mse, combined, logcosh")
    if not np.isfinite(cfg.channel_aux_weight) or cfg.channel_aux_weight < 0.0:
        raise ValueError("channel_aux_weight must be finite and nonnegative")

    model = QuantityNet(int(tensors["num"].shape[1]), cfg).to(DEVICE)
    learning_rate = effective_learning_rate(cfg)
    opt = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=cfg.weight_decay
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=epochs, eta_min=learning_rate * 0.01
    )
    backend = resolve_training_backend(cfg)
    started = time.perf_counter()
    _synchronize()

    model.train()
    try:
        if backend == "device_resident":
            final_loss, optimizer_steps, final_total_loss, final_aux_loss = (
                _train_device_resident(
                    model, tensors, y_target, opt, sched, cfg, epochs
                )
            )
        else:
            final_loss, optimizer_steps, final_total_loss, final_aux_loss = (
                _train_dataloader(
                    model, tensors, y_target, opt, sched, cfg, epochs
                )
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
            final_loss, optimizer_steps, final_total_loss, final_aux_loss = (
                _train_dataloader(
                    model, tensors, y_target, opt, sched, cfg, epochs
                )
            )
            backend = "dataloader_fallback"
        else:
            raise

    _synchronize()
    elapsed = time.perf_counter() - started
    model.eval()
    lower, upper = residual_support_bounds(y_target, cfg)
    model.residual_guard_lower = lower
    model.residual_guard_upper = upper
    model.target_mode = cfg.nn_target_mode
    if stats_out is not None:
        n_rows = len(y_target)
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
            "final_total_loss": float(final_total_loss),
            "final_aux_loss": float(final_aux_loss),
            "residual_guard_lower": lower,
            "residual_guard_upper": upper,
            "nn_loss": cfg.nn_loss,
            "nn_target_mode": cfg.nn_target_mode,
            "channel_aux_weight": float(cfg.channel_aux_weight),
            "estimated_device_tensor_mb": training_tensor_bytes(
                tensors, n_rows
            ) / (1024 ** 2),
        })
    return model


def predict_ensemble(
    models: list,
    tensors: dict,
    *,
    apply_residual_guard: bool = False,
    return_diagnostics: bool = False,
):
    """Reconstruct total demand and optionally average app-share heads."""
    baseline_log1p = tensors["baseline_log1p"].cpu().numpy().astype(float)
    x_num = tensors["num"].to(DEVICE)
    x_prod = tensors["prod"].to(DEVICE)
    x_cw = tensors["cw"].to(DEVICE)
    x_ca = tensors["ca"].to(DEVICE)
    x_horizon = tensors["horizon"].to(DEVICE)
    preds = []
    raw_targets = []
    share_preds = []
    guarded_rows = np.zeros(len(baseline_log1p), dtype=bool)
    max_log = np.log(np.finfo(np.float64).max) - 2.0

    for model in models:
        model.eval()
        with torch.inference_mode():
            if hasattr(model, "forward_multitask"):
                raw_tensor, share_logit = model.forward_multitask(
                    x_num, x_prod, x_cw, x_ca, x_horizon
                )
            else:
                raw_tensor = model(x_num, x_prod, x_cw, x_ca, x_horizon)
                share_logit = None
            raw = raw_tensor.cpu().numpy()
            if share_logit is not None:
                share_preds.append(torch.sigmoid(share_logit).cpu().numpy())
        raw = np.asarray(raw, dtype=float)
        raw_targets.append(raw)
        constrained = raw.copy()
        if apply_residual_guard:
            lower = float(getattr(model, "residual_guard_lower", -np.inf))
            upper = float(getattr(model, "residual_guard_upper", np.inf))
            finite_raw = np.isfinite(raw)
            clipped = np.clip(raw, lower, upper)
            guarded_rows |= finite_raw & (clipped != raw)
            constrained = np.where(finite_raw, clipped, np.nan)
        target_mode = getattr(model, "target_mode", "residual")
        if target_mode == "residual":
            log_prediction = constrained + baseline_log1p
        elif target_mode == "log1p":
            log_prediction = constrained
        else:
            raise ValueError(f"Unsupported NN target mode: {target_mode}")
        safe = np.isfinite(log_prediction) & (log_prediction <= max_log)
        reconstructed = np.full(len(log_prediction), np.nan, dtype=float)
        reconstructed[safe] = np.expm1(log_prediction[safe])
        preds.append(reconstructed)

    pred_matrix = np.vstack(preds)
    finite_pred = np.isfinite(pred_matrix)
    prediction_count = finite_pred.sum(axis=0)
    prediction_sum = np.where(finite_pred, pred_matrix, 0.0).sum(axis=0)
    prediction = np.divide(
        prediction_sum,
        prediction_count,
        out=np.full(prediction_count.shape, np.nan, dtype=float),
        where=prediction_count > 0,
    )
    prediction = np.clip(prediction, 0.0, None)

    raw_matrix = np.vstack(raw_targets)
    finite_raw = np.isfinite(raw_matrix)
    raw_min = np.min(np.where(finite_raw, raw_matrix, np.inf), axis=0)
    raw_max = np.max(np.where(finite_raw, raw_matrix, -np.inf), axis=0)
    raw_min[~finite_raw.any(axis=0)] = np.nan
    raw_max[~finite_raw.any(axis=0)] = np.nan
    diagnostics = {
        "residual_guard": guarded_rows,
        "residual_nonfinite": (~finite_raw).any(axis=0),
        "residual_raw_min": raw_min,
        "residual_raw_max": raw_max,
    }
    if share_preds:
        share_matrix = np.vstack(share_preds)
        app_share = np.nanmean(share_matrix, axis=0)
        app_share = np.clip(app_share, 0.0, 1.0)
        diagnostics.update({
            "app_share": app_share,
            "prediction_app": prediction * app_share,
            "prediction_web": prediction * (1.0 - app_share),
        })
    if return_diagnostics:
        return {"prediction": prediction, **diagnostics}
    return prediction


def predict_direct(
    models: list,
    scaler,
    panel: pd.DataFrame,
    cfg: Config = CFG,
    *,
    recursive_guard: bool = False,
    return_diagnostics: bool = False,
):
    tensors = make_tensors(panel, scaler, fit=False, cfg=cfg)
    return predict_ensemble(
        models, tensors,
        apply_residual_guard=recursive_guard,
        return_diagnostics=return_diagnostics,
    )
