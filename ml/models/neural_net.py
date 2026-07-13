"""Neural net model: feed-forward network with product, campaign & horizon
embeddings, trained on the direct multi-horizon panel
(`framework.build_direct_panel`) -- one forward pass predicts every
horizon directly from the origin, no recursion.

Tier B2: the network doesn't predict raw log1p(Quantity) -- it predicts a
*residual* on top of `target_baseline` (the weighted same-weekday baseline
for the target date, already computed by `build_direct_panel`), which is
also fed in as one of its own numeric input features. `residual_log1p_target`
builds the training target (log1p(actual) - log1p(baseline)); `predict_ensemble`
adds `log1p(baseline)` back before `expm1`, i.e. a skip connection straight
from the baseline to the output. Easier to learn than the raw target, since
the model only needs to capture the *deviation* from an already-strong
seasonal baseline instead of relearning the whole weekly pattern from
scratch.

This is the task brief's requested non-tree approach -- the actual
submission -- benchmarked in `pipeline.py` against the tree/naive models
defined alongside this one under `models/`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from framework import CFG, Config, NUM_CAMPAIGN_CATS, direct_panel_feature_names

torch.manual_seed(CFG.seed)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def numeric_feature_columns(cfg: Config = CFG) -> list[str]:
    """`direct_panel_feature_names` minus `horizon` -- horizon gets its own
    embedding below (like product/campaign) instead of being fed in as a
    raw numeric input."""
    return [c for c in direct_panel_feature_names(cfg) if c != "horizon"]


class QuantityNet(nn.Module):
    def __init__(self, num_numeric: int, cfg: Config):
        super().__init__()
        self.product_emb = nn.Embedding(cfg.num_products, cfg.embed_dim_product)
        self.campaign_emb_web = nn.Embedding(NUM_CAMPAIGN_CATS, cfg.embed_dim_campaign)
        self.campaign_emb_app = nn.Embedding(NUM_CAMPAIGN_CATS, cfg.embed_dim_campaign)
        self.horizon_emb = nn.Embedding(cfg.horizon, cfg.embed_dim_horizon)

        input_dim = (num_numeric + cfg.embed_dim_product + 2 * cfg.embed_dim_campaign
                     + cfg.embed_dim_horizon)
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden, p in zip(cfg.hidden_dims, cfg.dropout):
            layers += [nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(p)]
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
        x = torch.cat([x_num, emb], dim=1)
        return self.net(x).squeeze(-1)


def make_tensors(df: pd.DataFrame, scaler: StandardScaler, fit: bool, cfg: Config = CFG) -> dict[str, torch.Tensor]:
    num = df[numeric_feature_columns(cfg)].to_numpy(dtype=np.float32)
    num = scaler.fit_transform(num) if fit else scaler.transform(num)
    return {
        "num": torch.tensor(num, dtype=torch.float32),
        "prod": torch.tensor(df["product_idx"].to_numpy(dtype=np.int64)),
        "cw": torch.tensor(df["campaign_idx_web"].to_numpy(dtype=np.int64)),
        "ca": torch.tensor(df["campaign_idx_app"].to_numpy(dtype=np.int64)),
        "horizon": torch.tensor((df["horizon"].to_numpy(dtype=np.int64) - 1)),
        # Raw (unscaled) skip-connection reference, read straight off the
        # panel -- separate from `target_baseline`'s scaled copy inside
        # "num" above, which is a plain input feature like any other.
        "baseline_log1p": torch.tensor(np.log1p(df["target_baseline"].to_numpy(dtype=np.float32)),
                                       dtype=torch.float32),
    }


def residual_log1p_target(df: pd.DataFrame) -> np.ndarray:
    """The NN's actual training target: log1p(actual) - log1p(target_baseline)
    -- a residual on top of the weighted same-weekday baseline (Tier B2's
    skip-connection design, see `predict_ensemble`'s add-back), not raw
    log1p(Quantity). `df` must have both `target` and `target_baseline`
    columns (i.e. a `build_direct_panel` training panel)."""
    return (np.log1p(df["target"].to_numpy(dtype=np.float32))
            - np.log1p(df["target_baseline"].to_numpy(dtype=np.float32)))


def train_model(tensors: dict, y_residual: np.ndarray, cfg: Config, epochs: int, seed: int) -> QuantityNet:
    """Fixed-epoch training (no early stopping): every fold/final run gets a
    comparable, leakage-free training budget instead of peeking at the
    evaluation window to pick the "best" epoch. `y_residual` is a
    log-residual over `target_baseline` (see `residual_log1p_target`), not
    raw log1p(Quantity) -- the model only has to learn the deviation from
    the baseline, added back in `predict_ensemble`."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = QuantityNet(len(numeric_feature_columns(cfg)), cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.lr * 0.01)
    crit = nn.HuberLoss(delta=1.0)

    ds = TensorDataset(tensors["num"], tensors["prod"], tensors["cw"], tensors["ca"], tensors["horizon"],
                       torch.tensor(y_residual, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xn, xp, xcw, xca, xh, yt in dl:
            xn, xp, xcw, xca, xh, yt = (xn.to(DEVICE), xp.to(DEVICE), xcw.to(DEVICE),
                                         xca.to(DEVICE), xh.to(DEVICE), yt.to(DEVICE))
            pred = model(xn, xp, xcw, xca, xh)
            loss = crit(pred, yt)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * xn.size(0)
        sched.step()
        if epoch == 1 or epoch % 10 == 0 or epoch == epochs:
            print(f"      epoch {epoch:3d}/{epochs} | train loss {total / len(ds):.4f}")
    model.eval()
    return model


def predict_ensemble(models: list, tensors: dict) -> np.ndarray:
    """Skip connection: each model's raw output is a log-residual over
    `target_baseline` (see `residual_log1p_target`), so `baseline_log1p`
    is added back here -- before `expm1` -- to reconstruct an actual
    Quantity prediction. Ensembling still averages in the natural
    (post-expm1) scale across seeds, same as before B2."""
    baseline_log1p = tensors["baseline_log1p"].cpu().numpy()
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            p = m(tensors["num"].to(DEVICE), tensors["prod"].to(DEVICE),
                   tensors["cw"].to(DEVICE), tensors["ca"].to(DEVICE),
                   tensors["horizon"].to(DEVICE)).cpu().numpy()
        preds.append(np.expm1(p + baseline_log1p))
    return np.clip(np.mean(preds, axis=0), 0, None)


def predict_direct(models: list, scaler: StandardScaler, panel: pd.DataFrame,
                    cfg: Config = CFG) -> np.ndarray:
    """Direct multi-horizon prediction: one forward pass over every
    (origin, horizon, product) row of `panel` -- no recursion, since every
    horizon's features are already lookups into observed data (see
    `framework.build_direct_panel`)."""
    tensors = make_tensors(panel, scaler, fit=False, cfg=cfg)
    return predict_ensemble(models, tensors)
