"""Neural net model: feed-forward network with product & campaign embeddings.

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

from framework import CFG, Config, NUM_CAMPAIGN_CATS, feature_columns, recursive_forecast_generic

torch.manual_seed(CFG.seed)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class QuantityNet(nn.Module):
    def __init__(self, num_numeric: int, cfg: Config):
        super().__init__()
        self.product_emb = nn.Embedding(cfg.num_products, cfg.embed_dim_product)
        self.campaign_emb_web = nn.Embedding(NUM_CAMPAIGN_CATS, cfg.embed_dim_campaign)
        self.campaign_emb_app = nn.Embedding(NUM_CAMPAIGN_CATS, cfg.embed_dim_campaign)

        input_dim = num_numeric + cfg.embed_dim_product + 2 * cfg.embed_dim_campaign
        layers: list[nn.Module] = []
        prev = input_dim
        for hidden, p in zip(cfg.hidden_dims, cfg.dropout):
            layers += [nn.Linear(prev, hidden), nn.BatchNorm1d(hidden), nn.GELU(), nn.Dropout(p)]
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x_num, x_prod, x_camp_web, x_camp_app):
        emb = torch.cat([
            self.product_emb(x_prod),
            self.campaign_emb_web(x_camp_web),
            self.campaign_emb_app(x_camp_app),
        ], dim=1)
        x = torch.cat([x_num, emb], dim=1)
        return self.net(x).squeeze(-1)


def make_tensors(df: pd.DataFrame, scaler: StandardScaler, fit: bool, cfg: Config = CFG) -> dict[str, torch.Tensor]:
    num = df[feature_columns(cfg)].to_numpy(dtype=np.float32)
    num = scaler.fit_transform(num) if fit else scaler.transform(num)
    return {
        "num": torch.tensor(num, dtype=torch.float32),
        "prod": torch.tensor(df["product_idx"].to_numpy(dtype=np.int64)),
        "cw": torch.tensor(df["campaign_idx_web"].to_numpy(dtype=np.int64)),
        "ca": torch.tensor(df["campaign_idx_app"].to_numpy(dtype=np.int64)),
    }


def train_model(tensors: dict, y_log: np.ndarray, cfg: Config, epochs: int, seed: int) -> QuantityNet:
    """Fixed-epoch training (no early stopping): every fold/final run gets a
    comparable, leakage-free training budget instead of peeking at the
    evaluation window to pick the "best" epoch."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = QuantityNet(len(feature_columns(cfg)), cfg).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=cfg.lr * 0.01)
    crit = nn.HuberLoss(delta=1.0)

    ds = TensorDataset(tensors["num"], tensors["prod"], tensors["cw"], tensors["ca"],
                       torch.tensor(y_log, dtype=torch.float32))
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True)

    model.train()
    for epoch in range(1, epochs + 1):
        total = 0.0
        for xn, xp, xcw, xca, yt in dl:
            xn, xp, xcw, xca, yt = (xn.to(DEVICE), xp.to(DEVICE), xcw.to(DEVICE),
                                     xca.to(DEVICE), yt.to(DEVICE))
            pred = model(xn, xp, xcw, xca)
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
    preds = []
    for m in models:
        m.eval()
        with torch.no_grad():
            p = m(tensors["num"].to(DEVICE), tensors["prod"].to(DEVICE),
                   tensors["cw"].to(DEVICE), tensors["ca"].to(DEVICE)).cpu().numpy()
        preds.append(np.expm1(p))
    return np.clip(np.mean(preds, axis=0), 0, None)


def recursive_forecast(models: list, scaler: StandardScaler,
                        static_df: pd.DataFrame, history: dict,
                        cfg: Config = CFG) -> np.ndarray:
    """Neural-net ensemble convenience wrapper around `recursive_forecast_generic`."""
    def predict_fn(day_df: pd.DataFrame) -> np.ndarray:
        tensors = make_tensors(day_df, scaler, fit=False, cfg=cfg)
        return predict_ensemble(models, tensors)

    return recursive_forecast_generic(predict_fn, static_df, history, cfg)
