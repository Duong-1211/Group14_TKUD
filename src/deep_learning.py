from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - handled at runtime for optional dependency
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


def _require_torch() -> None:
    if torch is None:
        raise ImportError("PyTorch is required for deep-learning detectors. Install with: pip install torch")


@dataclass
class StandardScaler1D:
    mean_: float
    std_: float

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScaler1D":
        return cls(float(np.mean(values)), float(np.std(values) + 1e-8))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - self.mean_) / self.std_

    def inverse_transform(self, values: np.ndarray) -> np.ndarray:
        return values * self.std_ + self.mean_


def make_windows(values: np.ndarray, window_size: int = 288, stride: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding windows and return their ending point indices."""
    values = np.asarray(values, dtype=np.float32)
    windows = []
    end_indices = []
    for start in range(0, len(values) - window_size + 1, stride):
        end = start + window_size
        windows.append(values[start:end])
        end_indices.append(end - 1)
    return np.asarray(windows, dtype=np.float32), np.asarray(end_indices, dtype=np.int64)


def window_training_mask(point_mask: pd.Series | np.ndarray, end_indices: np.ndarray, window_size: int) -> np.ndarray:
    """Keep windows whose full span is marked trainable."""
    mask = np.asarray(point_mask, dtype=bool)
    keep = []
    for end in end_indices:
        start = end - window_size + 1
        keep.append(mask[start : end + 1].all())
    return np.asarray(keep, dtype=bool)


class WindowAutoencoder(nn.Module):
    def __init__(self, window_size: int = 288, latent_dim: int = 32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(window_size, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, window_size),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(
    df: pd.DataFrame,
    train_mask: pd.Series | np.ndarray,
    *,
    window_size: int = 288,
    stride: int = 1,
    epochs: int = 20,
    batch_size: int = 128,
    learning_rate: float = 1e-3,
    threshold_quantile: float = 0.995,
    seed: int = 42,
):
    """Train a window reconstruction autoencoder."""
    _require_torch()
    torch.manual_seed(seed)

    values = df["value"].to_numpy(dtype=np.float32)
    scaler = StandardScaler1D.fit(values[np.asarray(train_mask, dtype=bool)])
    scaled = scaler.transform(values)
    windows, end_indices = make_windows(scaled, window_size, stride)
    keep = window_training_mask(train_mask, end_indices, window_size)
    x_train = torch.tensor(windows[keep], dtype=torch.float32)

    model = WindowAutoencoder(window_size=window_size)
    loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(batch), batch)
            loss.backward()
            optimizer.step()

    train_scores = score_autoencoder(model, windows[keep], batch_size=batch_size)
    threshold = float(np.quantile(train_scores, threshold_quantile))
    return model, scaler, threshold


def score_autoencoder(model, windows: np.ndarray, *, batch_size: int = 256) -> np.ndarray:
    _require_torch()
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(windows, dtype=torch.float32)), batch_size=batch_size)
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            reconstructed = model(batch)
            scores.extend(torch.mean((reconstructed - batch) ** 2, dim=1).detach().cpu().tolist())
    return np.asarray(scores)


def detect_autoencoder(
    df: pd.DataFrame,
    model,
    scaler: StandardScaler1D,
    threshold: float,
    *,
    window_size: int = 288,
    stride: int = 1,
) -> pd.DataFrame:
    values = scaler.transform(df["value"].to_numpy(dtype=np.float32))
    windows, end_indices = make_windows(values, window_size, stride)
    scores = score_autoencoder(model, windows)

    out = df.loc[:, ["timestamp", "value"]].copy()
    out["score"] = 0.0
    out.loc[end_indices, "score"] = scores
    out["threshold"] = threshold
    out["is_anomaly_pred"] = out["score"] > threshold
    return out


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, attn_weights = self.attn(x, x, x, need_weights=True, average_attn_weights=False)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x, attn_weights


class CompactAnomalyTransformer(nn.Module):
    """A compact educational Anomaly-Transformer-style reconstruction model."""

    def __init__(self, window_size: int = 288, d_model: int = 32, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.window_size = window_size
        self.value_proj = nn.Linear(1, d_model)
        self.position = nn.Parameter(torch.zeros(1, window_size, d_model))
        self.block = TransformerBlock(d_model, n_heads, dropout)
        self.output = nn.Linear(d_model, 1)

    def forward(self, x):
        x = x.unsqueeze(-1)
        h = self.value_proj(x) + self.position[:, : x.shape[1], :]
        h, attn = self.block(h)
        reconstructed = self.output(h).squeeze(-1)
        return reconstructed, attn


def temporal_prior(window_size: int, sigma: float = 24.0):
    _require_torch()
    idx = torch.arange(window_size, dtype=torch.float32)
    dist = (idx[None, :] - idx[:, None]) ** 2
    prior = torch.exp(-dist / (2 * sigma**2))
    return prior / prior.sum(dim=-1, keepdim=True)


def association_discrepancy(attn, prior) -> torch.Tensor:
    prior = prior.to(attn.device)
    return torch.mean(torch.abs(attn - prior), dim=(1, 2, 3))


def train_anomaly_transformer(
    df: pd.DataFrame,
    train_mask: pd.Series | np.ndarray,
    *,
    window_size: int = 288,
    stride: int = 1,
    epochs: int = 10,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    threshold_quantile: float = 0.995,
    seed: int = 42,
):
    """Train a compact Anomaly-Transformer-style detector."""
    _require_torch()
    torch.manual_seed(seed)

    values = df["value"].to_numpy(dtype=np.float32)
    scaler = StandardScaler1D.fit(values[np.asarray(train_mask, dtype=bool)])
    scaled = scaler.transform(values)
    windows, end_indices = make_windows(scaled, window_size, stride)
    keep = window_training_mask(train_mask, end_indices, window_size)
    x_train = torch.tensor(windows[keep], dtype=torch.float32)

    model = CompactAnomalyTransformer(window_size=window_size)
    loader = DataLoader(TensorDataset(x_train), batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()
    prior = temporal_prior(window_size)

    model.train()
    for _ in range(epochs):
        for (batch,) in loader:
            optimizer.zero_grad()
            reconstructed, attn = model(batch)
            discrepancy = association_discrepancy(attn, prior).mean()
            loss = loss_fn(reconstructed, batch) + 0.05 * discrepancy
            loss.backward()
            optimizer.step()

    train_scores = score_anomaly_transformer(model, windows[keep], batch_size=batch_size)
    threshold = float(np.quantile(train_scores, threshold_quantile))
    return model, scaler, threshold


def score_anomaly_transformer(model, windows: np.ndarray, *, batch_size: int = 128) -> np.ndarray:
    _require_torch()
    model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(windows, dtype=torch.float32)), batch_size=batch_size)
    prior = temporal_prior(windows.shape[1])
    scores = []
    with torch.no_grad():
        for (batch,) in loader:
            reconstructed, attn = model(batch)
            recon = torch.mean((reconstructed - batch) ** 2, dim=1)
            discrepancy = association_discrepancy(attn, prior)
            scores.extend((recon + 0.05 * discrepancy).detach().cpu().tolist())
    return np.asarray(scores)


def detect_anomaly_transformer(
    df: pd.DataFrame,
    model,
    scaler: StandardScaler1D,
    threshold: float,
    *,
    window_size: int = 288,
    stride: int = 1,
) -> pd.DataFrame:
    values = scaler.transform(df["value"].to_numpy(dtype=np.float32))
    windows, end_indices = make_windows(values, window_size, stride)
    scores = score_anomaly_transformer(model, windows)

    out = df.loc[:, ["timestamp", "value"]].copy()
    out["score"] = 0.0
    out.loc[end_indices, "score"] = scores
    out["threshold"] = threshold
    out["is_anomaly_pred"] = out["score"] > threshold
    return out
