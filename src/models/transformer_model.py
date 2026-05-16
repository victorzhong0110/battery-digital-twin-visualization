"""Transformer-based time-series model for battery capacity prediction.

Key innovations over LSTM:
    1. Self-attention captures long-range cycle dependencies
    2. Positional encoding preserves cycle ordering
    3. Attention weights are extractable for visualization
       (shows which historical cycles the model focuses on)
    4. MC Dropout for uncertainty estimation

Architecture:
    Input (window_size, n_features) -> Positional Encoding
    -> TransformerEncoder (N layers) -> Pooling -> FC -> capacity prediction
"""

from __future__ import annotations

import logging
import math

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class CyclePositionalEncoding(nn.Module):
    """Sinusoidal positional encoding adapted for battery cycle sequences.

    Encodes both absolute position within the window and the cycle index
    as a feature, allowing the model to reason about aging progression.
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, : x.size(1)]


class BatteryTransformer(nn.Module):
    """Transformer encoder for battery degradation prediction.

    Stores attention weights for visualization.
    """

    def __init__(
        self,
        input_dim: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.d_model = d_model

        # Project input features to d_model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
        )

        self.pos_encoder = CyclePositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
        )

        self.dropout = nn.Dropout(dropout)

        # Prediction head: weighted pooling + FC
        self.attention_pool = nn.Sequential(
            nn.Linear(d_model, 1),
            nn.Softmax(dim=1),
        )
        self.fc_out = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )

        # Storage for attention weights (set during forward)
        self._attention_weights: list[torch.Tensor] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: (batch, seq_len, input_dim)

        Returns:
            predictions: (batch, 1)
        """
        # Project to d_model
        h = self.input_proj(x)  # (batch, seq_len, d_model)
        h = self.pos_encoder(h)

        # Transformer encoding
        h = self.transformer_encoder(h)  # (batch, seq_len, d_model)

        # Attention-weighted pooling (learn which cycles matter most)
        attn_weights = self.attention_pool(h)  # (batch, seq_len, 1)
        self._attention_weights = [attn_weights.detach().cpu()]

        # Weighted sum
        pooled = (h * attn_weights).sum(dim=1)  # (batch, d_model)
        pooled = self.dropout(pooled)

        return self.fc_out(pooled)

    def get_attention_weights(self) -> np.ndarray | None:
        """Get the last computed attention pooling weights for visualization.

        Returns: (seq_len,) array showing how much each historical cycle
        contributed to the prediction. Higher = more important.
        """
        if self._attention_weights:
            return self._attention_weights[0][0, :, 0].numpy()
        return None


class SequenceDataset(Dataset):
    """Sliding window dataset for Transformer input."""

    def __init__(self, features: np.ndarray, targets: np.ndarray, window_size: int = 15):
        self.window_size = window_size
        self.x_windows: list[np.ndarray] = []
        self.y_targets: list[float] = []

        for i in range(window_size, len(features)):
            self.x_windows.append(features[i - window_size: i])
            self.y_targets.append(targets[i])

    def __len__(self) -> int:
        return len(self.x_windows)

    def __getitem__(self, idx: int):
        return (
            torch.FloatTensor(self.x_windows[idx]),
            torch.FloatTensor([self.y_targets[idx]]),
        )


def train_transformer(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    window_size: int = 15,
    d_model: int = 64,
    nhead: int = 4,
    num_layers: int = 3,
    epochs: int = 150,
    batch_size: int = 16,
    lr: float = 5e-4,
    patience: int = 20,
) -> tuple[BatteryTransformer, dict]:
    """Train a Transformer model with cosine annealing + early stopping."""
    input_dim = train_features.shape[1]
    model = BatteryTransformer(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dropout=0.2,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    criterion = nn.HuberLoss(delta=0.05)  # Robust to outliers

    train_ds = SequenceDataset(train_features, train_targets, window_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_features is not None and val_targets is not None:
        val_ds = SequenceDataset(val_features, val_targets, window_size)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=batch_size)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)

        if val_loader is not None:
            model.eval()
            val_losses = []
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)
                    pred = model(x_batch)
                    val_losses.append(criterion(pred, y_batch).item())
            avg_val = np.mean(val_losses)
            history["val_loss"].append(avg_val)

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info("Transformer early stop at epoch %d", epoch + 1)
                    break
        else:
            if avg_train < best_val_loss:
                best_val_loss = avg_train
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    return model, history


def predict_transformer(
    model: BatteryTransformer,
    features: np.ndarray,
    window_size: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """Predict with attention weights extraction.

    Returns: (predictions, attention_weights_per_sample)
    """
    model.eval()
    predictions = []
    all_attn = []

    with torch.no_grad():
        for i in range(window_size, len(features)):
            x = torch.FloatTensor(features[i - window_size: i]).unsqueeze(0).to(DEVICE)
            pred = model(x).cpu().item()
            predictions.append(pred)
            attn = model.get_attention_weights()
            if attn is not None:
                all_attn.append(attn)

    return np.array(predictions), np.array(all_attn) if all_attn else np.array([])


def predict_with_uncertainty(
    model: BatteryTransformer,
    features: np.ndarray,
    window_size: int = 15,
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MC Dropout uncertainty estimation for Transformer."""
    model.train()  # Enable dropout
    all_preds = []

    for _ in range(n_samples):
        preds = []
        with torch.no_grad():
            for i in range(window_size, len(features)):
                x = torch.FloatTensor(features[i - window_size: i]).unsqueeze(0).to(DEVICE)
                pred = model(x).cpu().item()
                preds.append(pred)
        all_preds.append(preds)

    all_preds_arr = np.array(all_preds)
    mean_pred = all_preds_arr.mean(axis=0)
    lower = np.percentile(all_preds_arr, 5, axis=0)
    upper = np.percentile(all_preds_arr, 95, axis=0)

    model.eval()
    return mean_pred, lower, upper
