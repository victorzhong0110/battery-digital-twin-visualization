"""LSTM sequence-to-value model for battery capacity prediction.

Input: sliding window of W cycles of features.
Output: predicted capacity for the next cycle.
Supports MC Dropout for uncertainty estimation.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class BatterySequenceDataset(Dataset):
    """Dataset that creates sliding windows from battery cycle features."""

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        window_size: int = 10,
    ):
        self.window_size = window_size
        self.x_windows: list[np.ndarray] = []
        self.y_targets: list[float] = []

        n = len(features)
        for i in range(window_size, n):
            self.x_windows.append(features[i - window_size:i])
            self.y_targets.append(targets[i])

    def __len__(self) -> int:
        return len(self.x_windows)

    def __getitem__(self, idx: int):
        x = torch.FloatTensor(self.x_windows[idx])
        y = torch.FloatTensor([self.y_targets[idx]])
        return x, y


class LSTMPredictor(nn.Module):
    """2-layer LSTM with dropout for capacity prediction."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        lstm_out, _ = self.lstm(x)
        # Use last time step output
        last_hidden = lstm_out[:, -1, :]
        out = self.dropout(last_hidden)
        return self.fc(out)


def train_lstm(
    train_features: np.ndarray,
    train_targets: np.ndarray,
    val_features: np.ndarray | None = None,
    val_targets: np.ndarray | None = None,
    window_size: int = 10,
    hidden_dim: int = 64,
    epochs: int = 100,
    batch_size: int = 16,
    lr: float = 1e-3,
    patience: int = 15,
) -> tuple[LSTMPredictor, dict]:
    """Train an LSTM model with early stopping.

    Returns the trained model and training history.
    """
    input_dim = train_features.shape[1]
    model = LSTMPredictor(input_dim=input_dim, hidden_dim=hidden_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.MSELoss()

    train_ds = BatterySequenceDataset(train_features, train_targets, window_size)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    val_loader = None
    if val_features is not None and val_targets is not None:
        val_ds = BatterySequenceDataset(val_features, val_targets, window_size)
        if len(val_ds) > 0:
            val_loader = DataLoader(val_ds, batch_size=batch_size)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        # Train
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

        avg_train = np.mean(train_losses)
        history["train_loss"].append(avg_train)

        # Validate
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
                    logger.info("Early stopping at epoch %d", epoch + 1)
                    break
        else:
            if avg_train < best_val_loss:
                best_val_loss = avg_train
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(DEVICE)

    return model, history


def predict_lstm(
    model: LSTMPredictor,
    features: np.ndarray,
    window_size: int = 10,
) -> np.ndarray:
    """Generate predictions for all valid windows in the feature array."""
    model.eval()
    predictions = []

    with torch.no_grad():
        for i in range(window_size, len(features)):
            x = torch.FloatTensor(features[i - window_size:i]).unsqueeze(0).to(DEVICE)
            pred = model(x).cpu().item()
            predictions.append(pred)

    return np.array(predictions)


def predict_with_uncertainty(
    model: LSTMPredictor,
    features: np.ndarray,
    window_size: int = 10,
    n_samples: int = 50,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MC Dropout prediction with uncertainty estimation.

    Returns (mean_predictions, lower_bound, upper_bound).
    """
    model.train()  # Enable dropout at inference
    all_preds = []

    for _ in range(n_samples):
        preds = []
        with torch.no_grad():
            for i in range(window_size, len(features)):
                x = torch.FloatTensor(features[i - window_size:i]).unsqueeze(0).to(DEVICE)
                pred = model(x).cpu().item()
                preds.append(pred)
        all_preds.append(preds)

    all_preds = np.array(all_preds)  # (n_samples, n_predictions)
    mean_pred = all_preds.mean(axis=0)
    lower = np.percentile(all_preds, 5, axis=0)
    upper = np.percentile(all_preds, 95, axis=0)

    model.eval()
    return mean_pred, lower, upper
