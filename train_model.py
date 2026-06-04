"""
2_train_model.py
─────────────────
Gated Recurrent Unit (GRU) — Plant Health Index Forecaster

Trains a sequence-to-one GRU neural network on the hydro_data.csv dataset
to predict the continuous Plant Health Index (H ∈ [0, 1]).

Pipeline
────────
  1. Load & validate hydro_data.csv
  2. Construct overlapping sliding-window sequences
  3. Train/validation/test split (70 / 15 / 15)
  4. Train GRU with early stopping and learning-rate scheduling
  5. Evaluate on the hold-out test set (MSE, RMSE, MAE, R²)
  6. Serialise model weights → gru_health_model.pt

Author  : Hybrid Quantum-Classical CPS Research Group
License : MIT
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)s]  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
DATA_FILE        = Path("hydro_data.csv")
MODEL_FILE       = Path("gru_health_model.pt")

SEQUENCE_LENGTH  = 24          # look-back window (hours)
BATCH_SIZE       = 64
HIDDEN_SIZE      = 128
NUM_LAYERS       = 2
DROPOUT          = 0.25
LEARNING_RATE    = 1e-3
EPOCHS           = 100
PATIENCE         = 12          # early-stopping patience
LR_DECAY_FACTOR  = 0.5
LR_DECAY_PATIENCE= 6

TRAIN_RATIO      = 0.70
VAL_RATIO        = 0.15
# test_ratio = 1 - TRAIN_RATIO - VAL_RATIO = 0.15

FEATURE_COLS = [
    "pump_power_w", "led_power_w", "cooler_power_w",
    "water_ph", "water_ec", "ambient_temp",
    "water_temp", "humidity",
    "maceration_chlorophyll", "leaf_area_cm2",
]
TARGET_COL   = "health_index"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Using device: %s", DEVICE)


# ─────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────
class HydroSequenceDataset(Dataset):
    """
    Sliding-window dataset.

    Each sample is (X, y) where:
        X : float32 tensor  [seq_len × n_features]
        y : float32 scalar  (health_index at time t + seq_len)
    """

    def __init__(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        seq_len: int,
    ) -> None:
        self.features = torch.tensor(features, dtype=torch.float32)
        self.targets  = torch.tensor(targets,  dtype=torch.float32)
        self.seq_len  = seq_len

    def __len__(self) -> int:
        return len(self.targets) - self.seq_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        X = self.features[idx : idx + self.seq_len]          # [seq_len, n_feat]
        y = self.targets [idx + self.seq_len]                 # scalar
        return X, y


# ─────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────
class GRUHealthPredictor(nn.Module):
    """
    Multi-layer GRU followed by a fully-connected regression head.

    Architecture
    ────────────
        GRU (bidirectional=False, batch_first=True)
          ↓
        Layer Norm
          ↓
        Dropout
          ↓
        Linear(hidden_size → 64) + ReLU
          ↓
        Linear(64 → 1) + Sigmoid   → H ∈ (0, 1)
    """

    def __init__(
        self,
        input_size:  int,
        hidden_size: int = HIDDEN_SIZE,
        num_layers:  int = NUM_LAYERS,
        dropout:     float = DROPOUT,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout    = nn.Dropout(dropout)
        self.fc1        = nn.Linear(hidden_size, 64)
        self.relu       = nn.ReLU()
        self.fc2        = nn.Linear(64, 1)
        self.sigmoid    = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : [batch, seq_len, input_size]

        Returns
        -------
        Tensor of shape [batch] — predicted health index
        """
        out, _ = self.gru(x)                    # [batch, seq_len, hidden]
        last    = out[:, -1, :]                  # take final time-step
        last    = self.layer_norm(last)
        last    = self.dropout(last)
        last    = self.relu(self.fc1(last))
        pred    = self.sigmoid(self.fc2(last))   # [batch, 1]
        return pred.squeeze(-1)                  # [batch]


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────
def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    mse  = float(np.mean((y_true - y_pred) ** 2))
    rmse = math.sqrt(mse)
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2   = 1.0 - ss_res / (ss_tot + 1e-12)
    return {"MSE": mse, "RMSE": rmse, "MAE": mae, "R²": r2}


# ─────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────
def train_one_epoch(
    model:     GRUHealthPredictor,
    loader:    DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model:     GRUHealthPredictor,
    loader:    DataLoader,
    criterion: nn.Module,
) -> Tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    all_preds, all_true = [], []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(DEVICE)
        preds   = model(X_batch).cpu().numpy()
        all_preds.append(preds)
        all_true.append(y_batch.numpy())
        loss     = criterion(
            torch.tensor(preds),
            y_batch,
        )
        total_loss += loss.item() * len(y_batch)
    return (
        total_loss / len(loader.dataset),
        np.concatenate(all_preds),
        np.concatenate(all_true),
    )


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
def main() -> None:
    # ── Load data ────────────────────────────────────────────────────────
    log.info("Loading dataset: %s", DATA_FILE)
    df = pd.read_csv(DATA_FILE, parse_dates=["timestamp"])
    df = df.dropna(subset=FEATURE_COLS + [TARGET_COL])
    log.info("  Rows after NaN-drop: %d", len(df))

    features = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    targets  = df[TARGET_COL].to_numpy(dtype=np.float32)

    # ── Train / val / test split (chronological) ─────────────────────────
    n        = len(targets)
    n_train  = int(n * TRAIN_RATIO)
    n_val    = int(n * VAL_RATIO)

    train_ds = HydroSequenceDataset(features[:n_train],           targets[:n_train],           SEQUENCE_LENGTH)
    val_ds   = HydroSequenceDataset(features[n_train:n_train+n_val], targets[n_train:n_train+n_val], SEQUENCE_LENGTH)
    test_ds  = HydroSequenceDataset(features[n_train+n_val:],     targets[n_train+n_val:],     SEQUENCE_LENGTH)

    log.info("Split sizes → train: %d | val: %d | test: %d",
             len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

    # ── Model, loss, optimiser ───────────────────────────────────────────
    model     = GRUHealthPredictor(input_size=len(FEATURE_COLS)).to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=LR_DECAY_FACTOR,
        patience=LR_DECAY_PATIENCE,
    )

    log.info("Model parameter count: %d",
             sum(p.numel() for p in model.parameters() if p.requires_grad))

    # ── Training loop ────────────────────────────────────────────────────
    best_val_loss  = float("inf")
    patience_count = 0
    best_state     = None

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, _, _ = evaluate(model, val_loader, criterion)
        scheduler.step(val_loss)

        log.info(
            "Epoch %3d/%d  |  train_MSE: %.6f  |  val_MSE: %.6f  |  LR: %.2e",
            epoch, EPOCHS, train_loss, val_loss,
            optimizer.param_groups[0]["lr"],
        )

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            patience_count = 0
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_count += 1
            if patience_count >= PATIENCE:
                log.info("Early stopping triggered at epoch %d.", epoch)
                break

    # ── Restore best weights & test evaluation ───────────────────────────
    model.load_state_dict(best_state)
    _, y_pred, y_true = evaluate(model, test_loader, criterion)
    metrics           = compute_metrics(y_true, y_pred)

    log.info("─── Test-Set Performance ───────────────────────────")
    for name, val in metrics.items():
        log.info("  %-6s : %.6f", name, val)
    log.info("────────────────────────────────────────────────────")

    # ── Save model ───────────────────────────────────────────────────────
    torch.save(
        {
            "model_state_dict" : best_state,
            "config"           : {
                "input_size"  : len(FEATURE_COLS),
                "hidden_size" : HIDDEN_SIZE,
                "num_layers"  : NUM_LAYERS,
                "dropout"     : DROPOUT,
                "seq_len"     : SEQUENCE_LENGTH,
                "feature_cols": FEATURE_COLS,
            },
            "test_metrics": metrics,
        },
        MODEL_FILE,
    )
    log.info("Model checkpoint saved → %s", MODEL_FILE)


if __name__ == "__main__":
    main()