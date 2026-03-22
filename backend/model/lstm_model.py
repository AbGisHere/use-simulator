"""
LSTM sequential model for NSE stock direction prediction.

Why LSTM on top of XGBoost?
    XGBoost sees each day as independent features. LSTM sees a SEQUENCE of days —
    it can recognise patterns like "3 days of tightening range then volume spike"
    or "price compressing below resistance for 10 days" that are invisible to
    a tree model that has no memory.

Architecture:
    Input  : (batch, sequence_len=20, num_features)
    Layer 1: Bidirectional LSTM (hidden=64, 2 layers, dropout=0.2)
    Layer 2: LayerNorm + Linear(64 → 1) + Sigmoid
    Output : P(up tomorrow)  ∈ [0, 1]

Training:
    - Walk-forward folds (same split as XGBoost) to stay honest
    - 30 epochs per fold (fast on CPU: ~5s per stock)
    - BCELoss, Adam lr=1e-3, weight decay 1e-4
    - Feature normalisation with StandardScaler fitted on training fold only
      (important: scaler must NOT see validation data)

Saved artefact: {MODEL_DIR}/{TICKER}_lstm.pt  (state dict + scaler)
"""

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

from config import MODEL_DIR

logger = logging.getLogger(__name__)

SEQ_LEN  = 20   # look-back window (trading days)
HIDDEN   = 64
LAYERS   = 2
DROPOUT  = 0.2
EPOCHS   = 40
LR       = 1e-3
BATCH    = 32


# ── Model definition ──────────────────────────────────────────────────────────

class DirectionLSTM(nn.Module):
    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size, HIDDEN, num_layers=LAYERS,
            batch_first=True, dropout=DROPOUT, bidirectional=True,
        )
        self.norm = nn.LayerNorm(HIDDEN * 2)
        self.fc   = nn.Linear(HIDDEN * 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        out = self.norm(out[:, -1, :])   # last time step
        return torch.sigmoid(self.fc(out)).squeeze(-1)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _lstm_path(ticker: str) -> Path:
    return MODEL_DIR / f"{ticker.upper()}_lstm.pt"


def _make_sequences(X: np.ndarray, y: np.ndarray, seq_len: int = SEQ_LEN):
    """Convert flat arrays into overlapping sequences for LSTM input."""
    xs, ys = [], []
    for i in range(seq_len, len(X)):
        xs.append(X[i - seq_len : i])
        ys.append(y[i])
    if not xs:
        return np.empty((0, seq_len, X.shape[1])), np.empty(0)
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


def _train_one_epoch(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    X_seq: torch.Tensor,
    y_seq: torch.Tensor,
) -> float:
    model.train()
    perm = torch.randperm(len(X_seq))
    total_loss = 0.0
    for i in range(0, len(X_seq), BATCH):
        idx  = perm[i : i + BATCH]
        xb   = X_seq[idx]
        yb   = y_seq[idx]
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(idx)
    return total_loss / len(X_seq)


# ── Public API ────────────────────────────────────────────────────────────────

def train_lstm(
    ticker: str,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """
    Train the LSTM on walk-forward folds, then save a final model on all data.

    Returns metadata dict with walk-forward accuracy.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    df = feature_df.copy()
    df[feature_cols] = df[feature_cols].ffill().fillna(0.0)
    df = df.dropna(subset=["target"])

    X_all = df[feature_cols].values.astype(np.float32)
    y_all = df["target"].values.astype(np.float32)

    WALK_START = 252
    FOLD_SIZE  = 63
    input_size = len(feature_cols)

    fold_accuracies: list[float] = []

    logger.info("Training LSTM for %s (%d rows, %d features)…", symbol, len(df), input_size)

    for fold_start in range(WALK_START, len(df) - FOLD_SIZE, FOLD_SIZE):
        X_tr = X_all[:fold_start]
        y_tr = y_all[:fold_start]
        X_vl = X_all[fold_start : fold_start + FOLD_SIZE]
        y_vl = y_all[fold_start : fold_start + FOLD_SIZE]

        # Fit scaler on training fold only
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        # Need SEQ_LEN rows of training data to produce the first validation sequence
        X_vl_s = scaler.transform(X_vl)

        # Build sequences  (include the last SEQ_LEN training rows as context for val)
        context = scaler.transform(X_tr[-SEQ_LEN:])
        X_vl_ctx = np.vstack([context, X_vl_s])

        X_tr_seq, y_tr_seq = _make_sequences(X_tr_s, y_tr)
        X_vl_seq, y_vl_seq = _make_sequences(X_vl_ctx, np.concatenate([y_tr[-SEQ_LEN:], y_vl]))

        if len(X_tr_seq) < BATCH or len(X_vl_seq) == 0:
            continue

        model = DirectionLSTM(input_size)
        opt   = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
        crit  = nn.BCELoss()

        X_tr_t = torch.tensor(X_tr_seq)
        y_tr_t = torch.tensor(y_tr_seq)

        for _ in range(EPOCHS):
            _train_one_epoch(model, opt, crit, X_tr_t, y_tr_t)

        # Evaluate on validation sequences
        model.eval()
        with torch.no_grad():
            X_vl_t = torch.tensor(X_vl_seq)
            probs   = model(X_vl_t).numpy()
            preds   = (probs >= 0.5).astype(int)
            acc     = (preds == y_vl_seq.astype(int)).mean()
            fold_accuracies.append(float(acc))

    mean_acc = float(np.mean(fold_accuracies)) if fold_accuracies else 0.5
    logger.info(
        "LSTM walk-forward accuracy for %s: %.4f (%d folds)",
        symbol, mean_acc, len(fold_accuracies),
    )

    # ── Final model: train on ALL data ────────────────────────────────────────
    final_scaler = StandardScaler()
    X_all_s = final_scaler.fit_transform(X_all)
    X_seq, y_seq = _make_sequences(X_all_s, y_all)

    if len(X_seq) > 0:
        final_model = DirectionLSTM(input_size)
        opt  = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
        crit = nn.BCELoss()
        X_t  = torch.tensor(X_seq)
        y_t  = torch.tensor(y_seq)
        for epoch in range(EPOCHS):
            loss = _train_one_epoch(final_model, opt, crit, X_t, y_t)
            if (epoch + 1) % 10 == 0:
                logger.debug("LSTM epoch %d/%d — loss %.4f", epoch + 1, EPOCHS, loss)

        # Save state dict + scaler
        save_path = _lstm_path(symbol)
        joblib.dump(
            {
                "state_dict": final_model.state_dict(),
                "scaler": final_scaler,
                "feature_cols": feature_cols,
                "input_size": input_size,
                "accuracy": mean_acc,
            },
            save_path,
        )
        logger.info("Saved LSTM for %s to %s", symbol, save_path)
    else:
        logger.warning("Not enough sequences to train final LSTM for %s", symbol)

    return {"accuracy": mean_acc, "n_folds": len(fold_accuracies)}


def lstm_exists(ticker: str) -> bool:
    return _lstm_path(ticker.upper().replace(".NS", "").replace(".BO", "")).exists()


def predict_lstm(
    ticker: str,
    feature_df: pd.DataFrame,
    feature_cols: list[str],
) -> float | None:
    """
    Run the saved LSTM on the last SEQ_LEN rows and return P(up) ∈ [0,1].
    Returns None if no LSTM model exists.
    """
    symbol = ticker.upper().replace(".NS", "").replace(".BO", "")
    path = _lstm_path(symbol)
    if not path.exists():
        return None

    try:
        saved = joblib.load(path)
        scaler: StandardScaler = saved["scaler"]
        saved_cols: list[str]  = saved["feature_cols"]
        input_size: int        = saved["input_size"]

        # Use only the columns the LSTM was trained on
        avail = [c for c in saved_cols if c in feature_df.columns]
        if len(avail) < input_size:
            logger.warning("LSTM: %d/%d feature columns available", len(avail), input_size)
            return None

        df = feature_df.copy()
        df[avail] = df[avail].ffill().fillna(0.0)
        df = df.dropna(subset=["Close"])

        if len(df) < SEQ_LEN:
            return None

        X_raw = df[avail].values[-SEQ_LEN:]
        X_scaled = scaler.transform(X_raw)
        X_tensor  = torch.tensor(X_scaled[np.newaxis], dtype=torch.float32)

        model = DirectionLSTM(input_size)
        model.load_state_dict(saved["state_dict"])
        model.eval()
        with torch.no_grad():
            prob = float(model(X_tensor).item())

        return prob
    except Exception as exc:
        logger.warning("LSTM inference failed for %s: %s", symbol, exc)
        return None
