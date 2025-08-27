#!/usr/bin/env python3
"""
distance_lstm.py

Train and validate a multi-output regressor to estimate distances (range) from CSI amplitude features.

Usage:
  python distance_lstm.py \
    --X_train X_train.npy --y_train y_train.npy \
    --X_val   X_val.npy   --y_val   y_val.npy   \
    --epochs  50 --batch-size 32 --lr 1e-3 \
    --hidden-size 64 --num-layers 2 --dropout 0.5 \
    --checkpoint distance_lstm.pth

This version supports predicting distances to multiple anchors per sample.
"""
import argparse
import logging
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error


def configure_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
    )


class CSIDataset(Dataset):
    """Dataset for loading CSI features (flattened) and distance arrays."""

    def __init__(self, X_path, y_path):
        self.X = np.load(X_path)  # shape: (N, F)
        self.y = np.load(y_path)  # shape: (N, A)
        assert len(self.X) == len(self.y), "X and y must have same first dimension"

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x = torch.tensor(self.X[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x, y


class DistanceRegressor(nn.Module):
    """Feedforward network for multi-output distance regression."""

    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super().__init__()
        layers = []
        in_feat = input_size
        for i in range(num_layers):
            layers.append(nn.Linear(in_feat, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_feat = hidden_size
        layers.append(nn.Linear(hidden_size, output_size))  # output_size = num_anchors
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # shape: (batch, output_size)


def train_and_validate(args):
    # Load data
    train_ds = CSIDataset(args.X_train, args.y_train)
    val_ds = CSIDataset(args.X_val, args.y_val)
    logging.info(f"Train samples = {len(train_ds)}, Val samples = {len(val_ds)}")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Determine sizes
    sample_x, sample_y = train_ds[0]
    input_size = sample_x.shape[0]
    output_size = sample_y.shape[0]
    logging.info(f"Input features: {input_size}, Output dims: {output_size}")

    # Model
    model = DistanceRegressor(
        input_size=input_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        output_size=output_size,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    best_mae = float("inf")

    # Training
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            preds = model(x_batch)  # shape (batch, output_size)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        # Validation
        model.eval()
        val_losses = []
        all_preds, all_truths = [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                preds = model(x_batch)
                val_losses.append(criterion(preds, y_batch).item())
                all_preds.append(preds.cpu().numpy())
                all_truths.append(y_batch.cpu().numpy())
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        preds_arr = np.vstack(all_preds)
        truths_arr = np.vstack(all_truths)
        mae = mean_absolute_error(truths_arr, preds_arr)
        rmse = np.sqrt(mean_squared_error(truths_arr, preds_arr))
        logging.info(
            f"Epoch {epoch}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Val MAE: {mae:.4f} | Val RMSE: {rmse:.4f}"
        )
        # Checkpoint
        if mae < best_mae:
            best_mae = mae
            os.makedirs(os.path.dirname(args.checkpoint) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.checkpoint)
            logging.info(f"Saved best model to {args.checkpoint} (MAE {mae:.4f})")
    logging.info("Training complete.")


def parse_args():
    p = argparse.ArgumentParser(description="Distance LSTM/FNN Regressor")
    p.add_argument("--X_train", required=True)
    p.add_argument("--y_train", required=True)
    p.add_argument("--X_val", required=True)
    p.add_argument("--y_val", required=True)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-size", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--checkpoint", required=True)
    args = p.parse_args()
    # Validate
    for path in [args.X_train, args.y_train, args.X_val, args.y_val]:
        if not os.path.isfile(path):
            p.error(f"File not found: {path}")
    return args


if __name__ == "__main__":
    configure_logging()
    args = parse_args()
    train_and_validate(args)
