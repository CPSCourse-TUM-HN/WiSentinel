#!/usr/bin/env python3
"""
models/train.py

Batch training and cross-validation for multiple CSI-ranging models.
Supports LSTM, CNN-RNN, Sensor-Enhanced, and Adversarial architectures.

Usage:
  python models/train.py \
    --features X.npy --labels y.npy \
    --model-types lstm cnn_rnn sen adv \
    --kfolds 5 \
    --epochs 50 --batch-size 32 --lr 1e-3 \
    --output-dir results/train

Outputs:
  - Saved checkpoints for each model on full data: {output_dir}/{model_type}_final.pth
  - JSON report of cross-val metrics: {output_dir}/cv_metrics.json
"""
import os
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Model factories: mapping names to (display_name, constructor)
<<<<<<<< HEAD:Localized_Engine/Trash/train.py
from distance_lstm import DistanceRegressor
from CNN_RNN import CNNRNNRegressor
from SEN_trainer import SENRegressor
from Adversarial import AdvRegressor
========
from localized_engine.engine.distance_lstm import DistanceRegressor
from localized_engine.engine.CNN_RNN import CNNRNNRegressor
from localized_engine.engine.SEN_trainer import SENRegressor
from localized_engine.engine.Adversarial import AdvRegressor
>>>>>>>> origin/second_prototype:src/localized_engine/engine/models/train.py


MODEL_FACTORIES = {
    'lstm': ('LSTM', DistanceRegressor),
    'cnn_rnn': ('CNN-RNN', CNNRNNRegressor),
    'sen': ('Sensor-Enhanced', SENRegressor),
    'adv': ('Adversarial', AdvRegressor),
}

class CSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        # Handle complex values by taking the absolute value
        x_data = self.X[idx]
        if np.iscomplexobj(x_data):
            # Convert complex values to magnitude (absolute value)
            x_data = np.abs(x_data)
            # Ensure no NaN values
            x_data = np.nan_to_num(x_data, nan=0.0)

        # Ensure y values don't have NaNs
        y_data = self.y[idx]
        y_data = np.nan_to_num(y_data, nan=0.0)

        return torch.tensor(x_data, dtype=torch.float32), torch.tensor(y_data, dtype=torch.float32)


def train_one(model, train_loader, criterion, optimizer, device):
    model.train()

    losses = []
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return np.mean(losses)


def eval_one(model, val_loader, criterion, device):
    model.eval()
    losses, preds, truths = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            out = model(X_batch)
            losses.append(criterion(out, y_batch).item())
            preds.append(out.cpu().numpy())
            truths.append(y_batch.cpu().numpy())
    loss = np.mean(losses)
    preds = np.vstack(preds)
    truths = np.vstack(truths)

    # Handle NaN values before calculating metrics
    nan_mask = np.isnan(preds) | np.isnan(truths)
    if np.any(nan_mask):
        print(f"Warning: Found {np.sum(nan_mask)} NaN values in predictions or ground truth. Removing affected samples.")
        # Create a mask for rows that contain any NaN values
        row_has_nan = np.any(nan_mask, axis=1)
        # Filter out rows with NaN values
        if np.all(row_has_nan):
            print("Error: All predictions contain NaN values. Cannot calculate metrics.")
            return loss, float('nan'), float('nan')
        preds = preds[~row_has_nan]
        truths = truths[~row_has_nan]
        print(f"Remaining samples for evaluation: {len(preds)}/{len(row_has_nan)}")

    mae = mean_absolute_error(truths, preds)
    rmse = np.sqrt(mean_squared_error(truths, preds))
    return loss, mae, rmse


def main():
    parser = argparse.ArgumentParser(description="Batch train CSI-ranging models with CV")
    parser.add_argument('--features', required=True)
    parser.add_argument('--labels',   required=True)
    parser.add_argument('--model-types', nargs='+', default=['lstm'], choices=MODEL_FACTORIES.keys())
    parser.add_argument('--kfolds', type=int, default=5)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Load data
    X = np.load(args.features)
    y = np.load(args.labels)

    # Check for NaN or Inf values in the loaded data
    if np.any(np.isnan(X)) or np.any(np.isinf(X)):
        print(f"Warning: Input features contain {np.sum(np.isnan(X))} NaN values and {np.sum(np.isinf(X))} Inf values.")
        print("Replacing NaN and Inf values with zeros.")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    if np.any(np.isnan(y)) or np.any(np.isinf(y)):
        print(f"Warning: Labels contain {np.sum(np.isnan(y))} NaN values and {np.sum(np.isinf(y))} Inf values.")
        print("Replacing NaN and Inf values with zeros.")
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

    # Check if input features are complex
    if np.iscomplexobj(X):
        print("Warning: Input features contain complex values. Converting to magnitude.")
        X = np.abs(X)

    P, F = X.shape
    # Determine output dims
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    _, D = y.shape

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cv_metrics = {}
    # Adjust CV splits to not exceed number of samples
    n_samples = X.shape[0]
    n_splits = min(args.kfolds, n_samples) if n_samples >= 2 else 0
    if n_splits >= 2:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        kf = None  # Not enough samples for CV


    for mtype in args.model_types:
        disp, constructor = MODEL_FACTORIES[mtype]
        if constructor is None:
            print(f"Skipping {mtype}: constructor not available")
            continue
        print(f"=== Training model: {disp} ({mtype}) ===")
        # If CV unavailable, train only final model
        if kf is None:
            full_ds = CSIDataset(X, y)
            full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True)
            final_model = constructor(input_size=F, hidden_size=64, num_layers=2, dropout=0.5, output_size=D)
            final_model.to(device)
            optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr)
            criterion = nn.MSELoss()
            print("Training on full dataset only...")
            for epoch in range(1, args.epochs+1):
                train_loss = train_one(final_model, full_loader, criterion, optimizer, device)
            ckpt_path = os.path.join(args.output_dir, f"{mtype}_final.pth")
            torch.save(final_model.state_dict(), ckpt_path)
            print(f"Saved model to {ckpt_path}")
            continue
        disp, constructor = MODEL_FACTORIES[mtype]
        if constructor is None:
            print(f"Skipping {mtype}: constructor not available")
            continue
        print(f"=== Training model: {disp} ({mtype}) ===")
        fold_maes, fold_rmses = [], []
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            print(f"  Fold {fold+1}/{args.kfolds}...")
            X_tr, X_va = X[train_idx], X[val_idx]
            y_tr, y_va = y[train_idx], y[val_idx]
            train_ds = CSIDataset(X_tr, y_tr)
            val_ds   = CSIDataset(X_va, y_va)
            train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
            val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False)

            # Instantiate model
            model = constructor(input_size=F, hidden_size=64, num_layers=2, dropout=0.5, output_size=D)
            model.to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
            criterion = nn.MSELoss()

            # Train
            for epoch in range(1, args.epochs + 1):
                # DEBUG: show real-time progress for CNN-RNN
                print(f"[{disp}] Epoch {epoch}/{args.epochs}", end='\\r', flush=True)
                train_loss = train_one(model, train_loader, criterion, optimizer, device)
            # Evaluate
            val_loss, val_mae, val_rmse = eval_one(model, val_loader, criterion, device)
            print(f"    Val MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}")
            fold_maes.append(val_mae)
            fold_rmses.append(val_rmse)

        # Aggregate CV metrics
        cv_metrics[mtype] = {
            'mae_mean': float(np.mean(fold_maes)),
            'mae_std':  float(np.std(fold_maes)),
            'rmse_mean':float(np.mean(fold_rmses)),
            'rmse_std': float(np.std(fold_rmses)),
        }
        # Train final model on full dataset
        print(f"Training final {disp} on full data...")
        full_ds = CSIDataset(X, y)
        full_loader = DataLoader(full_ds, batch_size=args.batch_size, shuffle=True)
        final_model = constructor(input_size=F, hidden_size=64, num_layers=2, dropout=0.5, output_size=D)
        final_model.to(device)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=args.lr)
        criterion = nn.MSELoss()
        for epoch in range(1, args.epochs+1):
            train_one(final_model, full_loader, criterion, optimizer, device)
        ckpt_path = os.path.join(args.output_dir, f"{mtype}_final.pth")
        torch.save(final_model.state_dict(), ckpt_path)
        print(f"Saved final model to {ckpt_path}")

    # Save CV report
    report_path = os.path.join(args.output_dir, 'cv_metrics.json')
    with open(report_path, 'w') as f:
        json.dump(cv_metrics, f, indent=2)
    print(f"Cross-validation metrics saved to {report_path}")

if __name__ == '__main__':
    main()
