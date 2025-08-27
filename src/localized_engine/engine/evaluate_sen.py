#!/usr/bin/env python3
"""
evaluate_sen.py

Batch evaluation of the Sensor-Enhanced (SEN) model on the WIDAR dataset.
Computes coordinate MAE/RMSE and true spatial MAE/RMSE.
"""

import numpy as np
import torch
from SEN_trainer import SENRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


def main():
    # 1) Load features & labels
    X = np.load("/data_files/phase2/X.npy")  # shape: (N, F)
    true_xy = np.load("/data_files/phase2/y.npy")  # shape: (N, 2)

    # 2) Build SEN model & load checkpoint
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SENRegressor(
        input_size=X.shape[1],
        hidden_size=64,  # adjust if you tuned
        num_layers=2,
        dropout=0.5,
        output_size=2,  # x and y
    ).to(device)
    checkpoint = (
        "/Users/shivamsingh/PycharmProjects/WiSentinel/results/train/sen_final.pth"
    )
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.eval()

    # 3) Run predictions
    preds = np.zeros_like(true_xy)
    bs = 128
    with torch.no_grad():
        for i in range(0, len(X), bs):
            xb = torch.tensor(X[i : i + bs], dtype=torch.float32).to(device)
            out = model(xb).cpu().numpy()  # shape: (b, 2)
            preds[i : i + bs] = out

    # 4) Evaluate coordinate error
    coord_mae = mean_absolute_error(true_xy, preds)
    coord_rmse = np.sqrt(mean_squared_error(true_xy, preds))

    # 5) Evaluate true spatial error
    dist_errors = np.linalg.norm(preds - true_xy, axis=1)
    pos_mae = dist_errors.mean()
    pos_rmse = np.sqrt((dist_errors**2).mean())

    # 6) Print results
    print(f"SEN model — coordinate MAE/RMSE: {coord_mae:.3f}/{coord_rmse:.3f}")
    print(f"SEN model — position MAE/RMSE:   {pos_mae:.3f}/{pos_rmse:.3f} meters")


if __name__ == "__main__":
    main()
