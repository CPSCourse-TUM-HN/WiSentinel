import csv
import numpy as np

# Parameters: adjust these to your apartment
x_max = 6.0  # total width in meters
y_max = 4.0  # total depth in meters
spacing = 1.0  # grid spacing in meters

# Compute grid coordinates
x_coords = np.arange(0, x_max + 0.001, spacing)
y_coords = np.arange(0, y_max + 0.001, spacing)

# Write CSV
with open("../Localized_Engine/grid_points.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["x", "y", "point_name"])
    for j, y in enumerate(y_coords):
        for i, x in enumerate(x_coords):
            writer.writerow([f"{x:.1f}", f"{y:.1f}", f"pt_{i}{j}"])
print("Wrote grid_points.csv with", len(x_coords) * len(y_coords), "points.")
