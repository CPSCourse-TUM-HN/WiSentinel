from scipy.io import loadmat

# 1. Point this to your calibration .mat
mat = loadmat("/data_files/csi_src_test.mat")

# 2. Find the CSI variable and print its shape
for var_name, arr in mat.items():
    if not var_name.startswith("__"):
        print(f"Variable '{var_name}': shape = {arr.shape}")
