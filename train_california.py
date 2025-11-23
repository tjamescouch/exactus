import numpy as np
import mc_network
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import math
import struct
import time

print("--- California Housing (Single Layer) ---")

# 1. DATA
housing = fetch_california_housing()
X, y = housing.data, housing.target.astype(np.float32)

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
X = np.hstack([X, np.ones((X.shape[0], 1))]).astype(np.float32)

N_SAMPLES, D = X.shape
N = 4
M = math.comb(D + N - 1, N)
print(f"Data: {N_SAMPLES} samples. Model: {M} Params.")

# 2. SETUP
X_flat = X.flatten()

def load_pascal(filename="pascal.bin"):
    with open(filename, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    return np.array(struct.unpack(f'{count}I', data), dtype=np.uint32)

pascal_table = load_pascal()
coefficients = np.zeros(M, dtype=np.float32)

# 3. CONFIG
learning_rate = 1e-4
epochs = 1000
batch_size = 32

# 4. TRAIN
print("\n[Python] Calling Metal Engine...")
mc_network.reset_gpu()
start = time.time()

# Pass explicit Degree N=4
raw_coeffs = mc_network.fit(
    X_flat, 
    y, 
    coefficients, 
    learning_rate, 
    epochs, 
    batch_size, 
    N, # Degree
    pascal_table
)
coefficients = np.array(raw_coeffs, dtype=np.float32)

end = time.time()
print(f"[Python] Training took {end - start:.4f} seconds.")

# 5. VALIDATE
print("\n[Python] Validating...")
# Fast Batch Inference
y_pred = mc_network.predict(X_flat, coefficients, D, N, pascal_table)
mae = mean_absolute_error(y, y_pred)

print(f"Validation MAE: {mae:.4f}")