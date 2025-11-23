import numpy as np
import mc_network
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import math
import struct
import time

print("--- California Housing Turbo Training ---")

housing = fetch_california_housing()
X, y = housing.data, housing.target.astype(np.float32)

scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
X = np.hstack([X, np.ones((X.shape[0], 1))]).astype(np.float32)

N_SAMPLES, D = X.shape
N = 4
M = math.comb(D + N - 1, N)

X_flat = X.flatten()

def load_pascal(filename="pascal.bin"):
    with open(filename, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    return struct.unpack(f'{count}I', data)
pascal_table = load_pascal()

coefficients = np.zeros(M, dtype=np.float32)

# TUNING:
# D=9, N=4, Inputs < 1.0. 
# We need a higher LR to learn.
learning_rate = 1e-5 
epochs = 500

print(f"Data: {N_SAMPLES} samples. Model: {M} Params.")
print("Calling Metal Engine...")
mc_network.reset_gpu()
start = time.time()

mc_network.fit(
    X_flat, 
    y, 
    coefficients, 
    learning_rate, 
    epochs, 
    pascal_table
)

end = time.time()
print(f"Training Complete in {end - start:.4f} seconds.")

print("Validating...")
y_preds = []
CHECK_LIMIT = 2000
for i in range(CHECK_LIMIT):
    update_indices = np.arange(M, dtype=np.uint32)
    mc_network.process_step(X[i], float(y[i]), coefficients, 0.0, update_indices, pascal_table, False)
    y_preds.append(mc_network.get_debug_output()[0])

mae = mean_absolute_error(y[:CHECK_LIMIT], y_preds)
print(f"Validation MAE: {mae:.4f}")