import numpy as np
import mc_network
import math
import struct
import time

print("--- Learning Logic Gates (XOR) ---")

# 1. DATA (Raw XOR)
# NO BIAS COLUMN. We rely on x^2 terms becoming constants (1.0) to act as bias.
X = np.array([
    [-1.0, -1.0],
    [-1.0,  1.0],
    [ 1.0, -1.0],
    [ 1.0,  1.0]
], dtype=np.float32)

y = np.array([-1.0, 1.0, 1.0, -1.0], dtype=np.float32)

N_SAMPLES, D = X.shape
N = 2 # Degree 2 gives us terms: x1^2, x1*x2, x2^2
M = math.comb(D + N - 1, N)
X_flat = X.flatten()

print(f"Data: {N_SAMPLES} samples. Model: {M} Params (Degree {N}).")
# Expected Features:
# x1^2 (Constant 1), x1*x2 (The Answer), x2^2 (Constant 1)

# 2. SETUP
def load_pascal(filename="pascal.bin"):
    with open(filename, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    return np.array(struct.unpack(f'{count}I', data), dtype=np.uint32)

pascal_table = load_pascal()

# Random init is crucial to break symmetry between the two constant terms
coefficients = np.random.uniform(-0.1, 0.1, M).astype(np.float32)

# TUNING
learning_rate = 0.05 # Aggressive LR for simple logic
epochs = 500
batch_size = 1 # STOCHASTIC MODE: Force it to learn row-by-row

# 3. TRAIN
print("Calling Metal Engine...")
mc_network.reset_gpu()
start = time.time()

raw_coeffs = mc_network.fit(
    X_flat, 
    y, 
    coefficients, 
    learning_rate, 
    epochs, 
    batch_size,
    N,
    pascal_table
)
coefficients = np.array(raw_coeffs, dtype=np.float32)

end = time.time()
print(f"Training Complete in {end - start:.4f} seconds.")

# 4. VERIFY
print("\n--- Truth Table Verification ---")
print(f"{'A':>4} {'B':>4} | {'Pred':>6} | {'Target':>6} | {'Logic'}")
print("-" * 35)

preds = mc_network.predict(X_flat, coefficients, D, N, pascal_table)

for i in range(N_SAMPLES):
    pred = preds[i]
    target = y[i]
    logic_state = "PASS" if (np.sign(pred) == np.sign(target)) else "FAIL"
    print(f"{X[i][0]:4.1f} {X[i][1]:4.1f} | {pred:6.3f} | {target:6.1f} | {logic_state}")

# 5. INSPECT WEIGHTS
# We expect the interaction term (x1*x2) to be roughly -1.0
# The bias terms (x1^2, x2^2) should sum to roughly 0.0
print("\nLearned Polynomial Coefficients:")
for i, w in enumerate(coefficients):
    print(f"Index {i}: {w:.4f}")