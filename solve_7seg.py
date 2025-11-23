import numpy as np
import mc_network
import math
import struct
import time
from numpy.linalg import lstsq

print("--- 7-Segment Hardware Solver (Algebraic + Bias) ---")

# 1. DATA (Truth Table 0-F)
X_raw = np.array([
    [-1, -1, -1, -1], [-1, -1, -1,  1], [-1, -1,  1, -1], [-1, -1,  1,  1],
    [-1,  1, -1, -1], [-1,  1, -1,  1], [-1,  1,  1, -1], [-1,  1,  1,  1],
    [ 1, -1, -1, -1], [ 1, -1, -1,  1], [ 1, -1,  1, -1], [ 1, -1,  1,  1],
    [ 1,  1, -1, -1], [ 1,  1, -1,  1], [ 1,  1,  1, -1], [ 1,  1,  1,  1]
], dtype=np.float32)

# CRITICAL FIX: Add Bias Column to allow Odd-Degree terms
# Input becomes 5D: [D, C, B, A, 1.0]
X = np.hstack([X_raw, np.ones((16, 1), dtype=np.float32)])

segments = {
    'a': [ 1, -1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1],
    'b': [ 1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1, -1, -1],
    'c': [ 1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1, -1, -1],
    'd': [ 1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1],
    'e': [ 1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1,  1,  1],
    'f': [ 1, -1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1],
    'g': [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1,  1]
}

N_SAMPLES, D = X.shape # D is now 5
N = 4 
M = math.comb(D + N - 1, N)

print(f"Truth Table: 16 rows. Model: {M} Params (Degree {N}, Inputs {D}).")

def load_pascal(filename="pascal.bin"):
    with open(filename, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    return np.array(struct.unpack(f'{count}I', data), dtype=np.uint32)

pascal_table = load_pascal()

# 2. GENERATE FEATURE MATRIX (H)
print("\n[Metal] Generating Feature Matrix (H)...")
H = np.zeros((N_SAMPLES, M), dtype=np.float32)

mc_network.reset_gpu()
probe_coeffs = np.zeros(M, dtype=np.float32)

# Extract features column by column
# (Note: For larger M, we would do this in batches or via a dedicated C++ kernel,
# but for M=70 it's instant)
for j in range(M):
    probe_coeffs[j] = 1.0
    if j > 0: probe_coeffs[j-1] = 0.0
    
    feature_col = mc_network.predict(X.flatten(), probe_coeffs, D, N, pascal_table)
    H[:, j] = feature_col

print(f"Feature Matrix H Shape: {H.shape} (Rank: {np.linalg.matrix_rank(H)})")

# 3. SOLVE
seg_weights = {}
print("\n[Algebra] Solving Logic Gates...")

for seg_name, target_list in segments.items():
    y = np.array(target_list, dtype=np.float32)
    
    # Least Squares
    w, residuals, rank, s = lstsq(H, y, rcond=None)
    
    # Snap to grid to clean up float noise
    w[np.abs(w) < 0.001] = 0
    
    seg_weights[seg_name] = w.astype(np.float32)
    
    # Verify
    preds = mc_network.predict(X.flatten(), seg_weights[seg_name], D, N, pascal_table)
    correct = np.sum(np.sign(preds) == np.sign(y))
    print(f"Segment '{seg_name}': Solved {correct}/16")

# 4. DEMO
print("\n--- FINAL DISPLAY TEST ---")
print("Hex | D C B A | Display")
print("-" * 30)

hex_chars = "0123456789ABCDEF"
all_preds = {}
for seg in segments:
    all_preds[seg] = mc_network.predict(X.flatten(), seg_weights[seg], D, N, pascal_table)

for i in range(N_SAMPLES):
    d, c, b, a = [1 if x>0 else 0 for x in X_raw[i]] # Use raw input for display
    on_segs = ""
    for seg in ['a','b','c','d','e','f','g']:
        if all_preds[seg][i] > 0: on_segs += seg
    print(f"  {hex_chars[i]} | {d} {c} {b} {a} | {on_segs}")
print("-" * 30)