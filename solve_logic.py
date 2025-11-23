import numpy as np
import mc_network
import math
import struct
import time

print("--- 7-Segment Logic Solver (Ridge Regression) ---")

# 1. DATA (Truth Table 0-F)
# Inputs: D C B A
X_raw = np.array([
    [-1, -1, -1, -1], [-1, -1, -1,  1], [-1, -1,  1, -1], [-1, -1,  1,  1],
    [-1,  1, -1, -1], [-1,  1, -1,  1], [-1,  1,  1, -1], [-1,  1,  1,  1],
    [ 1, -1, -1, -1], [ 1, -1, -1,  1], [ 1, -1,  1, -1], [ 1, -1,  1,  1],
    [ 1,  1, -1, -1], [ 1,  1, -1,  1], [ 1,  1,  1, -1], [ 1,  1,  1,  1]
], dtype=np.float32)

# ADD BIAS COLUMN (Critical for polynomial flexibility)
# New shape: (16, 5)
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

N_SAMPLES, D = X.shape # D is 5 now
N = 4 
M = math.comb(D + N - 1, N)

print(f"Truth Table: 16 rows. Model: {M} Params (Degree {N}, Inputs {D}).")

def load_pascal(filename="pascal.bin"):
    with open(filename, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    return np.array(struct.unpack(f'{count}I', data), dtype=np.uint32)

pascal_table = load_pascal()

# 2. GENERATE FEATURE MATRIX (H) ON GPU
# We probe the engine to fill H
print("\n[Metal] Generating Feature Matrix (H)...")
start_h = time.time()

H = np.zeros((N_SAMPLES, M), dtype=np.float32)
mc_network.reset_gpu()
probe_coeffs = np.zeros(M, dtype=np.float32)

# For M=70, we can iterate. For M=1M, we would tile this.
for j in range(M):
    probe_coeffs[j] = 1.0
    if j > 0: probe_coeffs[j-1] = 0.0
    
    # The engine computes: Sum(w_i * feature_i). 
    # Since only w_j = 1, result = feature_j.
    feature_col = mc_network.predict(X.flatten(), probe_coeffs, D, N, pascal_table)
    H[:, j] = feature_col

print(f"Matrix Generation: {time.time() - start_h:.4f}s")
print(f"Feature Matrix Shape: {H.shape}")

# 3. RIDGE REGRESSION SOLVER
# Solves (H.T @ H + alpha * I) w = H.T @ y
def solve_ridge(H, y, alpha=1e-4):
    dim = H.shape[1]
    
    # Normal Equation: LHS = H^T * H
    A = H.T @ H
    
    # Regularization: Add alpha to diagonal
    # This ensures the matrix is invertible even if features are correlated
    A.flat[::dim + 1] += alpha 
    
    # RHS = H^T * y
    b = H.T @ y
    
    try:
        # Cholesky solve (Fast & Stable for positive-definite matrices)
        w = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        # Fallback
        print("   > Warning: Singular matrix, using pseudo-inverse.")
        w = np.linalg.lstsq(A, b, rcond=None)[0]
        
    return w.astype(np.float32)

# 4. SOLVE ALL SEGMENTS
seg_weights = {}
print("\n[Algebra] Solving Logic Gates (Ridge)...")

for seg_name, target_list in segments.items():
    y = np.array(target_list, dtype=np.float32)
    
    # Instant Solve
    w = solve_ridge(H, y, alpha=1e-5)
    
    # Sparsify (Clean up floating point noise)
    w[np.abs(w) < 0.001] = 0.0
    
    seg_weights[seg_name] = w
    
    # Verify
    preds = mc_network.predict(X.flatten(), w, D, N, pascal_table)
    correct = np.sum(np.sign(preds) == np.sign(y))
    print(f"Segment '{seg_name}': Solved {correct}/16")

# 5. DEMO
print("\n--- FINAL DISPLAY TEST ---")
print("Hex | D C B A | Display")
print("-" * 30)

hex_chars = "0123456789ABCDEF"
all_preds = {}
for seg in segments:
    all_preds[seg] = mc_network.predict(X.flatten(), seg_weights[seg], D, N, pascal_table)

for i in range(N_SAMPLES):
    d, c, b, a = [1 if x>0 else 0 for x in X_raw[i]]
    on_segs = ""
    for seg in ['a','b','c','d','e','f','g']:
        # Use a slight tolerance for float comparison
        if all_preds[seg][i] > 0.0: on_segs += seg
    print(f"  {hex_chars[i]} | {d} {c} {b} {a} | {on_segs}")
print("-" * 30)