import numpy as np
import mc_network
import math
import struct
import time

print("--- Learning Hardware: Full Adder ---")

# 1. DATA (Full Adder Truth Table)
# Inputs: A, B, Cin
# Logic: -1 = 0, 1 = 1
X = np.array([
    [-1, -1, -1], # 0 0 0
    [-1, -1,  1], # 0 0 1
    [-1,  1, -1], # 0 1 0
    [-1,  1,  1], # 0 1 1
    [ 1, -1, -1], # 1 0 0
    [ 1, -1,  1], # 1 0 1
    [ 1,  1, -1], # 1 1 0
    [ 1,  1,  1]  # 1 1 1
], dtype=np.float32)

# Targets: [Sum, Cout]
# Sum  = A ^ B ^ Cin (Odd parity is 1)
# Cout = Majority (At least two 1s)
# We train two separate polynomials (one per output bit)
Y_sum  = np.array([-1,  1,  1, -1,  1, -1, -1,  1], dtype=np.float32)
Y_cout = np.array([-1, -1, -1,  1, -1,  1,  1,  1], dtype=np.float32)

N_SAMPLES, D = X.shape
N = 3 # Degree 3 needed for Sum (x1*x2*x3 interaction)
M = math.comb(D + N - 1, N)

print(f"Data: {N_SAMPLES} samples. Model: {M} Params (Degree {N}).")

# 2. SETUP
def load_pascal(filename="pascal.bin"):
    with open(filename, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    return np.array(struct.unpack(f'{count}I', data), dtype=np.uint32)

pascal_table = load_pascal()

# Config
learning_rate = 0.05 
epochs = 1000
batch_size = 1 # Stochastic works best for logic

# 3. TRAIN LOOP (Function to train one bit)
def train_bit(target_y, bit_name):
    print(f"\n>>> Training {bit_name} Bit...")
    mc_network.reset_gpu()
    
    # Random init to break symmetry
    coeffs = np.random.uniform(-0.1, 0.1, M).astype(np.float32)
    
    start = time.time()
    raw_coeffs = mc_network.fit(
        X.flatten(), 
        target_y, 
        coeffs, 
        learning_rate, 
        epochs, 
        batch_size, 
        N, 
        pascal_table
    )
    dt = time.time() - start
    print(f"    Finished in {dt:.4f}s.")
    return np.array(raw_coeffs, dtype=np.float32)

# Train both outputs
w_sum = train_bit(Y_sum, "SUM")
w_cout = train_bit(Y_cout, "CARRY OUT")

# 4. VERIFY
print("\n--- Full Adder Verification ---")
print(f" A  B  C | {'Sum':<4} {'(Pred)':<6} | {'Cout':<4} {'(Pred)':<6} | Result")
print("-" * 55)

pred_sum  = mc_network.predict(X.flatten(), w_sum, D, N, pascal_table)
pred_cout = mc_network.predict(X.flatten(), w_cout, D, N, pascal_table)

passes = 0
for i in range(N_SAMPLES):
    # Logic Check
    s_pass = (np.sign(pred_sum[i]) == np.sign(Y_sum[i]))
    c_pass = (np.sign(pred_cout[i]) == np.sign(Y_cout[i]))
    
    status = "PASS" if (s_pass and c_pass) else "FAIL"
    if status == "PASS": passes += 1
    
    # Format inputs for readability (-1 -> 0, 1 -> 1)
    a, b, c = [int(x > 0) for x in X[i]]
    tgt_s = int(Y_sum[i] > 0)
    tgt_c = int(Y_cout[i] > 0)
    
    print(f" {a}  {b}  {c} | {tgt_s:<4} {pred_sum[i]:6.2f} | {tgt_c:<4} {pred_cout[i]:6.2f} | {status}")

print(f"\nScore: {passes}/{N_SAMPLES}")

# 5. INSPECT SUM LOGIC
# Sum should be dominated by the x1*x2*x3 term (Index 0 in co-lex usually)
print("\nDominant Terms for SUM (Expected: Interaction x1*x2*x3):")
indices = np.argsort(-np.abs(w_sum))[:3]
for idx in indices:
    print(f"Idx {idx}: {w_sum[idx]:.4f}")