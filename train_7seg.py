import numpy as np
import mc_network
import math
import struct
import time

print("--- Learning Hardware: 7-Segment Hex Decoder (Balanced) ---")

# 1. DATA
X = np.array([
    [-1, -1, -1, -1], [-1, -1, -1,  1], [-1, -1,  1, -1], [-1, -1,  1,  1],
    [-1,  1, -1, -1], [-1,  1, -1,  1], [-1,  1,  1, -1], [-1,  1,  1,  1],
    [ 1, -1, -1, -1], [ 1, -1, -1,  1], [ 1, -1,  1, -1], [ 1, -1,  1,  1],
    [ 1,  1, -1, -1], [ 1,  1, -1,  1], [ 1,  1,  1, -1], [ 1,  1,  1,  1]
], dtype=np.float32)

segments = {
    'a': [ 1, -1,  1,  1, -1,  1,  1,  1,  1,  1,  1, -1,  1, -1,  1,  1],
    'b': [ 1,  1,  1,  1,  1, -1, -1,  1,  1,  1,  1, -1, -1,  1, -1, -1],
    'c': [ 1,  1, -1,  1,  1,  1,  1,  1,  1,  1,  1,  1, -1,  1, -1, -1],
    'd': [ 1, -1,  1,  1, -1,  1,  1, -1,  1,  1, -1,  1,  1,  1,  1, -1],
    'e': [ 1, -1,  1, -1, -1, -1,  1, -1,  1, -1,  1,  1,  1,  1,  1,  1],
    'f': [ 1, -1, -1, -1,  1,  1,  1, -1,  1,  1,  1,  1,  1, -1,  1,  1],
    'g': [-1, -1,  1,  1,  1,  1,  1, -1,  1,  1,  1,  1, -1,  1,  1,  1]
}

N_SAMPLES, D = X.shape
N = 4 
M = math.comb(D + N - 1, N)

print(f"Truth Table: 16 rows. Model: {M} Params (Degree {N}).")

def load_pascal(filename="pascal.bin"):
    with open(filename, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    return np.array(struct.unpack(f'{count}I', data), dtype=np.uint32)

pascal_table = load_pascal()

# 2. BALANCED TRAINER
def train_segment_balanced(seg_name, raw_targets):
    # Calculate Balance
    # We want sum(pos) == sum(neg)
    # pos_weight * count_pos = neg_weight * count_neg
    
    pos_mask = raw_targets > 0
    neg_mask = raw_targets < 0
    count_pos = np.sum(pos_mask)
    count_neg = np.sum(neg_mask)
    
    # Base voltage
    V = 2.0 
    
    # Determine weights
    # If count is 0 (impossible here but safe), weight is 1
    w_pos = V * (count_neg / N_SAMPLES) if count_pos > 0 else 1.0
    w_neg = V * (count_pos / N_SAMPLES) if count_neg > 0 else 1.0
    
    # Normalize to keep gradients reasonable (approx magnitude 1-3)
    # Scale so max weight is V
    scale = V / max(w_pos, w_neg)
    w_pos *= scale
    w_neg *= scale
    
    # Apply weights to targets
    y_balanced = raw_targets.copy()
    y_balanced[pos_mask] *= w_pos
    y_balanced[neg_mask] *= w_neg # neg targets are already negative, so just scale magnitude
    
    # print(f"  > {seg_name}: Pos x{count_pos} (Target {w_pos:.2f}), Neg x{count_neg} (Target -{w_neg:.2f})")

    max_retries = 10
    
    for attempt in range(max_retries):
        # Tuning: Batch 4 is a good compromise for Logic
        lr = 0.005 
        epochs = 3000
        batch_size = 4 
        
        coeffs = np.random.uniform(-0.1, 0.1, M).astype(np.float32)
        mc_network.reset_gpu()
        
        raw_coeffs = mc_network.fit(
            X.flatten(), y_balanced, coeffs, 
            lr, epochs, batch_size, N, pascal_table
        )
        w = np.array(raw_coeffs, dtype=np.float32)
        
        # Verify against ORIGINAL raw targets (+/- 1)
        preds = mc_network.predict(X.flatten(), w, D, N, pascal_table)
        correct = np.sum(np.sign(preds) == np.sign(raw_targets))
        
        if correct == 16:
            print(f"Segment '{seg_name}' Solved! (Attempt {attempt+1})")
            return w
            
    print(f"!!! Failed '{seg_name}'. Best: {correct}/16")
    return w

# 3. MAIN LOOP
seg_weights = {}
print("\n--- Synthesizing Circuits (Balanced) ---")

for seg_name, target_list in segments.items():
    y = np.array(target_list, dtype=np.float32)
    seg_weights[seg_name] = train_segment_balanced(seg_name, y)

# 4. DEMO
print("\n--- FINAL DISPLAY TEST ---")
print("Hex | D C B A | Display")
print("-" * 30)

hex_chars = "0123456789ABCDEF"
all_preds = {}
for seg in segments:
    all_preds[seg] = mc_network.predict(X.flatten(), seg_weights[seg], D, N, pascal_table)

for i in range(N_SAMPLES):
    d, c, b, a = [1 if x>0 else 0 for x in X[i]]
    on_segs = ""
    for seg in ['a','b','c','d','e','f','g']:
        if all_preds[seg][i] > 0: on_segs += seg
    
    print(f"  {hex_chars[i]} | {d} {c} {b} {a} | {on_segs}")

print("-" * 30)