import numpy as np
import mc_network
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import math
import struct
import time

# --- 1. LOAD MNIST ---
print("Loading MNIST...")
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X_raw = mnist.data.astype(np.float32)
y_raw = mnist.target.astype(np.int32)

scaler = MinMaxScaler(feature_range=(-1, 1))
X_raw = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X_raw, y_raw, test_size=10000, random_state=42)

N_SAMPLES, D = X_train.shape
N_DEGREE = 2 
M = math.comb(D + N_DEGREE - 1, N_DEGREE)

print(f"Training Set: {N_SAMPLES} samples. Model: {M} Params (Degree {N_DEGREE}).")

# --- 2. SETUP ---
def load_pascal(filename="pascal.bin"):
    with open(filename, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    return np.array(struct.unpack(f'{count}I', data), dtype=np.uint32)

pascal_table = load_pascal()

# --- 3. TUNING ---
# Dense Mode updates 300k weights at once. 
# Safety Limit = 0.1 / (Batch * M)
# Batch 128 * 300,000 = 38,000,000. 
# Max LR approx 2.0e-9.
learning_rate = 5e-9 
epochs = 5 
batch_size = 128

print(f"Config: LR={learning_rate}, Batch={batch_size}")

# --- 4. TRAINING ---
digit_weights = []
total_start = time.time()

for digit in range(10):
    print(f"--- Digit {digit} ---")
    
    # One-vs-All Target
    y_binary = np.where(y_train == digit, 1.0, -1.0).astype(np.float32)
    X_flat = X_train.flatten()
    
    coeffs = np.zeros(M, dtype=np.float32)
    mc_network.reset_gpu()
    
    start = time.time()
    raw_coeffs = mc_network.fit(
        X_flat, 
        y_binary, 
        coeffs, 
        learning_rate, 
        epochs, 
        batch_size, 
        N_DEGREE,
        pascal_table
    )
    dt = time.time() - start
    
    trained_coeffs = np.array(raw_coeffs, dtype=np.float32)
    digit_weights.append(trained_coeffs)
    
    # Check Magnitude
    mag = np.sum(np.abs(trained_coeffs))
    print(f"Finished in {dt:.2f}s. Weight Mag: {mag:.2f}")

print(f"Total Training Time: {time.time() - total_start:.2f}s")

# --- 5. INFERENCE ---
print("\n[MNIST] Inference on Test Set...")
test_scores = np.zeros((10, len(y_test)), dtype=np.float32)
X_test_flat = X_test.flatten()

for digit in range(10):
    # Pass Explicit D and N
    scores = mc_network.predict(X_test_flat, digit_weights[digit], D, N_DEGREE, pascal_table)
    test_scores[digit] = scores

final_predictions = np.argmax(test_scores, axis=0)
accuracy = accuracy_score(y_test, final_predictions)

print(f"==============================")
print(f"FINAL MNIST ACCURACY: {accuracy*100:.2f}%")
print(f"==============================")