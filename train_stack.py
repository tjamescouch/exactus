import numpy as np
import mc_network
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error
import math
import struct
import time
import sys

class PolynomialStack:
    def __init__(self, M, D, degree=4, layers=5, target_lr=1e-4, epochs=500, batch_size=32): 
        self.M = M
        self.D = D
        self.N = degree
        self.num_layers = layers
        self.batch_size = batch_size
        
        # Physics-based Safety Limit for LR
        # (Prevent weights from exploding in a single update)
        physics_limit = 0.1 / (batch_size * M)
        self.lr = min(target_lr, physics_limit)
        
        print(f"[Stack] Batch Size: {batch_size}")
        print(f"[Stack] Target LR: {target_lr:.1e} | Safety Limit: {physics_limit:.1e}")
        print(f"[Stack] Active LR: {self.lr:.1e}")
        
        self.epochs = epochs
        self.nets = [] 
        self.pascal_table = self._load_pascal()
        
    def _load_pascal(self, filename="pascal.bin"):
        with open(filename, 'rb') as f:
            data = f.read()
        count = len(data) // 4
        # CRITICAL: Explicitly cast to uint32 for C++
        return np.array(struct.unpack(f'{count}I', data), dtype=np.uint32)

    def fit(self, X_flat, y):
        # Sanity Check
        if np.isnan(X_flat).any() or np.isinf(X_flat).any():
            print("!!! ERROR: Input X contains NaNs or Infs!")
            return
        if np.isnan(y).any() or np.isinf(y).any():
            print("!!! ERROR: Target y contains NaNs or Infs!")
            return

        current_target = y.copy()
        
        print(f"\n[Stack] Training {self.num_layers} Layers (Degree {self.N})...")
        
        for i in range(self.num_layers):
            print(f"\n--- Layer {i} ---")
            
            # Layer 0 gets full rate, subsequent layers get 50% to refine
            layer_lr = self.lr if i == 0 else self.lr * 0.5
            
            # Retry Loop for Stability
            attempt = 0
            success = False
            
            while not success and attempt < 3:
                if attempt > 0:
                    print(f"   > Retry {attempt}. Reducing LR to {layer_lr:.1e}...")
                
                mc_network.reset_gpu()
                coeffs = np.zeros(self.M, dtype=np.float32)
                
                start = time.time()
                
                # TRAIN (C++ returns vector<float>)
                raw_coeffs = mc_network.fit(
                    X_flat, 
                    current_target, 
                    coeffs, 
                    layer_lr, 
                    self.epochs, 
                    self.batch_size, 
                    self.N, # Degree
                    self.pascal_table 
                )
                dt = time.time() - start
                
                coeffs = np.array(raw_coeffs, dtype=np.float32)
                weight_mag = np.sum(np.abs(coeffs))
                
                # Fallback Checks
                if np.isnan(weight_mag) or np.isinf(weight_mag):
                    print(f"   > Layer exploded (Nan). Retrying with Batch=1...")
                    mc_network.reset_gpu()
                    coeffs = np.zeros(self.M, dtype=np.float32)
                    
                    # Emergency Fit (Serial)
                    raw_coeffs = mc_network.fit(
                        X_flat, 
                        current_target, 
                        coeffs, 
                        layer_lr * 0.1, 
                        self.epochs, 
                        1, # Batch 1
                        self.N,
                        self.pascal_table
                    )
                    coeffs = np.array(raw_coeffs, dtype=np.float32)
                    weight_mag = np.sum(np.abs(coeffs))
                    
                    if np.isnan(weight_mag):
                        print("   > Critical Failure even at Batch=1.")
                        attempt += 1
                        continue

                if weight_mag == 0.0:
                    print(f"   > Layer failed (Weights 0).")
                    attempt += 1
                    continue
                    
                # Success
                success = True
                self.nets.append(coeffs)
                
                # PREDICT (GPU Batch Inference)
                # Update residual: Target = Target - Prediction
                layer_pred = np.array(mc_network.predict(X_flat, coeffs, self.D, self.N, self.pascal_table))
                
                old_mae = np.mean(np.abs(current_target))
                current_target = current_target - layer_pred
                new_mae = np.mean(np.abs(current_target))
                
                print(f"Layer {i} Finished in {dt:.2f}s | Mag: {weight_mag:.2f}")
                print(f"Residual MAE: {old_mae:.4f} -> {new_mae:.4f}")

            if not success:
                print("!!! CRITICAL: Layer failed 3 times. Stopping Stack training.")
                break

    def predict(self, X_flat, n_samples):
        print(f"\n[Stack] Inference on {n_samples} samples...")
        final_pred = np.zeros(n_samples, dtype=np.float32)
        
        for i, coeffs in enumerate(self.nets):
            # Sum predictions from all layers
            layer_pred = np.array(mc_network.predict(X_flat, coeffs, self.D, self.N, self.pascal_table))
            final_pred += layer_pred
            
        return final_pred

# --- MAIN ---
print("Loading Data...")
housing = fetch_california_housing()
X, y = housing.data, housing.target.astype(np.float32)

# Scale Inputs (-1 to 1)
scaler = MinMaxScaler(feature_range=(-1, 1))
X = scaler.fit_transform(X)
X = np.hstack([X, np.ones((X.shape[0], 1))]).astype(np.float32)

N_SAMPLES, D = X.shape
N = 4
M = math.comb(D + N - 1, N)
X_flat = X.flatten()

print(f"Setup: {N_SAMPLES} samples. M={M}.")

# Use Batch 32 for safety, or 1024 for speed if stable
stack = PolynomialStack(M, D, degree=N, layers=10, target_lr=1e-4, epochs=1000, batch_size=32)
stack.fit(X_flat, y)

print("\n--- Final Evaluation ---")
y_pred = stack.predict(X_flat, N_SAMPLES)
mae = mean_absolute_error(y, y_pred)
print(f"Final Stack MAE: {mae:.4f}")