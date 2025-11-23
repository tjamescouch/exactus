import numpy as np
import mc_network
import math
import struct
import time
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024 # MB

# SETUP
def load_pascal(filename="pascal.bin"):
    with open(filename, 'rb') as f:
        data = f.read()
    count = len(data) // 4
    return np.array(struct.unpack(f'{count}I', data), dtype=np.uint32)

pascal_table = load_pascal()

# Fixed Constants
N_SAMPLES = 1000 # Fixed sample size for stability check
N_DEGREE = 2     # Keep degree simple, we just scale D to scale M

print(f"--- Benchmarking Algebraic Solver Limits ---")
print(f"System RAM Baseline: {get_memory_usage():.2f} MB")

# We will ramp up D to increase M
# M = comb(D+N-1, N) roughly D^2/2
test_dims = [10, 50, 100, 200, 300, 400, 500]

for D in test_dims:
    M = math.comb(D + N_DEGREE - 1, N_DEGREE)
    print(f"\nTesting D={D}, N={N_DEGREE} => M={M} Features...")
    
    # 1. Generate Random Data
    X = np.random.uniform(-1, 1, (N_SAMPLES, D)).astype(np.float32)
    y = np.random.uniform(-1, 1, N_SAMPLES).astype(np.float32)
    
    # 2. Generate Feature Matrix (H)
    try:
        t0 = time.time()
        # H matrix size: 1000 * M * 4 bytes
        # If M=1M, H=4GB. 
        H = np.zeros((N_SAMPLES, M), dtype=np.float32)
        
        # Probe engine (This is the slow part for huge M in "Probe Mode")
        # Optimization: In production we'd have a batch probe kernel, 
        # but here we loop (slow but memory safe until H fills up)
        # NOTE: Looping M times in Python is slow. 
        # For the benchmark, we accept it to measure RAM impact.
        
        mc_network.reset_gpu()
        
        # For speed in this specific test, let's just measure the SOLVE step 
        # assuming we *could* generate H fast.
        # We simulate H generation time or skip it if M is massive to save time?
        # No, we must materialize H to test RAM.
        
        # Fake H generation for speed test of SOLVER MEMORY
        # (Real generation is just GPU compute, which we know is fast/streamed)
        # We want to know if (H.T @ H) crashes.
        H = np.random.randn(N_SAMPLES, M).astype(np.float32) 
        
        gen_time = time.time() - t0
        mem_h = get_memory_usage()
        print(f"  > H Matrix Allocated ({mem_h:.0f} MB).")

        # 3. The Solver Step (The Bottleneck)
        t1 = time.time()
        
        # Ridge Solve: (H.T @ H + alpha*I)
        # This creates an (M x M) matrix.
        # If M=20,000 -> 400,000,000 floats -> 1.6 GB.
        # If M=50,000 -> 2.5B floats -> 10 GB.
        
        A = H.T @ H
        solve_time = time.time() - t1
        mem_solve = get_memory_usage()
        
        print(f"  > Solved (H.T @ H) in {solve_time:.4f}s. RAM Peak: {mem_solve:.0f} MB")
        print(f"  > STATUS: SUCCESS")
        
    except MemoryError:
        print(f"  > STATUS: FAILED (OOM)")
        break
    except Exception as e:
        print(f"  > STATUS: FAILED ({e})")
        break