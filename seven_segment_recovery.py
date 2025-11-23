#!/usr/bin/env python3
"""
Seven-segment recovery from partial supervision (Exactus).

Features:
  - Degree-stacked polynomial training via mc_network on Apple Silicon (Metal).
  - Identifiability diagnostics: per-segment σ_min of design matrix on labeled rows.
  - Exhaustive search over 6-digit subsets (10 choose 6 = 210) to maximize worst-case σ_min.
  - Reproducible benchmarks: prints timing and accuracy vs full 10-digit truth.

Examples
--------
# Basic: pick 60% of digits at random, degree up to 4
python seven_segment_recovery.py --ps 0.6 --degmax 4

# Fixed digits (comma-separated)
python seven_segment_recovery.py --digits 0,1,2,4,6,9 --degmax 4

# Search best 6-digit subset by identifiability (σ_min)
python seven_segment_recovery.py --search-k6 --degmax 4

# Compare 60% and 100%
python seven_segment_recovery.py --ps 0.6,1.0 --degmax 3

Notes
-----
- Inputs: 4-bit BCD for digits 0..9, encoded to {-1,+1}.
- Outputs: 7 segments a..g, encoded to {-1,+1}.
- No augmentation: BCD bits have no relevant symmetry; identifiability is driven by which digits are labeled.
"""

import os, sys, time, argparse, itertools, random
from typing import List, Tuple, Dict, Sequence, Optional
import numpy as np

# --- mc_network import (built module) ---
try:
    import mc_network  # type: ignore
    mc_network.set_train_hyperparams(lambda_=1e-4, gmax=50.0)
except Exception:
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    if os.path.isdir(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import mc_network  # type: ignore
    mc_network.set_train_hyperparams(lambda_=1e-4, gmax=50.0)

# -----------------------------
# Seven-segment truth (common)
# Order of segments: a,b,c,d,e,f,g (1=on, 0=off)
SEG7 = {
    0: [1,1,1,1,1,1,0],
    1: [0,1,1,0,0,0,0],
    2: [1,1,0,1,1,0,1],
    3: [1,1,1,1,0,0,1],
    4: [0,1,1,0,0,1,1],
    5: [1,0,1,1,0,1,1],
    6: [1,0,1,1,1,1,1],
    7: [1,1,1,0,0,0,0],
    8: [1,1,1,1,1,1,1],
    9: [1,1,1,1,0,1,1],
}
SEG_NAMES = ["a","b","c","d","e","f","g"]

# -----------------------------
# Combinatorics / features
def gen_pascal(rows: int, cols: int) -> np.ndarray:
    P = np.zeros((rows, cols), dtype=np.uint32)
    P[0, 0] = 1
    for n in range(1, rows):
        P[n, 0] = 1
        kmax = min(n, cols - 1)
        if kmax >= 1:
            a = P[n - 1, 1:kmax+1]
            b = P[n - 1, 0:kmax]
            c = a + b
            if np.any((c < a) | (c < b)):
                raise OverflowError(f"uint32 overflow at row n={n}")
            P[n, 1:kmax+1] = c
        if n < cols:
            P[n, n] = 1
    return P

def all_multicomb_indices(D: int, deg: int) -> List[Tuple[int,...]]:
    """All multisets of size 'deg' from [0..D-1], lexicographic."""
    if deg == 0:
        return [tuple()]
    out: List[Tuple[int,...]] = []
    def rec(start: int, left: int, cur: List[int]):
        if left == 0:
            out.append(tuple(cur))
            return
        for v in range(start, D):
            cur.append(v)
            rec(v, left-1, cur)
            cur.pop()
    rec(0, deg, [])
    return out

def build_feature_matrix_pm1(X_pm1: np.ndarray, deg_schedule: Sequence[int]) -> np.ndarray:
    """Explicit feature matrix Φ for identifiability diagnostics (small N,M)."""
    N, D = X_pm1.shape
    columns: List[np.ndarray] = []
    for deg in deg_schedule:
        idxs = all_multicomb_indices(D, deg)
        for idx in idxs:
            if len(idx) == 0:
                columns.append(np.ones(N, dtype=np.float32))
            else:
                v = np.ones(N, dtype=np.float32)
                for j in idx:
                    v *= X_pm1[:, j]
                columns.append(v.astype(np.float32))
    return np.stack(columns, axis=1)  # [N, M_total]

def monomial_count(D: int, deg: int) -> int:
    from math import comb
    return comb(D + deg - 1, deg)

# -----------------------------
# BCD inputs, ±1 labels
def bcd4(n: int) -> Tuple[int,int,int,int]:
    assert 0 <= n <= 15
    # bits: b3 b2 b1 b0 (MSB..LSB)
    return ((n>>3)&1, (n>>2)&1, (n>>1)&1, n&1)

def bits01_to_pm1(bits: Sequence[int]) -> Tuple[int,...]:
    return tuple(1 if b==1 else -1 for b in bits)

def y01_to_pm1(y: int) -> int:
    return 1 if y==1 else -1

def build_dataset_pm1(digits: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """X: [N,4] in ±1, Y: [N,7] in ±1 for given digit list."""
    X = np.zeros((len(digits), 4), dtype=np.float32)
    Y = np.zeros((len(digits), 7), dtype=np.float32)
    for i, d in enumerate(digits):
        X[i, :] = np.array(bits01_to_pm1(bcd4(d)), dtype=np.float32)
        Y[i, :] = np.array([y01_to_pm1(v) for v in SEG7[d]], dtype=np.float32)
    return X, Y

# -----------------------------
# mc_network wrappers
def fit_degree(D: int, deg: int, X: np.ndarray, target: np.ndarray,
               lr: float, epochs: int, bs: int):
    M = monomial_count(D, deg)
    P = gen_pascal(D + deg + 4, deg + 1)
    w0 = np.zeros(M, dtype=np.float32)
    w = mc_network.fit(X.flatten().tolist(),
                       target.astype(np.float32).tolist(),
                       w0.tolist(),
                       float(lr), int(epochs), int(bs), int(deg), P)
    return np.array(w, dtype=np.float32), P

def predict_degree(D: int, deg: int, X: np.ndarray, w: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.array(mc_network.predict(X.flatten().tolist(),
                                       w.tolist(), int(D), int(deg), P),
                    dtype=np.float32)

def train_segment_stack(X: np.ndarray, y: np.ndarray,
                        deg_schedule: Sequence[int],
                        lr: float, epochs0: int, epochs: int, bs: int,
                        l1_tau_small: float=0.02, l1_tau_large: float=0.05) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """
    Residual stack over degrees; returns {deg: (w,P)}.
    """
    D = X.shape[1]
    residual = y.copy().astype(np.float32)
    out: Dict[int, Tuple[np.ndarray,np.ndarray]] = {}
    for i, deg in enumerate(deg_schedule):
        epo = epochs0 if (deg == 0 and i == 0) else epochs
        w, P = fit_degree(D, deg, X, residual, lr, epo, bs)
        # soft L1 shrink (except deg=0)
        if deg > 0:
            tau = l1_tau_small if len(X) <= 6 else l1_tau_large
            s = np.sign(w); a = np.abs(w) - tau; a[a<0]=0.0; w = s * a
        yhat = predict_degree(D, deg, X, w, P)
        residual = residual - yhat
        out[deg] = (w, P)
    return out

def predict_segment_stack(params: Dict[int, Tuple[np.ndarray, np.ndarray]],
                          D: int, X: np.ndarray, deg_schedule: Sequence[int]) -> np.ndarray:
    y = np.zeros(X.shape[0], dtype=np.float32)
    for deg in deg_schedule:
        if deg in params:
            w, P = params[deg]
            y += predict_degree(D, deg, X, w, P)
    return y

# -----------------------------
# Identifiability: σ_min on labeled rows
def design_sigma_min(X_labeled: np.ndarray, deg_schedule: Sequence[int]) -> float:
    Phi = build_feature_matrix_pm1(X_labeled, deg_schedule)  # [k, M]
    # robust to tiny k<M: compute SVD skinny
    s = np.linalg.svd(Phi, full_matrices=False, compute_uv=False)
    return float(s.min()) if s.size else 0.0

# Exhaustive search for the best 6-digit set (maximize worst σ_min across 7 segments)
def search_best_k6(digits_full: Sequence[int], deg_schedule: Sequence[int]) -> List[int]:
    best_set = None
    best_score = -1.0
    X_full, Y_full = build_dataset_pm1(digits_full)
    for combo in itertools.combinations(digits_full, 6):
        idx = [digits_full.index(d) for d in combo]
        X_l = X_full[idx, :]
        # one σ_min for the shared Φ (same for all segments)
        smin = design_sigma_min(X_l, deg_schedule)
        if smin > best_score:
            best_score = smin
            best_set = list(combo)
    return best_set if best_set is not None else list(digits_full[:6])

# -----------------------------
def acc_from_logits(y_pred_pm1: np.ndarray, y_true_pm1: np.ndarray) -> float:
    return float((np.sign(y_pred_pm1) == np.sign(y_true_pm1)).mean())

def run_once(labeled_digits: List[int],
             degmax: int,
             lr: float, epochs0: int, epochs: int, bs: int,
             verbose: bool=True) -> Tuple[List[float], List[float], float]:
    digits_full = list(range(10))
    deg_schedule = list(range(0, degmax+1))

    # Build datasets
    X_lab, Y_lab = build_dataset_pm1(labeled_digits)
    X_all, Y_all = build_dataset_pm1(digits_full)

    # Identifiability metric for this choice (shared for all segments)
    smin = design_sigma_min(X_lab, deg_schedule)

    # Train each segment independently (±1 regression)
    per_seg_acc: List[float] = []
    per_seg_sigma: List[float] = []

    t0 = time.time()
    for j in range(7):
        y_lab = Y_lab[:, j]
        params = train_segment_stack(X_lab, y_lab, deg_schedule, lr, epochs0, epochs, bs)
        y_pred_all = predict_segment_stack(params, D=4, X=X_all, deg_schedule=deg_schedule)
        acc = acc_from_logits(y_pred_all, Y_all[:, j])
        per_seg_acc.append(acc)
        per_seg_sigma.append(smin)
    t1 = time.time()

    if verbose:
        msg = " | ".join([f"{SEG_NAMES[j]}: {per_seg_acc[j]*100:5.2f}%" for j in range(7)])
        print(f"  digits={labeled_digits}  {msg}  || overall: {np.mean(per_seg_acc)*100:5.2f}%")
        print(f"  timing: {t1-t0:.3f}s  σ_min={smin:.4f}")

    return per_seg_acc, per_seg_sigma, (t1 - t0)

# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ps", type=str, default="0.6",
                    help="Comma-separated fractions in (0,1], e.g. '0.6,1.0'. Mutually exclusive with --digits and --search-k6.")
    ap.add_argument("--digits", type=str, default="",
                    help="Comma-separated explicit labeled digits, e.g. '0,1,2,4,6,9'.")
    ap.add_argument("--search-k6", action="store_true",
                    help="Search all 6-digit subsets to maximize σ_min (ignores --ps and --digits).")
    ap.add_argument("--degmax", type=int, default=4)
    ap.add_argument("--epochs0", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=220)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    digits_full = list(range(10))
    deg_schedule = list(range(0, args.degmax+1))

    print(f"\n=== 7-seg decoder (digits 0-9 only) ===")
    if args.degmax >= 4:
        print(f"Degrees: {deg_schedule}, train lr={args.lr}, epochs0/mid/deg4={args.epochs0}/{args.epochs}/{max(args.epochs+100, args.epochs)}}}, bs={args.bs}")
    else:
        print(f"Degrees: {deg_schedule}, train lr={args.lr}, epochs0/mid={args.epochs0}/{args.epochs}, bs={args.bs}")

    # Baseline: full supervision (sanity)
    print("\n[Baseline p=100%]")
    acc_full, _, t_full = run_once(digits_full, args.degmax, args.lr, args.epochs0, args.epochs, args.bs, verbose=True)

    # Partial: according to mode
    if args.search_k6:
        best = search_best_k6(digits_full, deg_schedule)
        print("\n[Search k=6 max σ_min]  (expect higher stability than random 6)")
        run_once(best, args.degmax, args.lr, args.epochs0, args.epochs, args.bs, verbose=True)
        return

    if args.digits.strip():
        sel = [int(x) for x in args.digits.strip().split(",") if x.strip()!=""]
        print("\n[Fixed selection]")
        run_once(sel, args.degmax, args.lr, args.epochs0, args.epochs, args.bs, verbose=True)
        return

    # Percent selections
    ps = [float(p.strip()) for p in args.ps.split(",")]
    print("")
    for p in ps:
        k = max(1, int(round(p * len(digits_full))))
        # choose k deterministic but shuffled by seed for reproducibility
        pool = digits_full[:]
        random.shuffle(pool)
        sel = sorted(pool[:k])
        label = f"[p={int(p*100):2d}%  k={k}]"
        print(label)
        run_once(sel, args.degmax, args.lr, args.epochs0, args.epochs, args.bs, verbose=True)

    # Micro-benchmark footer
    print("\n--- Benchmark summary ---")
    print(f"Full supervision time: {t_full:.3f}s for 7 segments (degmax={args.degmax}) on 10 rows")
    print("Expected behavior:")
    print("  - 100% at p=1.0 (confirming representation & training).")
    print("  - ~75–90% at k=6 unless digits are chosen for high σ_min.")
    print("  - '--search-k6' should beat random-6 on average.")
    print("Tip: if a run under k=6 looks low, check σ_min printed by the script;")
    print("     if it's tiny, identifiability is the limiter—not the optimizer.")

if __name__ == "__main__":
    main()
