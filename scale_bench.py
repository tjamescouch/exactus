#!/usr/bin/env python3
"""
Scalability benchmark: Metal/SGD (mc_network) vs CPU ridge (explicit features).

- Sweeps (D, degree) -> model size M = C(D+deg-1, deg).
- Teacher data: y = Phi(X) @ w_true + noise.
- Measures fit time and accuracy (MSE) on held-out.
- Saves CSV + plots (time vs M; mse vs M) per degree.

Requirements:
  - Built mc_network extension (cmake && make).
  - numpy, matplotlib, pandas (for plotting/CSV).
"""

import os, sys, time, argparse, math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np

# --- import mc_network from repo root or ./build ---
try:
    import mc_network  # type: ignore
except Exception:
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    if os.path.isdir(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import mc_network  # type: ignore

# ---------- combinatorics / indexing ----------
def nck(n: int, k: int) -> int:
    if k < 0 or k > n: return 0
    if k == 0 or k == n: return 1
    k = min(k, n - k)
    num = 1
    den = 1
    for i in range(1, k + 1):
        num *= (n - (k - i))
        den *= i
    return num // den

def monomial_count(D: int, deg: int) -> int:
    return nck(D + deg - 1, deg)

def gen_pascal(rows: int, cols: int) -> np.ndarray:
    """uint32 Pascal needed by mc_network backend."""
    P = np.zeros((rows, cols), dtype=np.uint32)
    P[0, 0] = 1
    for n in range(1, rows):
        P[n, 0] = 1
        kmax = min(n, cols - 1)
        if kmax >= 1:
            a = P[n - 1, 1:kmax + 1]
            b = P[n - 1, 0:kmax]
            c = a + b
            # overflow guard (should not hit with rows small)
            if np.any((c < a) | (c < b)):
                raise OverflowError(f"uint32 overflow at row n={n}")
            P[n, 1:kmax + 1] = c
        if n < cols:
            P[n, n] = 1
    return P

def decode_monomial_indices(k: int, D: int, deg: int) -> Tuple[int, ...]:
    """Graded lex decode for weakly increasing multi-index of length 'deg' in [0..D-1]."""
    if deg == 0:
        return tuple()
    idx = []
    prev = 0
    rem = k
    for i in range(deg):
        for v in range(prev, D):
            rest = nck((D - v) + (deg - i - 1) - 1, (deg - i - 1))
            if rest <= rem:
                rem -= rest
                continue
            idx.append(v)
            prev = v
            break
    return tuple(idx)

def build_feature_matrix_explicit(X: np.ndarray, D: int, deg: int) -> np.ndarray:
    """
    Explicit feature matrix H (N x M) for CPU ridge baseline.
    Monomials are products of X columns according to decode_monomial_indices.
    """
    N = X.shape[0]
    M = monomial_count(D, deg)
    H = np.empty((N, M), dtype=np.float32)
    if deg == 0:
        H[:, 0] = 1.0
        return H
    # Pre-cache columns to avoid repeated slicing
    cols = [X[:, j] for j in range(D)]
    for k in range(M):
        idx = decode_monomial_indices(k, D, deg)
        v = np.ones(N, dtype=np.float32)
        for j in idx:
            v = v * cols[j]
        H[:, k] = v
    return H

# ---------- experiment scaffolding ----------
@dataclass
class RunCfg:
    degree: int
    D_list: List[int]
    # Data sizes relative to M (so we scale with model size)
    N_train_per_M: float = 4.0
    N_test: int = 4096
    # Teacher distribution
    w_scale: float = 0.2
    noise_std: float = 0.01
    # Metal SGD hyperparams
    lr: float = 1e-4
    epochs: int = 20
    batch_size: int = 256
    # CPU ridge
    ridge_alpha: float = 1e-6
    # Safety caps for CPU baseline
    max_explicit_H_elems: int = 30_000_000  # ~30M floats ~120MB (float32)
    max_explicit_M: int = 50_000

def metal_fit_predict(X_tr: np.ndarray, y_tr: np.ndarray,
                      X_te: np.ndarray, D: int, degree: int,
                      lr: float, epochs: int, batch_size: int) -> Tuple[np.ndarray, float, np.ndarray]:
    """
    Returns (w, elapsed_s, yhat_te)
    """
    M = monomial_count(D, degree)
    pascal = gen_pascal(D + degree + 4, degree + 1)
    w0 = np.zeros(M, dtype=np.float32)
    t0 = time.time()
    w = mc_network.fit(
        X_tr.flatten().tolist(),
        y_tr.astype(np.float32).tolist(),
        w0.tolist(),
        float(lr),
        int(epochs),
        int(batch_size),
        int(degree),
        pascal,
    )
    t1 = time.time()
    w = np.array(w, dtype=np.float32)

    yhat_te = np.array(
        mc_network.predict(
            X_te.flatten().tolist(),
            w.tolist(),
            int(D),
            int(degree),
            pascal,
        ),
        dtype=np.float32,
    )
    return w, (t1 - t0), yhat_te

def ridge_fit_predict(X_tr: np.ndarray, y_tr: np.ndarray,
                      X_te: np.ndarray, D: int, degree: int,
                      alpha: float, max_elems: int, max_M: int) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray], str]:
    """
    CPU closed-form ridge with explicit design matrix.
    Returns (w, elapsed_s, yhat_te, status) where any may be None if skipped.
    """
    M = monomial_count(D, degree)
    N_tr = X_tr.shape[0]
    # Skip if too big
    if M > max_M:
        return None, None, None, f"SKIP (M={M} > max_M={max_M})"
    if N_tr * M > max_elems:
        return None, None, None, f"SKIP (H elems {N_tr*M} > cap {max_elems})"

    t0 = time.time()
    H_tr = build_feature_matrix_explicit(X_tr, D, degree)  # (N_tr x M)
    # Solve (H^T H + alpha I) w = H^T y
    # Use np.linalg.solve on normal equations (OK for well-conditioned smallish M)
    HtH = H_tr.T @ H_tr
    Hty = H_tr.T @ y_tr
    HtH.flat[::M+1] += alpha  # add alpha to diagonal
    w = np.linalg.solve(HtH.astype(np.float64, copy=False), Hty.astype(np.float64, copy=False)).astype(np.float32)
    # Predict
    H_te = build_feature_matrix_explicit(X_te, D, degree)
    yhat_te = H_te @ w
    t1 = time.time()

    return w, (t1 - t0), yhat_te.astype(np.float32), "OK"

def make_teacher(D: int, degree: int, w_scale: float, rng: np.random.Generator) -> np.ndarray:
    M = monomial_count(D, degree)
    return rng.normal(0.0, w_scale, size=M).astype(np.float32)

def synth_targets_with_metal(X: np.ndarray, w_true: np.ndarray, D: int, degree: int) -> np.ndarray:
    """Use the mc_network predictor to compute y = Phi(X) @ w_true exactly."""
    pascal = gen_pascal(D + degree + 4, degree + 1)
    y = np.array(
        mc_network.predict(
            X.flatten().tolist(),
            w_true.tolist(),
            int(D),
            int(degree),
            pascal,
        ),
        dtype=np.float32,
    )
    return y

def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    var = float(np.var(y_true))
    if var == 0.0:
        return 1.0 if np.allclose(y_true, y_pred) else 0.0
    return 1.0 - float(np.mean((y_true - y_pred) ** 2)) / var

# ---------- plotting ----------
def plot_time_and_mse(csv_path: str, out_dir: str):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    os.makedirs(out_dir, exist_ok=True)

    for deg in sorted(df['degree'].unique()):
        dsub = df[df['degree'] == deg]

        # Time vs M
        plt.figure()
        for backend in ['metal', 'ridge']:
            s = dsub[dsub['backend'] == backend]
            if len(s) == 0: continue
            plt.plot(s['M'], s['fit_time_s'], marker='o', label=backend)
        plt.xlabel('Model size M = C(D+deg-1, deg)')
        plt.ylabel('Fit time (s)')
        plt.title(f'Fit time vs M (degree={deg})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'time_vs_M_deg{deg}.png'), dpi=150)
        plt.close()

        # MSE vs M
        plt.figure()
        for backend in ['metal', 'ridge']:
            s = dsub[dsub['backend'] == backend]
            if len(s) == 0: continue
            plt.plot(s['M'], s['mse'], marker='o', label=backend)
        plt.xlabel('Model size M = C(D+deg-1, deg)')
        plt.ylabel('Test MSE')
        plt.title(f'Test MSE vs M (degree={deg})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'mse_vs_M_deg{deg}.png'), dpi=150)
        plt.close()

# ---------- main sweep ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=str, default='results/scale', help='output directory')
    ap.add_argument('--degrees', type=int, nargs='+', default=[1,2,3], help='degrees to test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--noise', type=float, default=0.01)
    ap.add_argument('--wscale', type=float, default=0.2)
    ap.add_argument('--epochs', type=int, default=20)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--bs', type=int, default=256)
    ap.add_argument('--ridge_alpha', type=float, default=1e-6)
    ap.add_argument('--csv', type=str, default='scale_results.csv')
    ap.add_argument('--maxH', type=int, default=30_000_000, help='max explicit H elements for ridge')
    ap.add_argument('--maxM', type=int, default=50_000, help='max M for ridge')
    args = ap.parse_args()

    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, args.csv)

    rng = np.random.default_rng(args.seed)

    # Reasonable D lists per degree to span a range of M
    D_lists: Dict[int, List[int]] = {
        1: [16, 64, 256, 1024, 4096],           # M = D (linear)
        2: [16, 32, 64, 128, 256],              # M ~ O(D^2/2)
        3: [8, 12, 16, 24, 32],                 # M ~ O(D^3/6)
    }

    # Set backend hyperparams
    mc_network.set_train_hyperparams(lambda_=1e-3, gmax=10.0)

    # CSV header
    with open(csv_path, 'w') as f:
        f.write(','.join([
            'backend','degree','D','M','N_train','N_test',
            'epochs','lr','batch_size','ridge_alpha',
            'noise_std','fit_time_s','mse','r2','w_corr'
        ]) + '\n')

    for deg in args.degrees:
        D_list = D_lists.get(deg, [])
        if not D_list:
            print(f"[warn] No D_list preset for degree={deg}, skipping.")
            continue

        for D in D_list:
            M = monomial_count(D, deg)
            # choose N_train relative to M
            N_train = int(max(1024, math.ceil(4.0 * M)))
            N_test = 4096

            # Data
            Xtr = rng.normal(size=(N_train, D)).astype(np.float32)
            Xte = rng.normal(size=(N_test,  D)).astype(np.float32)
            w_true = make_teacher(D, deg, args.wscale, rng)
            y_tr_clean = synth_targets_with_metal(Xtr, w_true, D, deg)
            y_te_clean = synth_targets_with_metal(Xte, w_true, D, deg)
            y_tr = y_tr_clean + args.noise * rng.normal(size=y_tr_clean.shape).astype(np.float32)
            y_te = y_te_clean + args.noise * rng.normal(size=y_te_clean.shape).astype(np.float32)

            # ---- Metal/SGD ----
            try:
                w_m, t_m, yhat_m = metal_fit_predict(
                    Xtr, y_tr, Xte, D, deg,
                    lr=args.lr, epochs=args.epochs, batch_size=args.bs
                )
                mse_m = float(np.mean((yhat_m - y_te)**2))
                r2_m  = r2_score(y_te, yhat_m)
                corr_m = float(np.corrcoef(w_m, w_true)[0,1]) if len(w_m)==len(w_true) else float('nan')
                with open(csv_path, 'a') as f:
                    f.write(','.join(map(str, [
                        'metal', deg, D, M, N_train, N_test,
                        args.epochs, args.lr, args.bs, args.ridge_alpha,
                        args.noise, t_m, mse_m, r2_m, corr_m
                    ])) + '\n')
                print(f"[metal] deg={deg} D={D} M={M} time={t_m:.3f}s mse={mse_m:.4g}")
            except Exception as e:
                print(f"[metal] deg={deg} D={D} FAILED: {e}")

            # ---- CPU ridge (skip if too big) ----
            try:
                w_r, t_r, yhat_r, status = ridge_fit_predict(
                    Xtr, y_tr, Xte, D, deg,
                    alpha=args.ridge_alpha,
                    max_elems=args.maxH,
                    max_M=args.maxM
                )
                if status == "OK":
                    mse_r = float(np.mean((yhat_r - y_te)**2))
                    r2_r  = r2_score(y_te, yhat_r)
                    corr_r = float(np.corrcoef(w_r, w_true)[0,1]) if len(w_r)==len(w_true) else float('nan')
                    with open(csv_path, 'a') as f:
                        f.write(','.join(map(str, [
                            'ridge', deg, D, M, N_train, N_test,
                            args.epochs, args.lr, args.bs, args.ridge_alpha,
                            args.noise, t_r, mse_r, r2_r, corr_r
                        ])) + '\n')
                    print(f"[ridge] deg={deg} D={D} M={M} time={t_r:.3f}s mse={mse_r:.4g}")
                else:
                    print(f"[ridge] deg={deg} D={D} M={M} {status}")
            except Exception as e:
                print(f"[ridge] deg={deg} D={D} FAILED: {e}")

    # Plots
    try:
        plot_time_and_mse(csv_path, out_dir)
        print(f"\nWrote CSV: {csv_path}")
        print(f"Plots: {out_dir}/time_vs_M_deg*.png and mse_vs_M_deg*.png")
    except Exception as e:
        print(f"[plot] skipped due to error: {e}")

if __name__ == "__main__":
    main()
