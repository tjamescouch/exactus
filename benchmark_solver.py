#!/usr/bin/env python3
import os, sys, time, csv, argparse
from math import comb
import numpy as np

# --- import mc_network and set hyperparams ---
try:
    import mc_network  # type: ignore
    mc_network.set_train_hyperparams(lambda_=1e-4, gmax=50.0)
except Exception:
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    if os.path.isdir(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import mc_network  # type: ignore
    mc_network.set_train_hyperparams(lambda_=1e-4, gmax=50.0)

# --- utils ---
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

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    an = np.linalg.norm(a); bn = np.linalg.norm(b)
    if an == 0.0 or bn == 0.0: return 0.0
    return float(np.dot(a, b) / (an * bn))

def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    bn = np.linalg.norm(b)
    if bn == 0.0: return float('inf')
    return float(np.linalg.norm(a - b) / bn)

# --- experiments ---
def run_noise(D=16, degree=3, N=4096, epochs=6, bs=256, lr=5e-6, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D)).astype(np.float32)
    y = rng.normal(size=(N,)).astype(np.float32)

    M = comb(D + degree - 1, degree)
    pascal = gen_pascal(D + degree + 4, degree + 1)
    w0 = np.zeros(M, dtype=np.float32)

    t0 = time.time()
    w = mc_network.fit(
        X.flatten().tolist(), y.tolist(), w0.tolist(),
        float(lr), int(epochs), int(bs), int(degree), pascal
    )
    t1 = time.time()

    yhat = np.array(
        mc_network.predict(X.flatten().tolist(), w, int(D), int(degree), pascal),
        dtype=np.float32,
    )

    mse = float(np.mean((yhat - y) ** 2))
    mse_baseline = float(np.mean(y**2))
    dbg = mc_network.debug_output()
    y0 = float(dbg[0]) if len(dbg) > 0 else float("nan")
    g0 = float(dbg[1]) if len(dbg) > 1 else float("nan")

    print(f"[NOISE] baseline_zero_pred_mse={mse_baseline:.6f}")
    print(
        f"[NOISE] D={D}, deg={degree}, M={M}, N={N}, epochs={epochs}, "
        f"time={t1 - t0:.3f}s, mse={mse:.6f}, pascal=({pascal.shape[0]}, {pascal.shape[1]}) {pascal.dtype}, "
        f"debug=[yhat0={y0:.6f}, grad0={g0:.6f}]"
    )

def run_teacher_once(D, degree, N, epochs, bs, lr, seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D)).astype(np.float32)

    M = comb(D + degree - 1, degree)
    pascal = gen_pascal(D + degree + 4, degree + 1)

    # teacher
    w_true = rng.normal(0, 0.2, size=M).astype(np.float32)
    y = np.array(
        mc_network.predict(X.flatten().tolist(), w_true.tolist(), int(D), int(degree), pascal),
        dtype=np.float32,
    )
    y += 0.01 * rng.normal(size=y.shape).astype(np.float32)

    # train
    w0 = np.zeros(M, dtype=np.float32)
    w = mc_network.fit(
        X.flatten().tolist(), y.tolist(), w0.tolist(),
        float(lr), int(epochs), int(bs), int(degree), pascal
    )

    yhat = np.array(
        mc_network.predict(X.flatten().tolist(), w, int(D), int(degree), pascal),
        dtype=np.float32,
    )
    mse = float(np.mean((yhat - y) ** 2))
    cos = cosine_similarity(np.array(w, dtype=np.float32), w_true)
    rel = relative_l2(np.array(w, dtype=np.float32), w_true)
    print(f"[TEACHER] D={D}, deg={degree}, M={M}, N={N}, epochs={epochs}, lr={lr:g}, "
          f"mse={mse:.6f}, cos_sim={cos:.6f}, rel_l2={rel:.6e}")
    return mse, cos, rel

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--reset_csv", action="store_true")
    ap.add_argument("--out", type=str, default="results")
    ap.add_argument("--D", type=int, default=16)
    ap.add_argument("--deg", type=int, default=3)
    ap.add_argument("--N", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    args = ap.parse_args()

    ensure_dir(args.out)
    csv_path = os.path.join(args.out, "teacher_recovery.csv")

    if args.reset_csv or not os.path.exists(csv_path):
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["D","degree","N","epochs","lr","bs","mse","cos_sim","rel_L2"])

    # Noise sanity (unchanged)
    run_noise()

    # Trials
    for i in range(args.trials):
        mse, cos, rel = run_teacher_once(args.D, args.deg, args.N, args.epochs, args.bs, args.lr, seed=i)
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([args.D, args.deg, args.N, args.epochs, args.lr, args.bs, mse, cos, rel])

if __name__ == "__main__":
    main()
