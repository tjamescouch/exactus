#!/usr/bin/env python3
import os, sys, time
from math import comb
import numpy as np

# Import built module (repo root or ./build)
try:
    import mc_network  # type: ignore

    mc_network.set_train_hyperparams(lambda_=1e-3, gmax=3.0) 
except Exception:
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    if os.path.isdir(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import mc_network  # type: ignore

    mc_network.set_train_hyperparams(lambda_=1e-3, gmax=3.0) 


def gen_pascal(rows: int, cols: int) -> np.ndarray:
    """Build a uint32 Pascal table with shape [rows, cols], cols == degree+1."""
    P = np.zeros((rows, cols), dtype=np.uint32)
    P[0, 0] = 1
    for n in range(1, rows):
        P[n, 0] = 1
        kmax = min(n, cols - 1)
        if kmax >= 1:
            a = P[n - 1, 1 : kmax + 1]
            b = P[n - 1, 0 : kmax]
            c = a + b  # uint32 add (wraps on overflow)
            if np.any((c < a) | (c < b)):
                raise OverflowError(f"uint32 overflow at row n={n}; reduce D/degree or rows.")
            P[n, 1 : kmax + 1] = c
        if n < cols:
            P[n, n] = 1
    return P


def run_noise(D=16, degree=3, N=4096, epochs=6, bs=256, lr=5e-6, seed=123):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D)).astype(np.float32)
    y = rng.normal(size=(N,)).astype(np.float32)

    M = comb(D + degree - 1, degree)
    rows = D + degree + 4
    pascal = gen_pascal(rows, degree + 1)

    w0 = np.zeros(M, dtype=np.float32)

    t0 = time.time()
    w = mc_network.fit(
        X.flatten().tolist(),
        y.tolist(),
        w0.tolist(),
        float(lr),
        int(epochs),
        int(bs),
        int(degree),
        pascal,  # numpy uint32
    )
    t1 = time.time()

    yhat = np.array(
        mc_network.predict(
            X.flatten().tolist(),
            w,
            int(D),
            int(degree),
            pascal,
        ),
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
        f"time={t1 - t0:.3f}s, mse={mse:.6f}, pascal={pascal.shape} {pascal.dtype}, "
        f"debug=[yhat0={y0:.6f}, grad0={g0:.6f}]"
    )


def run_teacher(D=16, degree=3, N=4096, epochs=20, bs=128, lr=1e-4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, D)).astype(np.float32)

    M = comb(D + degree - 1, degree)
    rows = D + degree + 4
    pascal = gen_pascal(rows, degree + 1)

    # Teacher weights and synthetic targets
    w_true = rng.normal(0, 0.2, size=M).astype(np.float32)
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
    y += 0.01 * rng.normal(size=y.shape).astype(np.float32)  # tiny noise

    # Train from zeros
    w0 = np.zeros(M, dtype=np.float32)
    w = mc_network.fit(
        X.flatten().tolist(),
        y.tolist(),
        w0.tolist(),
        float(lr),
        int(epochs),
        int(bs),
        int(degree),
        pascal,
    )

    yhat = np.array(
        mc_network.predict(
            X.flatten().tolist(),
            w,
            int(D),
            int(degree),
            pascal,
        ),
        dtype=np.float32,
    )
    teacher_mse = float(np.mean((yhat - y) ** 2))
    print(f"[TEACHER] D={D}, deg={degree}, M={M}, N={N}, epochs={epochs}, lr={lr:g}, mse={teacher_mse:.6f}")


def main():
    run_noise()
    run_teacher()


if __name__ == "__main__":
    main()
