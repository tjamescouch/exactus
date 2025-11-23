# tests/test_teacher_recovery.py
import os, sys, numpy as np
from math import comb

try:
    import mc_network  # built module
except Exception:
    bd = os.path.join(os.path.dirname(__file__), "..", "build")
    if os.path.isdir(bd) and bd not in sys.path:
        sys.path.insert(0, bd)
    import mc_network

def gen_pascal(rows, cols):
    P = np.zeros((rows, cols), dtype=np.uint32)
    P[0, 0] = 1
    for n in range(1, rows):
        P[n, 0] = 1
        kmax = min(n, cols - 1)
        if kmax >= 1:
            a = P[n-1, 1:kmax+1]; b = P[n-1, 0:kmax]
            c = a + b
            if np.any((c < a) | (c < b)):
                raise OverflowError
            P[n, 1:kmax+1] = c
        if n < cols:
            P[n, n] = 1
    return P

def test_teacher_exact_recovery():
    D, degree, N = 16, 3, 4096
    M = comb(D + degree - 1, degree)
    P = gen_pascal(D + degree + 4, degree + 1)

    rng = np.random.default_rng(0)
    X = rng.normal(size=(N, D)).astype(np.float32)

    w_true = rng.normal(0, 0.2, size=M).astype(np.float32)
    y = np.array(mc_network.predict(X.flatten().tolist(), w_true.tolist(), D, degree, P), dtype=np.float32)

    mc_network.set_train_hyperparams(lambda_=1e-4, gmax=50.0)
    w = mc_network.fit(X.flatten().tolist(), y.tolist(),
                       np.zeros(M, np.float32).tolist(),
                       3e-3, 200, 256, degree, P)
    yhat = np.array(mc_network.predict(X.flatten().tolist(), w, D, degree, P), dtype=np.float32)
    mse = float(((yhat - y)**2).mean())
    assert mse < 1e-3, f"teacher recovery too high: mse={mse}"
