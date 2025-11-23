#!/usr/bin/env python3
import os, sys, time
import numpy as np
from dataclasses import dataclass
from typing import Callable, Tuple

# import built module (repo root or ./build)
try:
    import mc_network  # type: ignore
except Exception:
    b = os.path.join(os.path.dirname(__file__), "build")
    if os.path.isdir(b) and b not in sys.path:
        sys.path.insert(0, b)
    import mc_network  # type: ignore

def poly_kernel_mv(X: np.ndarray, u: np.ndarray, degree: int) -> np.ndarray:
    N, D = X.shape
    v = mc_network.kernel_poly_matvec(X.astype(np.float32).ravel().tolist(),
                                      int(N), int(D),
                                      u.astype(np.float32).tolist(),
                                      int(degree))
    return np.asarray(v, dtype=np.float32)

def cg_solve(apply_A: Callable[[np.ndarray], np.ndarray],
             y: np.ndarray, tol: float=1e-6, maxit: int=1000) -> Tuple[np.ndarray, int, float]:
    """
    Solve A alpha = y with conjugate gradient, where apply_A(v) computes A v.
    Returns (alpha, iters, rel_res).
    """
    y = y.astype(np.float32)
    alpha = np.zeros_like(y)
    r = y - apply_A(alpha)
    p = r.copy()
    rs0 = float(np.dot(r, r))
    rel = 1.0
    it = 0
    while it < maxit and rel > tol:
        Ap = apply_A(p)
        denom = float(np.dot(p, Ap))
        if denom <= 1e-30:
            break
        a = rs0 / denom
        alpha = alpha + a * p
        r = r - a * Ap
        rs1 = float(np.dot(r, r))
        rel = (rs1 ** 0.5) / (np.linalg.norm(y) + 1e-12)
        if rel <= tol:
            it += 1
            break
        beta = rs1 / rs0
        p = r + beta * p
        rs0 = rs1
        it += 1
    return alpha, it, rel

@dataclass
class KernelRidgeModel:
    X: np.ndarray
    alpha: np.ndarray
    degree: int
    lam: float

def fit_kernel_ridge(X: np.ndarray, y: np.ndarray,
                     degree: int, lam: float=1e-3,
                     tol: float=1e-6, maxit: int=500) -> KernelRidgeModel:
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    N = X.shape[0]

    def apply_A(u: np.ndarray) -> np.ndarray:
        # (K + lam I) u
        Ku = poly_kernel_mv(X, u, degree)
        return Ku + lam * u

    alpha, it, rel = cg_solve(apply_A, y, tol=tol, maxit=maxit)
    # print(f"[CG] it={it} rel={rel:.2e}")
    return KernelRidgeModel(X=X, alpha=alpha, degree=degree, lam=lam)

def predict(model: KernelRidgeModel, Xtest: np.ndarray) -> np.ndarray:
    Xtest = Xtest.astype(np.float32)
    # v_i = sum_j alpha_j * (1 + x_i^T x_j)^degree
    # We can reuse the same mat-vec by swapping roles: for each test point i, compute
    # v_i = K_test_train alpha. Implement a small loop for now.
    Xt = model.X
    deg = model.degree
    Ntest = Xtest.shape[0]
    N = Xt.shape[0]
    out = np.zeros((Ntest,), dtype=np.float32)
    # simple, correct baseline; can batch later
    for i in range(Ntest):
        xi = Xtest[i]
        dots = Xt @ xi  # shape (N,)
        kij = np.power(1.0 + dots, deg, dtype=np.float32)
        out[i] = float(np.dot(kij, model.alpha))
    return out

def _demo():
    rng = np.random.default_rng(0)
    N, D, deg = 4096, 16, 3
    X = rng.normal(size=(N, D)).astype(np.float32)
    # teacher in kernel space: use random alpha on a subset to keep target smooth
    J = min(N, 1024)
    mask = rng.choice(N, size=J, replace=False)
    alpha_true = np.zeros((N,), dtype=np.float32)
    alpha_true[mask] = rng.normal(0, 0.1, size=(J,)).astype(np.float32)
    # y = K alpha_true
    y = poly_kernel_mv(X, alpha_true, deg)

    t0 = time.time()
    mdl = fit_kernel_ridge(X, y, degree=deg, lam=1e-4, tol=1e-6, maxit=300)
    t1 = time.time()

    yhat = predict(mdl, X)
    mse = float(np.mean((yhat - y)**2))
    print(f"[kernel-ridge] N={N} D={D} deg={deg} time={t1-t0:.3f}s mse={mse:.3e}")

if __name__ == "__main__":
    _demo()
