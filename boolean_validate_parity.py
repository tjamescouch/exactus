#!/usr/bin/env python3
# Exact ANF extraction from learned truth tables using a GF(2) Möbius transform.
# Keeps the fast GPU fit, but derives the ANF from the *predicted* 0/1 table,
# guaranteeing correct parity polynomials without heuristic term picking.

import os, sys, time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
import numpy as np

# --- import built module (repo root or ./build) ---
try:
    import mc_network  # type: ignore
except Exception:
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    if os.path.isdir(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import mc_network  # type: ignore

# Reuse small helpers from the scalar-basis file if present; otherwise inline:
def gen_pascal(rows: int, cols: int) -> np.ndarray:
    P = np.zeros((rows, cols), dtype=np.uint32)
    P[0, 0] = 1
    for n in range(1, rows):
        P[n, 0] = 1
        kmax = min(n, cols - 1)
        if kmax >= 1:
            a = P[n - 1, 1 : kmax + 1]
            b = P[n - 1, 0 : kmax]
            c = a + b
            if np.any((c < a) | (c < b)):
                raise OverflowError(f"uint32 overflow at row n={n}")
            P[n, 1 : kmax + 1] = c
        if n < cols:
            P[n, n] = 1
    return P

def nck(n: int, k: int) -> int:
    if k < 0 or k > n: return 0
    if k == 0 or k == n: return 1
    k = min(k, n-k); num=1; den=1
    for i in range(1,k+1):
        num *= (n-(k-i)); den *= i
    return num//den

def monomial_count(D: int, deg: int) -> int:
    return nck(D + deg - 1, deg)

def decode_monomial_indices(k: int, D: int, deg: int) -> Tuple[int, ...]:
    if deg == 0: return tuple()
    idx = []; prev = 0; rem = k
    for i in range(deg):
        for v in range(prev, D):
            rest = nck((D - v) + (deg - i - 1) - 1, (deg - i - 1))
            if rest <= rem:
                rem -= rest; continue
            idx.append(v); prev = v; break
    return tuple(idx)

def to_bits(n: int, w: int) -> Tuple[int, ...]:
    return tuple((n >> i) & 1 for i in range(w))

def bits_to_pm1(bits: Tuple[int, ...]) -> Tuple[int, ...]:
    # map {0,1} -> {-1,+1}
    return tuple(1 if b==1 else -1 for b in bits)

# ---------- boolean tasks ----------
@dataclass
class BoolTask:
    name: str
    n_bits: int
    fn: Callable[[Tuple[int,...]], int]  # {0,1}^n -> {0,1}

def TT_XOR(a,b): return a ^ b
def TT_XNOR(a,b): return 1 - (a ^ b)
def TT_PAR3(a,b,c): return (a ^ b ^ c) & 1

TASKS: List[BoolTask] = [
    BoolTask("XOR2",  2, lambda x: TT_XOR(x[0],x[1])),
    BoolTask("XNOR2", 2, lambda x: TT_XNOR(x[0],x[1])),
    BoolTask("PAR3",  3, lambda x: TT_PAR3(x[0],x[1],x[2])),
]

# ---------- training over ±1 inputs (keeps nice parity features) ----------
def build_truth_pm1(task: BoolTask) -> Tuple[np.ndarray, np.ndarray]:
    n = task.n_bits
    X = np.zeros((1<<n, n), dtype=np.float32)
    y = np.zeros((1<<n,), dtype=np.float32)
    for a in range(1<<n):
        bits = to_bits(a, n)
        X[a,:] = bits_to_pm1(bits)  # inputs in {-1,+1}
        y[a]   = task.fn(bits)      # targets in {0,1}
    return X, y

def fit_degree(D: int, deg: int, X: np.ndarray, target: np.ndarray,
               lr: float, epochs: int, bs: int):
    M = monomial_count(D, deg)
    P = gen_pascal(D + deg + 4, deg + 1)
    w0 = np.zeros(M, dtype=np.float32)
    w = mc_network.fit(X.flatten().tolist(),
                       target.astype(np.float32).tolist(),
                       w0.tolist(),
                       float(lr), int(epochs), int(bs), int(deg), P)
    return np.array(w, dtype=np.float32), [decode_monomial_indices(k, D, deg) for k in range(M)], P

def predict_degree(D: int, deg: int, X: np.ndarray, w: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.array(mc_network.predict(X.flatten().tolist(), w.tolist(), int(D), int(deg), P), dtype=np.float32)

def best_threshold_accuracy(y_pred: np.ndarray, y_true01: np.ndarray) -> Tuple[float,float]:
    ys = np.unique(y_pred)
    cuts = []
    for i in range(len(ys)-1):
        cuts.append(0.5*(ys[i]+ys[i+1]))
    if len(ys)>0: cuts = [ys[0]-1e-6] + cuts + [ys[-1]+1e-6]
    best, tstar = 0.0, 0.5
    for t in cuts:
        acc = float(((y_pred >= t).astype(np.float32) == y_true01).mean())
        if acc > best: best, tstar = acc, t
    return best, tstar

# ---------- exact ANF via GF(2) Möbius transform ----------
def anf_from_truth_table(y01: np.ndarray) -> np.ndarray:
    """
    Input: y01 length 2^n array over {0,1} in lex order with index=bitmask (little-endian).
    Output: a (length 2^n) over {0,1}, where a[mask]=1 means include monomial Π_{i in mask} x_i in ANF.
    Uses in-place subset (zeta) transform over GF(2).
    """
    y = (y01.astype(np.uint8) & 1).copy()
    n = int(np.log2(len(y)))
    # Möbius over subsets: for each bit i, for each mask with bit i set, a[mask] ^= a[mask ^ (1<<i)]
    for i in range(n):
        step = 1 << i
        for mask in range(1<<n):
            if mask & step:
                y[mask] ^= y[mask ^ step]
    return y  # coefficients mod 2

def anf_terms_from_coeffs(a: np.ndarray, n_bits: int) -> List[Tuple[int,...]]:
    terms = []
    for mask, coef in enumerate(a.tolist()):
        if coef == 0:
            continue
        if mask == 0:
            terms.append(tuple())  # constant 1
        else:
            idxs = [i for i in range(n_bits) if (mask >> i) & 1]
            terms.append(tuple(idxs))
    # sort by degree then lex
    terms.sort(key=lambda t: (len(t), t))
    return terms

def term_to_str(term: Tuple[int,...]) -> str:
    if len(term)==0: return "1"
    return "*".join(f"x{i}" for i in term)

# ---------- run one task ----------
def run_one(task: BoolTask, K: int, lr: float, epochs: int, bs: int):
    X, y = build_truth_pm1(task)
    D = task.n_bits
    mc_network.set_train_hyperparams(lambda_=0.0, gmax=10.0)

    # Degree-by-degree residual training (as before)
    ysum = np.zeros(X.shape[0], dtype=np.float32)
    for deg in range(0, K+1):
        w_d, idx_d, P_d = fit_degree(D, deg, X, y - ysum, lr, epochs, bs)
        ysum += predict_degree(D, deg, X, w_d, P_d)

    acc, tstar = best_threshold_accuracy(ysum, y)
    y_bin = (ysum >= tstar).astype(np.uint8)

    # Exact ANF from the (predicted) 0/1 truth table
    a = anf_from_truth_table(y_bin)                     # GF(2) coefficients
    terms = anf_terms_from_coeffs(a, task.n_bits)       # list of index tuples
    return acc, tstar, terms

def main():
    configs = [
        ("XOR2",  2, 1e-3, 300, 1),
        ("XNOR2", 2, 1e-3, 300, 1),
        ("PAR3",  3, 1e-3, 400, 1),
    ]
    name_to_task = {t.name: t for t in TASKS}

    t0 = time.time()
    for name, K, lr, epochs, bs in configs:
        task = name_to_task[name]
        acc, tstar, terms = run_one(task, K, lr, epochs, bs)
        print(f"\n=== {name} (n={task.n_bits}, K={K}) ===")
        print(f"acc={acc*100:.1f}% | t*={tstar:.4f} | ANF_terms={len(terms)}")
        if terms:
            print("ANF:", " ⊕ ".join(term_to_str(t) for t in terms))
    print(f"\nTotal time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
