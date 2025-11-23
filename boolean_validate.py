#!/usr/bin/env python3
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


# ========== utilities ==========
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

def n_choose_k(n: int, k: int) -> int:
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
    # with replacement: C(D + deg - 1, deg)
    return n_choose_k(D + deg - 1, deg)

def decode_monomial_indices(k: int, D: int, deg: int) -> Tuple[int, ...]:
    """
    Correct combinations-with-replacement decoder.
    Returns nondecreasing tuple (i0,...,i_{deg-1}) with 0<=i<=D-1.
    Enumerates in lexicographic order consistent with stars-and-bars.
    """
    if deg == 0:
        return tuple()
    idx = []
    prev = 0
    rem = k
    # positions i = 0..deg-1
    for i in range(deg):
        # choose smallest v in [prev..D-1] s.t. remaining count exceeds rem
        for v in range(prev, D):
            # number of multisets of size (deg-i-1) using labels v..D-1:
            # C( (D - v) + (deg-i-1) - 1, (deg-i-1) )
            rest = n_choose_k((D - v) + (deg - i - 1) - 1, (deg - i - 1))
            if rest <= rem:
                rem -= rest
                continue
            idx.append(v)
            prev = v  # nondecreasing (replacement allowed)
            break
    return tuple(idx)

def to_bits(n: int, width: int) -> Tuple[int, ...]:
    # little-endian bit order; consistent across build_truth_table
    return tuple((n >> i) & 1 for i in range(width))


# ========== boolean tasks ==========
@dataclass
class BoolTask:
    name: str
    n_bits: int
    fn: Callable[[Tuple[int,...]], int]  # {0,1}^n -> {0,1}

def TT_AND(a,b): return a & b
def TT_OR(a,b): return a | b
def TT_NAND(a,b): return 1 - (a & b)
def TT_NOR(a,b): return 1 - (a | b)
def TT_XOR(a,b): return a ^ b
def TT_XNOR(a,b): return 1 - (a ^ b)
def TT_MAJ3(a,b,c): return 1 if (a + b + c) >= 2 else 0
def TT_HA_SUM(a,b): return a ^ b
def TT_HA_CARRY(a,b): return a & b
def TT_MUX(sel,a,b): return a if sel==0 else b

TASKS: List[BoolTask] = [
    BoolTask("AND2",    2, lambda x: TT_AND(x[0],x[1])),
    BoolTask("OR2",     2, lambda x: TT_OR(x[0],x[1])),
    BoolTask("XOR2",    2, lambda x: TT_XOR(x[0],x[1])),
    BoolTask("XNOR2",   2, lambda x: TT_XNOR(x[0],x[1])),
    BoolTask("NAND2",   2, lambda x: TT_NAND(x[0],x[1])),
    BoolTask("NOR2",    2, lambda x: TT_NOR(x[0],x[1])),
    BoolTask("MAJ3",    3, lambda x: TT_MAJ3(x[0],x[1],x[2])),
    BoolTask("HA_SUM",  2, lambda x: TT_HA_SUM(x[0],x[1])),
    BoolTask("HA_CARRY",2, lambda x: TT_HA_CARRY(x[0],x[1])),
    BoolTask("MUX3",    3, lambda x: TT_MUX(x[0],x[1],x[2])),  # sel,a,b
]


# ========== training (degree-by-degree residual) ==========
@dataclass
class TrainedPoly:
    D: int
    max_deg: int
    pascals: Dict[int, np.ndarray]                 # degree -> Pascal table
    weights: Dict[int, np.ndarray]                 # degree -> (M_d,)
    index_map: Dict[int, List[Tuple[int,...]]]     # degree -> monomial indices

def fit_degree(D: int, deg: int, X: np.ndarray, target: np.ndarray,
               lr: float, epochs: int, bs: int) -> Tuple[np.ndarray, List[Tuple[int,...]], np.ndarray]:
    M = monomial_count(D, deg)
    P = gen_pascal(D + deg + 4, deg + 1)
    w0 = np.zeros(M, dtype=np.float32)

    w = mc_network.fit(
        X.flatten().tolist(),
        target.astype(np.float32).tolist(),
        w0.tolist(),
        float(lr), int(epochs), int(bs), int(deg), P
    )
    w = np.array(w, dtype=np.float32)

    idx_list: List[Tuple[int,...]] = [decode_monomial_indices(k, D, deg) for k in range(M)]
    return w, idx_list, P

def predict_degree(D: int, deg: int, X: np.ndarray, w: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.array(
        mc_network.predict(X.flatten().tolist(),
                           w.tolist(),
                           int(D), int(deg), P),
        dtype=np.float32
    )

def train_poly_logic(D: int, max_deg: int, X01: np.ndarray, y01: np.ndarray,
                     lr: float, epochs: int, bs: int) -> TrainedPoly:
    # noiseless → remove brakes for exact fits
    mc_network.set_train_hyperparams(lambda_=0.0, gmax=10.0)
    X = X01.astype(np.float32)
    residual = y01.astype(np.float32).copy()

    weights: Dict[int, np.ndarray] = {}
    pascals: Dict[int, np.ndarray] = {}
    index_map: Dict[int, List[Tuple[int,...]]] = {}

    for deg in range(0, max_deg + 1):
        w_d, idx_d, P_d = fit_degree(D, deg, X, residual, lr, epochs, bs)
        yhat_d = predict_degree(D, deg, X, w_d, P_d)
        residual -= yhat_d
        weights[deg]   = w_d
        pascals[deg]   = P_d
        index_map[deg] = idx_d

    return TrainedPoly(D=D, max_deg=max_deg, pascals=pascals, weights=weights, index_map=index_map)

def predict_poly(model: TrainedPoly, X: np.ndarray) -> np.ndarray:
    y = np.zeros(X.shape[0], dtype=np.float32)
    for deg in range(0, model.max_deg + 1):
        w = model.weights.get(deg)
        if w is None: continue
        P = model.pascals[deg]
        y += predict_degree(model.D, deg, X, w, P)
    return y


# ========== symbolic extraction & optimal threshold ==========
def extract_dnf(model: TrainedPoly, tau: float = 0.05) -> List[str]:
    clauses: List[str] = []
    for deg in range(0, model.max_deg + 1):
        w = model.weights.get(deg)
        if w is None: continue
        idx_map = model.index_map[deg]
        for k, wk in enumerate(w):
            if abs(wk) < tau: continue
            inds = idx_map[k]
            if len(inds) == 0:
                clauses.append(f"(BIAS≈{wk:.2f})")
            else:
                lits = [f"x{j}" for j in inds]
                mark = "" if wk > 0 else "-"
                clauses.append(f"{mark}{'*'.join(lits)} (≈{wk:.2f})")
    return clauses

def best_threshold_accuracy(y_pred: np.ndarray, y_true01: np.ndarray) -> Tuple[float, float]:
    """
    Scan thresholds between unique y_pred values; return (best_acc, best_tau).
    """
    ys = np.unique(y_pred)
    # add midpoints to separate ties better
    cuts = []
    for i in range(len(ys) - 1):
        cuts.append((ys[i] + ys[i+1]) * 0.5)
    # also test extremes
    if len(ys) > 0:
        cuts = [ys[0] - 1e-6] + cuts + [ys[-1] + 1e-6]
    best_acc, best_t = 0.0, 0.5
    for t in cuts:
        acc = float(((y_pred >= t).astype(np.float32) == y_true01).mean())
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_acc, best_t


# ========== harness ==========
def build_truth_table(task: BoolTask) -> Tuple[np.ndarray, np.ndarray]:
    n = task.n_bits
    X01 = np.zeros((1 << n, n), dtype=np.float32)
    y01 = np.zeros((1 << n,), dtype=np.float32)
    for a in range(1 << n):
        bits = to_bits(a, n)
        X01[a, :] = bits
        y01[a] = task.fn(bits)
    return X01, y01

def run_one(task: BoolTask, max_deg: int, lr: float, epochs: int, bs: int, tau_print: float):
    X01, y01 = build_truth_table(task)
    model = train_poly_logic(task.n_bits, max_deg, X01, y01, lr=lr, epochs=epochs, bs=bs)
    y_pred = predict_poly(model, X01)
    acc, tstar = best_threshold_accuracy(y_pred, y01)
    clauses = extract_dnf(model, tau=tau_print)
    return acc, tstar, clauses

def main():
    # 1-bit sanity: y = x0 must be exact
    X01 = np.array([[0],[1]], dtype=np.float32)
    y01 = np.array([0,1], dtype=np.float32)
    sanity = train_poly_logic(1, 1, X01, y01, lr=1e-3, epochs=200, bs=1)
    pred   = predict_poly(sanity, X01)
    acc_s, t_s = best_threshold_accuracy(pred, y01)
    print("Sanity y=x0 ->",
          (pred>=t_s).astype(int).tolist(),
          " expected [0, 1], acc=", acc_s, " t*=", round(float(t_s), 4))

    # robust knobs (exact tables)
    mc_network.set_train_hyperparams(lambda_=0.0, gmax=10.0)

    configs = [
        ("AND2",     2, 1e-3, 200, 1, 0.05),
        ("OR2",      2, 1e-3, 200, 1, 0.05),
        ("XOR2",     2, 1e-3, 300, 1, 0.05),
        ("XNOR2",    2, 1e-3, 300, 1, 0.05),
        ("NAND2",    2, 1e-3, 200, 1, 0.05),
        ("NOR2",     2, 1e-3, 200, 1, 0.05),
        ("MAJ3",     3, 1e-3, 300, 1, 0.05),
        ("HA_SUM",   2, 1e-3, 300, 1, 0.05),
        ("HA_CARRY", 2, 1e-3, 200, 1, 0.05),
        ("MUX3",     3, 1e-3, 300, 1, 0.05),
    ]

    name_to_task: Dict[str, BoolTask] = {t.name: t for t in TASKS}

    results = []
    t0 = time.time()
    for (name, max_deg, lr, epochs, bs, tau) in configs:
        task = name_to_task[name]
        acc, tstar, clauses = run_one(task, max_deg, lr, epochs, bs, tau)
        results.append((name, task.n_bits, max_deg, acc, len(clauses), tstar))
        print(f"\n=== {name} (n={task.n_bits}, max_deg={max_deg}) ===")
        print(f"accuracy={acc*100:.1f}%  | t*={tstar:.4f} | printed_terms={len(clauses)}")
        if clauses:
            print("Top terms:")
            for s in clauses[:8]:
                print("  ", s)

    t1 = time.time()
    print("\nSummary:")
    for (name, n, K, acc, nterms, tstar) in results:
        print(f"{name:8s}  n={n}  K={K}  acc={acc*100:5.1f}%  terms_shown={nterms:2d}  t*={tstar:.4f}")
    print(f"\nTotal time: {t1 - t0:.2f}s")


if __name__ == "__main__":
    main()
