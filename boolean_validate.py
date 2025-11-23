#!/usr/bin/env python3
import os, sys, time
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable
import numpy as np

try:
    import mc_network  # type: ignore
except Exception:
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    if os.path.isdir(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import mc_network  # type: ignore

from qm_minimize import minimize_to_dnf, literals_to_str

# ---------- utilities ----------
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
            idx.append(v); prev = v; break
    return tuple(idx)

def to_bits(n: int, w: int) -> Tuple[int, ...]:
    return tuple((n >> i) & 1 for i in range(w))

def expand_with_complements(X01: np.ndarray) -> np.ndarray:
    """Expand each bit x -> [x, 1-x]. Shape: [N, 2n]."""
    Xc = 1.0 - X01
    return np.concatenate([X01, Xc], axis=1)

def literal_name(j: int, n_bits: int) -> str:
    if j < n_bits:
        return f"x{j}"
    else:
        return f"¬x{j - n_bits}"

# ---------- boolean tasks ----------
@dataclass
class BoolTask:
    name: str
    n_bits: int
    fn: Callable[[Tuple[int,...]], int]

def TT_AND(a,b): return a & b
def TT_OR(a,b): return a | b
def TT_NAND(a,b): return 1 - (a & b)
def TT_NOR(a,b): return 1 - (a | b)
def TT_XOR(a,b): return a ^ b
def TT_XNOR(a,b): return 1 - (a ^ b)
def TT_MAJ3(a,b,c): return 1 if (a+b+c) >= 2 else 0
def TT_HA_SUM(a,b): return a ^ b
def TT_HA_CARRY(a,b): return a & b
def TT_MUX(s,a,b): return a if s==0 else b

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
    BoolTask("MUX3",    3, lambda x: TT_MUX(x[0],x[1],x[2])),
]

# ---------- training ----------
@dataclass
class TrainedPoly:
    D_effective: int           # 2*n_bits if complements used, else n_bits
    n_bits: int                # original bits
    max_deg: int
    pascals: Dict[int, np.ndarray]
    weights: Dict[int, np.ndarray]
    index_map: Dict[int, List[Tuple[int,...]]]
    use_complements: bool

def fit_degree(D: int, deg: int, X: np.ndarray, target: np.ndarray,
               lr: float, epochs: int, bs: int) -> Tuple[np.ndarray, List[Tuple[int,...]], np.ndarray]:
    M = monomial_count(D, deg)
    P = gen_pascal(D + deg + 4, deg + 1)
    w0 = np.zeros(M, dtype=np.float32)
    w = mc_network.fit(X.flatten().tolist(),
                       target.astype(np.float32).tolist(),
                       w0.tolist(),
                       float(lr), int(epochs), int(bs), int(deg), P)
    w = np.array(w, dtype=np.float32)
    idx = [decode_monomial_indices(k, D, deg) for k in range(M)]
    return w, idx, P

def predict_degree(D: int, deg: int, X: np.ndarray, w: np.ndarray, P: np.ndarray) -> np.ndarray:
    return np.array(mc_network.predict(X.flatten().tolist(),
                                       w.tolist(),
                                       int(D), int(deg), P), dtype=np.float32)

def soft_threshold_(w: np.ndarray, tau: float) -> None:
    if tau <= 0: return
    s = np.sign(w); a = np.abs(w) - tau
    a[a < 0] = 0.0
    w[:] = s * a

def train_poly_logic(X01: np.ndarray, y01: np.ndarray,
                     max_deg: int, lr: float, epochs: int, bs: int,
                     l1_tau: float, use_complements: bool) -> TrainedPoly:
    mc_network.set_train_hyperparams(lambda_=0.0, gmax=10.0)

    if use_complements:
        X = expand_with_complements(X01).astype(np.float32)
        D_eff = X.shape[1]
    else:
        X = X01.astype(np.float32)
        D_eff = X.shape[1]

    residual = y01.astype(np.float32).copy()

    weights: Dict[int, np.ndarray] = {}
    pascals: Dict[int, np.ndarray] = {}
    index_map: Dict[int, List[Tuple[int,...]]] = {}

    for deg in range(0, max_deg + 1):
        w_d, idx_d, P_d = fit_degree(D_eff, deg, X, residual, lr, epochs, bs)
        if l1_tau > 0.0:
            soft_threshold_(w_d, l1_tau)
        yhat_d = predict_degree(D_eff, deg, X, w_d, P_d)
        residual -= yhat_d
        weights[deg] = w_d; pascals[deg] = P_d; index_map[deg] = idx_d

    return TrainedPoly(D_effective=D_eff,
                       n_bits=X01.shape[1],
                       max_deg=max_deg,
                       pascals=pascals,
                       weights=weights,
                       index_map=index_map,
                       use_complements=use_complements)

def predict_poly(model: TrainedPoly, X01: np.ndarray) -> np.ndarray:
    X = expand_with_complements(X01).astype(np.float32) if model.use_complements else X01.astype(np.float32)
    y = np.zeros(X.shape[0], dtype=np.float32)
    for deg in range(0, model.max_deg + 1):
        w = model.weights.get(deg)
        if w is None: continue
        y += predict_degree(model.D_effective, deg, X, w, model.pascals[deg])
    return y

# --- drop-in: replace your current extract_dnf with this version ---

def _simplify_product(inds: Tuple[int, ...], n_bits: int) -> Tuple[bool, Tuple[Tuple[int,int], ...]]:
    """
    inds: monomial indices in the expanded feature space:
          [0..n_bits-1]  -> x_j
          [n_bits..2n-1] -> ¬x_j  (encoded as 1 - x_j)
    Returns:
      (is_valid, literals)
      where literals is a sorted tuple of (var_idx, polarity), polarity ∈ {+1 (x_j), -1 (¬x_j)}.
      If the product contains both x_j and ¬x_j, it's a contradiction -> (False, ()).
      Repeated occurrences collapse (idempotence).
    """
    pos = set()
    neg = set()
    for t in inds:
        if t < n_bits:
            pos.add(t)
        else:
            neg.add(t - n_bits)
    # contradiction?
    if any(j in neg for j in pos):
        return False, ()
    # collapse repeats (idempotence) by using sets
    lits = []
    for j in sorted(pos):
        lits.append((j, +1))
    for j in sorted(neg):
        lits.append((j, -1))
    return True, tuple(lits)

def _lits_to_str(lits: Tuple[Tuple[int,int], ...]) -> str:
    if not lits:
        return "1"  # tautological product (rare; corresponds to pure bias term)
    parts = []
    for (j, pol) in lits:
        parts.append(f"x{j}" if pol > 0 else f"¬x{j}")
    return "*".join(parts)

def extract_dnf(model: TrainedPoly, tau: float = 0.05) -> List[str]:
    """
    Prints simplified Boolean products:
      - collapse powers: x*x -> x
      - remove contradictions: x*¬x -> 0 (drop clause)
      - dedup literals
    """
    clauses: List[str] = []
    n_bits = model.n_bits
    for deg in range(0, model.max_deg + 1):
        w = model.weights.get(deg)
        if w is None: 
            continue
        idx_map = model.index_map[deg]
        for k, wk in enumerate(w):
            if abs(wk) < tau:
                continue
            inds = idx_map[k]
            if len(inds) == 0:
                # bias term
                clauses.append(f"(BIAS≈{wk:.2f})")
                continue
            ok, lits = _simplify_product(inds, n_bits)
            if not ok:
                # contradiction -> skip
                continue
            s = _lits_to_str(lits)
            # sign of weight indicates contribution direction
            clauses.append((s if wk > 0 else f"-{s}") + f" (≈{wk:.2f})")
    return clauses


def best_threshold_accuracy(y_pred: np.ndarray, y_true01: np.ndarray) -> Tuple[float, float]:
    ys = np.unique(y_pred)
    cuts = []
    for i in range(len(ys) - 1):
        cuts.append(0.5*(ys[i] + ys[i+1]))
    if len(ys) > 0:
        cuts = [ys[0]-1e-6] + cuts + [ys[-1]+1e-6]
    best_acc, best_t = 0.0, 0.5
    for t in cuts:
        acc = float(((y_pred >= t).astype(np.float32) == y_true01).mean())
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_acc, best_t

# ---------- harness ----------
def build_truth_table(task: BoolTask) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    n = task.n_bits
    X01 = np.zeros((1 << n, n), dtype=np.float32)
    y01 = np.zeros((1 << n,), dtype=np.float32)
    on_set = []
    for a in range(1 << n):
        bits = to_bits(a, n)
        X01[a, :] = bits
        y = task.fn(bits)
        y01[a] = y
        if y == 1:
            on_set.append(a)
    return X01, y01, on_set

def run_one(task: BoolTask, max_deg: int, lr: float, epochs: int, bs: int,
            tau_print: float, l1_tau: float, use_complements: bool):
    X01, y01, on_set = build_truth_table(task)
    model = train_poly_logic(X01, y01, max_deg, lr, epochs, bs, l1_tau, use_complements)
    y_pred = predict_poly(model, X01)
    acc, tstar = best_threshold_accuracy(y_pred, y01)
    clauses = extract_dnf(model, tau=tau_print)
    dnf_min = minimize_to_dnf(task.n_bits, on_set, dc_set=[])
    return acc, tstar, clauses, dnf_min

def main():
    # sanity
    X01 = np.array([[0],[1]], dtype=np.float32)
    y01 = np.array([0,1], dtype=np.float32)
    m = train_poly_logic(X01, y01, max_deg=1, lr=1e-3, epochs=200, bs=1, l1_tau=0.05, use_complements=True)
    pred = predict_poly(m, X01)
    acc_s, t_s = best_threshold_accuracy(pred, y01)
    print("Sanity y=x0 ->", (pred>=t_s).astype(int).tolist(), " acc=", acc_s, " t*=", round(float(t_s), 4))

    mc_network.set_train_hyperparams(lambda_=0.0, gmax=10.0)

    # Strong settings + complements for exact DNF, n<=3 demos
    configs = [
        ("AND2",     2, 1e-3, 200, 1, 0.05, 0.05, True),
        ("OR2",      2, 1e-3, 200, 1, 0.05, 0.05, True),
        ("XOR2",     2, 1e-3, 300, 1, 0.05, 0.05, True),
        ("XNOR2",    2, 1e-3, 300, 1, 0.05, 0.05, True),  # now solvable exactly
        ("NAND2",    2, 1e-3, 200, 1, 0.05, 0.05, True),
        ("NOR2",     2, 1e-3, 200, 1, 0.05, 0.05, True),
        ("MAJ3",     3, 1e-3, 300, 1, 0.05, 0.05, True),
        ("HA_SUM",   2, 1e-3, 300, 1, 0.05, 0.05, True),
        ("HA_CARRY", 2, 1e-3, 200, 1, 0.05, 0.05, True),
        ("MUX3",     3, 1e-3, 300, 1, 0.05, 0.05, True),
    ]

    name_to_task: Dict[str, BoolTask] = {t.name: t for t in TASKS}

    t0 = time.time()
    for (name, K, lr, epochs, bs, tau, l1_tau, use_comp) in configs:
        task = name_to_task[name]
        acc, tstar, clauses, dnf_min = run_one(task, K, lr, epochs, bs, tau, l1_tau, use_comp)
        print(f"\n=== {name} (n={task.n_bits}, K={K}) ===")
        print(f"acc={acc*100:.1f}% | t*={tstar:.4f} | printed_terms={len(clauses)} | minimal_terms={len(dnf_min)}")
        if clauses:
            print("Extracted terms:")
            for s in clauses[:8]:
                print("  ", s)
        if dnf_min:
            print("Minimal DNF:")
            for lits in dnf_min:
                print("  ", literals_to_str(lits))

    t1 = time.time()
    print(f"\nTotal time: {t1 - t0:.2f}s")

if __name__ == "__main__":
    main()
