#!/usr/bin/env python3
"""
compose_recovery.py
Partial-supervision logic reconstruction with label-preserving augmentation
and *deterministic informative samplers* for FA and 2-bit ripple adders.

Run:
  python compose_recovery.py
"""

import os, sys, random
from dataclasses import dataclass
from typing import Callable, List, Tuple, Dict
import numpy as np

# ---- import built module (repo root or ./build) ----
try:
    import mc_network  # type: ignore
except Exception:
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    if os.path.isdir(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import mc_network  # type: ignore

# keep step sizes conservative
mc_network.set_train_hyperparams(lambda_=1e-3, gmax=10.0)

# ---------- small combinatorics ----------
def nck(n: int, k: int) -> int:
    if k < 0 or k > n: return 0
    if k == 0 or k == n: return 1
    k = min(k, n - k)
    num = 1; den = 1
    for i in range(1, k + 1):
        num *= (n - (k - i))
        den *= i
    return num // den

def monomial_count(D: int, deg: int) -> int:
    return nck(D + deg - 1, deg)

def gen_pascal(rows: int, cols: int) -> np.ndarray:
    P = np.zeros((rows, cols), dtype=np.uint32)
    P[0, 0] = 1
    for r in range(1, rows):
        P[r, 0] = 1
        kmax = min(r, cols - 1)
        if kmax >= 1:
            a = P[r - 1, 1:kmax + 1]
            b = P[r - 1, 0:kmax]
            c = a + b
            if np.any((c < a) | (c < b)):
                raise OverflowError("uint32 overflow in Pascal table")
            P[r, 1:kmax + 1] = c
        if r < cols:
            P[r, r] = 1
    return P

# ---------- encoding helpers ----------
def to_bits(n: int, w: int) -> Tuple[int, ...]:
    return tuple((n >> i) & 1 for i in range(w))

def bits01_to_pm1(bits: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(1 if b == 1 else -1 for b in bits)

def y01_to_pm1(y: int) -> int:
    return 1 if y == 1 else -1

# ---------- tiny polynomial stack ----------
@dataclass
class PolyModel:
    D: int
    deg_schedule: List[int]
    W: Dict[int, np.ndarray]
    P: Dict[int, np.ndarray]

def fit_degree(D: int, deg: int, X: np.ndarray, r: np.ndarray,
               lr: float, epochs: int, bs: int, P: np.ndarray) -> np.ndarray:
    M = monomial_count(D, deg)
    w0 = np.zeros(M, dtype=np.float32)
    w = mc_network.fit(
        X.flatten().tolist(),
        r.astype(np.float32).tolist(),
        w0.tolist(),
        float(lr), int(epochs), int(bs), int(deg),
        P
    )
    return np.array(w, dtype=np.float32)

def predict_degree(D: int, deg: int, X: np.ndarray, w: np.ndarray, P: np.ndarray) -> np.ndarray:
    y = mc_network.predict(X.flatten().tolist(), w.tolist(), int(D), int(deg), P)
    return np.array(y, dtype=np.float32)

def train_poly_pm1(X: np.ndarray, y: np.ndarray, deg_schedule: List[int],
                   lr: float = 1e-3, epochs0: int = 160, epochs: int = 180, bs: int = 2) -> PolyModel:
    D = X.shape[1]
    W: Dict[int, np.ndarray] = {}
    P: Dict[int, np.ndarray] = {}
    ysum = np.zeros_like(y, dtype=np.float32)

    for i, deg in enumerate(deg_schedule):
        epo = epochs0 if (deg == 0 and i == 0) else epochs
        pascal = gen_pascal(D + deg + 4, deg + 1)
        w_d = fit_degree(D, deg, X, y - ysum, lr, epo, bs, pascal)
        ysum += predict_degree(D, deg, X, w_d, pascal)
        W[deg] = w_d; P[deg] = pascal
    return PolyModel(D, deg_schedule, W, P)

def predict_poly_pm1(model: PolyModel, X: np.ndarray) -> np.ndarray:
    y = np.zeros(X.shape[0], dtype=np.float32)
    for deg in model.deg_schedule:
        y += predict_degree(model.D, deg, X, model.W[deg], model.P[deg])
    return y

# ---------- gates & modules ----------
def AND(a, b): return a & b
def OR(a, b):  return a | b
def XOR(a, b): return a ^ b
def MAJ3(a,b,c): return 1 if (a+b+c) >= 2 else 0

def half_adder(a,b):
    s = XOR(a,b)
    c = AND(a,b)
    return s, c

def full_adder(a,b,cin):
    s1 = XOR(a,b)
    c1 = AND(a,b)
    s  = XOR(s1,cin)          # xor-of-three
    c2 = AND(s1,cin)
    cout = OR(c1,c2)          # = MAJ3(a,b,cin)
    return s, cout

def ripple_adder_2bit(a1,a0,b1,b0):
    s0, c0 = full_adder(a0,b0,0)
    s1, c1 = full_adder(a1,b1,c0)
    return s1, s0, c1  # (MSB sum, LSB sum, final carry)

@dataclass
class Module:
    name: str
    n_in: int
    fn: Callable[[Tuple[int,...]], Tuple[int,...]]
    deg_schedules: List[List[int]]  # per-output

# Single gates
GATES = [
    Module("AND2", 2, lambda x: (AND(x[0],x[1]),),        [[0,1,2]]),
    Module("XOR2", 2, lambda x: (XOR(x[0],x[1]),),        [[0,2]]),
    Module("MAJ3", 3, lambda x: (MAJ3(x[0],x[1],x[2]),),  [[0,1]]),
]

# Composed modules (per-output degree schedules)
MODULES = [
    Module("HA",   2, lambda x: half_adder(x[0],x[1]),                 [[0,2], [0,1,2]]),
    Module("FA",   3, lambda x: full_adder(x[0],x[1],x[2]),            [[0,1,2,3], [0,1]]),
    Module("RA2",  4, lambda x: ripple_adder_2bit(x[0],x[1],x[2],x[3]),[[0,1,2,3], [0,2], [0,1]]),
]

# ---------- data builders ----------
def build_truth_pm1(n_in: int, fn: Callable[[Tuple[int,...]], Tuple[int,...]]):
    X = []
    Ys: List[List[int]] = []
    out_arity = None
    for a in range(1 << n_in):
        bits = to_bits(a, n_in)
        outs = fn(bits)
        if out_arity is None:
            out_arity = len(outs)
            Ys = [[] for _ in range(out_arity)]
        X.append(bits01_to_pm1(bits))
        for j, yj in enumerate(outs):
            Ys[j].append(y01_to_pm1(yj))
    X = np.array(X, dtype=np.float32)
    Y = np.stack([np.array(col, dtype=np.float32) for col in Ys], axis=1)
    return X, Y  # ±1

# ---------- label-preserving augmentations ----------
def aug_inputs(mod_name: str, x_pm1: np.ndarray) -> List[np.ndarray]:
    if mod_name in ("AND2", "XOR2", "HA"):
        # swap a<->b
        return [x_pm1, x_pm1[[1,0]]]

    if mod_name == "MAJ3":
        # full S3 symmetry
        return [x_pm1,
                x_pm1[[1,0,2]], x_pm1[[0,2,1]],
                x_pm1[[2,0,1]], x_pm1[[1,2,0]], x_pm1[[2,1,0]]]

    if mod_name == "FA":
        # BOTH outputs symmetric in (a,b,cin): full S3 permutations
        i = [0,1,2]
        perms = [
            i, [1,0,2], [0,2,1], [2,0,1], [1,2,0], [2,1,0]
        ]
        return [x_pm1[p] for p in perms]

    if mod_name == "RA2":
        # swap operands A<->B; (a1,a0,b1,b0) -> (b1,b0,a1,a0)
        return [x_pm1, x_pm1[[2,3,0,1]]]

    return [x_pm1]

def augment_rows(X: np.ndarray, y: np.ndarray, idx: List[int], mod_name: str):
    Xa = []; ya = []
    for i in idx:
        for xr in aug_inputs(mod_name, X[i]):
            Xa.append(xr); ya.append(y[i])
    return np.stack(Xa,0), np.array(ya, dtype=np.float32)

# ---------- informative samplers ----------
def sampler_AND2(X: np.ndarray) -> List[int]:
    pos = [i for i,x in enumerate(X) if (x[0] > 0 and x[1] > 0)]
    neg = [i for i,x in enumerate(X) if not (x[0] > 0 and x[1] > 0)]
    sel = []
    if pos: sel.append(pos[0])
    if neg: sel.append(neg[0])
    return sorted(list(dict.fromkeys(sel)))

def sampler_XOR2(X: np.ndarray) -> List[int]:
    diff = [i for i,x in enumerate(X) if x[0] != x[1]]
    same = [i for i,x in enumerate(X) if x[0] == x[1]]
    sel = []
    if diff: sel.append(diff[0])
    if same: sel.append(same[0])
    return sorted(sel)

def sampler_MAJ3(X: np.ndarray) -> List[int]:
    ge2 = [i for i,x in enumerate(X) if (x>0).sum() >= 2]
    le1 = [i for i,x in enumerate(X) if (x>0).sum() <= 1]
    sel = []
    if ge2: sel.append(ge2[0])
    if le1: sel.append(le1[0])
    if len(ge2) > 1: sel.append(ge2[-1])
    if len(le1) > 1: sel.append(le1[-1])
    return sorted(list(dict.fromkeys(sel)))

def sampler_HA(X: np.ndarray) -> List[int]:
    a1b1 = [i for i,x in enumerate(X) if x[0] > 0 and x[1] > 0]    # carry=1
    diff  = [i for i,x in enumerate(X) if x[0] != x[1]]            # sum=1
    eq    = [i for i,x in enumerate(X) if x[0] == x[1]]            # sum=0
    sel = []
    if a1b1: sel.append(a1b1[0])
    if diff:  sel.append(diff[0])
    if eq:    sel.append(eq[0])
    return sorted(list(dict.fromkeys(sel)))

def _idx_RA2_find(X: np.ndarray, a1,a0,b1,b0) -> int:
    # X rows are ±1; map to bits; return index or -1
    for i,x in enumerate(X):
        if (x[0] > 0) == (a1==1) and (x[1] > 0) == (a0==1) and \
           (x[2] > 0) == (b1==1) and (x[3] > 0) == (b0==1):
            return i
    return -1

def sampler_FA(X: np.ndarray) -> List[int]:
    # Cover all Hamming classes for (a,b,cin): weight 1, weight 2, and all-same
    w1 = [i for i,x in enumerate(X) if (x>0).sum()==1]
    w2 = [i for i,x in enumerate(X) if (x>0).sum()==2]
    ws = [i for i,x in enumerate(X) if (x>0).sum() in (0,3)]
    sel = []
    if w1: sel.append(w1[0])   # xor=1, maj=0
    if w2: sel.append(w2[0])   # xor=1, maj=1
    if ws: sel.append(ws[0])   # xor=0, maj in {0,1}
    # Prefer a second ws example with opposite majority if available
    if len(ws) > 1:
        sel.append(ws[-1])
    return sorted(list(dict.fromkeys(sel)))

def sampler_RA2(X: np.ndarray) -> List[int]:
    # Inputs: (a1,a0,b1,b0). Ensure:
    #  (1) LSB K/P/G classes with MSB zeros,
    #  (2) a true propagate into MSB (c0=1 & a1 xor b1 = 1),
    #  (3) a no-carry MSB contrast so s1 flips with/without c0.
    idxs: List[int] = []

    # (1) LSB classes with a1=b1=0: K(00), P(01), P(10), G(11)
    for (a0,b0) in [(0,0), (0,1), (1,0), (1,1)]:
        j = _idx_RA2_find(X, 0,a0, 0,b0)
        if j != -1: idxs.append(j)

    # (2) Propagate into MSB: make c0=1 (LSB=11) and choose (a1,b1)=(0,1) so MSB depends on c0
    j2 = _idx_RA2_find(X, 0,1, 1,1)
    if j2 == -1:
        j2 = _idx_RA2_find(X, 1,1, 0,1)
    if j2 != -1: idxs.append(j2)

    # (3) No-carry MSB contrast: c0=0 & a1 xor b1 = 1 (e.g., LSB=00, a1=0,b1=1)
    j3 = _idx_RA2_find(X, 0,0, 0,1)
    if j3 != -1: idxs.append(j3)

    # Dedup + sort
    return sorted(list(dict.fromkeys(idxs)))

SAMPLER = {
    "AND2": sampler_AND2,
    "XOR2": sampler_XOR2,
    "MAJ3": sampler_MAJ3,
    "HA":   sampler_HA,
    "FA":   sampler_FA,
    "RA2":  sampler_RA2,
}

# ---------- experiment core ----------
def augment_rows(X: np.ndarray, y: np.ndarray, idx: List[int], mod_name: str):
    Xa = []; ya = []
    for i in idx:
        for xr in aug_inputs(mod_name, X[i]):
            Xa.append(xr); ya.append(y[i])
    return np.stack(Xa,0), np.array(ya, dtype=np.float32)

def train_output_bit(module: Module, X: np.ndarray, y_bit: np.ndarray,
                     deg_schedule: List[int], p: float, seed: int,
                     lr=1e-3, e0=160, e=180, bs=2):
    N = X.shape[0]
    k = max(2, int(round(p * N)))

    base = SAMPLER.get(module.name, None)
    if base is not None:
        idx_seed = base(X)
    else:
        rng = random.Random(seed)
        idx_neg = [i for i,v in enumerate(y_bit) if v < 0]
        idx_pos = [i for i,v in enumerate(y_bit) if v > 0]
        idx_seed = []
        if idx_neg: idx_seed.append(idx_neg[0])
        if idx_pos: idx_seed.append(idx_pos[0])
        pool = [i for i in range(N) if i not in idx_seed]
        rng.shuffle(pool)
        idx_seed += pool[:max(0, k - len(idx_seed))]

    # trim/extend deterministically to k
    if len(idx_seed) >= k:
        idx = idx_seed[:k]
    else:
        extra = [i for i in range(N) if i not in idx_seed]
        random.Random(seed).shuffle(extra)
        idx = idx_seed + extra[:(k - len(idx_seed))]

    Xtr, ytr = augment_rows(X, y_bit, idx, module.name)
    model = train_poly_pm1(Xtr, ytr, deg_schedule, lr=lr, epochs0=e0, epochs=e, bs=bs)
    ypred = np.sign(predict_poly_pm1(model, X))
    acc = float((ypred == np.sign(y_bit)).mean())
    return {"train_rows": len(idx), "aug_rows": int(Xtr.shape[0]), "acc": acc}

def run_demo(ps=(0.25, 0.50, 0.75), seed=123, lr=1e-3, e0=160, e=180, bs=2):
    print("\n=== Single gates ===")
    for g in GATES:
        X, Y = build_truth_pm1(g.n_in, g.fn)
        print(f"\n[{g.name}] n_in={g.n_in}   truth_rows={len(X)}")
        for p in ps:
            r = train_output_bit(g, X, Y[:,0], g.deg_schedules[0], p, seed, lr, e0, e, bs)
            print(f"  p={int(100*p):2d}%  labeled={r['train_rows']:2d}→aug={r['aug_rows']:2d}  acc={r['acc']*100:6.2f}%")

    print("\n=== Composed modules ===")
    for m in MODULES:
        X, Y = build_truth_pm1(m.n_in, m.fn)
        n_out = Y.shape[1]
        print(f"\n[{m.name}] n_in={m.n_in}  outputs={n_out}  truth_rows={len(X)}")
        for p in ps:
            row = []
            for j in range(n_out):
                r = train_output_bit(m, X, Y[:,j], m.deg_schedules[j], p, seed, lr, e0, e, bs)
                row.append((r['train_rows'], r['aug_rows'], r['acc']))
            outs = " | ".join(
                [f"y{j}: p={int(100*p):2d}%  L={L:2d}→A={A:2d}  acc={acc*100:6.2f}%"
                 for j,(L,A,acc) in enumerate(row)]
            )
            print("  " + outs)

if __name__ == "__main__":
    run_demo(ps=(0.25, 0.50, 0.75), seed=123, lr=1e-3, e0=160, e=180, bs=2)
