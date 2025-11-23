#!/usr/bin/env python3
import os, sys, time, random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Callable, Optional
import numpy as np

# ---- mc_network import (repo root or ./build) ----
try:
    import mc_network  # type: ignore
except Exception:
    build_dir = os.path.join(os.path.dirname(__file__), "build")
    if os.path.isdir(build_dir) and build_dir not in sys.path:
        sys.path.insert(0, build_dir)
    import mc_network  # type: ignore

# ---------- combinatorics ----------
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

def nck(n: int, k: int) -> int:
    if k<0 or k>n: return 0
    if k==0 or k==n: return 1
    k=min(k,n-k); num=1; den=1
    for i in range(1,k+1): num *= (n-(k-i)); den *= i
    return num//den

def monomial_count(D: int, deg: int) -> int:
    return nck(D + deg - 1, deg)

def decode_monomial_indices(k: int, D: int, deg: int) -> Tuple[int, ...]:
    if deg == 0: return tuple()
    idx=[]; prev=0; rem=k
    for i in range(deg):
        for v in range(prev, D):
            rest = nck((D - v) + (deg - i - 1) - 1, (deg - i - 1))
            if rest <= rem: rem -= rest; continue
            idx.append(v); prev=v; break
    return tuple(idx)

# ---------- utils ----------
def to_bits(n: int, w: int) -> Tuple[int, ...]:
    return tuple((n >> i) & 1 for i in range(w))

def bits01_to_pm1(bits: Tuple[int, ...]) -> Tuple[int, ...]:
    return tuple(1 if b==1 else -1 for b in bits)

def y01_to_pm1(y01: int) -> int:
    return 1 if y01==1 else -1

# ---------- ±1 truth ----------
def TT_AND_01(a,b): return a & b
def TT_OR_01(a,b):  return a | b
def TT_XOR_01(a,b): return a ^ b
def TT_XNOR_01(a,b):return 1 - (a ^ b)
def TT_MAJ3_01(a,b,c): return 1 if (a+b+c)>=2 else 0

def TT_MUX_PM1(s: int, a: int, b: int) -> int:
    val = 0.5*(a+b) + 0.5*(b-a)*s
    return 1 if val >= 0 else -1

P2 = [(0,1),(1,0)]
P3 = [(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]

# ---------- tasks ----------
@dataclass
class BoolTask:
    name: str
    n_bits: int
    fn_pm1: Callable[[Tuple[int,...]], int]  # expects ±1 input, returns ±1
    perms: List[Tuple[int, ...]]
    allow_global_flip: bool
    deg_schedule: List[int]
    custom_aug: Optional[Callable[[np.ndarray], List[np.ndarray]]] = None
    custom_sampler: Optional[Callable[[np.ndarray, int, int], List[int]]] = None

TASKS: List[BoolTask] = [
    BoolTask("AND2", 2, lambda x: y01_to_pm1(TT_AND_01(x[0],x[1])), perms=P2, allow_global_flip=False, deg_schedule=[0,1,2]),
    BoolTask("OR2",  2, lambda x: y01_to_pm1(TT_OR_01(x[0],x[1])),  perms=P2, allow_global_flip=False, deg_schedule=[0,1,2]),
    BoolTask("XOR2", 2, lambda x: y01_to_pm1(TT_XOR_01(x[0],x[1])), perms=P2, allow_global_flip=True,  deg_schedule=[0,2]),
    BoolTask("XNOR2",2, lambda x: y01_to_pm1(TT_XNOR_01(x[0],x[1])),perms=P2, allow_global_flip=True,  deg_schedule=[0,2]),
    BoolTask("MAJ3", 3, lambda x: y01_to_pm1(TT_MAJ3_01(x[0],x[1],x[2])), perms=P3,
             allow_global_flip=False, deg_schedule=[0,1]),
    BoolTask("MUX3", 3, lambda x: TT_MUX_PM1(x[0],x[1],x[2]), perms=[(0,1,2)],
             allow_global_flip=False, deg_schedule=[0,1]),
]

# ---------- samplers & aug ----------
def stratified_indices_pm1(y_pm1: np.ndarray, k: int, seed: int) -> List[int]:
    rng = random.Random(seed)
    idx_neg = [i for i,v in enumerate(y_pm1) if v < 0]
    idx_pos = [i for i,v in enumerate(y_pm1) if v > 0]
    sel=[]
    if idx_neg: sel.append(rng.choice(idx_neg))
    if idx_pos and len(sel)<k: sel.append(rng.choice(idx_pos))
    pool=[i for i in range(len(y_pm1)) if i not in sel]
    rng.shuffle(pool)
    while len(sel)<min(k,len(y_pm1)) and pool:
        sel.append(pool.pop())
    sel.sort()
    return sel

def mux_sampler_rankaware(X_full: np.ndarray, k: int, seed: int) -> List[int]:
    rng = np.random.RandomState(seed)
    idx_p = [i for i,x in enumerate(X_full) if x[0] > 0]
    idx_n = [i for i,x in enumerate(X_full) if x[0] < 0]

    def pick_contrast(pool):
        if not pool: return []
        r = pool.copy(); rng.shuffle(r)
        i0 = r.pop()
        a0,b0 = X_full[i0][1], X_full[i0][2]
        for j in r:
            a1,b1 = X_full[j][1], X_full[j][2]
            if (a1 != a0) or (b1 != b0):
                return [i0, j]
        return [i0]

    sel = list(dict.fromkeys(pick_contrast(idx_p) + pick_contrast(idx_n)))
    rest = [i for i in range(len(X_full)) if i not in sel]
    rng.shuffle(rest)
    sel += rest[: max(0, k - len(sel))]
    sel = sel[:min(k, len(X_full))]
    sel.sort()
    return sel

def maj3_sampler_rankaware(X_full: np.ndarray, k: int, seed: int) -> List[int]:
    def is_pos(x): return (int(x[0]>0)+int(x[1]>0)+int(x[2]>0)) >= 2
    pos_idx = [i for i,x in enumerate(X_full) if is_pos(x)]
    neg_idx = [i for i,x in enumerate(X_full) if not is_pos(x)]
    sel = (pos_idx[:2] + neg_idx[:2])[:k]
    rest = [i for i in range(len(X_full)) if i not in sel]
    sel += rest[: max(0, k - len(sel))]
    sel = sel[:min(k, len(X_full))]
    sel.sort()
    return sel

def mux_aug_pm1(x: np.ndarray) -> List[np.ndarray]:
    s,a,b = x.tolist()
    return [x, np.array([-s, b, a], dtype=np.float32)]

# attach samplers/aug to tasks
for t in TASKS:
    if t.name == "MUX3":
        t.custom_sampler = mux_sampler_rankaware
        t.custom_aug = mux_aug_pm1
    if t.name == "MAJ3":
        t.custom_sampler = maj3_sampler_rankaware

# ---------- model ----------
@dataclass
class TrainedPoly:
    D: int
    deg_schedule: List[int]
    pascals: Dict[int, np.ndarray]
    weights: Dict[int, np.ndarray]
    index_map: Dict[int, List[Tuple[int,...]]]

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

def train_pm1(X_pm1: np.ndarray, y_pm1: np.ndarray, deg_schedule: List[int],
              lr: float, epochs0: int, epochs: int, bs: int,
              l1_tau: float) -> TrainedPoly:
    mc_network.set_train_hyperparams(lambda_=1e-3, gmax=10.0)
    D = X_pm1.shape[1]
    pascals: Dict[int, np.ndarray] = {}
    weights: Dict[int, np.ndarray] = {}
    index_map: Dict[int, List[Tuple[int,...]]] = {}

    ysum = np.zeros_like(y_pm1, dtype=np.float32)
    for i, deg in enumerate(deg_schedule):
        epo = epochs0 if (deg == 0 and i == 0) else epochs
        w_d, idx_d, P_d = fit_degree(D, deg, X_pm1, y_pm1 - ysum, lr, epo, bs)
        if l1_tau>0 and deg>0:
            s=np.sign(w_d); a=np.abs(w_d)-l1_tau; a[a<0]=0.0; w_d[:] = s*a
        ysum += predict_degree(D, deg, X_pm1, w_d, P_d)
        weights[deg]=w_d; index_map[deg]=idx_d; pascals[deg]=P_d

    return TrainedPoly(D, deg_schedule, pascals, weights, index_map)

def predict_pm1(model: TrainedPoly, X_pm1: np.ndarray) -> np.ndarray:
    y = np.zeros(X_pm1.shape[0], dtype=np.float32)
    for deg in model.deg_schedule:
        w = model.weights.get(deg)
        if w is None: continue
        y += predict_degree(model.D, deg, X_pm1, w, model.pascals[deg])
    return y

# ---------- data builders & sampling ----------
def build_truth_pm1(task: BoolTask):
    n = task.n_bits
    X = np.zeros((1<<n, n), dtype=np.float32)
    y = np.zeros((1<<n,), dtype=np.float32)
    for a in range(1<<n):
        bits = to_bits(a, n)
        X[a,:] = np.array(bits01_to_pm1(bits), dtype=np.float32)
        y[a] = float(task.fn_pm1(tuple(int(v) for v in X[a,:])))
    return X, y

def run_partial_pm1(task: BoolTask, lr: float, epochs0: int, epochs: int, bs: int,
                    l1_tau_small: float, l1_tau_large: float, p: float, seed: int=0):
    X_full, y_full = build_truth_pm1(task)
    N = X_full.shape[0]
    k = max(2, int(round(p*N)))
    if task.custom_sampler is not None:
        idx_train = task.custom_sampler(X_full, k, seed)
    else:
        idx_train = stratified_indices_pm1(y_full, k, seed)

    X_tr, y_tr = augment_orbit(X_full, y_full, idx_train, task.perms,
                               task.allow_global_flip, task.custom_aug)

    l1_tau = l1_tau_small if p <= 0.25 else l1_tau_large
    model = train_pm1(X_tr, y_tr, task.deg_schedule, lr, epochs0, epochs, bs, l1_tau)

    y_pred = predict_pm1(model, X_full)
    acc_full = float((np.sign(y_pred) == np.sign(y_full)).mean())
    pos = (y_full>0); neg = (y_full<0)
    bacc = 0.5*((np.sign(y_pred[pos])==1).mean() + (np.sign(y_pred[neg])==-1).mean())

    return acc_full, bacc, len(idx_train), X_tr.shape[0]

# ---------- experiment ----------
def main():
    configs = [
        ("AND2", 1e-3, 200, 200, 1),
        ("OR2",  1e-3, 200, 200, 1),
        ("XOR2", 1e-3, 250, 200, 1),
        ("XNOR2",1e-3, 250, 200, 1),
        ("MAJ3", 1e-3, 200, 150, 1),
        ("MUX3", 1e-3, 300, 300, 1),
    ]
    ps = [0.10, 0.25, 0.50, 0.75]
    l1_tau_small = 0.02
    l1_tau_large = 0.05

    t0 = time.time()
    for name, lr, e0, e, bs in configs:
        task = next(t for t in TASKS if t.name==name)
        print(f"\n== {name} (n={task.n_bits}, K={max(task.deg_schedule)}) ==")
        for p in ps:
            acc, bacc, k, k_aug = run_partial_pm1(task, lr, e0, e, bs,
                                                  l1_tau_small, l1_tau_large, p, seed=42)
            print(f"  p={int(p*100):2d}%  train_rows={k}→aug={k_aug:2d}  acc_full={acc*100:5.1f}%  bAcc={bacc*100:5.1f}%")
    print(f"\nTotal time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
