#!/usr/bin/env python3
import os, sys, time, random
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

# --- small utils (same as partial_supervision script) ---
def gen_pascal(rows: int, cols: int) -> np.ndarray:
    P = np.zeros((rows, cols), dtype=np.uint32); P[0,0]=1
    for n in range(1, rows):
        P[n,0]=1; kmax=min(n, cols-1)
        if kmax>=1:
            a=P[n-1,1:kmax+1]; b=P[n-1,0:kmax]; c=a+b
            if np.any((c<a)|(c<b)): raise OverflowError(f"uint32 overflow at n={n}")
            P[n,1:kmax+1]=c
        if n<cols: P[n,n]=1
    return P

def nck(n: int, k: int) -> int:
    if k<0 or k>n: return 0
    if k==0 or k==n: return 1
    k=min(k,n-k); num=1; den=1
    for i in range(1,k+1): num*= (n-(k-i)); den*=i
    return num//den

def monomial_count(D: int, deg: int) -> int:
    return nck(D + deg - 1, deg)

def decode_monomial_indices(k: int, D: int, deg: int) -> Tuple[int, ...]:
    if deg==0: return tuple()
    idx=[]; prev=0; rem=k
    for i in range(deg):
        for v in range(prev, D):
            rest = nck((D - v) + (deg - i - 1) - 1, (deg - i - 1))
            if rest <= rem: rem -= rest; continue
            idx.append(v); prev=v; break
    return tuple(idx)

def to_bits(n: int, w: int) -> Tuple[int, ...]:
    return tuple((n >> i) & 1 for i in range(w))

def expand_with_complements(X01: np.ndarray) -> np.ndarray:
    return np.concatenate([X01, 1.0 - X01], axis=1)

@dataclass
class BoolTask:
    name: str
    n_bits: int
    fn: Callable[[Tuple[int,...]], int]

def TT_AND(a,b): return a & b
def TT_OR(a,b): return a | b
def TT_XOR(a,b): return a ^ b
def TT_MAJ3(a,b,c): return 1 if (a+b+c)>=2 else 0

TASKS: List[BoolTask] = [
    BoolTask("AND2", 2, lambda x: TT_AND(x[0],x[1])),
    BoolTask("OR2",  2, lambda x: TT_OR(x[0],x[1])),
    BoolTask("XOR2", 2, lambda x: TT_XOR(x[0],x[1])),
    BoolTask("MAJ3", 3, lambda x: TT_MAJ3(x[0],x[1],x[2])),
]

@dataclass
class TrainedPoly:
    D_eff: int
    n_bits: int
    max_deg: int
    pascals: Dict[int, np.ndarray]
    weights: Dict[int, np.ndarray]
    index_map: Dict[int, List[Tuple[int,...]]]
    use_complements: bool

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

def soft_threshold_(w: np.ndarray, tau: float) -> None:
    if tau<=0: return
    s=np.sign(w); a=np.abs(w)-tau; a[a<0]=0.0; w[:]=s*a

def train_poly_logic(X01: np.ndarray, y01: np.ndarray, max_deg: int,
                     lr: float, epochs: int, bs: int, l1_tau: float,
                     use_complements: bool) -> TrainedPoly:
    mc_network.set_train_hyperparams(lambda_=0.0, gmax=10.0)
    X = expand_with_complements(X01).astype(np.float32) if use_complements else X01.astype(np.float32)
    D_eff = X.shape[1]; residual = y01.astype(np.float32).copy()
    weights: Dict[int,np.ndarray] = {}; pascals: Dict[int,np.ndarray] = {}; index_map: Dict[int,List[Tuple[int,...]]] = {}
    for deg in range(0, max_deg+1):
        w_d, idx_d, P_d = fit_degree(D_eff, deg, X, residual, lr, epochs, bs)
        if l1_tau>0: soft_threshold_(w_d, l1_tau)
        residual -= predict_degree(D_eff, deg, X, w_d, P_d)
        weights[deg]=w_d; pascals[deg]=P_d; index_map[deg]=idx_d
    return TrainedPoly(D_eff, X01.shape[1], max_deg, pascals, weights, index_map, use_complements)

def _simplify_product(inds: Tuple[int, ...], n_bits: int):
    pos, neg = set(), set()
    for t in inds:
        (pos if t<n_bits else neg).add(t if t<n_bits else t-n_bits)
    if any(j in neg for j in pos): return False, ()
    lits = [(j,+1) for j in sorted(pos)] + [(j,-1) for j in sorted(neg)]
    return True, tuple(lits)

def extract_term_set(model: TrainedPoly, tau: float=0.08) -> set:
    out=set(); n_bits=model.n_bits
    for deg in range(0, model.max_deg+1):
        w=model.weights.get(deg); idx_map=model.index_map.get(deg)
        if w is None: continue
        for k, wk in enumerate(w):
            if abs(wk) < tau: continue
            inds = idx_map[k]
            if len(inds)==0: continue
            ok, lits = _simplify_product(inds, n_bits)
            if not ok: continue
            out.add(lits)
    return out

def build_truth_table(task: BoolTask):
    n = task.n_bits
    X01 = np.zeros((1<<n, n), dtype=np.float32)
    y01 = np.zeros((1<<n,), dtype=np.float32)
    for a in range(1<<n):
        bits = to_bits(a, n)
        X01[a,:] = bits
        y01[a] = task.fn(bits)
    return X01, y01

def flip_labels(y: np.ndarray, eps: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y = y.copy().astype(np.float32)
    N = y.shape[0]
    m = int(round(eps*N))
    idx = rng.choice(N, size=m, replace=False)
    y[idx] = 1.0 - y[idx]
    return y

def run_noise(task: BoolTask, K: int, lr: float, epochs: int, bs: int,
              l1_tau: float, use_complements: bool,
              eps: float, trials: int=10, seed: int=0):
    X01, y01 = build_truth_table(task)

    # reference (noise-free) term set
    model_ref = train_poly_logic(X01, y01, K, lr, epochs, bs, l1_tau, use_complements)
    ref_terms = extract_term_set(model_ref)

    accs = []; overlaps = []
    for t in range(trials):
        y_noisy = flip_labels(y01, eps, seed+t)
        model = train_poly_logic(X01, y_noisy, K, lr, epochs, bs, l1_tau, use_complements)

        # predict full table and choose best threshold on full table for simplicity
        X_eval = expand_with_complements(X01) if use_complements else X01
        y_pred = np.zeros(X01.shape[0], dtype=np.float32)
        for deg in range(0, K+1):
            w = model.weights.get(deg)
            if w is None: continue
            y_pred += predict_degree(model.D_eff, deg, X_eval, w, model.pascals[deg])

        ys = np.unique(y_pred); cuts=[]
        for i in range(len(ys)-1): cuts.append(0.5*(ys[i]+ys[i+1]))
        if len(ys)>0: cuts = [ys[0]-1e-6] + cuts + [ys[-1]+1e-6]
        best_acc, best_t = 0.0, 0.5
        for c in cuts:
            acc = float(((y_pred >= c).astype(np.float32) == y01).mean())
            if acc>best_acc: best_acc, best_t = acc, c

        accs.append(best_acc)

        terms = extract_term_set(model)
        inter = len(terms & ref_terms); union = len(terms | ref_terms) if (terms or ref_terms) else 1
        overlaps.append(inter/union)

    return float(np.mean(accs)), float(np.std(accs)), float(np.mean(overlaps)), float(np.std(overlaps))

def main():
    mc_network.set_train_hyperparams(lambda_=0.0, gmax=10.0)
    configs = [
        ("AND2", 2, 2, 1e-3, 200, 1, True, 0.05),
        ("OR2",  2, 2, 1e-3, 200, 1, True, 0.05),
        ("XOR2", 2, 2, 1e-3, 300, 1, True, 0.05),
        ("MAJ3", 3, 3, 1e-3, 300, 1, True, 0.05),
    ]
    epsilons = [0.01, 0.03, 0.05]

    t0 = time.time()
    for name, n_bits, K, lr, epochs, bs, use_comp, l1 in configs:
        print(f"\n== {name} (n={n_bits}, K={K}) noise trials ==")
        for eps in epsilons:
            mean_acc, sd_acc, mean_overlap, sd_overlap = run_noise(
                next(t for t in TASKS if t.name==name),
                K, lr, epochs, bs, l1, use_comp, eps=eps, trials=10, seed=0
            )
            print(f"  eps={eps:.2f}  acc={mean_acc*100:5.1f}%±{sd_acc*100:3.1f}  term_overlap={mean_overlap:0.2f}±{sd_overlap:0.2f}")
    print(f"\nTotal time: {time.time()-t0:.2f}s")

if __name__ == "__main__":
    main()
