
# Exactus — Bitwise Polynomials on Metal (Logic Recovery + High-D Poly)

> **Break the Memory Wall.** Train and analyze sparse polynomial models on Apple Silicon (Metal) with interpretable logic extraction, partial-supervision recovery, and reproducible benchmarks.

This repo combines two complementary stories:

1. **High-Dimensional Polynomial Engine (Bitwise/Combinadics).** Efficient index decoding and feature generation without materializing huge design matrices (the “memory-oblivious” angle).
2. **Exact / Near-Exact Logic Recovery from Partial Truth Tables.** Small Boolean blocks (AND/OR/XOR/XNOR/MAJ/MUX) are recovered from a handful of rows using symmetry-aware augmentation and low-degree polynomial fits (K≤2) with a minimal Metal SGD backend.

Your original framing (trainer vs solver, combinadics, algebraic ridge) is preserved; this README adds **concrete scripts, artifacts, and tables** that make the logic-recovery story verifiable at a glance. :contentReference[oaicite:0]{index=0}

---

## Contents

- [Key Results](#key-results)
- [Quickstart](#quickstart)
- [Interpretable Formulas (DNF / ANF)](#interpretable-formulas-dnf--anf)
- [Partial Supervision: Data Efficiency](#partial-supervision-data-efficiency)
- [How It Works](#how-it-works)
- [Scaling & Performance](#scaling--performance)
- [Reproducibility & Artifacts](#reproducibility--artifacts)
- [Roadmap](#roadmap)
- [Cite This Work](#cite-this-work)
- [License](#license)

---

## Key Results

Run the one-shot benchmark to reproduce the table and write CSV/JSON artifacts:

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..

python bench_logic.py --out results --timestamp 1
````

You should see outputs of this shape (representative):

```
== AND2 (n=2, K=2) ==
  p=10%  train_rows=2→aug= 4  acc_full=100.0%  bAcc=100.0%
  p=25%  train_rows=2→aug= 4  acc_full=100.0%  bAcc=100.0%
  p=50%  train_rows=2→aug= 4  acc_full=100.0%  bAcc=100.0%
  p=75%  train_rows=3→aug= 6  acc_full=100.0%  bAcc=100.0%

== OR2 (n=2, K=2) ==
  p=10%  train_rows=2→aug= 4  acc_full=100.0%  bAcc=100.0%
  ...

== XOR2 (n=2, K=2) ==
  p=10%  train_rows=2→aug= 8  acc_full=100.0%  bAcc=100.0%
  ...

== XNOR2 (n=2, K=2) ==
  p=10%  train_rows=2→aug= 8  acc_full=100.0%  bAcc=100.0%
  ...

== MAJ3 (n=3, K=1) ==
  p=10%  train_rows=2→aug=12  acc_full= 50.0%  bAcc= 50.0%
  p=50%  train_rows=4→aug=24  acc_full=100.0%  bAcc=100.0%

== MUX3 (n=3, K=2) ==
  p=10%  train_rows=2→aug= 4  acc_full= 75.0%  bAcc= 75.0%
  p=75%  train_rows=6→aug=12  acc_full=100.0%  bAcc=100.0%
```

**Takeaways**

* **Exact 100% recovery** for AND/OR/XOR/XNOR at degree ≤2 using symmetry-aware augmentation, from as few as **2 distinct rows**.
* **MAJ3** is linearly separable in ±1 with bias (K=1): hits 100% once training coverage is adequate (≥50%).
* **MUX3** requires **degree-2** interactions (selector × payload). With rank-aware sampling and label-preserving augmentation, it reaches 100%.

Artifacts (CSV/JSON) are saved in `results/`. See [Reproducibility](#reproducibility--artifacts).

---

## Quickstart

### Build core + Python module

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..
```

### Benchmark logic recovery

```bash
python bench_logic.py --out results --timestamp 1
```

### Summarize to Markdown table

```bash
python scripts/summarize_results.py --dir results > RESULTS_TABLE.md
```

### Sanity checks (human-readable terms)

```bash
python boolean_validate.py            # DNF-like inspection
python boolean_validate_parity.py     # ANF / parity form
```

---

## Interpretable Formulas (DNF / ANF)

* **DNF-style** extraction (bias/linear/quadratic terms) is printed by `boolean_validate.py`.
  Examples:

  * `AND2` → minimal DNF: `x0*x1`
  * `NOR2` → minimal DNF: `¬x0*¬x1`
  * `MUX3` (selector `s`, payloads `a,b`) → terms like `¬s*a` and `s*b` appear at degree-2.

* **ANF (algebraic normal form / XOR-polynomial)** is printed by `boolean_validate_parity.py`.
  Examples:

  * `XOR2` → `x0 ⊕ x1`
  * `XNOR2` → `1 ⊕ x0 ⊕ x1`
  * `PAR3` → `x0 ⊕ x1 ⊕ x2`

These scripts use the same Metal-accelerated core but present the result in logic-friendly algebra.

---

## Partial Supervision: Data Efficiency

`bench_logic.py` evaluates **fractional coverage** `p ∈ {10%, 25%, 50%, 75%}` with **equivariant augmentation**:

* **Permutations** (e.g., swap inputs for symmetric gates),
* **Global flips** for parity-invariant tasks (XOR/XNOR),
* **Task-aware flips** for `MUX3`: `(s,a,b) → (−s, b, a)` preserves the label in ±1 encoding and injects the right cross-terms.

This turns a tiny set of labeled rows into a full-rank, informative training set **without** hallucinating labels.

---

## How It Works

* **±1 Encoding.** Inputs and labels are mapped to {−1, +1}. Many Boolean functions linearize or become low-degree polynomials in this domain.
* **Low-Degree Polynomial Stack.** Fit degrees in a schedule (e.g., `[0,1,2]`) with residual stacking: solve bias first, then add linear, then quadratic.
* **Metal Back-End (SGD).** Tiny stepper with gradient clipping and small L2; fast on Apple Silicon.
* **Bitwise/Combinadics (for scale).** For large D/degree, we use combinadic tricks to **decode monomial indices on-device** without materializing the feature map (the “memory-oblivious” approach). Algebraic ridge (closed form) is also available for smaller M.

---

## Scaling & Performance

Two operating modes:

| Mode        | Method                                           |                               Scale | Use Cases                                |
| ----------- | ------------------------------------------------ | ----------------------------------: | ---------------------------------------- |
| **Trainer** | SGD (Metal) with combinadic decoding             | Up to billions of implicit features | High-D regression, large synthetic tasks |
| **Solver**  | Algebraic ridge (w=(H^T H + \alpha I)^{-1}H^T y) |         ≤ ~45k features (RAM-bound) | Exact logic/circuits, small blocks       |

Representative times (Apple Silicon) and limits are discussed in your original write-up (preserved conceptually here).

---

## Reproducibility & Artifacts

* **Benchmark runner:** `bench_logic.py` → writes `{csv,json}` to `results/logic_bench_*.{csv,json}`
* **Summary:** `scripts/summarize_results.py` → prints a README-ready table
* **DNF / ANF:** `boolean_validate.py`, `boolean_validate_parity.py`
* **One-liner:** `./experiments/run_logic.sh results 1`

Version your `results/` and paste `RESULTS_TABLE.md` into PRs to keep the story verifiable.

---

## Roadmap

* **Exporters:** minimal **Verilog** / **BLIF** from recovered terms (ANF or DNF).
* **Multi-bit modules:** ripple adders, small ALUs (compose learned blocks).
* **GPU-first ridge:** block-wise WHT/ANF pipelines for parity-heavy tasks.
* **Visualization:** web demo rendering recovered logic and truth tables.

---

## Cite This Work

```bibtex
@misc{exactus2025,
  author = {Couch, James},
  title = {Exactus: Bitwise Polynomial Networks on Metal with Logic Recovery},
  year = {2025},
  howpublished = {\url{https://github.com/tjamescouch/mc-network}}
}
```

(Replace repo URLs/titles with your preferred canonical references if you consolidate.)

---

## License

MIT (see `LICENSE`).


::contentReference[oaicite:1]{index=1}
```
