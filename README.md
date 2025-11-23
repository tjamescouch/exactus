Here’s a full drop-in README that’s lean and centered on **algebraic logic reconstruction** while keeping only what’s necessary.

````markdown
# Exactus — Algebraic Logic Reconstruction on Metal

Exactus recovers Boolean logic from partial truth tables using **low-degree ±1 polynomial models** trained on Apple Silicon (Metal). It extracts **interpretable formulas** (DNF and ANF), validates with exhaustive truth tables, and writes reproducible artifacts.

> Core idea: encode inputs/labels in {−1,+1}; many gates become **linear or low-degree polynomials**. Fit tiny models, apply **symmetry-aware augmentation**, and read back logic.

---

## Contents
- [Key Results](#key-results)
- [Quickstart](#quickstart)
- [Logic Extraction (DNF / ANF)](#logic-extraction-dnf--anf)
- [Partial Supervision (Data Efficiency)](#partial-supervision-data-efficiency)
- [How It Works](#how-it-works)
- [Reproducibility & Artifacts](#reproducibility--artifacts)
- [Roadmap (Logic-first)](#roadmap-logicfirst)
- [Cite](#cite)
- [License](#license)

---

## Key Results

Run the benchmark; it prints accuracy at different label coverages `p` and saves artifacts.

```bash
# build Metal core + py module
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..

# logic recovery benchmark (writes to results/)
python bench_logic.py --out results --timestamp 1
````

Representative output:

```
== AND2 (n=2, K=2) ==
  p=10%  train_rows=2→aug= 4  acc_full=100.0%  bAcc=100.0%
  ...

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
  p=50%  train_rows=4→aug=24  acc_full=100.0%  bAcc=100.0%

== MUX3 (n=3, K=2) ==
  p=75%  train_rows=6→aug=12  acc_full=100.0%  bAcc=100.0%
```

**Takeaways**

* **AND/OR/XOR/XNOR:** 100% exact recovery at **degree ≤ 2**, from **2 labeled rows** + augmentation.
* **MAJ3:** linear with bias in ±1; reaches 100% once augmented coverage is adequate.
* **MUX3:** needs **degree-2 selector×payload** terms; rank-aware sampling + label-preserving aug hits 100%.

---

## Quickstart

### Build

```bash
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..
```

### Benchmarks

```bash
# logic recovery table
python bench_logic.py --out results --timestamp 1

# human-readable checks
python boolean_validate.py            # prints DNF-like minimal terms
python boolean_validate_parity.py     # prints ANF (⊕ form)
```

### Optional: teacher recovery sanity (tiny real-valued poly)

```bash
python benchmark_solver.py
python plot_teacher_recovery.py --csv results/teacher_recovery.csv \
  --out results/teacher_recovery_plot.png
```

---

## Logic Extraction (DNF / ANF)

**DNF-style** (from `boolean_validate.py`)

* `AND2` → minimal DNF: `x0 * x1`
* `NOR2` → minimal DNF: `¬x0 * ¬x1`
* `MUX3(s,a,b)` → minimal DNF: `¬s*a` and `s*b` (appears via degree-2 terms)

**ANF / parity** (from `boolean_validate_parity.py`)

* `XOR2` → `x0 ⊕ x1`
* `XNOR2` → `1 ⊕ x0 ⊕ x1`
* `PAR3` → `x0 ⊕ x1 ⊕ x2`

Both scripts use the same fitted coefficients, but display them in logic-friendly bases.

---

## Partial Supervision (Data Efficiency)

`bench_logic.py` evaluates fractional coverage `p ∈ {10%, 25%, 50%, 75%}` with **equivariant augmentation**:

* **Permutations** for symmetric gates (e.g., swap inputs).
* **Global flips** for parity-invariant tasks (XOR/XNOR).
* **MUX3 label-preserving aug**: `(s,a,b) → (−s, b, a)` in ±1 keeps the label and injects the correct cross-terms.

This converts a few labeled rows into an informative, full-rank training set **without fabricated labels**.

---

## How It Works

* **±1 encoding.** Map bits/labels to {−1,+1}. Several Boolean functions become linear/low-degree polynomials.
* **Low-degree stack.** Fit degree schedule `[0,1,2]` (or `[0,1]` for MAJ3) by residual stacking: bias → linear → quadratic.
* **Tiny SGD on Metal.** Small stepper with gradient clipping + mild L2. Fast enough for these blocks; deterministic and reproducible.
* **Extraction.** Coefficients are thresholded/grouped to print DNF or ANF terms; verification is by exhaustive truth tables.

> High-D “combinadics/bitwise” machinery exists in the repo for larger polynomial problems, but the **logic story** stands alone and is fully verifiable here.

---

## Reproducibility & Artifacts

* `bench_logic.py` → `results/logic_bench_*.{csv,json}`
* `boolean_validate*.py` → prints formulas, also verifies accuracy
* `benchmark_solver.py` → `results/teacher_recovery.csv`
* `plot_teacher_recovery.py` → `results/teacher_recovery_plot.png`

Summarize to a README-ready table:

```bash
python scripts/summarize_results.py --dir results > RESULTS_TABLE.md
```

---

## Roadmap (Logic-first)

* Export learned blocks to **Verilog/BLIF** (from DNF/ANF).
* Compose blocks: half/full adders, small ALU slices, ripple constructions.
* Robustness: label noise, adversarial missing rows, conflicting labels.
* Speed: batched many-gate recovery; multi-block joint fitting.

---

## Cite

If this code or the logic-recovery results help you, please cite:

```bibtex
@misc{exactus2025,
  author = {Couch, James},
  title  = {Exactus: Algebraic Logic Reconstruction on Metal},
  year   = {2025},
  howpublished = {\url{https://github.com/tjamescouch/mc-network}}
}
```

**Software credits:** Apple Metal; pybind11 for Python bindings.

---

## License

MIT (see `LICENSE`).

```
