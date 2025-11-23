# Exactus — Learn Digital Circuits from Partial Truth Tables

Exactus recovers **Boolean circuits from partial truth tables** using low‑degree polynomial models in ±1 encoding, trained on Apple Silicon (Metal). It extracts **interpretable logic** (DNF / ANF), verifies against **exhaustive truth tables**, and writes reproducible artifacts.

> Why ±1? Mapping bits to {−1,+1} turns many Boolean functions into **linear or low‑degree polynomials**, so we can learn exact logic with standard regression—no discrete search.

---

## What it does

- **Logic reconstruction from incomplete truth tables.**
- **Interpretable output:** prints minimal DNF‑style terms and ANF (parity/XOR) forms.
- **Partial supervision:** symmetry‑aware augmentation (permutations, global flips, task‑preserving transforms) to get full‑rank training from a few rows.
- **Reproducible artifacts:** CSV/JSON logs for tables, plus plots where relevant.

This repository is focused on **logic recovery**. The high‑dimensional “bitwise/combinadic” engine still exists here as infrastructure but is **not** the product story of Exactus.

---

## Key results

Run the benchmark; it prints accuracy at different label coverages `p` and saves artifacts in `results/`.

```bash
# 1) build Metal core + Python module
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
cd ..

# 2) logic recovery benchmark
python bench_logic.py --out results --timestamp 1
```

Representative output (truth‑table recovery under partial labels):

```
== AND2 (n=2, K=2) ==
  p=10%  train_rows=2→aug= 4  acc_full=100.0%  bAcc=100.0%
  ...

== XOR2 (n=2, K=2) ==
  p=10%  train_rows=2→aug= 8  acc_full=100.0%  bAcc=100.0%
  ...

== MAJ3 (n=3, K=1) ==
  p=50%  train_rows=4→aug=24  acc_full=100.0%  bAcc=100.0%

== MUX3 (n=3, K=2) ==
  p=75%  train_rows=6→aug=12  acc_full=100.0%  bAcc=100.0%
```

### Seven‑segment decoder (digits 0–9)

We model each segment (a–g) as a low‑degree polynomial over 4‑bit BCD inputs.

```bash
# Random 60% of digits labeled
python seven_segment_recovery.py --ps 0.6 --degmax 4

# Full supervision (sanity)
python seven_segment_recovery.py --ps 1.0 --degmax 4
```

Typical behavior on a Mac with Metal backend:

```
Full (10/10): a..g = 100% each, overall 100.00%   (σ_min=4.00)
Random 6/10 : overall ≈ 75–85%                    (σ_min≈5–7)
```

**Interpretation.** With all digits labeled the decoder is exactly recovered. With only 6/10 digits, accuracy depends on **which** digits you labeled. We report the **smallest singular value** σ_min of the labeled design matrix as an identifiability diagnostic: larger σ_min ⇒ better conditioning ⇒ fewer spurious fits.

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

# human‑readable checks
python boolean_validate.py            # prints DNF‑like minimal terms
python boolean_validate_parity.py     # prints ANF (⊕ form)
```

### Optional: teacher‑recovery sanity (tiny real‑valued poly)

```bash
python benchmark_solver.py
python plot_teacher_recovery.py --csv results/teacher_recovery.csv \
  --out results/teacher_recovery_plot.png
```

---

## Logic extraction (DNF / ANF)

**DNF‑style** (from `boolean_validate.py`)

- `AND2` → `x0 · x1`
- `NOR2` → `¬x0 · ¬x1`
- `MUX3(s,a,b)` → `¬s·a ∨ s·b` (appears via degree‑2 terms)

**ANF / parity** (from `boolean_validate_parity.py`)

- `XOR2` → `x0 ⊕ x1`
- `XNOR2` → `1 ⊕ x0 ⊕ x1`
- `PAR3` → `x0 ⊕ x1 ⊕ x2`

Both views are derived from the **same fitted coefficients**, just different logical bases.

---

## Partial supervision (data efficiency)

Fractional coverage `p ∈ {10%, 25%, 50%, 75%}` is evaluated with **equivariant augmentation**:

- **Permutations** of symmetric inputs (e.g., swap inputs for AND/OR/XOR).
- **Global flips** for parity‑invariant tasks (XOR/XNOR in ±1).
- **MUX3 label‑preserving transform:** `(s,a,b) → (−s, b, a)` leaves labels unchanged in ±1 and injects selector×payload terms.

This yields full‑rank, informative training without fabricating labels.

---

## How it works

- **±1 encoding.** Bits → {−1,+1}. Many gates become linear/low‑degree polynomials.
- **Degree stacking.** Fit `[0,1,2]` (or `[0,1]` for MAJ3; `[0,1,2,3,4]` for 7‑seg) sequentially as residuals.
- **Tiny SGD on Metal.** Small stepper with gradient clipping + mild L2. Deterministic and fast for small blocks.
- **Extraction.** Threshold/group coefficients to print DNF or ANF; verify on the full truth table.

> The high‑D combinadic/bitwise feature machinery remains available but is not the focus here.

---

## Reproducibility & artifacts

- `bench_logic.py` → `results/logic_bench_*.{csv,json}`
- `boolean_validate*.py` → prints formulas and verifies accuracy
- `seven_segment_recovery.py` → 7‑seg experiments with σ_min diagnostics
- `benchmark_solver.py` → `results/teacher_recovery.csv`
- `plot_teacher_recovery.py` → `results/teacher_recovery_plot.png`

Summarize to a README‑ready table:

```bash
python scripts/summarize_results.py --dir results > RESULTS_TABLE.md
```

---

## Roadmap (logic‑first)

- Export learned blocks to **Verilog/BLIF**.
- Compose larger modules: half/full adders, ripple adders, ALU slices.
- Robustness: label noise, missing/conflicting labels.
- Batched multi‑block fitting; visualization for failure modes.

---

## Cite

If this code or the logic‑recovery results help you, please cite:

```bibtex
@misc{exactus2025,
  author = {Couch, James},
  title  = {Exactus: Algebraic Logic Reconstruction on Metal},
  year   = {2025},
  howpublished = {\url{https://github.com/tjamescouch/exactus}}
}
```

**Software credits:** Apple Metal; pybind11 for Python bindings.

## License

MIT (see `LICENSE`).

