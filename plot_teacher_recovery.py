#!/usr/bin/env python3
import os, csv
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = os.path.join("results", "teacher_recovery.csv")
OUT_PATH  = os.path.join("results", "teacher_recovery_plot.png")

def load_rows(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No CSV at {path}")
    rows = []
    with open(path, "r") as f:
        R = list(csv.DictReader(f))
        for r in R:
            try:
                mse = float(r.get("mse", "nan"))
                rel = float(r.get("rel_L2", "nan"))
                cos = float(r.get("cos_sim", "nan"))
                rows.append((mse, rel, cos))
            except Exception:
                continue
    return rows

def main():
    rows = load_rows(CSV_PATH)
    if not rows:
        print("No valid rows.")
        return
    mse = np.array([x[0] for x in rows], dtype=float)
    rel = np.array([x[1] for x in rows], dtype=float)
    cos = np.array([x[2] for x in rows], dtype=float)

    # Filter non-finite
    good_mse = np.isfinite(mse) & (mse >= 0)
    good_rel = np.isfinite(rel) & (rel >= 0)

    x = np.arange(len(mse))[good_mse]
    mse_good = mse[good_mse]

    fig = plt.figure(figsize=(12,4.5))

    ax1 = fig.add_subplot(1,2,1)
    ax1.scatter(x, mse_good, s=20)
    ax1.set_yscale("log")
    ax1.set_xlabel("trial index")
    ax1.set_ylabel("MSE")
    ax1.set_title("Teacher recovery MSE")
    # Show a simple success threshold line (e.g., 1e-3)
    ax1.axhline(1e-3, ls="--", lw=1)

    ax2 = fig.add_subplot(1,2,2)
    if good_rel.any():
        ax2.scatter(np.arange(len(rel))[good_rel], rel[good_rel], s=20)
        ax2.set_ylim(0.0, 0.05)  # tight window for “near-identical”
    else:
        ax2.text(0.5, 0.5, "no rel_L2 in CSV", ha="center", va="center", transform=ax2.transAxes)
    ax2.set_xlabel("trial index")
    ax2.set_ylabel("‖w − w_true‖ / ‖w_true‖")
    ax2.set_title("Relative L2 error on weights")

    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=180)
    print(f"wrote {OUT_PATH}")

if __name__ == "__main__":
    main()
