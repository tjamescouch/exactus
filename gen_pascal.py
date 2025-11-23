#!/usr/bin/env python3
"""
Generate a Pascal table for combinadics (with repetition) lookups.

The Metal kernels expect:
  - a 2D, row-major Pascal table `P` with shape [rows, cols],
  - dtype = uint32,
  - `cols = degree + 1`,
  - `rows >= D + degree` (often add a small margin, e.g., +4).

Entry semantics:
  P[n, k] = C(n, k), with P[n, 0] = 1 and P[n, n] = 1 when n < cols.

This script:
  * Builds P with safe overflow checks for uint32.
  * Emits .npy (default), .bin (raw little-endian), or .txt (CSV).
  * Can be imported and used from Python.

Examples
--------
# Minimal: D=128, degree=3 → rows auto = D+degree+4, cols=4
python gen_pascal.py --D 128 --degree 3 -o pascal.npy

# Explicit rows and txt output
python gen_pascal.py --D 256 --degree 4 --rows 300 --fmt txt -o pascal_deg4.txt

# Use as a library
from gen_pascal import build_pascal_for
P = build_pascal_for(D=128, degree=3, extra_rows=4)  # shape [rows, 4], dtype=uint32
"""

from __future__ import annotations
import argparse
import sys
from typing import Tuple, Optional
import numpy as np


def _will_u32_overflow(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> bool:
    """Return True if c = a + b overflowed uint32 for any element."""
    # In uint32, overflow wraps. If c < a or c < b for any element, it overflowed.
    return np.any(c < a) | np.any(c < b)


def build_pascal(rows: int, cols: int, *, dtype=np.uint32, strict_u32: bool = True) -> np.ndarray:
    """
    Build a Pascal table P[n,k] with shape [rows, cols].

    - dtype must be np.uint32 (the Metal path binds 'uint' = 32-bit).
    - strict_u32=True: error out if an addition overflows uint32.

    Returns
    -------
    P : np.ndarray (rows, cols), dtype=uint32
    """
    if dtype is not np.uint32:
        raise ValueError("This engine expects dtype=uint32. (Match Metal 'uint' in constant buffer.)")

    if rows <= 0 or cols <= 0:
        raise ValueError("rows and cols must be positive")

    P = np.zeros((rows, cols), dtype=np.uint32)
    P[0, 0] = 1

    for n in range(1, rows):
        P[n, 0] = 1
        kmax = min(n, cols - 1)
        # Standard Pascal recurrence where valid
        if kmax >= 1:
            a = P[n - 1, 1 : kmax + 1]   # P[n-1, k-1] shifted right
            b = P[n - 1, 0 : kmax]       # P[n-1, k]
            c = a + b                    # uint32 add (may wrap)
            if strict_u32 and _will_u32_overflow(a, b, c):
                max_ok = np.iinfo(np.uint32).max
                raise OverflowError(
                    f"uint32 overflow detected while building Pascal at row n={n}. "
                    f"Try reducing D+degree or use smaller degree. "
                    f"(uint32 max={max_ok})"
                )
            P[n, 1 : kmax + 1] = c

        # Put the diagonal '1' if within column bounds
        if n < cols:
            P[n, n] = 1

    return P


def build_pascal_for(*, D: int, degree: int, extra_rows: int = 4, strict_u32: bool = True) -> np.ndarray:
    """
    Convenience wrapper: choose rows automatically for given (D, degree).

    rows = D + degree + extra_rows
    cols = degree + 1

    Returns
    -------
    P : np.ndarray (rows, cols), dtype=uint32
    """
    if degree < 0:
        raise ValueError("degree must be non-negative")
    if D <= 0:
        raise ValueError("D (input dimension) must be positive")

    rows = int(D + degree + max(0, extra_rows))
    cols = int(degree + 1)
    return build_pascal(rows=rows, cols=cols, dtype=np.uint32, strict_u32=strict_u32)


def save_pascal(P: np.ndarray, path: str, fmt: str = "npy") -> None:
    """
    Save the Pascal table.

    fmt:
      - 'npy' : NumPy .npy with shape [rows, cols], dtype=uint32 (default)
      - 'bin' : raw little-endian uint32 (no header), row-major
      - 'txt' : CSV (rows lines, comma-separated)
    """
    fmt = fmt.lower()
    if fmt == "npy":
        np.save(path, P)
    elif fmt == "bin":
        P.tofile(path)  # raw binary
    elif fmt == "txt":
        np.savetxt(path, P, fmt="%u", delimiter=",")
    else:
        raise ValueError(f"Unsupported format: {fmt!r}")


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Generate a uint32 Pascal table for Metal combinadics.")
    ap.add_argument("--D", type=int, required=True, help="Input dimension (feature dimension).")
    ap.add_argument("--degree", type=int, required=True, help="Polynomial degree (N).")
    ap.add_argument("--rows", type=int, default=None,
                    help="Rows of Pascal (>= D+degree). If omitted, uses D+degree+extra_rows.")
    ap.add_argument("--extra-rows", type=int, default=4,
                    help="If --rows is omitted, add this many margin rows (default: 4).")
    ap.add_argument("-o", "--out", type=str, default="pascal.npy", help="Output path (default: pascal.npy).")
    ap.add_argument("--fmt", type=str, choices=("npy", "bin", "txt"), default="npy",
                    help="Output format (npy|bin|txt). Default: npy")
    ap.add_argument("--allow-overflow", action="store_true",
                    help="Do NOT error on uint32 overflow (not recommended).")
    return ap.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)

    if args.rows is None:
        rows = args.D + args.degree + max(0, args.extra_rows)
    else:
        rows = args.rows
        if rows < args.D + args.degree:
            print(f"[warn] rows ({rows}) < D+degree ({args.D + args.degree}). "
                  f"You should use at least rows >= D+degree.", file=sys.stderr)

    cols = args.degree + 1

    try:
        P = build_pascal(rows=rows, cols=cols, dtype=np.uint32, strict_u32=not args.allow_overflow)
    except OverflowError as e:
        print(f"[error] {e}", file=sys.stderr)
        return 2

    save_pascal(P, args.out, fmt=args.fmt)
    print(f"[ok] Wrote Pascal: shape=({rows}, {cols}), dtype=uint32 → {args.out} (fmt={args.fmt})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
