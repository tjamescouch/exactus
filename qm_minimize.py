#!/usr/bin/env python3
# Pure-Python Quine–McCluskey (DNF) with Petrick's method (small n).
# Input: n (vars), on_set (list of integers 0..2^n-1), optional dc_set.
# Output: list of implicants; each implicant is tuple of literals: (var_idx, polarity)
# where polarity = +1 for x_i, -1 for ¬x_i.

from collections import defaultdict
from itertools import combinations, product
from typing import List, Tuple, Set

Bit = int
Cube = Tuple[int, ...]  # tuple of {0,1,2}; 2 = don't care

def int_to_bits(x: int, n: int) -> Tuple[int, ...]:
    return tuple((x >> i) & 1 for i in range(n))  # little-endian

def bits_to_int(b: Tuple[int, ...]) -> int:
    v = 0
    for i, bit in enumerate(b):
        v |= (bit & 1) << i
    return v

def count_ones(b: Tuple[int, ...]) -> int:
    return sum(1 for z in b if z == 1)

def mergeable(a: Cube, b: Cube) -> bool:
    diff = 0
    for x, y in zip(a, b):
        if x != y:
            if x == 2 or y == 2: return False
            diff += 1
            if diff > 1: return False
    return diff == 1

def merge(a: Cube, b: Cube) -> Cube:
    return tuple(2 if x != y else x for x, y in zip(a, b))

def covers(c: Cube, m: Tuple[int, ...]) -> bool:
    for ci, mi in zip(c, m):
        if ci != 2 and ci != mi:
            return False
    return True

def expand(c: Cube) -> List[Tuple[int, ...]]:
    # enumerate minterms covered by cube c (for small n)
    slots = []
    for v in c:
        if v == 2:
            slots.append((0,1))
        else:
            slots.append((v,))
    return [tuple(t) for t in product(*slots)]

def qm_prime_implicants(n: int, on: List[int], dc: List[int]) -> List[Cube]:
    minterms = [int_to_bits(x, n) for x in sorted(set(on + dc))]
    groups = defaultdict(list)
    for m in minterms:
        groups[count_ones(m)].append(m)

    marked = set()
    next_groups = defaultdict(list)
    primes: Set[Cube] = set()

    # Initial cubes are exact minterms (0/1 only)
    level = {k: [tuple(m) for m in v] for k, v in groups.items()}

    while True:
        used = set()
        next_level = defaultdict(list)

        keys = sorted(level.keys())
        for k in keys:
            if (k+1) not in level: continue
            for a in level[k]:
                for b in level[k+1]:
                    if mergeable(a, b):
                        c = merge(a, b)
                        used.add(a); used.add(b)
                        if c not in next_level[k]:
                            next_level[k].append(c)

        # any cube not used in a merge becomes prime
        for group in level.values():
            for c in group:
                if c not in used:
                    primes.add(tuple(c))  # type: ignore

        if not next_level:
            break
        # canonicalize next_level by hashing/unique and regroup by #ones ignoring don't cares
        canon = defaultdict(list)
        seen = set()
        for lst in next_level.values():
            for c in lst:
                if c in seen: continue
                seen.add(c)
                # count ones using any assignment (treat 2 as both); just bucket together
                canon[sum(1 for v in c if v == 1)].append(c)
        level = canon

    # Keep only primes that actually cover some ON minterm (ignore primes that cover DC only).
    on_bits = [int_to_bits(x, n) for x in on]
    real_primes = []
    for p in primes:
        if any(covers(p, m) for m in on_bits):
            real_primes.append(p)
    return real_primes

def petrick_minimize(n: int, primes: List[Cube], on: List[int]) -> List[Cube]:
    """Select a minimal set of primes that cover all ON minterms (exact).
       Brute-force subset search is OK for small problems (n ≤ 8)."""
    on_bits = [int_to_bits(x, n) for x in on]
    # Build coverage map: for each ON minterm index j, which primes cover it
    cover = []
    for m in on_bits:
        idxs = [i for i, p in enumerate(primes) if covers(p, m)]
        cover.append(set(idxs))

    # Quick essential selection
    chosen: Set[int] = set()
    remaining = cover[:]
    changed = True
    while changed:
        changed = False
        # If any minterm is covered by a single prime, take it
        for j, s in enumerate(remaining):
            if len(s) == 1:
                i = next(iter(s))
                if i not in chosen:
                    chosen.add(i); changed = True
        if changed:
            # remove all minterms covered by newly chosen primes
            new_remaining = []
            for s in remaining:
                if any(i in s for i in chosen):
                    continue
                new_remaining.append(s)
            remaining = new_remaining

    if not remaining:
        return [primes[i] for i in sorted(chosen)]

    # Small residual → brute-force minimal subset
    candidates = list(set(i for s in remaining for i in s))
    best = None
    for r in range(1, len(candidates) + 1):
        for combo in combinations(candidates, r):
            S = set(combo) | chosen
            # check covers all remaining
            ok = True
            for s in remaining:
                if not (S & s):
                    ok = False; break
            if ok:
                best = S
                break
        if best is not None:
            break

    if best is None:
        # fallback: return all (shouldn't happen for small cases)
        return [primes[i] for i in range(len(primes))]
    return [primes[i] for i in sorted(best)]

def cube_to_literals(c: Cube) -> Tuple[Tuple[int,int], ...]:
    # return tuple of (var_index, polarity) where polarity=+1 for x_i, -1 for ¬x_i
    out = []
    for i, v in enumerate(c):
        if v == 2: continue
        out.append((i, +1 if v == 1 else -1))
    return tuple(out)

def minimize_to_dnf(n: int, on_set: List[int], dc_set: List[int] = None) -> List[Tuple[Tuple[int,int], ...]]:
    dc = dc_set or []
    primes = qm_prime_implicants(n, on_set, dc)
    chosen = petrick_minimize(n, primes, on_set)
    return [cube_to_literals(c) for c in chosen]

# Pretty-printer
def literals_to_str(lits: Tuple[Tuple[int,int], ...]) -> str:
    if not lits: return "1"  # tautology
    parts = []
    for i, pol in lits:
        parts.append(f"x{i}" if pol > 0 else f"¬x{i}")
    return "*".join(parts)

if __name__ == "__main__":
    # tiny demo: XOR2 truth table (n=2 → ON = {01,10})
    print(minimize_to_dnf(2, [1,2]))
