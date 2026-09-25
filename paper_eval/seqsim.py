"""Fast sequence similarity for many-pair comparisons.

``lcs_fast`` is the bit-parallel LCS of Allison-Dix / Hyyro (2004): one big-integer add, or
and mask per symbol of ``b``, so a 400x400 comparison costs ~400 word-vector operations instead
of 160,000 Python steps. It returns exactly ``qwen_abc.metrics.lcs_len`` (tested).
"""

from __future__ import annotations

from typing import Dict, Hashable, Sequence


def _masks(a: Sequence[Hashable]) -> Dict[Hashable, int]:
    m: Dict[Hashable, int] = {}
    for i, x in enumerate(a):
        m[x] = m.get(x, 0) | (1 << i)
    return m


def lcs_fast(a: Sequence[Hashable], b: Sequence[Hashable], masks: Dict[Hashable, int] = None) -> int:
    n = len(a)
    if not n or not b:
        return 0
    m = masks if masks is not None else _masks(a)
    full = (1 << n) - 1
    v = full
    for y in b:
        u = v & m.get(y, 0)
        v = ((v + u) | (v - u)) & full
    return n - bin(v).count("1")


def lcs_sim(a: Sequence[Hashable], b: Sequence[Hashable]) -> float:
    """LCS / max length (1 = identical, 0 = nothing in common)."""
    if not a and not b:
        return 1.0
    return lcs_fast(a, b) / max(len(a), len(b))


def lcs_dist(a, b) -> float:
    return 1.0 - lcs_sim(a, b)


def lcs_pairs(a: Sequence[Hashable], b: Sequence[Hashable]):
    """Matched (i, j) index pairs of one longest common subsequence of a and b (O(len(a)*len(b)))."""
    n, m = len(a), len(b)
    if not n or not m:
        return []
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        ai, row, nxt = a[i], dp[i], dp[i + 1]
        for j in range(m - 1, -1, -1):
            row[j] = nxt[j + 1] + 1 if ai == b[j] else (nxt[j] if nxt[j] >= row[j + 1] else row[j + 1])
    out, i, j = [], 0, 0
    while i < n and j < m:
        if a[i] == b[j]:
            out.append((i, j))
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out
