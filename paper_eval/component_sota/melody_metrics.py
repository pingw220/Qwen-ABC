"""Lyrics->melody metrics: CSL-L2M's published PD / DD / MD, plus diagnostics.

Published metrics (CSL-L2M, AAAI-25; following SongMASS / TeleMelody), reimplemented here:

* **PD** (pitch distribution similarity): overlap area sum_p min(P_gen(p), P_ref(p)) of the
  two songs' pitch histograms (MIDI pitch, note-count weighted), per song; higher is better.
* **DD** (duration distribution similarity): the same overlap area over note-duration
  histograms (durations in 16th notes, capped at 2 bars); higher is better.
* **DD_tb** (diagnostic, ours): max of DD over duration rescalings x0.5 / x1 / x2 of the generated
  song. CSL-L2M's corpus notates melodies about twice as dense per bar as ours (5-8 vs 3.5 notes
  per bar), so plain DD partly measures the notation convention; DD_tb removes that factor.
* **MD** (melody distance): both melodies are rendered as pitch per eighth-note frame (rests
  hold the last pitch), mean-centred (key/register invariant), and compared by DTW with
  |x - y| cost; MD = DTW cost / path length (semitones per frame); lower is better.

Reference-free diagnostics reuse qwen_abc.metrics / paper_eval.musicality so numbers match
the paper-final reports.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional

import numpy as np

from qwen_abc.canonical import Song


def _hist(values) -> Dict:
    c = Counter(values)
    n = sum(c.values()) or 1
    return {k: v / n for k, v in c.items()}


def overlap_area(a: Dict, b: Dict) -> float:
    return float(sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in set(a) | set(b)))


def pd_score(gen: Song, ref: Song) -> Optional[float]:
    if not gen.notes or not ref.notes:
        return None
    return overlap_area(_hist(n.pitch for n in gen.notes), _hist(n.pitch for n in ref.notes))


def pd_pc_score(gen: Song, ref: Song) -> Optional[float]:
    """Octave-invariant diagnostic variant of PD (pitch classes relative to C, not absolute pitch)."""
    if not gen.notes or not ref.notes:
        return None
    return overlap_area(_hist(n.pitch % 12 for n in gen.notes), _hist(n.pitch % 12 for n in ref.notes))


def dd_score(gen: Song, ref: Song) -> Optional[float]:
    if not gen.notes or not ref.notes:
        return None
    return overlap_area(_hist(min(n.duration, 32) for n in gen.notes), _hist(min(n.duration, 32) for n in ref.notes))


def dd_tb_score(gen: Song, ref: Song) -> Optional[float]:
    if not gen.notes or not ref.notes:
        return None
    r = _hist(min(n.duration, 32) for n in ref.notes)
    return max(overlap_area(_hist(min(max(int(round(n.duration * f)), 1), 32) for n in gen.notes), r) for f in (0.5, 1.0, 2.0))


def frames(song: Song, step: int = 2) -> np.ndarray:
    """Pitch per frame of ``step`` ticks (default 2 = eighth note); rests hold the previous pitch."""
    if not song.notes:
        return np.zeros(0)
    end = max(n.onset + n.duration for n in song.notes)
    out = np.full(end // step + 1, np.nan)
    for n in song.notes:
        out[n.onset // step: (n.onset + n.duration) // step + 1] = n.pitch
    # forward fill, then back fill the leading rest
    for i in range(1, len(out)):
        if np.isnan(out[i]):
            out[i] = out[i - 1]
    first = out[~np.isnan(out)][0]
    out[np.isnan(out)] = first
    # drop leading/trailing silence effects: keep from the first onset
    start = min(n.onset for n in song.notes) // step
    return out[start:]


def dtw_cost(x: np.ndarray, y: np.ndarray) -> float:
    n, m = len(x), len(y)
    prev = np.full(m + 1, np.inf)
    prev[0] = 0.0
    prev_len = np.zeros(m + 1)
    for i in range(1, n + 1):
        cost = np.abs(x[i - 1] - y)
        cur = np.full(m + 1, np.inf)
        cur_len = np.zeros(m + 1)
        # vectorized over the diagonal/up moves, then a left-to-right scan for the "left" move
        diag, up = prev[:-1], prev[1:]
        best = np.minimum(diag, up)
        blen = np.where(diag <= up, prev_len[:-1], prev_len[1:])
        for j in range(1, m + 1):
            c = best[j - 1]
            l = blen[j - 1]
            if cur[j - 1] < c:
                c, l = cur[j - 1], cur_len[j - 1]
            cur[j] = cost[j - 1] + c
            cur_len[j] = l + 1
        prev, prev_len = cur, cur_len
    return float(prev[m] / max(prev_len[m], 1))


def md_score(gen: Song, ref: Song, step: int = 2, max_frames: int = 1200) -> Optional[float]:
    x, y = frames(gen, step), frames(ref, step)
    if not len(x) or not len(y):
        return None
    # long songs: compare at a coarser grid so DTW stays O(1e6)
    while max(len(x), len(y)) > max_frames:
        x, y = x[::2], y[::2]
    return dtw_cost(x - x.mean(), y - y.mean())


def phrase_lengths(song: Song, rest_ticks: int = 4) -> List[int]:
    """Notes per phrase, a phrase ending at a rest of >= one beat."""
    out, cur = [], 0
    notes = song.notes
    for a, b in zip(notes, notes[1:] + [None]):
        cur += 1
        if b is None or b.onset - (a.onset + a.duration) >= rest_ticks:
            out.append(cur)
            cur = 0
    return out
