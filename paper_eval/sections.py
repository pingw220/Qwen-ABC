"""Per-requested-section outcomes of one generation, for position-resolved analysis.

A requested section i is placed at normalized song progress p = (start_bar + bars/2) / total_bars
of the *requested* plan, so every system is bucketed on the same axis. Section i of the output is
the i-th ``P:`` section written (index alignment, as ``section_plan_exact`` uses).
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional

from qwen_abc.abc import parse_abc
from qwen_abc.canonical import Song
from qwen_abc.prompt import split_syllables

from .seqsim import lcs_fast

_HEADER_RE = re.compile(r"^% section (\d+)/(\d+) \| (\d+) bars\s*$")
_COUNTER_RE = re.compile(r"\[r:\s*(\d+)\s*\]")
_COUNTER_STRIP_RE = re.compile(r"\[r:[^\]]*\] ?")


def counters_by_section(text: str) -> List[List[Optional[int]]]:
    """[r:k] values per written section (same bar segmentation as abc_v2.counter_report)."""
    secs: List[List[Optional[int]]] = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("P:"):
            secs.append([])
            continue
        if not s or s.startswith("%") or re.match(r"^[A-Za-z]:", s):
            continue
        if not secs:
            secs.append([])
        for seg in re.split(r"\|+\]?|\[\|", s):
            if not re.search(r"[A-Ga-gz]", _COUNTER_STRIP_RE.sub("", seg)):
                continue
            m = _COUNTER_RE.search(seg)
            secs[-1].append(int(m.group(1)) if m else None)
    return secs


def section_records(row: dict, spec: dict, with_counters: bool = True) -> List[Dict]:
    want = spec["sections"]
    total = sum(s["bars"] for s in want) or 1
    starts, acc = [], 0
    for s in want:
        starts.append(acc)
        acc += s["bars"]
    song = Song.from_json(row["song"]) if row.get("song") else None
    hit_eos = bool(row.get("hit_eos"))
    got = song.sections if song else []
    # bar-duration errors: the parser's written ticks vs the declared meter, per written bar
    bad_bar = []
    if song is not None and row.get("generation"):
        pr = parse_abc(row["generation"], "x")
        if pr.song is not None and pr.bar_ticks and pr.declared_beats:
            bad_bar = [int(t != 4 * b) for t, b in zip(pr.bar_ticks, pr.declared_beats)]
    counters = counters_by_section(row.get("generation", "")) if with_counters else []
    sung_by_sec: List[List[str]] = []
    cram_by_sec: List[Optional[float]] = []
    if song is not None:
        bs = song.bar_starts()
        for sec in got:
            lo = bs[sec.start_bar] if sec.start_bar < len(bs) else song.total_ticks
            e = sec.start_bar + sec.num_bars
            hi = bs[e] if e < len(bs) else song.total_ticks
            att = [n for n in song.notes if lo <= n.onset < hi and n.lyric]
            sy = [x for n in att for x in n.lyric]
            sung_by_sec.append(sy)
            cram_by_sec.append(sum(len(n.lyric) for n in att if len(n.lyric) > 1) / len(sy) if sy else None)
    out = []
    for i, w in enumerate(want):
        pos = (starts[i] + w["bars"] / 2) / total
        wsy = [x for line in w["lines"] for x in split_syllables(line)]
        rec = {"sec_index": i, "n_sections": len(want), "position": pos, "bucket": min(int(pos * 5), 4),
               "label": w["label"], "req_bars": w["bars"], "req_syllables": len(wsy),
               "present": float(i < len(got)), "parse_ok": float(song is not None)}
        if song is None:
            rec.update(exact=0.0, label_ok=0.0, bars_ok=0.0)
            out.append(rec)
            continue
        if i < len(got):
            g = got[i]
            rec.update(label_ok=float(g.label == w["label"]), bars_ok=float(g.num_bars == w["bars"]),
                       exact=float(g.label == w["label"] and g.num_bars == w["bars"]),
                       bar_err=abs(g.num_bars - w["bars"]))
            bars_i = bad_bar[g.start_bar: g.start_bar + g.num_bars] if bad_bar else []
            rec["bad_bar_frac"] = sum(bars_i) / len(bars_i) if bars_i else None
            if wsy:
                rec["lyric_recall"] = lcs_fast(wsy, sung_by_sec[i]) / len(wsy)
                rec["cram"] = cram_by_sec[i]
            if with_counters and i < len(counters) and counters[i]:
                cs = counters[i]
                n = len(cs)
                rec["counter_self_ok"] = sum(1 for j, k in enumerate(cs) if k == n - j) / n
                rec["counter_plan_ok"] = sum(1 for j, k in enumerate(cs) if k == w["bars"] - j) / n
        else:
            rec.update(exact=0.0, label_ok=0.0, bars_ok=0.0, premature_eos=float(hit_eos))
            if wsy:
                rec["lyric_recall"] = 0.0
        out.append(rec)
    return out
