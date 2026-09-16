"""ABC-v2: ABC-v1 plus explicit long-range structural state (reports/ABC_V2_SCHEMA.md).

Two additions, both invisible to ABC tools and to ``parse_abc``:

* after every ``P:label`` line a comment ``% section i/N | B bars`` (section
  index, section count, requested bar count);
* before every bar a remark field ``[r:k]``: bars remaining in the section,
  counting the current bar (``[r:B]`` on the first bar, ``[r:1]`` on the last).

``[r:...]`` is the ABC 2.1 inline *remark* field and ``%`` starts a comment, so
standard ABC software ignores both, the v1 parser reads v2 text unchanged, and
``strip_v2`` returns the exact v1 text.

The prompt adds ``| section i/N`` to every structure line so the header can be
copied instead of counted.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .abc import song_to_abc
from .canonical import Section, Song
from .prompt import COMPLETION_MARKER, _beats_text

ABC_V2_VERSION = "qwen_abc_leadsheet_v2"
PROMPT_HEADER_V2 = "Task: write a lead sheet in ABC notation (melody, chord symbols, aligned lyrics, bar countdown)."

_HEADER_RE = re.compile(r"^% section (\d+)/(\d+) \| (\d+) bars\s*$")
_COUNTER_RE = re.compile(r"\[r:\s*(\d+)\s*\]")
_COUNTER_STRIP_RE = re.compile(r"\[r:[^\]]*\] ?")


def section_header(si: int, n_sections: int, sec: Section) -> str:
    return f"% section {si + 1}/{n_sections} | {sec.num_bars} bars"


def bar_prefix(si: int, n_sections: int, bar_in_sec: int, sec_bars: int, line_start: bool) -> str:
    return f"[r:{sec_bars - bar_in_sec}]"


def song_to_abc_v2(song: Song, bars_per_line: int = 4) -> str:
    return song_to_abc(song, bars_per_line, bar_prefix=bar_prefix, section_header=section_header)


def strip_v2(text: str) -> str:
    """ABC-v2 -> ABC-v1 text (drops the section comments and bar countdown remarks)."""
    out = []
    for line in text.split("\n"):
        if _HEADER_RE.match(line):
            continue
        out.append(_COUNTER_STRIP_RE.sub("", line))
    return "\n".join(out)


def spec_to_prompt_v2(spec: Dict[str, Any]) -> str:
    nominal = int(str(spec["meter"]).split("/")[0])
    out = [
        PROMPT_HEADER_V2,
        f"Language: {spec['language']}",
        f"Meter: {spec['meter']}",
        f"Tempo: {spec['tempo_bpm']} BPM",
    ]
    if spec.get("key"):
        out.append(f"Key: {spec['key']}")
    out.append("Structure:")
    n = len(spec["sections"])
    for i, sec in enumerate(spec["sections"]):
        out.append(f"P:{sec['label']} | {sec['bars']} bars{_beats_text(sec['beats'], nominal)} | section {i + 1}/{n}")
        out.extend(sec["lines"])
    return "\n".join(out) + "\n\n" + COMPLETION_MARKER


def counter_report(text: str, spec: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """How well the explicit state in a (generated) ABC-v2 text tracks reality and the plan.

    Bars are the ``|``-separated segments of music lines, grouped by ``P:`` lines.
    * ``counter_present_frac``: bars that carry ``[r:k]``;
    * ``counter_self_consistent_frac``: k == bars actually remaining in the written section;
    * ``counter_plan_frac``: k == bars remaining under the requested plan (same section index);
    * ``header_plan_frac``: section comments equal to the plan (index, count, bars);
    * ``sections_ending_on_counter_1``: sections whose last bar says ``[r:1]``.
    """
    sections: List[Dict[str, Any]] = []
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith("P:"):
            sections.append({"label": s[2:].strip().lower(), "header": None, "counters": []})
            continue
        hm = _HEADER_RE.match(s)
        if hm:
            if sections:
                sections[-1]["header"] = tuple(int(x) for x in hm.groups())
            continue
        if not s or s.startswith("%") or re.match(r"^[A-Za-z]:", s):
            continue
        if not sections:
            sections.append({"label": "other", "header": None, "counters": []})
        for seg in re.split(r"\|+\]?|\[\|", s):
            if not re.search(r"[A-Ga-gz]", _COUNTER_STRIP_RE.sub("", seg)):
                continue
            m = _COUNTER_RE.search(seg)
            sections[-1]["counters"].append(int(m.group(1)) if m else None)
    bars = present = consistent = plan_ok = plan_n = 0
    headers = header_ok = ends1 = nonempty = 0
    want = [(i + 1, len(spec["sections"]), sec["bars"]) for i, sec in enumerate(spec["sections"])] if spec else None
    for si, sec in enumerate(sections):
        cs = sec["counters"]
        if cs:
            nonempty += 1
            ends1 += cs[-1] == 1
        for bi, k in enumerate(cs):
            bars += 1
            if k is None:
                continue
            present += 1
            consistent += k == len(cs) - bi
            if want is not None and si < len(want):
                plan_n += 1
                plan_ok += k == want[si][2] - bi
        if sec["header"] is not None:
            headers += 1
            if want is not None and si < len(want):
                header_ok += sec["header"] == want[si]
    out = {
        "counter_bars": bars,
        "counter_present_frac": present / max(bars, 1),
        "counter_self_consistent_frac": consistent / max(present, 1),
        "section_header_frac": headers / max(len(sections), 1),
        "sections_ending_on_counter_1": ends1 / max(nonempty, 1),
    }
    if want is not None:
        out["counter_plan_frac"] = plan_ok / max(plan_n, 1)
        out["header_plan_frac"] = header_ok / max(len(want), 1)
    return out
