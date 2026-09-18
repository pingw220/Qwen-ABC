"""ABC-v3 prompt: a syllable budget per section, and lyric lines marked where they continue.

The completion side is unchanged — v3 writes the same ABC-v2 text, counters and
all. Only the prompt changes, so a v3 run against E3b is a single-variable
comparison of how the *request* is phrased.

Two things are added, both aimed at defects measured on round-2 output:

**A syllable budget per section.** Models put 11.4% of syllables on a note that
already carries one, against 5.7% in the corpus, and the worst case measured was
19 syllables inside a 0.19 s note. The section header already states the bar
count; stating the syllable count with it gives the model the arithmetic it needs
to spread a line over the bars it has, instead of discovering halfway through a
bar that it has run out of notes.

**Continuation marks.** A lyric line often starts before the section it belongs
to — the singer's pickup lands in the last bar of the previous section. The
round-2 cleaning tolerates that deliberately (moving the boundary would break the
bar plan), but the prompt then shows a one- or two-character line hanging alone
in a section: 29.2% of training songs have one. The model learned to sing those
fragments as if they were phrases. Marking them (``…`` at the end of a line that
continues, at the start of the line that continues it) says what is actually
happening, and costs two characters.

``strip_v3_marks`` recovers the plain lyric text, so scoring and lyric metrics
are unaffected.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .abc_v2 import PROMPT_HEADER_V2
from .prompt import COMPLETION_MARKER, _beats_text, split_syllables

CONTINUES = "…"

ABC_V3_VERSION = "abc_v3_syllable_budget_and_continuation"


def strip_v3_marks(text: str) -> str:
    """Plain lyric text: the continuation marks are prompt notation, not lyrics."""
    return text.replace(CONTINUES, "")


def section_syllables(sec: Dict[str, Any]) -> int:
    return sum(len(split_syllables(line)) for line in sec.get("lines") or [])


def _continuation_flags(spec: Dict[str, Any]) -> List[List[tuple]]:
    """Per section, per line: (continues_from_previous, continues_into_next).

    A line was split by a section boundary when the last line of one section and
    the first line of the next came from the same lyric line of the song.
    Sections with no lyrics at all are skipped when looking for the neighbour: an
    instrumental can sit between the pickup and the phrase it belongs to.
    """
    secs = spec["sections"]

    def neighbour_ids(i: int, step: int) -> List[Any]:
        j = i + step
        while 0 <= j < len(secs):
            ids = secs[j].get("line_ids") or []
            if ids:
                return ids
            j += step
        return []

    flags: List[List[tuple]] = []
    for i, sec in enumerate(secs):
        lines = sec.get("lines") or []
        ids = sec.get("line_ids") or [None] * len(lines)
        prev_ids = neighbour_ids(i, -1)
        next_ids = neighbour_ids(i, +1)
        out = []
        for j in range(len(lines)):
            ident = ids[j] if j < len(ids) else None
            from_prev = bool(j == 0 and ident is not None and prev_ids and prev_ids[-1] == ident)
            into_next = bool(j == len(lines) - 1 and ident is not None and next_ids and next_ids[0] == ident)
            out.append((from_prev, into_next))
        flags.append(out)
    return flags


def spec_to_prompt_v3(spec: Dict[str, Any]) -> str:
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
    flags = _continuation_flags(spec)
    n = len(spec["sections"])
    for i, sec in enumerate(spec["sections"]):
        syllables = section_syllables(sec)
        head = (f"P:{sec['label']} | {sec['bars']} bars{_beats_text(sec['beats'], nominal)}"
                f" | section {i + 1}/{n} | {syllables} syllable{'' if syllables == 1 else 's'}")
        out.append(head)
        for line, (from_prev, into_next) in zip(sec.get("lines") or [], flags[i]):
            out.append((CONTINUES if from_prev else "") + line + (CONTINUES if into_next else ""))
    return "\n".join(out) + "\n\n" + COMPLETION_MARKER


def budget_report(spec: Dict[str, Any]) -> Dict[str, float]:
    """What the added information looks like on one song, for the dataset report."""
    secs = spec["sections"]
    flags = _continuation_flags(spec)
    per_section = [section_syllables(s) for s in secs]
    sung = [s for s in per_section if s]
    split_lines = sum(1 for f in flags for from_prev, into_next in f if from_prev or into_next)
    fragments = sum(1 for s in secs for line in (s.get("lines") or []) if len(split_syllables(line)) <= 2)
    return {
        "sections": len(secs),
        "sections_with_lyrics": len(sung),
        "syllables": sum(per_section),
        "syllables_per_bar": sum(per_section) / max(sum(s["bars"] for s in secs), 1),
        "lines_split_by_a_boundary": split_lines,
        "fragment_lines": fragments,
    }
