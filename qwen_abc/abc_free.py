"""ABC-free prompt: the section plan, the whole lyric, and no instruction about which goes where.

Every prompt so far has told the model which lyric lines belong to which section.
That assignment comes from the corpus labels, and the labels are wrong often
enough to be audible: a section boundary lands mid-phrase, so 29.2% of training
songs contain a section whose only lyric line is one or two characters. A
listener hears the opening of such a song as almost wordless, and the model is
doing exactly as it was told.

This prompt removes the assignment from the request. It gives:

* the section plan — labels, bars, `section i/N` — exactly as before;
* the song's lyric lines **rejoined**, in order, as one block.

Rejoining is the point. A "line" in the spec is a run of notes sharing one lyric
line id; when a boundary splits that run, the spec shows two fragments in two
sections. Here the fragments are put back together, so the model is asked to
place `无法阻止心流感扩散`, not `无` then `法阻止心` then `流感扩散`.

The completion is unchanged: still the reference lead sheet, with the reference's
own assignment. So training asks the model to *infer* the placement it was
previously told, and generation lets it choose. Two metrics then say different
things and both are worth reading:

* **lyric recall** — did it sing all the words, in order? Should hold up.
* **section-local lyric recall** — did it place them where the corpus did?
  May fall, and a fall is not automatically a loss: the corpus placement is the
  thing suspected of being wrong.
"""

from __future__ import annotations

from typing import Any, Dict, List

from .abc_v2 import PROMPT_HEADER_V2
from .prompt import COMPLETION_MARKER, _beats_text

LYRIC_HEADER = "Lyrics, in order (decide for yourself which section sings each line):"

ABC_FREE_VERSION = "abc_free_lyric_assignment"


def rejoined_lines(spec: Dict[str, Any]) -> List[str]:
    """The song's lyric lines, with boundary-split fragments put back together."""
    out: List[str] = []
    last_id = object()
    for sec in spec["sections"]:
        lines = sec.get("lines") or []
        ids = sec.get("line_ids") or [None] * len(lines)
        for line, ident in zip(lines, ids):
            if out and ident is not None and ident == last_id:
                out[-1] += line
            else:
                out.append(line)
            last_id = ident
    return out


def spec_to_prompt_free(spec: Dict[str, Any]) -> str:
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
    lines = rejoined_lines(spec)
    if lines:
        out.append(LYRIC_HEADER)
        out.extend(lines)
    return "\n".join(out) + "\n\n" + COMPLETION_MARKER


def assignment_report(spec: Dict[str, Any]) -> Dict[str, float]:
    """How much the corpus assignment fragments this song, for the build report."""
    per_section = [len(sec.get("lines") or []) for sec in spec["sections"]]
    fragments = sum(1 for sec in spec["sections"] for line in (sec.get("lines") or []) if len(line) <= 2)
    joined = len(rejoined_lines(spec))
    return {
        "sections": len(spec["sections"]),
        "lines_as_labelled": sum(per_section),
        "lines_rejoined": joined,
        "fragments_removed": sum(per_section) - joined,
        "fragment_lines_as_labelled": fragments,
    }
