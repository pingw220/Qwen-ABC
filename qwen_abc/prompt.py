"""SFT conditioning: a structured spec and its plain-text prompt.

The spec is what a user of the system would author (compare MIDI-LLM's
``--from-spec`` JSON): meter, tempo, key, and per section a label, a bar
count, the beat count of any irregular bar, and the lyric lines to sing.
It deliberately mirrors MIDI-LLM, where the click track (bars, tempo) is
always given and never generated.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .canonical import Song

PROMPT_HEADER = "Task: write a lead sheet in ABC notation (melody, chord symbols, aligned lyrics)."
COMPLETION_MARKER = "ABC:\n"
_CJK = re.compile(r"[㐀-鿿豈-﫿]")


def join_syllables(sylls: List[str]) -> str:
    out = ""
    for s in sylls:
        if out and not (_CJK.search(out[-1]) and _CJK.match(s[0])):
            out += " "
        out += s
    return out


def song_to_spec(song: Song) -> Dict[str, Any]:
    starts = song.bar_starts()
    total = song.total_ticks
    sections = []
    for sec in song.sections:
        begin = starts[sec.start_bar]
        end_bar = sec.start_bar + sec.num_bars
        end = starts[end_bar] if end_bar < len(starts) else total
        lines: List[List[str]] = []
        line_ids: List[Any] = []
        last_line: Optional[int] = object()  # sentinel never equal to a line index
        for n in song.notes:
            if not (begin <= n.onset < end) or not n.lyric:
                continue
            if n.line != last_line or not lines:
                lines.append([])
                line_ids.append(n.line)
                last_line = n.line
            lines[-1].extend(n.lyric)
        beats = song.bar_beats[sec.start_bar:end_bar]
        sections.append({
            "label": sec.label,
            "bars": sec.num_bars,
            "beats": beats,
            "lines": [join_syllables(l) for l in lines],
            # which lyric line each entry came from: equal ids either side of a
            # section boundary mean one line was split by it (usually a pickup)
            "line_ids": line_ids,
        })
    return {
        "language": song.language,
        "meter": song.meter,
        "tempo_bpm": song.tempo_bpm,
        "key": song.key,
        "sections": sections,
    }


def _beats_text(beats: List[int], nominal: int) -> str:
    if all(b == nominal for b in beats):
        return ""
    runs: List[List[int]] = []
    for b in beats:
        if runs and runs[-1][0] == b:
            runs[-1][1] += 1
        else:
            runs.append([b, 1])
    return " | beats " + " ".join(f"{b}x{n}" if n > 1 else str(b) for b, n in runs)


def spec_to_prompt(spec: Dict[str, Any]) -> str:
    nominal = int(str(spec["meter"]).split("/")[0])
    out = [
        PROMPT_HEADER,
        f"Language: {spec['language']}",
        f"Meter: {spec['meter']}",
        f"Tempo: {spec['tempo_bpm']} BPM",
    ]
    if spec.get("key"):
        out.append(f"Key: {spec['key']}")
    out.append("Structure:")
    for sec in spec["sections"]:
        out.append(f"P:{sec['label']} | {sec['bars']} bars{_beats_text(sec['beats'], nominal)}")
        out.extend(sec["lines"])
    return "\n".join(out) + "\n\n" + COMPLETION_MARKER


def spec_syllables(spec: Dict[str, Any]) -> List[str]:
    """Flatten the lyric lines back into syllables (CJK characters / latin words)."""
    out: List[str] = []
    for sec in spec["sections"]:
        for line in sec["lines"]:
            out.extend(split_syllables(line))
    return out


def split_syllables(text: str) -> List[str]:
    return re.findall(r"[㐀-鿿豈-﫿]|[^\s㐀-鿿豈-﫿]+", text)
