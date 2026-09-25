"""Control interventions on a song spec, a check that exactly one control changed, and
the paired distances used to compare an intervention with a reseed.

A *spec* is what the prompt is built from (qwen_abc/prompt.py): meter, tempo, key and
per section a label, a bar count, per-bar beats and lyric lines. Every intervention
changes one control and leaves every other control byte-identical; ``controls_changed``
verifies that on every generated task, and the generation driver refuses a task that
fails it.
"""

from __future__ import annotations

import bisect
import copy
import math
import re
from typing import Dict, List, Optional, Sequence, Tuple

from qwen_abc.canonical import TICKS_PER_BEAT, Song
from qwen_abc.metrics import chord_at, lcs_len
from qwen_abc.prompt import join_syllables, split_syllables
from qwen_abc.theory import canonical_key, parse_chord_symbol, parse_key_name

from .common import seeded_choice

NAMES = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
LYRIC_LABELS = ("verse", "chorus", "bridge", "prechorus")
_CJK_ONE = re.compile(r"[㐀-鿿豈-﫿]")


# ------------------------------------------------------------------ what counts as a control
def control_view(spec: dict) -> Dict[str, object]:
    """Flatten a spec into named controls so two specs can be diffed control by control."""
    v: Dict[str, object] = {"meter": spec["meter"], "tempo": spec["tempo_bpm"], "key": spec.get("key"),
                            "n_sections": len(spec["sections"]), "language": spec.get("language")}
    for i, s in enumerate(spec["sections"]):
        v[f"sec{i}.label"] = s["label"]
        v[f"sec{i}.bars"] = (s["bars"], tuple(s["beats"]))
        v[f"sec{i}.lyrics"] = tuple(s["lines"])
    return v


def controls_changed(a: dict, b: dict) -> List[str]:
    va, vb = control_view(a), control_view(b)
    return sorted(k for k in set(va) | set(vb) if va.get(k) != vb.get(k))


def control_family(changed: List[str]) -> str:
    """'tempo' / 'key' / 'bars' / 'label' / 'lyrics' / 'mixed' / 'none'."""
    fams = set()
    for k in changed:
        fams.add(k.split(".")[-1] if "." in k else k)
    if not fams:
        return "none"
    return fams.pop() if len(fams) == 1 else "mixed"


# ------------------------------------------------------------------ target section
def syllables_of(sec: dict) -> List[str]:
    return [x for line in sec["lines"] for x in split_syllables(line)]


def pick_target_section(spec: dict, song_id: str, min_bars: int = 6) -> Optional[int]:
    """A lyric-bearing verse/chorus section with room for a -4 change, chosen deterministically.

    Prefers sections that are not the first or last section, so collateral change can be
    measured on both sides of the edit.
    """
    n = len(spec["sections"])
    cand = [i for i, s in enumerate(spec["sections"])
            if s["label"] in ("verse", "chorus") and syllables_of(s) and s["bars"] >= min_bars
            and sum(1 for b in s["beats"] if b == int(spec["meter"].split("/")[0])) >= 4]
    inner = [i for i in cand if 0 < i < n - 1]
    pool = inner or cand
    if not pool:
        return None
    return seeded_choice(song_id, pool, salt="target_section")


# ------------------------------------------------------------------ interventions
def change_bars(spec: dict, sec_i: int, delta: int) -> dict:
    """Add/remove ``delta`` nominal-length bars in one section; lyrics, label and all else unchanged."""
    out = copy.deepcopy(spec)
    sec = out["sections"][sec_i]
    nominal = int(spec["meter"].split("/")[0])
    beats = list(sec["beats"])
    if delta > 0:
        # insert before a trailing irregular (pickup/partial) bar, otherwise at the end
        pos = len(beats) - 1 if beats and beats[-1] != nominal else len(beats)
        beats[pos:pos] = [nominal] * delta
    elif delta < 0:
        k = -delta
        for j in range(len(beats) - 1, -1, -1):
            if k and beats[j] == nominal:
                del beats[j]
                k -= 1
        if k:
            raise ValueError("not enough nominal bars to remove")
    sec["beats"] = beats
    sec["bars"] = len(beats)
    return out


def change_label(spec: dict, sec_i: int, new_label: str) -> dict:
    out = copy.deepcopy(spec)
    out["sections"][sec_i]["label"] = new_label
    return out


def transpose_key(spec: dict, semitones: int) -> dict:
    parsed = parse_key_name(spec.get("key"))
    if parsed is None:
        raise ValueError("no key")
    pc, mode = parsed
    out = copy.deepcopy(spec)
    out["key"] = canonical_key(f"{NAMES[(pc + semitones) % 12]} {mode}")
    return out


def scale_tempo(spec: dict, factor: float) -> dict:
    out = copy.deepcopy(spec)
    out["tempo_bpm"] = int(round(spec["tempo_bpm"] * factor))
    return out


def _fill_lines(template_lines: List[str], sylls: List[str]) -> List[str]:
    """Pour ``sylls`` into as many lines as the template has, keeping each line's share."""
    counts = [len(split_syllables(l)) for l in template_lines]
    total = sum(counts)
    if not total:
        return []
    out, pos = [], 0
    for i, c in enumerate(counts):
        take = round(c * len(sylls) / total) if i < len(counts) - 1 else len(sylls) - pos
        chunk = sylls[pos: pos + take]
        pos += take
        if chunk:
            out.append(join_syllables(chunk))
    return out


def replace_lyrics(spec: dict, donor_sylls: List[str], sec_i: Optional[int] = None) -> dict:
    """Replace every section's lyrics (sec_i None) or one section's, syllable count preserved.

    The donor stream is cut to the syllable count being replaced (cycling if short), so the
    requested syllable load per section stays identical and only lexical identity changes.
    """
    out = copy.deepcopy(spec)
    idx = [sec_i] if sec_i is not None else range(len(out["sections"]))
    need = sum(len(syllables_of(out["sections"][i])) for i in idx)
    if not donor_sylls:
        raise ValueError("empty donor")
    stream = [donor_sylls[j % len(donor_sylls)] for j in range(need)]
    pos = 0
    for i in idx:
        sec = out["sections"][i]
        k = len(syllables_of(sec))
        if not k:
            continue
        sec["lines"] = _fill_lines(sec["lines"], stream[pos: pos + k])
        pos += k
    return out


def pick_donor(song_id: str, spec: dict, pool: Dict[str, dict]) -> Optional[str]:
    """Another held-out song whose total syllable count is closest (no train/test duplicate by
    construction of the split), with all-Chinese lyrics so no model's front end can refuse the
    swapped prompt; ties broken deterministically."""
    n = sum(len(syllables_of(s)) for s in spec["sections"])
    best = sorted(((abs(sum(len(syllables_of(s)) for s in p["sections"]) - n), sid) for sid, p in pool.items()
                   if sid != song_id and all(_CJK_ONE.fullmatch(x) for s in p["sections"] for x in syllables_of(s))),
                  key=lambda t: (t[0], seeded_choice(song_id + t[1], list(range(1000)), "donor")))
    return best[0][1] if best else None


# ------------------------------------------------------------------ paired distances between two generations
def _bar_index(song: Song):
    starts = song.bar_starts()
    return starts, (lambda t: max(bisect.bisect_right(starts, t) - 1, 0))


def note_tokens(song: Song, kind: str, notes=None, transpose: int = 0) -> List[str]:
    notes = song.notes if notes is None else notes
    starts, bar_of = _bar_index(song)
    if kind == "melody":      # onset-in-bar, pitch, duration
        return [f"{n.onset - starts[bar_of(n.onset)]}:{n.pitch + transpose}:{n.duration}" for n in notes]
    if kind == "pitch":
        return [str(n.pitch + transpose) for n in notes]
    if kind == "contour":     # interval sequence, transposition invariant
        return [str(max(-12, min(12, y.pitch - x.pitch))) for x, y in zip(notes, notes[1:])]
    if kind == "rhythm":      # onset-in-bar and duration, pitch ignored
        return [f"{n.onset - starts[bar_of(n.onset)]}:{n.duration}" for n in notes]
    raise ValueError(kind)


def seq_distance(a: Sequence[str], b: Sequence[str]) -> float:
    """1 - LCS / max length; 0 = identical, 1 = nothing in common."""
    from .seqsim import lcs_dist
    return lcs_dist(list(a), list(b))


def chord_tokens(song: Song, transpose: int = 0) -> List[str]:
    """One chord (root pitch class + quality) per beat, 'N' if none; transposable."""
    out = []
    starts = song.bar_starts()
    for b, beats in zip(starts, song.bar_beats):
        for k in range(beats):
            c = chord_at(song, b + k * TICKS_PER_BEAT)
            info = parse_chord_symbol(c.symbol) if c else None
            if not info or info["root_pc"] is None:
                out.append("N")
            else:
                pcs = sorted((pc - info["root_pc"]) % 12 for pc in info["pcs"])
                out.append(f"{(info['root_pc'] + transpose) % 12}:{'.'.join(map(str, pcs))}")
    return out


def chord_seq_by_change(song: Song, transpose: int = 0) -> List[str]:
    seq = []
    for c in song.chords:
        info = parse_chord_symbol(c.symbol)
        if not info or info["root_pc"] is None:
            continue
        tok = f"{(info['root_pc'] + transpose) % 12}:{'.'.join(map(str, sorted((pc - info['root_pc']) % 12 for pc in info['pcs'])))}"
        if not seq or seq[-1] != tok:
            seq.append(tok)
    return seq


def pc_hist(song: Song) -> List[float]:
    h = [0.0] * 12
    for n in song.notes:
        h[n.pitch % 12] += n.duration
    s = sum(h) or 1.0
    return [x / s for x in h]


def best_pc_shift(a: Song, b: Song) -> Tuple[int, float]:
    """Circular shift k maximizing the correlation of b's pitch-class histogram with a's shifted by k."""
    ha, hb = pc_hist(a), pc_hist(b)
    best = (0, -2.0)
    for k in range(12):
        sa = [ha[(i - k) % 12] for i in range(12)]
        ma, mb = sum(sa) / 12, sum(hb) / 12
        num = sum((x - ma) * (y - mb) for x, y in zip(sa, hb))
        den = math.sqrt(sum((x - ma) ** 2 for x in sa) * sum((y - mb) ** 2 for y in hb)) or 1e-9
        c = num / den
        if c > best[1] + 1e-12:
            best = (k, c)
    return best


def section_notes(song: Song) -> List[list]:
    starts = song.bar_starts()
    total = song.total_ticks
    out = []
    for s in song.sections:
        a = starts[s.start_bar] if s.start_bar < len(starts) else total
        e = s.start_bar + s.num_bars
        b = starts[e] if e < len(starts) else total
        out.append([n for n in song.notes if a <= n.onset < b])
    return out


def pair_distances(a: Song, b: Song, transpose: int = 0) -> Dict[str, float]:
    """Whole-song distances between two generations (b compared with a transposed by ``transpose``)."""
    d = {
        "d_melody": seq_distance(note_tokens(a, "melody", transpose=transpose), note_tokens(b, "melody")),
        "d_pitch": seq_distance(note_tokens(a, "pitch", transpose=transpose), note_tokens(b, "pitch")),
        "d_contour": seq_distance(note_tokens(a, "contour"), note_tokens(b, "contour")),
        "d_rhythm": seq_distance(note_tokens(a, "rhythm"), note_tokens(b, "rhythm")),
        "d_chord": seq_distance(chord_seq_by_change(a, transpose), chord_seq_by_change(b)),
        "d_chord_beats": seq_distance(chord_tokens(a, transpose), chord_tokens(b)),
    }
    na, nb = max(len(a.bar_beats), 1), max(len(b.bar_beats), 1)
    d["d_note_density"] = abs(len(b.notes) / nb - len(a.notes) / na)
    d["d_structure"] = float([(s.label, s.num_bars) for s in a.sections] != [(s.label, s.num_bars) for s in b.sections])
    return d


def section_pair_distances(a: Song, b: Song, kind: str = "melody") -> List[Optional[float]]:
    """Per section index: distance of the notes a and b wrote in that section (None if missing)."""
    sa, sb = section_notes(a), section_notes(b)
    out = []
    for i in range(max(len(sa), len(sb))):
        if i >= len(sa) or i >= len(sb):
            out.append(None)
            continue
        ta = note_tokens(a, kind, notes=sa[i])
        tb = note_tokens(b, kind, notes=sb[i])
        out.append(seq_distance(ta, tb) if (ta or tb) else None)
    return out


def recall_of(song: Song, sylls: List[str]) -> float:
    got = [x for n in song.notes if n.lyric for x in n.lyric]
    return lcs_len(sylls, got) / max(len(sylls), 1)


def section_lyric_recall_of(song: Song, sec_i: int, sylls: List[str]) -> Optional[float]:
    sn = section_notes(song)
    if sec_i >= len(sn):
        return 0.0
    got = [x for n in sn[sec_i] if n.lyric for x in n.lyric]
    return lcs_len(sylls, got) / max(len(sylls), 1)
