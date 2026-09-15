"""Evaluation metrics for generated lyrics-aware ABC lead sheets.

Every metric is computed identically for generations and for references, so
a model is judged against the corpus' own band rather than an absolute ideal
(pseudo-labelled data: chord-tone agreement is ~0.63 in the references).
"""

from __future__ import annotations

import bisect
import gzip
import math
from collections import Counter
from typing import Dict, List, Optional, Sequence

from .abc import ParseResult
from .canonical import TICKS_PER_BEAT, Song
from .prompt import spec_syllables
from .theory import parse_chord_symbol, parse_key_name


# ----------------------------------------------------------- distributions
def histogram(values: Sequence, keys: Sequence) -> List[float]:
    c = Counter(values)
    total = sum(c[k] for k in keys)
    return [c[k] / total if total else 0.0 for k in keys]


def js_divergence(p: Sequence[float], q: Sequence[float]) -> Optional[float]:
    if not sum(p) or not sum(q):
        return None
    m = [(a + b) / 2 for a, b in zip(p, q)]

    def kl(a, b):
        return sum(x * math.log2(x / y) for x, y in zip(a, b) if x > 0)

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


PC_KEYS = list(range(12))
INTERVAL_KEYS = list(range(-12, 13))
ONSET_KEYS = [0, 1, 2, 3]
DUR_KEYS = [1, 2, 3, 4, 6, 8, 12, 16, "long"]


def _dur_bucket(d: int):
    return d if d in (1, 2, 3, 4, 6, 8, 12, 16) else ("long" if d > 16 else min((1, 2, 3, 4, 6, 8, 12, 16), key=lambda k: abs(k - d)))


def melody_features(song: Song) -> Dict[str, list]:
    notes = song.notes
    ints = [max(-12, min(12, b.pitch - a.pitch)) for a, b in zip(notes, notes[1:])]
    bar_starts = song.bar_starts()
    onset_pos = []
    for n in notes:
        i = bisect.bisect_right(bar_starts, n.onset) - 1
        onset_pos.append((n.onset - bar_starts[max(i, 0)]) % TICKS_PER_BEAT)
    key = parse_key_name(song.key)
    tonic = key[0] if key else 0
    return {
        "pc": [n.pitch % 12 for n in notes],
        "pc_rel": [(n.pitch - tonic) % 12 for n in notes],  # scale degree w.r.t. the declared key
        "interval": ints,
        "onset": onset_pos,
        "duration": [_dur_bucket(n.duration) for n in notes],
    }


# ----------------------------------------------------------------- lyrics
def lcs_len(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0]
        for j, y in enumerate(b):
            cur.append(prev[j] + 1 if x == y else max(prev[j + 1], cur[j]))
        prev = cur
    return prev[-1]


def chord_at(song: Song, tick: int):
    onsets = [c.onset for c in song.chords]
    i = bisect.bisect_right(onsets, tick) - 1
    if i >= 0 and tick < song.chords[i].offset:
        return song.chords[i]
    return None


# ------------------------------------------------------------- per song
def song_metrics(song: Song, spec: Optional[dict] = None, parse: Optional[ParseResult] = None) -> Dict[str, float]:
    m: Dict[str, float] = {}
    notes = song.notes
    nbars = max(len(song.bar_beats), 1)
    m["num_notes"] = len(notes)
    m["num_bars"] = len(song.bar_beats)
    m["notes_per_bar"] = len(notes) / nbars
    if notes:
        pitches = [n.pitch for n in notes]
        m["pitch_range"] = max(pitches) - min(pitches)
        m["pitch_mean"] = sum(pitches) / len(pitches)
        ints = [abs(b.pitch - a.pitch) for a, b in zip(notes, notes[1:])]
        m["mean_abs_interval"] = sum(ints) / max(len(ints), 1)
        m["large_leap_frac"] = sum(1 for x in ints if x > 7) / max(len(ints), 1)
        m["mean_duration_beats"] = sum(n.duration for n in notes) / len(notes) / TICKS_PER_BEAT
        m["offbeat_16th_frac"] = sum(1 for x in melody_features(song)["onset"] if x in (1, 3)) / len(notes)
        # degeneration: compressibility of the note stream and repeated bars
        stream = ",".join(f"{n.pitch}:{n.duration}:{n.onset % (TICKS_PER_BEAT * 4)}" for n in notes).encode()
        m["gzip_ratio"] = len(gzip.compress(stream)) / max(len(stream), 1)
        bars = _bar_signatures(song)
        nonempty = [b for b in bars if b]
        m["distinct_bar_frac"] = len(set(nonempty)) / max(len(nonempty), 1)
        run, best = 1, 1
        for a, b in zip(bars, bars[1:]):
            run = run + 1 if (a == b and a) else 1
            best = max(best, run)
        m["max_identical_consecutive_bars"] = best
    total = max(song.total_ticks, 1)
    m["melody_time_frac"] = sum(n.duration for n in notes) / total

    # chords
    chords = song.chords
    infos = [parse_chord_symbol(c.symbol) for c in chords]
    m["num_chords"] = len(chords)
    m["chords_per_bar"] = len(chords) / nbars
    m["chord_valid_frac"] = sum(1 for i in infos if i is not None) / max(len(chords), 1)
    m["chord_time_coverage"] = sum(c.duration for c, i in zip(chords, infos) if i and i["root_pc"] is not None) / total
    tone = dur = under = 0
    for n in notes:
        c = chord_at(song, n.onset)
        info = parse_chord_symbol(c.symbol) if c else None
        if info and info["root_pc"] is not None:
            under += 1
            dur += n.duration
            tone += n.duration * (n.pitch % 12 in info["pcs"])
    m["notes_under_chord_frac"] = under / max(len(notes), 1)
    m["chord_tone_frac"] = tone / dur if dur else float("nan")

    # lyrics
    attacks = [n for n in notes if n.lyric]
    sylls = [s for n in attacks for s in n.lyric]
    melisma = sum(1 for n in notes if n.melisma)
    m["syllables_sung"] = len(sylls)
    m["notes_per_syllable"] = len(notes) / max(len(sylls), 1)
    m["melisma_note_frac"] = melisma / max(len(notes), 1)
    m["wordless_note_frac"] = sum(1 for n in notes if not n.lyric and not n.melisma) / max(len(notes), 1)
    m["multi_syllable_note_frac"] = sum(1 for n in attacks if len(n.lyric) > 1) / max(len(notes), 1)
    if spec is not None:
        want = spec_syllables(spec)
        l = lcs_len(want, sylls)
        m["lyric_recall"] = l / max(len(want), 1)  # prompt syllables sung, in order
        m["lyric_precision"] = l / max(len(sylls), 1)  # sung syllables that belong, in order
        m["lyric_exact"] = float(want == sylls)
        m["lyric_missing_frac"] = 1 - m["lyric_recall"]
        m["lyric_extra_frac"] = 1 - m["lyric_precision"] if sylls else 0.0
        # structure conformance with the requested plan
        want_bars = [b for s in spec["sections"] for b in s["beats"]]
        m["bar_count_match"] = float(len(want_bars) == len(song.bar_beats))
        m["bar_beats_match_frac"] = sum(1 for a, b in zip(want_bars, song.bar_beats) if a == b) / max(len(want_bars), 1)
        want_secs = [(s["label"], s["bars"]) for s in spec["sections"]]
        got_secs = [(s.label, s.num_bars) for s in song.sections]
        m["section_plan_exact"] = float(want_secs == got_secs)
        m["section_label_seq_match"] = float([x[0] for x in want_secs] == [x[0] for x in got_secs])
        m["tempo_match"] = float(song.tempo_bpm == spec["tempo_bpm"])
        m["key_match"] = float((song.key or "") == (spec.get("key") or ""))
        m["meter_match"] = float(song.meter == spec["meter"])
        m["section_lyric_recall"] = _section_lyric_recall(song, spec)
    if parse is not None:
        m["strict_valid"] = float(parse.strict_ok)
        m["bar_duration_ok_frac"] = 1 - parse.errors.get("bar_duration_mismatch", 0) / max(len(parse.bar_ticks), 1)
        m["lyric_alignment_errors"] = float(parse.errors.get("lyric_overflow", 0) + parse.errors.get("orphan_melisma", 0)
                                            + parse.errors.get("lyric_bar_overflow", 0))
        m["lyric_alignment_valid"] = float(m["lyric_alignment_errors"] == 0)
    return m


def _bar_signatures(song: Song) -> List[tuple]:
    starts = song.bar_starts()
    sig: List[list] = [[] for _ in starts]
    for n in song.notes:
        i = bisect.bisect_right(starts, n.onset) - 1
        sig[max(i, 0)].append((n.onset - starts[max(i, 0)], n.duration, n.pitch))
    return [tuple(s) for s in sig]


def _section_lyric_recall(song: Song, spec: dict) -> float:
    """Fraction of prompt syllables sung inside the section that requested them."""
    starts = song.bar_starts()
    total = song.total_ticks
    got_sections = []
    for s in song.sections:
        a = starts[s.start_bar] if s.start_bar < len(starts) else total
        end_bar = s.start_bar + s.num_bars
        b = starts[end_bar] if end_bar < len(starts) else total
        got_sections.append([x for n in song.notes if a <= n.onset < b and n.lyric for x in n.lyric])
    hit = want_total = 0
    from .prompt import split_syllables

    for i, sec in enumerate(spec["sections"]):
        want = [x for line in sec["lines"] for x in split_syllables(line)]
        want_total += len(want)
        if i < len(got_sections):
            hit += lcs_len(want, got_sections[i])
    return hit / max(want_total, 1)


# ------------------------------------------------------------ aggregate
def aggregate(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = sorted({k for r in rows for k in r})
    out = {}
    for k in keys:
        vals = sorted(r[k] for r in rows if k in r and r[k] is not None and not (isinstance(r[k], float) and math.isnan(r[k])))
        if not vals:
            continue
        out[k] = {
            "mean": sum(vals) / len(vals),
            "median": vals[len(vals) // 2],
            "p05": vals[int(0.05 * (len(vals) - 1))],
            "p95": vals[int(0.95 * (len(vals) - 1))],
            "n": len(vals),
        }
    return out


def corpus_distributions(songs: List[Song]) -> Dict[str, List[float]]:
    feats = {"pc": [], "pc_rel": [], "interval": [], "onset": [], "duration": []}
    for s in songs:
        f = melody_features(s)
        for k in feats:
            feats[k].extend(f[k])
    return {
        "pc": histogram(feats["pc"], PC_KEYS),
        "pc_rel": histogram(feats["pc_rel"], PC_KEYS),
        "interval": histogram(feats["interval"], INTERVAL_KEYS),
        "onset": histogram(feats["onset"], ONSET_KEYS),
        "duration": histogram(feats["duration"], DUR_KEYS),
    }


def distribution_distances(gen: List[Song], ref: List[Song]) -> Dict[str, Optional[float]]:
    g, r = corpus_distributions(gen), corpus_distributions(ref)
    return {f"js_{k}": js_divergence(g[k], r[k]) for k in g}
