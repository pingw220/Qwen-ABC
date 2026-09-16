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
    m.update(long_structure_metrics(song, spec))
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


# ================================================================ long-structure metrics (round 2)
def lcs_matches(a: Sequence[str], b: Sequence[str]) -> List[int]:
    """Indices into ``a`` matched by one longest common subsequence with ``b``."""
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
            out.append(i)
            i += 1
            j += 1
        elif dp[i + 1][j] >= dp[i][j + 1]:
            i += 1
        else:
            j += 1
    return out


def _section_notes(song: Song) -> List[List]:
    starts = song.bar_starts()
    total = song.total_ticks
    out = []
    for s in song.sections:
        a = starts[s.start_bar] if s.start_bar < len(starts) else total
        e = s.start_bar + s.num_bars
        b = starts[e] if e < len(starts) else total
        out.append([n for n in song.notes if a <= n.onset < b])
    return out


def motif_tokens(notes) -> List[str]:
    """Transposition-invariant motif tokens: (interval, inter-onset interval) per note transition."""
    return [f"{max(-12, min(12, y.pitch - x.pitch))}:{min(y.onset - x.onset, 16)}" for x, y in zip(notes, notes[1:])]


def motif_similarity(a_notes, b_notes) -> Optional[float]:
    ta, tb = motif_tokens(a_notes), motif_tokens(b_notes)
    if len(ta) < 3 or len(tb) < 3:
        return None
    return lcs_len(ta, tb) / max(len(ta), len(tb))


def _ngrams(seq, n):
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


SCALE_STEPS = {"major": (0, 2, 4, 5, 7, 9, 11), "minor": (0, 2, 3, 5, 7, 8, 10, 11)}


def long_structure_metrics(song: Song, spec: Optional[dict] = None) -> Dict[str, float]:
    m: Dict[str, float] = {}
    # tonal consistency with the declared key: high temperature buys variety partly by leaving the key
    key = parse_key_name(song.key)
    if key and song.notes:
        tonic, mode = key
        steps = SCALE_STEPS[mode]
        dur = sum(n.duration for n in song.notes)
        m["melody_in_key_frac"] = sum(n.duration for n in song.notes if (n.pitch - tonic) % 12 in steps) / max(dur, 1)
        rooted = [(c, parse_chord_symbol(c.symbol)) for c in song.chords]
        rooted = [(c, i) for c, i in rooted if i and i["root_pc"] is not None]
        tot = sum(c.duration for c, _ in rooted)
        if tot:
            m["chord_root_in_key_frac"] = sum(c.duration for c, i in rooted if (i["root_pc"] - tonic) % 12 in steps) / tot
            m["chord_all_tones_in_key_frac"] = sum(c.duration for c, i in rooted
                                                   if all((pc - tonic) % 12 in steps for pc in i["pcs"])) / tot
    notes = song.notes
    bars = _bar_signatures(song)
    nonempty = [b for b in bars if b]
    # ---------------------------------------------- repetition / diversity
    seen, rep = set(), 0
    for b in nonempty:
        rep += b in seen
        seen.add(b)
    m["repeated_bar_frac"] = rep / max(len(nonempty), 1)
    m["consecutive_repeat_bar_frac"] = sum(1 for a, b in zip(bars, bars[1:]) if a and a == b) / max(len(nonempty), 1)
    pitches = [n.pitch for n in notes]
    ints = [b - a for a, b in zip(pitches, pitches[1:])]
    for name, seq in (("pitch", pitches), ("interval", ints)):
        g = _ngrams(seq, 4)
        m[f"{name}_4gram_repeat_frac"] = 1 - len(set(g)) / len(g) if g else 0.0
    # transposition-invariant bar motifs (interval + rhythm inside the bar)
    starts = song.bar_starts()
    bar_notes: List[list] = [[] for _ in starts]
    for n in notes:
        i = bisect.bisect_right(starts, n.onset) - 1
        bar_notes[max(i, 0)].append(n)
    motifs = []
    for i, bn in enumerate(bar_notes):
        if len(bn) >= 2:
            motifs.append(tuple((y.pitch - x.pitch, x.onset - starts[i], x.duration) for x, y in zip(bn, bn[1:])))
        else:
            motifs.append(None)
    m["consecutive_motif_repeat_frac"] = sum(1 for a, b in zip(motifs, motifs[1:]) if a and a == b) / max(sum(1 for x in motifs if x), 1)
    # ---------------------------------------------- musical content
    if len(notes) >= 2:
        absint = [abs(x) for x in ints]
        k = len(absint)
        m["int_repeat_frac"] = sum(1 for x in absint if x == 0) / k
        m["int_step_frac"] = sum(1 for x in absint if 1 <= x <= 2) / k
        m["int_skip_frac"] = sum(1 for x in absint if 3 <= x <= 4) / k
        m["int_leap_frac"] = sum(1 for x in absint if 5 <= x <= 7) / k
        m["int_large_frac"] = sum(1 for x in absint if x > 7) / k
    if notes:
        pos = []
        sync = 0
        for n in notes:
            i = bisect.bisect_right(starts, n.onset) - 1
            p = (n.onset - starts[max(i, 0)]) % TICKS_PER_BEAT
            pos.append(p)
            # syncopation proxy: an off-beat attack that sustains across the next beat
            if p != 0 and n.onset + n.duration > n.onset - p + TICKS_PER_BEAT:
                sync += 1
        m["offbeat_8th_frac"] = sum(1 for p in pos if p == 2) / len(notes)
        m["syncopation_frac"] = sync / len(notes)
        durs = [n.duration for n in notes]
        m["dur_16th_frac"] = sum(1 for d in durs if d <= 1) / len(durs)
        m["dur_8th_frac"] = sum(1 for d in durs if d == 2) / len(durs)
        m["dur_long_frac"] = sum(1 for d in durs if d >= 8) / len(durs)
        attacks = [n for n in notes if n.lyric]
        sylls = sum(len(n.lyric) for n in attacks)
        m["cram_syllable_frac"] = sum(len(n.lyric) for n in attacks if len(n.lyric) > 1) / max(sylls, 1)
        # early vs late half of the song (by time)
        mid = song.total_ticks / 2
        first = [n for n in notes if n.onset < mid]
        second = [n for n in notes if n.onset >= mid]
        if len(first) >= 4 and len(second) >= 4:
            r1 = max(x.pitch for x in first) - min(x.pitch for x in first)
            r2 = max(x.pitch for x in second) - min(x.pitch for x in second)
            m["late_minus_early_pitch_range"] = r2 - r1

            def rhythm_div(ns):
                sig = {}
                for n in ns:
                    i = bisect.bisect_right(starts, n.onset) - 1
                    sig.setdefault(i, []).append((n.onset - starts[max(i, 0)], n.duration))
                pats = [tuple(v) for v in sig.values()]
                return len(set(pats)) / max(len(pats), 1)

            m["late_minus_early_rhythm_diversity"] = rhythm_div(second) - rhythm_div(first)
    # ---------------------------------------------- long-range coherence (same-label sections)
    sec_notes = _section_notes(song)
    labels = [s.label for s in song.sections]
    by_label: Dict[str, List[int]] = {}
    for i, l in enumerate(labels):
        if len(sec_notes[i]) >= 4:
            by_label.setdefault(l, []).append(i)
    for lab in ("verse", "chorus"):
        idx = by_label.get(lab, [])
        sims = [motif_similarity(sec_notes[a], sec_notes[b]) for k, a in enumerate(idx) for b in idx[k + 1:]]
        v = _mean(sims)
        if v is not None:
            m[f"{lab}_{lab}_motif_sim"] = v
    ch = by_label.get("chorus", [])
    if len(ch) >= 2:
        g0 = set(_ngrams([max(-12, min(12, y.pitch - x.pitch)) for x, y in zip(sec_notes[ch[0]], sec_notes[ch[0]][1:])], 4))
        pres = []
        for c in ch[1:]:
            g = set(_ngrams([max(-12, min(12, y.pitch - x.pitch)) for x, y in zip(sec_notes[c], sec_notes[c][1:])], 4))
            if g0 and g:
                pres.append(len(g0 & g) / len(g0))
        if pres:
            m["chorus_motif_preservation"] = sum(pres) / len(pres)
    cross = []
    labelled = [i for i in range(len(labels)) if len(sec_notes[i]) >= 4]
    for k, a in enumerate(labelled):
        for b in labelled[k + 1:]:
            if labels[a] != labels[b] and {labels[a], labels[b]} <= {"verse", "chorus", "prechorus", "bridge"}:
                cross.append(motif_similarity(sec_notes[a], sec_notes[b]))
    v = _mean(cross)
    if v is not None:
        m["cross_label_motif_sim"] = v
    within = []
    for i, s in enumerate(song.sections):
        bm = [bar_notes[b] for b in range(s.start_bar, min(s.start_bar + s.num_bars, len(bar_notes)))]
        for x, y in zip(bm, bm[1:]):
            within.append(motif_similarity(x, y) if len(x) >= 4 and len(y) >= 4 else None)
    v = _mean(within)
    if v is not None:
        m["within_section_adjacent_bar_motif_sim"] = v
    # ---------------------------------------------- structure vs the request
    if spec is not None:
        want = spec["sections"]
        got = song.sections
        n_want = len(want)
        m["section_count_match"] = float(len(got) == n_want)
        m["section_bars_seq_match"] = float([s["bars"] for s in want] == [s.num_bars for s in got])
        m["section_completion_ratio"] = min(len(got), n_want) / max(n_want, 1)
        m["eos_section_index"] = len(got)
        m["fewer_sections_than_requested"] = float(len(got) < n_want)
        m["more_sections_than_requested"] = float(len(got) > n_want)
        same_idx = [(w, got[i] if i < len(got) else None) for i, w in enumerate(want)]
        m["section_bars_exact_frac"] = sum(1 for w, g in same_idx if g is not None and g.num_bars == w["bars"] and g.label == w["label"]) / max(n_want, 1)
        want_bars = sum(s["bars"] for s in want)
        m["total_bars_ratio"] = len(song.bar_beats) / max(want_bars, 1)
        m["abs_total_bar_error"] = abs(len(song.bar_beats) - want_bars)
        late = [i for i in range(n_want) if i >= math.ceil(2 * n_want / 3)]
        if late:
            m["late_section_present"] = sum(1 for i in late if i < len(got) and got[i].label == want[i]["label"]) / len(late)
            m["late_section_exact"] = sum(1 for i in late if i < len(got) and got[i].label == want[i]["label"]
                                          and got[i].num_bars == want[i]["bars"]) / len(late)
        for lab in ("bridge", "outro"):
            idx = [i for i in range(n_want) if want[i]["label"] == lab]
            if idx:
                m[f"{lab}_completion"] = sum(1 for i in idx if i < len(got) and got[i].label == lab) / len(idx)
        # lyrics: which requested syllables were sung (one LCS alignment over the whole song)
        from .prompt import split_syllables

        wanted = []
        for i, s in enumerate(want):
            for line in s["lines"]:
                wanted.extend((i, x) for x in split_syllables(line))
        sung = [x for n in notes if n.lyric for x in n.lyric]
        matched = set(lcs_matches([x for _, x in wanted], sung))
        if wanted:
            cut = math.ceil(2 * len(wanted) / 3)
            tail = range(cut, len(wanted))
            m["late_lyric_recall"] = sum(1 for j in tail if j in matched) / max(len(tail), 1)
            lsec = [j for j, (i, _) in enumerate(wanted) if i in late] if late else []
            if lsec:
                m["late_section_lyric_recall"] = sum(1 for j in lsec if j in matched) / len(lsec)
    return m
