#!/usr/bin/env python
"""Does the melody follow the lyric's Mandarin tones? A content-level test of lyric conditioning.

  python -m paper_eval.tone_melody

For consecutive sung syllables on different notes, the tone pair predicts a pitch direction
(end height of the first tone vs start height of the second, on the 5-level Chao scale:
T1 55, T2 35, T3 214 -> sung as a low 21 before another syllable, T4 51; 3-3 sandhi applied).
Agreement = share of pairs whose melodic direction equals the predicted direction, among pairs
where both are non-zero (chance 0.5).

Crossover design on the `lyrics_all` intervention (same plan/key/tempo, lyrics replaced by
another song's with the same syllable count per section):
  A = unmodified samples, B = lyric-swapped samples; tones_old / tones_new aligned by syllable slot.
  own-lyric advantage(A) = agree(A, tones_old) - agree(A, tones_new)
  own-lyric advantage(B) = agree(B, tones_new) - agree(B, tones_old)
If the model shapes its melody to the lyric it is given, both advantages are > 0. The same
statistic on the pseudo-labelled references (own vs another song's tones) says whether the
training data carry that signal at all.
"""

from __future__ import annotations

import re
from collections import defaultdict
from typing import List, Optional

from qwen_abc.canonical import Song
from qwen_abc.prompt import split_syllables

from .collect import iter_new
from .common import DATA_OUT, bootstrap_mean_ci, fmt_ci, jdump, load_rows, mean, paired_bootstrap, write_table
from .interventions import pick_donor, syllables_of
from .seqsim import lcs_pairs

_CJK = re.compile(r"[㐀-鿿豈-﫿]")
START = {1: 5, 2: 3, 3: 2, 4: 5}
END = {1: 5, 2: 5, 3: 1, 4: 1}


def tones_of(sylls: List[str]) -> List[Optional[int]]:
    from pypinyin import Style, lazy_pinyin
    out: List[Optional[int]] = [None] * len(sylls)
    i = 0
    while i < len(sylls):
        if not _CJK.fullmatch(sylls[i]):
            i += 1
            continue
        j = i
        while j < len(sylls) and _CJK.fullmatch(sylls[j]):
            j += 1
        py = lazy_pinyin("".join(sylls[i:j]), style=Style.TONE3, neutral_tone_with_five=True)
        if len(py) == j - i:
            for k, p in enumerate(py):
                t = int(p[-1]) if p and p[-1].isdigit() else None
                out[i + k] = t if t in (1, 2, 3, 4) else None
        i = j
    for k in range(len(out) - 1):          # third-tone sandhi
        if out[k] == 3 and out[k + 1] == 3:
            out[k] = 2
    return out


def slot_pitches(song: Song, requested: List[str]):
    """(pitch, note index) for every requested syllable slot the sample actually sang, else None."""
    sung, where = [], []
    for ni, n in enumerate(song.notes):
        if n.lyric:
            for s in n.lyric:
                sung.append(s)
                where.append((n.pitch, ni))
    if not sung:
        return [None] * len(requested)
    out = [None] * len(requested)
    for a, b in lcs_pairs(requested, sung):
        out[a] = where[b]
    return out


def agreement(slots, tones) -> Optional[tuple]:
    agree = total = 0
    for k in range(min(len(slots), len(tones)) - 1):
        a, b, ta, tb = slots[k], slots[k + 1], tones[k], tones[k + 1]
        if a is None or b is None or ta is None or tb is None or a[1] == b[1]:
            continue
        pred = (START[tb] > END[ta]) - (START[tb] < END[ta])
        mel = (b[0] > a[0]) - (b[0] < a[0])
        if pred == 0 or mel == 0:
            continue
        total += 1
        agree += pred == mel
    return (agree, total) if total else None


def rate(pairs):
    a = sum(x[0] for x in pairs if x)
    t = sum(x[1] for x in pairs if x)
    return a / t if t else None


def main() -> None:
    rows = {r["song_id"]: r for r in load_rows("test")}
    pool = {sid: r["spec"] for sid, r in rows.items()}
    req = {sid: [x for sec in r["spec"]["sections"] for x in syllables_of(sec)] for sid, r in rows.items()}
    tones = {sid: tones_of(v) for sid, v in req.items()}
    out = {}
    # ---- references: own tones vs the donor song's tones at the same slots
    ref_own, ref_other = {}, {}
    for sid, r in rows.items():
        s = Song.from_json(r["song"])
        slots = slot_pitches(s, req[sid])
        donor = pick_donor(sid, r["spec"], pool)
        ag_o = agreement(slots, tones[sid])
        ag_d = agreement(slots, tones_of([req[donor][k % len(req[donor])] for k in range(len(req[sid]))]) if donor else [])
        if ag_o and ag_d:
            ref_own[sid], ref_other[sid] = ag_o[0] / ag_o[1], ag_d[0] / ag_d[1]
    out["reference"] = paired_bootstrap(ref_own, ref_other)
    print("reference own vs other:", out["reference"])
    table = [{"melody": "pseudo-GT reference", "tones": "own lyric vs another song's",
              "agreement (own)": f"{out['reference']['a']:.3f}", "agreement (other)": f"{out['reference']['b']:.3f}",
              "own − other [95% CI]": fmt_ci(out["reference"]["diff"], out["reference"]["lo"], out["reference"]["hi"], 3, True),
              "N songs": out["reference"]["n"]}]
    # ---- models: crossover on lyrics_all
    for model in ("qwen_e3b", "mupt"):
        A = defaultdict(list)
        for sid, s, row in iter_new(model, "orig"):
            if row.get("song"):
                A[sid].append(Song.from_json(row["song"]))
        B, newreq = defaultdict(list), {}
        for sid, s, row in iter_new(model, "lyrics_all"):
            if row.get("song"):
                B[sid].append(Song.from_json(row["song"]))
                newreq[sid] = [x for sec in row["spec"]["sections"] for x in syllables_of(sec)]
        adv_a, adv_b, a_old, a_new, b_new, b_old = {}, {}, {}, {}, {}, {}
        for sid in B:
            if sid not in A:
                continue
            t_old, t_new = tones[sid], tones_of(newreq[sid])
            ao = rate([agreement(slot_pitches(x, req[sid]), t_old) for x in A[sid]])
            an = rate([agreement(slot_pitches(x, req[sid]), t_new) for x in A[sid]])
            bn = rate([agreement(slot_pitches(x, newreq[sid]), t_new) for x in B[sid]])
            bo = rate([agreement(slot_pitches(x, newreq[sid]), t_old) for x in B[sid]])
            if None in (ao, an, bn, bo):
                continue
            a_old[sid], a_new[sid], b_new[sid], b_old[sid] = ao, an, bn, bo
            adv_a[sid], adv_b[sid] = ao - an, bn - bo
        for label, x, y in (("unmodified samples (A)", a_old, a_new), ("lyric-swapped samples (B)", b_new, b_old)):
            d = paired_bootstrap(x, y)
            out[f"{model}:{label}"] = d
            table.append({"melody": f"{model} {label}", "tones": "own (prompted) lyric vs the other lyric",
                          "agreement (own)": f"{d['a']:.3f}" if d["a"] is not None else "–",
                          "agreement (other)": f"{d['b']:.3f}" if d["b"] is not None else "–",
                          "own − other [95% CI]": fmt_ci(d["diff"], d["lo"], d["hi"], 3, True), "N songs": d["n"]})
    write_table(table, "tone_melody", "Mandarin tone-melody direction agreement (chance 0.5): own lyric vs counterfactual lyric at the same syllable slots; song-level paired bootstrap")
    jdump(out, DATA_OUT / "tone_melody.json")
    print("TONE_MELODY_DONE")


if __name__ == "__main__":
    main()
