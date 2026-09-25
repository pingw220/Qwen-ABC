"""Every generation task of the paper-final round, as data.

A task is (model, condition, song, seed name) plus the spec to prompt with. Conditions:

* ``orig``          the song's own spec (seeds S1..S4) -- single / selector@4 / oracle@4, reseed floor
* ``replay``        ``orig`` at S1 again, batched in a different composition -- numeric noise floor
* interventions (seeds S1, S2; lyrics_all also S3, S4), each changes exactly ONE control of ``orig``:
  ``bars_m4 bars_m2 bars_p2 bars_p4``   one target section's bar count
  ``label_bridge label_swap``          one target section's label (verse/chorus -> bridge; verse <-> chorus)
  ``key_p2 key_p5 key_m3``             the key
  ``tempo_x0.8 tempo_x1.25``           the tempo
  ``lyrics_all``                       every lyric replaced by another held-out song's (same syllable count per section)
  ``lyrics_sec``                       the target section's lyrics only
* OOD plans (seed S1), see paper_eval/ood.py.

Targets and donors are deterministic functions of the song id, so every model gets the
identical task list.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .common import SEEDS, load_rows, seeded_choice
from .interventions import (change_bars, change_label, control_family, controls_changed, pick_donor,
                            pick_target_section, replace_lyrics, scale_tempo, syllables_of, transpose_key)

INTERVENTIONS = {
    "bars_m4": ("bars", -4), "bars_m2": ("bars", -2), "bars_p2": ("bars", 2), "bars_p4": ("bars", 4),
    "label_bridge": ("label", "bridge"), "label_swap": ("label", "swap"),
    "key_p2": ("key", 2), "key_p5": ("key", 5), "key_m3": ("key", -3),
    "tempo_x0.8": ("tempo", 0.8), "tempo_x1.25": ("tempo", 1.25),
    "lyrics_all": ("lyrics", "all"), "lyrics_sec": ("lyrics", "sec"),
}
INTERVENTION_SEEDS = ("S1", "S2")
# the central lyric test gets four seeds so its within-condition spread is estimated as well as orig's
EXTRA_SEEDS = {"lyrics_all": ("S3", "S4")}
BON_SEEDS = ("S1", "S2", "S3", "S4")


def _task(model, cond, row, seed_name, spec, **meta):
    return {"model": model, "condition": cond, "song_id": row["song_id"], "seed_name": seed_name,
            "seed": SEEDS[seed_name], "spec": spec, "meta": meta}


def intervention_spec(row: dict, cond: str, pool: Dict[str, dict]):
    """(spec, meta) for one intervention on one song, or (None, reason) if the song is ineligible."""
    spec, sid = row["spec"], row["song_id"]
    kind, arg = INTERVENTIONS[cond]
    tgt = pick_target_section(spec, sid)
    meta = {"family": kind, "arg": arg, "target_section": tgt}
    if kind in ("bars", "label", "lyrics") and arg != "all" and tgt is None:
        return None, "no eligible target section"
    if kind == "bars":
        new = spec["sections"][tgt]["bars"] + arg
        if new < 2:
            return None, "section too short"
        meta.update(requested_bars=new, original_bars=spec["sections"][tgt]["bars"],
                    target_label=spec["sections"][tgt]["label"], target_syllables=len(syllables_of(spec["sections"][tgt])))
        try:
            return change_bars(spec, tgt, arg), meta
        except ValueError as e:
            return None, str(e)
    if kind == "label":
        old = spec["sections"][tgt]["label"]
        new = "bridge" if arg == "bridge" else {"verse": "chorus", "chorus": "verse"}[old]
        meta.update(original_label=old, requested_label=new)
        return change_label(spec, tgt, new), meta
    if kind == "key":
        try:
            out = transpose_key(spec, arg)
        except ValueError:
            return None, "no key"
        meta.update(original_key=spec["key"], requested_key=out["key"], semitones=arg)
        return out, meta
    if kind == "tempo":
        out = scale_tempo(spec, arg)
        meta.update(original_tempo=spec["tempo_bpm"], requested_tempo=out["tempo_bpm"])
        return out, meta
    if kind == "lyrics":
        donor = pick_donor(sid, spec, pool)
        dsyl = [x for s in pool[donor]["sections"] for x in syllables_of(s)]
        if arg == "all":
            out = replace_lyrics(spec, dsyl)
        else:
            out = replace_lyrics(spec, dsyl, tgt)
        meta.update(donor=donor)
        return out, meta
    raise KeyError(cond)


def build_tasks(model: str, suite: str, rows: Optional[List[dict]] = None,
                conditions: Optional[List[str]] = None) -> tuple:
    """(tasks, skipped) for a suite: 'bon', 'interventions', 'replay', 'ood'."""
    rows = rows if rows is not None else load_rows("test")
    pool = {r["song_id"]: r["spec"] for r in load_rows("test")}
    tasks, skipped = [], []
    if suite == "bon":
        for r in rows:
            for s in BON_SEEDS:
                tasks.append(_task(model, "orig", r, s, r["spec"], family="none"))
    elif suite == "replay":
        for r in rows:
            tasks.append(_task(model, "replay", r, "S1", r["spec"], family="none"))
    elif suite == "interventions":
        conds = conditions or list(INTERVENTIONS)
        for cond in conds:
            for r in rows:
                spec, meta = intervention_spec(r, cond, pool)
                if spec is None:
                    skipped.append({"condition": cond, "song_id": r["song_id"], "reason": meta})
                    continue
                changed = controls_changed(r["spec"], spec)
                fam = control_family(changed)
                if fam == "none":
                    skipped.append({"condition": cond, "song_id": r["song_id"], "reason": "intervention is a no-op"})
                    continue
                if fam != INTERVENTIONS[cond][0]:
                    raise AssertionError(f"{cond} {r['song_id']}: changed {changed} (family {fam})")
                meta["controls_changed"] = changed
                for s in INTERVENTION_SEEDS + EXTRA_SEEDS.get(cond, ()):
                    tasks.append(_task(model, cond, r, s, spec, **meta))
    elif suite == "ood":
        from .ood import ood_tasks_for_row, load_train_stats
        stats = load_train_stats()
        for r in rows:
            for cond, spec, meta in ood_tasks_for_row(r, stats, conditions):
                if spec is None:
                    skipped.append({"condition": cond, "song_id": r["song_id"], "reason": meta})
                    continue
                meta["controls_changed"] = controls_changed(r["spec"], spec)
                tasks.append(_task(model, cond, r, "S1", spec, **meta))
    else:
        raise KeyError(suite)
    return tasks, skipped
