#!/usr/bin/env python
"""Analyse listening-study responses (PENDING: no responses have been collected).

  python reports/paper_final/listening_study/analysis.py --responses reports/paper_final/listening_study/responses \
      [--out results.json] [--max-catch-errors 1]

Inputs: ``items.csv`` (answer key), ``lists/list_<k>.csv`` (which clip was A), and response CSVs
following ``response_schema.csv``. Reports, per condition and question:

* 2AFC accuracy (3 options: A / B / none) against chance 1/3, and the forced-choice rate among
  A/B answers against 0.5 -- each with a 95% song-level (cluster) bootstrap CI;
* for key items, accuracy restricted to items whose melody register moved in the requested
  direction (the declared key and the register can disagree);
* ``edit_happened`` yes/partly/no rates and their agreement with the automatic compliance flag;
* Likert means (plausibility of edited vs original clip, unrelated-content consistency);
* inter-rater agreement: Fleiss' kappa (nominal, variable raters per item) and Krippendorff's alpha
  (nominal for choices, interval for Likert).

Participants who miss more than ``--max-catch-errors`` catch items are excluded (reported).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional

HERE = Path(__file__).resolve().parent
EXPECTED = {  # which clip is the correct answer to the item's 2AFC question
    "bars_p4": "edited", "bars_m4": "original", "key_p5": "edited", "key_m3": "original", "lyrics_sec": "edited",
    "catch_identical": "none", "catch_transposed": "edited", "catch_doubled": "edited",
}
TWO_AFC = ("longer_section", "higher_key", "matches_lyrics")


def read_csv(p: Path) -> List[dict]:
    with open(p, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def load(study: Path, responses: Path):
    items = {r["item_id"]: r for r in read_csv(study / "items.csv")}
    lists = {}
    for p in sorted((study / "lists").glob("list_*.csv")):
        for r in read_csv(p):
            lists[(str(r["list_id"]), r["item_id"])] = r
    resp = []
    paths = [responses] if responses.is_file() else sorted(responses.glob("*.csv"))
    for p in paths:
        resp.extend(read_csv(p))
    return items, lists, resp


def expected_letter(item: dict, lst: dict) -> str:
    side = EXPECTED[item["condition"]]
    if side == "none":
        return "none"
    edited = lst["edited_is"]
    return edited if side == "edited" else ("B" if edited == "A" else "A")


def boot_ci(values_by_song: Dict[str, List[float]], n_boot: int = 10000, seed: int = 0):
    songs = [s for s, v in values_by_song.items() if v]
    if not songs:
        return None
    per = {s: sum(values_by_song[s]) / len(values_by_song[s]) for s in songs}
    m = sum(per.values()) / len(per)
    rng = random.Random(seed)
    bs = []
    for _ in range(n_boot):
        smp = [per[rng.choice(songs)] for _ in songs]
        bs.append(sum(smp) / len(smp))
    bs.sort()
    return {"mean": m, "lo": bs[int(0.025 * n_boot)], "hi": bs[int(0.975 * n_boot) - 1], "n_songs": len(songs),
            "n_responses": sum(len(values_by_song[s]) for s in songs)}


def fleiss_kappa(ratings: Dict[str, List[str]]) -> Optional[float]:
    """Fleiss' kappa with a variable number of raters per item (items with <2 ratings dropped)."""
    items = {k: v for k, v in ratings.items() if len(v) >= 2}
    if not items:
        return None
    cats = sorted({c for v in items.values() for c in v})
    p_i, totals, N = [], Counter(), 0
    for v in items.values():
        n = len(v)
        c = Counter(v)
        p_i.append((sum(x * x for x in c.values()) - n) / (n * (n - 1)))
        totals.update(c)
        N += n
    P_bar = sum(p_i) / len(p_i)
    P_e = sum((totals[c] / N) ** 2 for c in cats)
    return None if P_e == 1 else (P_bar - P_e) / (1 - P_e)


def krippendorff_alpha(ratings: Dict[str, List], metric: str = "nominal") -> Optional[float]:
    """Krippendorff's alpha from the coincidence matrix (nominal or interval)."""
    units = {k: v for k, v in ratings.items() if len(v) >= 2}
    if not units:
        return None
    vals = sorted({x for v in units.values() for x in v})
    if metric == "interval":
        d = lambda a, b: (float(a) - float(b)) ** 2  # noqa: E731
    else:
        d = lambda a, b: 0.0 if a == b else 1.0  # noqa: E731
    o = defaultdict(float)
    for v in units.values():
        m = len(v)
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[(v[i], v[j])] += 1.0 / (m - 1)
    n_c = defaultdict(float)
    for (a, _b), w in o.items():
        n_c[a] += w
    n = sum(n_c.values())
    D_o = sum(w * d(a, b) for (a, b), w in o.items()) / n
    D_e = sum(n_c[a] * n_c[b] * d(a, b) for a in vals for b in vals) / (n * (n - 1))
    return None if D_e == 0 else 1 - D_o / D_e


def analyse(study: Path, responses: Path, max_catch_errors: int = 1, n_boot: int = 10000) -> dict:
    items, lists, resp = load(study, responses)
    # catch-based exclusion
    catch_err = Counter()
    for r in resp:
        it = items.get(r["item_id"])
        if it and it["kind"] == "catch" and r["question_id"] in TWO_AFC:
            catch_err[r["participant_id"]] += r["response"] != expected_letter(it, lists[(str(r["list_id"]), r["item_id"])])
    participants = sorted({r["participant_id"] for r in resp})
    excluded = sorted(p for p in participants if catch_err[p] > max_catch_errors)
    keep = [r for r in resp if r["participant_id"] not in excluded and items.get(r["item_id"], {}).get("kind") == "test"]
    out = {"participants": len(participants), "excluded": excluded, "responses_used": len(keep), "conditions": {}}
    by_cond = defaultdict(list)
    for r in keep:
        by_cond[items[r["item_id"]]["condition"]].append(r)
    for cond, rows in sorted(by_cond.items()):
        res = {}
        acc, forced, reg_ok, ratings = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
        edit, edit_vs_auto, lik = defaultdict(list), [], defaultdict(lambda: defaultdict(list))
        for r in rows:
            it = items[r["item_id"]]
            lst = lists[(str(r["list_id"]), r["item_id"])]
            sid = it["song_id"]
            q, v = r["question_id"], r["response"]
            if q in TWO_AFC:
                exp = expected_letter(it, lst)
                acc[sid].append(float(v == exp))
                if v in ("A", "B"):
                    forced[sid].append(float(v == exp))
                if it.get("realized_register_agrees_with_request") == "True":
                    reg_ok[sid].append(float(v == exp))
                # rate the *side* (edited/original/none) so A/B order does not split agreement
                side = "none" if v not in ("A", "B") else ("edited" if v == lst["edited_is"] else "original")
                ratings[r["item_id"]].append(side)
            elif q == "edit_happened":
                edit[sid].append({"yes": 1.0, "partly": 0.5, "no": 0.0}.get(v, 0.0))
                edit_vs_auto.append((v, it.get("realized_complied") == "True"))
                ratings["edit|" + r["item_id"]].append(v)
            elif q in ("plausibility_A", "plausibility_B"):
                side = "edited" if q[-1] == lst["edited_is"] else "original"
                lik[f"plausibility_{side}"][sid].append(float(v))
            elif q == "unrelated_consistent":
                lik["unrelated_consistent"][sid].append(float(v))
        res["accuracy_3way"] = boot_ci(acc, n_boot)
        res["chance_3way"] = 1 / 3
        res["accuracy_forced_AB"] = boot_ci(forced, n_boot)
        res["chance_forced_AB"] = 0.5
        if reg_ok:
            res["accuracy_register_agrees"] = boot_ci(reg_ok, n_boot)
        res["edit_happened_score"] = boot_ci(edit, n_boot)
        if edit_vs_auto:
            agree = sum(1 for v, auto in edit_vs_auto if (v == "yes") == auto)
            res["edit_happened_agrees_with_automatic"] = agree / len(edit_vs_auto)
        for k, v in lik.items():
            res[k] = boot_ci(v, n_boot)
        res["fleiss_kappa_2afc"] = fleiss_kappa({k: v for k, v in ratings.items() if not k.startswith("edit|")})
        res["krippendorff_alpha_2afc"] = krippendorff_alpha({k: v for k, v in ratings.items() if not k.startswith("edit|")})
        res["fleiss_kappa_edit_happened"] = fleiss_kappa({k: v for k, v in ratings.items() if k.startswith("edit|")})
        out["conditions"][cond] = res
    lik_units = defaultdict(list)
    for r in keep:
        if r["question_id"] == "unrelated_consistent":
            lik_units[r["item_id"]].append(r["response"])
    out["krippendorff_alpha_consistency_interval"] = krippendorff_alpha(lik_units, "interval")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", type=Path, default=HERE)
    ap.add_argument("--responses", type=Path, default=HERE / "responses")
    ap.add_argument("--max-catch-errors", type=int, default=1)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not args.responses.exists() or (args.responses.is_dir() and not any(args.responses.glob("*.csv"))):
        raise SystemExit(f"no responses under {args.responses}: listening results are PENDING")
    res = analyse(args.study, args.responses, args.max_catch_errors)
    txt = json.dumps(res, indent=1)
    print(txt)
    if args.out:
        args.out.write_text(txt, encoding="utf-8")


if __name__ == "__main__":
    main()
