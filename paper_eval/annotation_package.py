#!/usr/bin/env python
"""Build the human-annotation package for pseudo-label verification (no human results inside).

  python -m paper_eval.annotation_package [--n 100] [--no-materials]

Writes ``reports/paper_final/human_annotation_package/``:

* ``candidates.csv``            song ids, selection stratum, priority score and every component (no lyrics)
* ``PRIORITIZATION.md``         the exact score, generated from the constants below
* ``materials/<song>/``         (git-ignored: lyrics) pseudo-label ABC + plan, E3b predictions (sample 0 and
                                selector@4 choice), reference MIDI, a piano-roll PNG of reference vs prediction
* ``materials/annotate.html`` + ``materials/songs_data.js``   an offline annotation UI

Selection (deterministic): all 34 songs of the existing clean subset (anchors: they were
screened as *likely clean*, so annotating them measures the precision of that screen), the
highest-priority remaining songs, and a seeded random control sample of the rest, so that
noise rates can also be estimated without the prioritization bias.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

from .common import CLEAN_IDS, OFFLOAD, REPORT_DIR, ROOT, jdump, load_rows

PKG = REPORT_DIR / "human_annotation_package"
MAT = PKG / "materials"
MANIFEST = Path("/mmfs1/gscratch/scrubbed/pingw220/music_acc/sheetsage-pro/dataset/processed/"
                "sheetsage_zh_training_views_v2_lvcr/manifest.jsonl")
E3B_S0 = OFFLOAD / "evals/bon_e3b_test_T1.0/generations"          # <song>_s{0..3}.json
E3B_SEL = ROOT / "experiments/bon_r3b_20260918_100234/test_sum/generations"   # <song>_s0.json

# ---------------------------------------------------------------- priority score (documented verbatim)
# Each component is in [0, 1]; the score is the weighted sum. Weights encode the brief's order:
# pseudo-label confidence first, then model/reference disagreement, then specific label defects.
WEIGHTS = {
    "low_confidence": 3.0,        # SheetSage-Pro tier B, medium key agreement, low note-lyric coverage, pathologies
    "structure_disagreement": 2.5,  # E3b samples that do not reproduce the pseudo plan (fraction of 4) + selector miss
    "tiny_lyric_sections": 1.5,   # sections holding 1-2 syllables (boundary cut through a lyric line)
    "suspicious_boundaries": 1.5, # lyric lines still cut by a section boundary after cleaning (cuts_after), capped at 4
    "label_cramming": 1.0,        # pseudo-label notes carrying several syllables (aligner cramming), /0.05 capped
    "abnormal_order": 1.0,        # intro not first / outro not last / bridge or outro first / >1 intro
    "model_disagreement": 1.5,    # E3b lyric recall < 1 or heavy cramming in the model output (selector choice)
}
N_RANDOM_CONTROL = 10
RANDOM_SEED = 20260925


def _clip(x: float) -> float:
    return max(0.0, min(1.0, x))


def load_manifest(ids) -> Dict[str, dict]:
    out = {}
    if not MANIFEST.exists():
        return out
    with open(MANIFEST, encoding="utf-8") as fh:
        for line in fh:
            d = json.loads(line)
            sid = d["song_id"]
            if sid in ids:
                out[sid] = {k: d.get(k) for k in ("quality_tier", "quality_tier_reasons", "note_lyric_coverage",
                                                   "note_phoneme_coverage", "num_unmatched_notes", "num_melody_notes")}
    return out


def _load(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def abnormal_order(labels: List[str]) -> float:
    bad = 0
    if "intro" in labels and labels[0] != "intro":
        bad += 1
    if "outro" in labels and labels[-1] != "outro":
        bad += 1
    if labels and labels[0] in ("bridge", "outro", "chorus"):
        bad += 1
    if labels.count("intro") > 1:
        bad += 1
    return _clip(bad / 2)


def components(row: dict, man: dict, preds: List[dict], sel: dict) -> Dict[str, float]:
    from qwen_abc.prompt import split_syllables
    spec = row["spec"]
    f = row["flags"]
    cov = man.get("note_lyric_coverage")
    lowc = 0.0
    lowc += 0.35 * (man.get("quality_tier", "A") != "A")
    lowc += 0.25 * (f.get("key_agreement") != "high")
    lowc += 0.25 * (_clip((0.97 - cov) / 0.07) if cov is not None else 0.0)
    lowc += 0.15 * (len(row.get("pathologies") or []) > 0)
    plan = [(s["label"], s["bars"]) for s in spec["sections"]]
    miss = [p for p in preds if p and p.get("song")]
    frac_off = (sum(1 for p in miss if [(s["label"], s["num_bars"]) for s in p["song"]["sections"]] != plan) / len(miss)
                if miss else 1.0)
    sel_off = float(bool(sel and sel.get("song")) and [(s["label"], s["num_bars"]) for s in sel["song"]["sections"]] != plan)
    tiny = sum(1 for s in spec["sections"] if 0 < sum(len(split_syllables(l)) for l in s["lines"]) <= 2)
    sm = (sel or {}).get("metrics") or {}
    return {
        "low_confidence": _clip(lowc),
        "structure_disagreement": _clip(0.7 * frac_off + 0.3 * sel_off),
        "tiny_lyric_sections": _clip(tiny / 2),
        "suspicious_boundaries": _clip(row["cleaning"].get("cuts_after", 0) / 4),
        "label_cramming": _clip(f.get("multi_syllable_note_frac", 0.0) / 0.05),
        "abnormal_order": abnormal_order([s["label"] for s in spec["sections"]]),
        "model_disagreement": _clip(0.5 * (1 - sm.get("lyric_recall", 0.0)) / 0.05 + 0.5 * sm.get("cram_syllable_frac", 0.0) / 0.3),
    }


def score(c: Dict[str, float]) -> float:
    return sum(WEIGHTS[k] * c[k] for k in WEIGHTS)


def select(n: int = 100):
    rows = load_rows("test")
    ids = {r["song_id"] for r in rows}
    man = load_manifest(ids)
    clean = set(l.strip() for l in open(CLEAN_IDS, encoding="utf-8") if l.strip())
    recs = []
    for r in rows:
        sid = r["song_id"]
        preds = [_load(E3B_S0 / f"{sid}_s{k}.json") for k in range(4)]
        sel = _load(E3B_SEL / f"{sid}_s0.json")
        c = components(r, man.get(sid, {}), preds, sel)
        recs.append({"song_id": sid, "score": round(score(c), 4), **{k: round(v, 4) for k, v in c.items()},
                     "quality_tier": man.get(sid, {}).get("quality_tier"), "key_agreement": r["flags"].get("key_agreement"),
                     "note_lyric_coverage": man.get(sid, {}).get("note_lyric_coverage"),
                     "pathologies": "|".join(r.get("pathologies") or []), "in_clean_subset": sid in clean,
                     "n_sections": len(r["spec"]["sections"]), "n_bars": sum(s["bars"] for s in r["spec"]["sections"])})
    anchors = sorted((x for x in recs if x["in_clean_subset"]), key=lambda x: x["song_id"])
    rest = sorted((x for x in recs if not x["in_clean_subset"]), key=lambda x: (-x["score"], x["song_id"]))
    n_prio = max(n - len(anchors) - N_RANDOM_CONTROL, 0)
    prio = rest[:n_prio]
    pool = sorted(x["song_id"] for x in rest[n_prio:])
    rnd = set(random.Random(RANDOM_SEED).sample(pool, min(N_RANDOM_CONTROL, len(pool))))
    control = [x for x in rest[n_prio:] if x["song_id"] in rnd]
    chosen = []
    for stratum, xs in (("anchor_clean_subset", anchors), ("priority", prio), ("random_control", control)):
        for rank, x in enumerate(xs):
            chosen.append({"stratum": stratum, "rank_in_stratum": rank + 1, **x})
    # annotation order: interleave strata so a partial annotation pass still covers all three
    return chosen, recs


def write_candidates(chosen, recs) -> None:
    PKG.mkdir(parents=True, exist_ok=True)
    cols = ["song_id", "stratum", "rank_in_stratum", "score", *WEIGHTS, "quality_tier", "key_agreement",
            "note_lyric_coverage", "pathologies", "in_clean_subset", "n_sections", "n_bars"]
    with open(PKG / "candidates.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(chosen)
    with open(PKG / "all_test_songs_priority.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=[c for c in cols if c not in ("stratum", "rank_in_stratum")], extrasaction="ignore")
        w.writeheader()
        w.writerows(sorted(recs, key=lambda x: (-x["score"], x["song_id"])))
    strata = Counter(x["stratum"] for x in chosen)
    lines = ["# Annotation candidate prioritization", "",
             "Generated by `python -m paper_eval.annotation_package`; do not edit by hand.", "",
             f"**{len(chosen)} songs**: " + ", ".join(f"{k} {v}" for k, v in strata.items()) + ".", "",
             "* `anchor_clean_subset`: all songs of `reports/clean_test_song_ids.txt` (rule filters + manual reading, "
             "never model output). Annotating them measures how clean the 'clean' screen really is.",
             "* `priority`: the highest-scoring remaining test songs under the score below.",
             f"* `random_control`: {N_RANDOM_CONTROL} songs drawn with `random.Random({RANDOM_SEED})` from the rest, "
             "so noise rates can be estimated without the prioritization bias.", "",
             "## Score", "", "score = Σ weight × component, every component clipped to [0, 1]:", "",
             "| component | weight | definition |", "|---|---|---|",
             "| low_confidence | 3.0 | 0.35·[SheetSage-Pro quality_tier ≠ A] + 0.25·[key_agreement ≠ high] + 0.25·clip((0.97 − note_lyric_coverage)/0.07) + 0.15·[any cleaning pathology] |",
             "| structure_disagreement | 2.5 | 0.7·(fraction of the 4 E3b T=1.0 samples whose (label, bars) plan ≠ pseudo plan) + 0.3·[selector@4 choice ≠ pseudo plan] |",
             "| tiny_lyric_sections | 1.5 | (sections holding 1-2 syllables) / 2 |",
             "| suspicious_boundaries | 1.5 | (lyric lines still cut by a section boundary after cleaning, `cleaning.cuts_after`) / 4 |",
             "| label_cramming | 1.0 | pseudo-label fraction of notes carrying several syllables / 0.05 |",
             "| abnormal_order | 1.0 | ([intro present but not first] + [outro present but not last] + [song opens on bridge/outro/chorus] + [>1 intro]) / 2 |",
             "| model_disagreement | 1.5 | 0.5·(1 − selector lyric recall)/0.05 + 0.5·(selector crammed-syllable fraction)/0.3 |",
             "", "Sources: `data/generated/abc_v2_20260915_120927/songs_test.jsonl` (flags, pathologies, cleaning), "
             "the SheetSage-Pro manifest (quality_tier, note_lyric_coverage; read-only), E3b generations "
             "`qwen_abc_r2_offload/evals/bon_e3b_test_T1.0` and the selector choice `bon_r3b_20260918_100234/test_sum`.",
             "", "Caveat: two components use model output, so the priority stratum over-represents songs where the model "
             "and the label disagree. Rates estimated on it are *conditional*; use the random control (and anchors) "
             "for population estimates, or weight by inverse selection probability.", "",
             "Score distribution over all 225 test songs: " +
             ", ".join(f"p{q}={sorted(x['score'] for x in recs)[int(q / 100 * (len(recs) - 1))]:.2f}" for q in (10, 50, 90)) + "."]
    (PKG / "PRIORITIZATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- materials (git-ignored: lyrics)
def pianoroll(ref, pred, path: Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    # no CJK font is installed on Hyak; Noto Sans SC (OFL) is fetched into the git-ignored materials/_fonts
    font = MAT / "_fonts/NotoSansSC-Regular.otf"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(font)).get_name()
    songs = [("pseudo-label (reference)", ref)] + ([("E3b selector@4 prediction", pred)] if pred else [])
    bars_per_row = 16
    nbars = max(len(s.bar_beats) for _, s in songs)
    nrows = math.ceil(nbars / bars_per_row)
    fig, axes = plt.subplots(nrows * len(songs), 1, figsize=(22, 2.6 * nrows * len(songs)), squeeze=False)
    colors = {"intro": "#dde", "verse": "#dfd", "chorus": "#fdd", "bridge": "#ffd", "instrumental": "#eee", "outro": "#dff", "other": "#fef", "prechorus": "#efd"}
    for si, (name, s) in enumerate(songs):
        starts = s.bar_starts() + [s.total_ticks]
        pitches = [n.pitch for n in s.notes] or [60]
        lo, hi = min(pitches) - 2, max(pitches) + 3
        for r in range(nrows):
            ax = axes[si * nrows + r][0]
            b0, b1 = r * bars_per_row, min((r + 1) * bars_per_row, len(s.bar_beats))
            if b0 >= len(s.bar_beats):
                ax.axis("off")
                continue
            t0, t1 = starts[b0], starts[b1]
            for sec in s.sections:
                a, e = starts[min(sec.start_bar, len(starts) - 1)], starts[min(sec.start_bar + sec.num_bars, len(starts) - 1)]
                if e > t0 and a < t1:
                    ax.axvspan(max(a, t0), min(e, t1), color=colors.get(sec.label, "#fff"), zorder=0)
                    if a >= t0:
                        ax.text(a, hi + 0.5, f"{sec.label} ({sec.num_bars})", fontsize=8, va="bottom", weight="bold")
            for b in range(b0, b1 + 1):
                ax.axvline(starts[b], color="#999", lw=0.4)
                if b < b1:
                    ax.text(starts[b] + 1, lo - 0.2, str(b + 1), fontsize=5, va="top", color="#666")
            for c in s.chords:
                if t0 <= c.onset < t1:
                    ax.text(c.onset, hi - 0.5, c.symbol, fontsize=6, color="#a40")
            for n in s.notes:
                if t0 <= n.onset < t1:
                    ax.add_patch(plt.Rectangle((n.onset, n.pitch - 0.4), n.duration, 0.8,
                                               color="#248" if not n.melisma else "#69c", lw=0))
                    if n.lyric:
                        ax.text(n.onset, n.pitch + 0.6, "".join(n.lyric), fontsize=5.5,
                                color="#c00" if len(n.lyric) > 1 else "#000")
            ax.set_xlim(t0, t0 + bars_per_row * 16)
            ax.set_ylim(lo - 1.5, hi + 2)
            ax.set_yticks([])
            ax.set_xticks([])
            if r == 0:
                ax.set_title(f"{title} — {name}", fontsize=9, loc="left")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def write_materials(chosen) -> None:
    from qwen_abc.canonical import Song
    from qwen_abc.midi import song_to_midi
    by_id = {r["song_id"]: r for r in load_rows("test", ids=[c["song_id"] for c in chosen])}
    data = []
    for c in chosen:
        sid = c["song_id"]
        r = by_id[sid]
        d = MAT / sid
        d.mkdir(parents=True, exist_ok=True)
        (d / "pseudo_label.abc").write_text(r["abc"], encoding="utf-8")
        jdump(r["spec"], d / "pseudo_plan.json")
        ref = Song.from_json(r["song"])
        song_to_midi(ref, str(d / "pseudo_label.mid"))
        pred_song = None
        for name, p in (("prediction_e3b_sample0", E3B_S0 / f"{sid}_s0.json"), ("prediction_e3b_selector4", E3B_SEL / f"{sid}_s0.json")):
            g = _load(p)
            if g:
                (d / f"{name}.abc").write_text(g["generation"], encoding="utf-8")
                if name.endswith("selector4") and g.get("song"):
                    pred_song = Song.from_json(g["song"])
        pianoroll(ref, pred_song, d / "pianoroll.png", sid)
        starts = ref.bar_starts()
        data.append({
            "song_id": sid, "stratum": c["stratum"], "priority": c["score"], "tempo": r["spec"]["tempo_bpm"],
            "key": r["spec"]["key"], "meter": r["spec"]["meter"], "png": f"{sid}/pianoroll.png",
            "abc": f"{sid}/pseudo_label.abc", "midi": f"{sid}/pseudo_label.mid",
            "prediction_sections": [[s["label"], s["num_bars"]] for s in (pred_song.to_json()["sections"] if pred_song else [])],
            "sections": [{"index": i, "label": s["label"], "bars": s["bars"], "start_bar": sec.start_bar + 1,
                          "start_sec": round(starts[sec.start_bar] / 4 * 60 / r["spec"]["tempo_bpm"], 2),
                          "lines": s["lines"]}
                         for i, (s, sec) in enumerate(zip(r["spec"]["sections"], ref.sections))],
        })
    (MAT / "songs_data.js").write_text("window.SONGS = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    (MAT / "annotate.html").write_text(ANNOTATE_HTML, encoding="utf-8")


ANNOTATE_HTML = r"""<!doctype html><html><head><meta charset="utf-8"><title>Lead-sheet label check</title>
<style>
body{font:14px system-ui,sans-serif;margin:0;padding:12px 16px;background:#fafafa;color:#222}
header{display:flex;gap:12px;align-items:center;flex-wrap:wrap}select,input,textarea{font:inherit}
table{border-collapse:collapse;margin:8px 0}td,th{border:1px solid #ccc;padding:3px 6px;vertical-align:top}
th{background:#eee}.lines{font-size:12px;color:#444;max-width:260px}img{max-width:100%;border:1px solid #ccc;background:#fff}
.done{color:#070}.todo{color:#a00}button{font:inherit;padding:4px 10px}
</style></head><body>
<header><b>Annotator</b><input id="ann" placeholder="your id" size="10">
<select id="song"></select><span id="status"></span>
<button id="prev">◀</button><button id="next">▶</button>
<button id="export">Export all (JSON)</button><input type="file" id="imp" accept=".json"></header>
<p>Listen to <a id="midi">the pseudo-label MIDI</a> (or the original recording, by song id), read
<a href="../INSTRUCTIONS.md">INSTRUCTIONS.md</a>, then correct each section. Work is saved in this browser automatically.</p>
<div id="meta"></div><table id="tbl"></table>
<p>Whole song: confidence <select id="conf"><option></option><option>1</option><option>2</option><option>3</option><option>4</option><option>5</option></select>
key correct? <select id="keyok"><option></option><option>yes</option><option>no</option></select>
correct key <input id="keyfix" size="8">
tempo/beat grid correct? <select id="gridok"><option></option><option>yes</option><option>half_time</option><option>double_time</option><option>no</option></select></p>
<p>Notes<br><textarea id="notes" rows="3" cols="100"></textarea></p>
<img id="png">
<script src="songs_data.js"></script>
<script>
const LABELS=["intro","verse","prechorus","chorus","bridge","instrumental","outro","other"];
const KEY="qwenabc_annotations_v1";
let store={};try{store=JSON.parse(localStorage.getItem(KEY)||"{}")}catch(e){store={}}
const $=id=>document.getElementById(id);let cur=0;
function save(){try{localStorage.setItem(KEY,JSON.stringify(store))}catch(e){}refreshList()}
function rec(s){const a=$("ann").value||"anon";const k=a+"|"+s.song_id;
 if(!store[k])store[k]={schema_version:"1.0",song_id:s.song_id,annotator_id:a,sections:s.sections.map(x=>({index:x.index,pseudo_label:x.label,pseudo_bars:x.bars})),corrected_sections:null};return store[k]}
function refreshList(){const a=$("ann").value||"anon";$("song").innerHTML=SONGS.map((s,i)=>`<option value="${i}" class="${store[a+"|"+s.song_id]&&store[a+"|"+s.song_id].complete?"done":"todo"}">${i+1}. ${s.song_id} [${s.stratum}]${store[a+"|"+s.song_id]&&store[a+"|"+s.song_id].complete?" ✓":""}</option>`).join("");$("song").value=cur}
function render(){const s=SONGS[cur];const r=rec(s);$("png").src=s.png;$("midi").href=s.midi;
 $("meta").innerHTML=`<b>${s.song_id}</b> key ${s.key}, ${s.meter}, ${s.tempo} BPM; model plan: ${s.prediction_sections.map(x=>x[0]+" "+x[1]).join(" · ")}`;
 let h="<tr><th>#</th><th>pseudo label</th><th>bars (start bar / sec)</th><th>lyrics</th><th>label ok?</th><th>correct label</th><th>bars ok?</th><th>correct bars</th><th>boundary</th><th>shift (bars, + = later)</th><th>lyric-alignment errors</th><th>wrong lyric syllables</th><th>melody errors</th><th>chord errors</th></tr>";
 s.sections.forEach((x,i)=>{const a=r.sections[i];
  const sel=(n,opts)=>`<select data-i="${i}" data-n="${n}">`+["",...opts].map(o=>`<option ${a[n]==o?"selected":""}>${o}</option>`).join("")+"</select>";
  const num=n=>`<input type="number" data-i="${i}" data-n="${n}" value="${a[n]??""}" style="width:4em">`;
  h+=`<tr><td>${i+1}</td><td>${x.label}</td><td>${x.bars} (${x.start_bar} / ${x.start_sec}s)</td><td class="lines">${x.lines.join("<br>")}</td><td>${sel("label_correct",["yes","no"])}</td><td>${sel("correct_label",LABELS)}</td><td>${sel("bars_correct",["yes","no"])}</td><td>${num("correct_bars")}</td><td>${sel("boundary_ok",["yes","no","unsure"])}</td><td>${num("boundary_shift_bars")}</td><td>${num("lyric_alignment_errors")}</td><td>${num("wrong_lyric_syllables")}</td><td>${sel("melody_error",["0","1","2","3"])}</td><td>${sel("chord_error",["0","1","2","3"])}</td></tr>`});
 $("tbl").innerHTML=h;$("conf").value=r.confidence||"";$("keyok").value=r.key_correct||"";$("keyfix").value=r.correct_key||"";$("gridok").value=r.grid_correct||"";$("notes").value=r.notes||"";
 $("tbl").querySelectorAll("select,input").forEach(el=>el.onchange=()=>{const a=r.sections[+el.dataset.i];const v=el.value;
  a[el.dataset.n]=el.type==="number"?(v===""?null:+v):(v===""?null:v);r.updated=new Date().toISOString();r.complete=r.sections.every(z=>z.label_correct&&z.bars_correct);save()});
 $("status").textContent=r.complete?"complete":"incomplete"}
["conf","keyok","keyfix","gridok","notes"].forEach(id=>$(id).onchange=()=>{const r=rec(SONGS[cur]);const m={conf:"confidence",keyok:"key_correct",keyfix:"correct_key",gridok:"grid_correct",notes:"notes"}[id];
 r[m]=$(id).value===""?null:(id==="conf"?+$(id).value:$(id).value);save()});
$("song").onchange=()=>{cur=+$("song").value;render()};$("prev").onclick=()=>{cur=Math.max(0,cur-1);render();refreshList()};
$("next").onclick=()=>{cur=Math.min(SONGS.length-1,cur+1);render();refreshList()};$("ann").onchange=()=>{refreshList();render()};
$("export").onclick=()=>{const a=$("ann").value||"anon";const recs=Object.values(store).filter(r=>r.annotator_id===a);
 const blob=new Blob([JSON.stringify(recs,null,1)],{type:"application/json"});const l=document.createElement("a");
 l.href=URL.createObjectURL(blob);l.download=`annotations_${a}.json`;l.click()};
$("imp").onchange=e=>{const f=e.target.files[0];const rd=new FileReader();rd.onload=()=>{JSON.parse(rd.result).forEach(r=>store[r.annotator_id+"|"+r.song_id]=r);save();render()};rd.readAsText(f)};
refreshList();render();
</script></body></html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--no-materials", action="store_true")
    args = ap.parse_args()
    chosen, recs = select(args.n)
    write_candidates(chosen, recs)
    (PKG / "annotations").mkdir(parents=True, exist_ok=True)
    if not args.no_materials:
        write_materials(chosen)
    print(f"{len(chosen)} candidates -> {PKG / 'candidates.csv'}; strata {Counter(c['stratum'] for c in chosen)}")


if __name__ == "__main__":
    main()
