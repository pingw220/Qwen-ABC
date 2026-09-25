#!/usr/bin/env python
"""Table 1 (dataset statistics) and Table 2 (main three-family comparison).

  python -m paper_eval.main_tables     # after paper_eval.decoding
"""

from __future__ import annotations

import json
import statistics as st
from collections import Counter

from .common import DATA_DIR, DATA_OUT, bootstrap_mean_ci, fmt_ci, jload, load_rows, paired_bootstrap, write_table
from .ood import load_train_stats
from .interventions import syllables_of

T2_METRICS = [("gen_success", "generation success"), ("strict_valid", "valid output (strict ABC / own validator‡)"),
              ("section_plan_exact", "exact structure"), ("section_count_match", "section count correct"),
              ("section_label_seq_match", "label sequence correct"), ("section_bars_exact_frac", "sections with exact label+bars"),
              ("abs_total_bar_error", "|total bars − request|"), ("early_eos", "early termination"),
              ("counter_self_consistent_frac", "countdown consistency"),
              ("lyric_recall", "lyric recall"), ("lyric_precision", "lyric precision"), ("lyric_exact", "lyrics exactly as requested"),
              ("section_lyric_recall", "section-local lyric recall"), ("cram_syllable_frac", "crammed syllables"),
              ("wordless_note_frac", "wordless notes"), ("pitch_range", "pitch range (semitones)"),
              ("notes_per_bar", "notes per bar"), ("chords_per_bar", "chords per bar"), ("chord_tone_frac", "chord-tone agreement"),
              ("melody_in_key_frac", "melody in declared key"), ("distinct_bar_frac", "distinct bars")]
STRUCTURAL = {"section_plan_exact", "section_count_match", "section_label_seq_match", "section_bars_exact_frac",
              "abs_total_bar_error", "early_eos"}


def table1():
    man = json.loads((DATA_DIR / "split_manifest.json").read_text())
    br = json.loads((DATA_DIR.parent / "abc_v1_20260915_012459/build_report.json").read_text())
    tr = load_train_stats()
    rows = []
    for split in ("train", "validation", "test"):
        if split == "train":
            it = __import__("paper_eval.common", fromlist=["iter_train_rows"]).iter_train_rows()
        else:
            it = load_rows(split)
        secs, bars, syl, tempo, minor, toks, n = [], [], [], [], 0, [], 0
        for r in it:
            s = r["spec"]
            n += 1
            secs.append(len(s["sections"]))
            bars.append(sum(x["bars"] for x in s["sections"]))
            syl.append(sum(len(syllables_of(x)) for x in s["sections"]))
            tempo.append(s["tempo_bpm"])
            minor += (s.get("key") or "").endswith("minor")
            toks.append(r.get("abc_tokens") or 0)
        q = lambda v: f"{st.median(v):.0f} [{sorted(v)[int(0.05 * (len(v) - 1))]:.0f}–{sorted(v)[int(0.95 * (len(v) - 1))]:.0f}]"
        rows.append({"split": split, "songs": n, "sections/song median [5–95%]": q(secs), "bars/song": q(bars),
                     "lyric syllables/song": q(syl), "tempo (BPM)": q(tempo), "minor-key share": f"{minor / n:.2f}",
                     "ABC-v2 completion tokens": q(toks)})
    write_table(rows, "table1_dataset", f"Pseudo-labelled corpus: {br['corpus_songs']:,} SheetSage-Pro songs; "
                f"inherited split {br['inherited_split_songs']}; held-out lyric duplicates of train removed "
                f"(test 355, validation 318); canonical split {({k: len(v) for k, v in man.items()})}")
    lab = Counter()
    for k, v in tr["label"].items():
        lab[k] = v
    write_table([{"label": k, "train sections": v, "share": f"{v / sum(lab.values()):.3f}"} for k, v in lab.most_common()],
                "table1b_section_labels", "Section labels in the training split")


def table2():
    d = jload(DATA_OUT / "decoding_per_song.json")
    systems = [("Qwen-ABC (E3b)", "E3b (paired-seed x4)"), ("MuPT", "MuPT (paired-seed x4)"), ("MIDI-LLM (mode A)", "MIDI-LLM (mode A x4)")]
    rows = []
    for mode, mlabel in (("single", "single sample (mean of 4)"), ("selector", "selector@4")):
        for met, label in T2_METRICS:
            row = {"decoding": mlabel, "metric": label}
            for disp, key in systems:
                v = d.get(key, {}).get(mode, {}).get(met, {})
                m, lo, hi, n = bootstrap_mean_ci(list(v.values()))
                cell = fmt_ci(m, lo, hi, 2 if met in ("pitch_range", "abs_total_bar_error", "notes_per_bar") else 3)
                if key.startswith("MIDI") and met in STRUCTURAL and m is not None:
                    cell += " †"
                row[disp] = cell
                row[f"N {disp.split()[0]}"] = n
            q, mu, ml = (d.get(k, {}).get(mode, {}).get(met, {}) for _, k in systems)
            if q and mu:
                p = paired_bootstrap(q, mu)
                row["Qwen − MuPT"] = fmt_ci(p["diff"], p["lo"], p["hi"], 3, True)
            if q and ml and met not in STRUCTURAL:
                p = paired_bootstrap(q, ml)
                row["Qwen − MIDI-LLM"] = fmt_ci(p["diff"], p["lo"], p["hi"], 3, True)
            rows.append(row)
    write_table(rows, "model_comparison", "Main comparison on the 225 test songs (T=1.0/top-p 0.95; MIDI-LLM its own sampler, mode A). "
                "Failures count as 0 in rates. † MIDI-LLM section count, bars and grid are inputs (true by construction). "
                "‡ MIDI-LLM validity is its own validator, not strict ABC. Paired song-level bootstrap.")


def main():
    table1()
    table2()
    print("MAIN_TABLES_DONE")


if __name__ == "__main__":
    main()
