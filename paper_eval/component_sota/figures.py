#!/usr/bin/env python
"""Figures 1-8 of the component round (PDF + PNG), deterministic, from reports/component_sota/data/.

  python -m paper_eval.component_sota.figures [--only fig2,fig3]

Colors follow the entity in a fixed categorical order (validated palette): Qwen Full/E3b blue,
Qwen Melody-Only aqua, CSL-L2M orange, pseudo-reference grey, AccoMontage2 magenta, FastSinger
violet, SoulX yellow. Every quantitative panel shows 95% song-level bootstrap CIs.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from ..common import bootstrap_mean_ci  # noqa: E402
from .melody_eval import CS_REPORT  # noqa: E402

FIG = CS_REPORT / "figures"
DATA = CS_REPORT / "data"
COL = {"e3b": "#2a78d6", "mel": "#1baf7a", "csl_offc": "#eb6834", "csl_off": "#f3a47f", "csl_rtc": "#a8481d", "csl_rt": "#d9936c",
       "ref": "#8a8984", "qwen": "#2a78d6", "qwen_lyr": "#6da7ec", "am2": "#e87ba4", "diatonic": "#b8b7b0", "joint": "#2a78d6",
       "fastsinger": "#4a3aa7", "soulx": "#eda100", "infill": "#1baf7a", "regenerate": "#eb6834"}
NAME = {"e3b": "Qwen Full → melody", "mel": "Qwen Melody-Only", "csl_offc": "CSL-L2M official (chunked)", "csl_off": "CSL-L2M official (whole)",
        "csl_rtc": "CSL-L2M retrained (chunked)", "csl_rt": "CSL-L2M retrained (whole)", "ref": "pseudo-reference",
        "qwen": "Qwen Melody→Chord", "qwen_lyr": "Qwen Melody+Lyrics→Chord", "am2": "AccoMontage2", "diatonic": "diatonic baseline",
        "fastsinger": "FastSinger", "soulx": "SoulX-Singer"}
INK, MUTED = "#1f1f1e", "#6b6a64"
plt.rcParams.update({"font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e4e3dc", "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
                     "axes.edgecolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "pdf.fonttype": 42})


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIG / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def ci_song(df, col, by="song_id"):
    v = df.groupby(by)[col].mean().dropna().tolist()
    return bootstrap_mean_ci(v)


def dots(ax, labels, stats, colors, title, fmt="{:.2f}"):
    for i, ((m, lo, hi, n), c) in enumerate(zip(stats, colors)):
        if m is None:
            continue
        ax.errorbar(m, i, xerr=[[m - lo], [hi - m]], fmt="o", color=c, ms=5, lw=1.5, capsize=2.5)
        ax.text(hi, i, "  " + fmt.format(m), va="center", fontsize=7, color=INK)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_title(title, fontsize=8.5)


def fig1():
    fig, ax = plt.subplots(figsize=(9.6, 3.0))
    ax.axis("off")
    B = lambda x, y, t, fc="#f4f3ee": ax.text(x, y, t, ha="center", va="center", fontsize=7, color=INK,
                                             bbox=dict(boxstyle="round,pad=0.35", fc=fc, ec=MUTED, lw=0.8))
    A = lambda x0, y0, x1, y1, c=MUTED: ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="->", color=c, lw=1))
    B(0.06, 0.5, "lyrics +\ncontrols\n(plan, key,\ntempo)")
    B(0.30, 0.85, "Qwen Full (E3b)\njoint melody + chords", "#cde2fb")
    B(0.27, 0.45, "melody model\nQwen Melody-Only |\nCSL-L2M")
    B(0.47, 0.45, "harmonizer\nQwen Melody→Chord |\nAccoMontage2")
    B(0.64, 0.65, "lead sheet\n(melody, lyrics,\nchords, sections)", "#cde2fb")
    B(0.84, 0.88, "SVS-A FastSinger\nSVS-B SoulX-Singer")
    B(0.84, 0.45, "backing renderer\nMuseControlLite\n(MIDI-SAG adapter)")
    B(0.84, 0.10, "detectors: ASR (PER),\nRMVPE F0, BTC chords,\nBeatNet, key, SongEval")
    B(0.64, 0.12, "section infill\n(local edit)", "#e7f6ef")
    for a in ((0.11, 0.55, 0.22, 0.83), (0.11, 0.48, 0.20, 0.45), (0.34, 0.45, 0.40, 0.45), (0.54, 0.47, 0.60, 0.60),
              (0.38, 0.83, 0.59, 0.70), (0.69, 0.70, 0.78, 0.86), (0.69, 0.62, 0.78, 0.48), (0.84, 0.78, 0.84, 0.20),
              (0.84, 0.36, 0.84, 0.20), (0.64, 0.56, 0.64, 0.20)):
        A(*a)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    save(fig, "fig1_decomposition")


def fig2():
    import pandas as pd
    df = pd.read_parquet(DATA / "melody_samples_orig.parquet")
    srcs = [s for s in ("ref", "e3b", "mel", "csl_offc", "csl_off", "csl_rtc", "csl_rt") if s in set(df.source)]
    panels = [("PD", "PD (pitch dist. overlap) ↑"), ("DD", "DD (duration dist. overlap) ↑"), ("MD", "MD (melody distance) ↓"),
              ("lyric_recall", "lyric recall ↑"), ("pitch_range", "pitch range (semitones)"), ("notes_per_bar", "notes per bar")]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 4.6), sharey=True)
    for ax, (c, t) in zip(axes.flat, panels):
        dots(ax, [NAME[s] for s in srcs], [ci_song(df[df.source == s], c) if c in df else (None,) * 4 for s in srcs],
             [COL[s] for s in srcs], t)
    fig.tight_layout()
    save(fig, "fig2_lyrics_to_melody")


def fig3():
    import pandas as pd
    df = pd.read_parquet(DATA / "lyric_conditioning.parquet")
    srcs = [s for s in ("e3b", "mel", "csl_offc", "csl_rtc", "csl_off", "csl_rt") if s in set(df.source)]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8), sharey=True)
    for ax, (d, t) in zip(axes, (("d_melody", "melody (1−LCS)"), ("d_contour", "contour (intervals)"), ("f_pc_js", "pitch-class JS"))):
        stats = []
        for s in srcs:
            g = df[df.source == s]
            stats.append(bootstrap_mean_ci((g[f"cross_{d}"] - g[f"within_{d}"]).dropna().tolist()))
        dots(ax, [NAME[s] for s in srcs], stats, [COL[s] for s in srcs], f"{t}\nlyric swap − reseed (> 0: lyrics move melody)", "{:+.3f}")
        ax.axvline(0, color=MUTED, lw=1)
    fig.tight_layout()
    save(fig, "fig3_lyric_vs_reseed")


def fig4():
    import pandas as pd
    df = pd.read_parquet(DATA / "infill.parquet")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.0))
    kinds = ["edit_resample", "edit_lyrics", "edit_extend"]
    lab = {"edit_resample": "regenerate one section", "edit_lyrics": "replace section lyrics", "edit_extend": "extend section +4 bars"}
    for ax, (c, t) in zip(axes, (("target_struct_ok", "target section as requested ↑"), ("nontarget_notes_identical", "untouched sections: notes identical ↑"))):
        for j, meth in enumerate(("infill", "regenerate")):
            for i, k in enumerate(kinds):
                g = df[(df.method == meth) & (df.kind == k)]
                if c not in g or not g[c].notna().any():
                    continue
                m, lo, hi, n = ci_song(g, c)
                y = i + (j - 0.5) * 0.3
                ax.errorbar(m, y, xerr=[[m - lo], [hi - m]], fmt="o", color=COL[meth], ms=5, capsize=2.5, label=meth if i == 0 else None)
        ax.set_yticks(range(3))
        ax.set_yticklabels([lab[k] for k in kinds])
        ax.invert_yaxis()
        ax.set_title(t, fontsize=8.5)
        ax.set_xlim(-0.02, 1.02)
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.tight_layout()
    save(fig, "fig4_infill_locality")


def fig5():
    import pandas as pd
    df = pd.read_parquet(DATA / "chords.parquet")
    d = df[df.melody == "ref"]
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    for cs in [c for c in ("qwen", "qwen_lyr", "am2", "diatonic", "ref") if c in set(d.chords)]:
        g = d[d.chords == cs]
        y = ci_song(g, "strong_chord_tone")
        x = ci_song(g, "ref_chord_f1") if cs != "ref" else (1.0, 1.0, 1.0, 0)
        if x[0] is None or y[0] is None:
            continue
        ax.errorbar(x[0], y[0], xerr=[[x[0] - x[1]], [x[2] - x[0]]], yerr=[[y[0] - y[1]], [y[2] - y[0]]], fmt="o", color=COL[cs], ms=6, capsize=2)
        ax.annotate(NAME[cs], (x[0], y[0]), textcoords="offset points", xytext=(6, 4), fontsize=7.5, color=INK)
    ax.set_xlabel("reference similarity: chord chroma F1 vs pseudo-reference")
    ax.set_ylabel("compatibility: strong-beat chord-tone ratio")
    ax.set_title("Harmonizing the same (reference) melodies", fontsize=8.5)
    save(fig, "fig5_melody_to_chord")


def fig6():
    import pandas as pd
    mel = pd.read_parquet(DATA / "melody_samples_orig.parquet")
    ch = pd.read_parquet(DATA / "chords.parquet")
    systems = [("Qwen Joint (E3b)", "e3b", "joint"), ("Qwen Mel → Qwen Chord", "mel", "qwen"), ("Qwen Mel → AccoMontage2", "mel", "am2"),
               ("CSL-L2M → Qwen Chord", "csl_offc", "qwen"), ("CSL-L2M → AccoMontage2", "csl_offc", "am2")]
    panels = [("mel", "lyric_recall", "lyric recall ↑"), ("mel", "MD", "melody distance to ref ↓"),
              ("ch", "strong_chord_tone", "strong-beat chord-tone ↑"), ("ch", "chroma_compat", "chroma compatibility ↑")]
    fig, axes = plt.subplots(1, 4, figsize=(10.5, 2.6), sharey=True)
    for ax, (src, c, t) in zip(axes, panels):
        stats, cols = [], []
        for name, ms, cs in systems:
            g = mel[(mel.source == ms) & (mel.seed == "S1")] if src == "mel" else ch[(ch.melody == ms) & (ch.chords == cs) & (ch.chord_seed == "S1")]
            stats.append(ci_song(g, c) if (len(g) and c in g) else (None,) * 4)
            cols.append(COL["e3b"] if ms == "e3b" else COL[ms] if cs == "qwen" else COL["am2"])
        dots(ax, [s[0] for s in systems], stats, cols, t)
    fig.tight_layout()
    save(fig, "fig6_joint_vs_cascade")


def fig7():
    import pandas as pd
    df = pd.read_parquet(DATA / "svs.parquet")
    srcs = [s for s in ("ref", "e3b", "mel", "csl_offc", "csl_rtc") if s in set(df.source)]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 2.8))
    for ax, (c, t) in zip(axes, (("PER", "PER ↓"), ("note_pitch_acc_octave_folded", "note pitch accuracy ↑"), ("voicing_f1", "voicing F1 vs score ↑"))):
        for r, off in (("fastsinger", -0.15), ("soulx", 0.15)):
            xs, ys, lo, hi = [], [], [], []
            for i, s in enumerate(srcs):
                g = df[(df.source == s) & (df.renderer == r)]
                if c not in g or not g[c].notna().any():
                    continue
                m, l, h, n = bootstrap_mean_ci(g[c].dropna().tolist())
                xs.append(i + off); ys.append(m); lo.append(m - l); hi.append(h - m)
            ax.errorbar(xs, ys, yerr=[lo, hi], fmt="o-", color=COL[r], ms=5, capsize=2, lw=1.2, label=NAME[r])
        ax.set_xticks(range(len(srcs)))
        ax.set_xticklabels([NAME[s] for s in srcs], rotation=30, ha="right", fontsize=7)
        ax.set_title(t, fontsize=8.5)
    axes[0].legend(fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig7_svs_cross_render")


def fig8():
    import pandas as pd
    df = pd.read_parquet(DATA / "backing.parquet")
    ctl = df[df.group == "control"]
    conds = [c for c in ("key_gen", "key_transpose", "tempo_gen", "tempo_direct") if c in set(ctl.condition)]
    lab = {"key_gen": "key +5: regenerate", "key_transpose": "key +5: transpose score", "tempo_gen": "tempo ×1.25: regenerate",
           "tempo_direct": "tempo ×1.25: edit Q:"}
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    for j, (c, name, color) in enumerate((("symbolic_success", "in the score", "#2a78d6"), ("audio_success", "detected in rendered audio", "#eb6834"))):
        for i, cond in enumerate(conds):
            g = ctl[ctl.condition == cond]
            if c not in g or not g[c].notna().any():
                continue
            m, lo, hi, n = bootstrap_mean_ci(g[c].dropna().tolist())
            ax.errorbar(m, i + (j - 0.5) * 0.3, xerr=[[m - lo], [hi - m]], fmt="o", color=color, ms=5, capsize=2.5, label=name if i == 0 else None)
    ax.set_yticks(range(len(conds)))
    ax.set_yticklabels([lab[c] for c in conds])
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1.05)
    ax.set_xlabel("requested value realized")
    ax.legend(fontsize=7.5, loc="lower left")
    save(fig, "fig8_symbolic_vs_audio_control")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5, "fig6": fig6, "fig7": fig7, "fig8": fig8}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    for k, f in FIGS.items():
        if a.only and k not in a.only.split(","):
            continue
        try:
            f()
        except (FileNotFoundError, KeyError) as e:
            print(f"skip {k}: {e}")


if __name__ == "__main__":
    main()
