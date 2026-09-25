#!/usr/bin/env python
"""Paper figures (PDF + PNG preview), deterministic, from the JSON/parquet the analyses write.

  python -m paper_eval.figures [--only fig2,fig3,...]

Colors follow the entity, in a fixed categorical order (validated palette: blue, orange, aqua,
yellow, magenta, green, violet); every quantitative panel shows 95% song-level bootstrap CIs.
Figure 6 (edit -> re-render) is drawn by paper_eval/edit_rerender.py.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .common import DATA_OUT, FIG_DIR  # noqa: E402

COL = {"E3b": "#2a78d6", "Qwen-ABC (E3b)": "#2a78d6", "qwen_e3b": "#2a78d6",
       "MuPT": "#eb6834", "mupt": "#eb6834",
       "MIDI-LLM": "#1baf7a", "midi_llm": "#1baf7a",
       "E0": "#eda100", "qwen_e0": "#eda100", "Qwen E0 (no ESS)": "#eda100",
       "E1": "#e87ba4", "qwen_e1": "#e87ba4",
       "E1c": "#008300",
       "E1-long": "#4a3aa7", "qwen_e1long": "#4a3aa7"}
INK, MUTED, GRID = "#1f1f1e", "#6b6a64", "#e4e3dc"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "pdf.fonttype": 42, "svg.hashsalt": "paper-final"})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{name}.pdf", bbox_inches="tight", metadata={"CreationDate": None})
    fig.savefig(FIG_DIR / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def err(ax, x, m, lo, hi, color, **kw):
    ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], fmt="o", color=color, ms=5, lw=1.5, capsize=3, **kw)


# ------------------------------------------------------------------ fig 1: pipeline
def fig1():
    fig, ax = plt.subplots(figsize=(9.5, 1.9))
    ax.axis("off")
    boxes = ["Mandarin pop\nrecordings\n(11,673 songs)", "SheetSage-Pro\n(automatic\ntranscription)",
             "pseudo-labelled\nlead sheets\n(melody, chords,\nsections, lyrics)", "LM finetuning\nQwen-ABC / MuPT\n(MIDI-LLM baseline)",
             "editable ABC\nlead sheet\n(controls: plan, key,\ntempo, lyrics)", "renderer\nFastSinger +\nMuseControlLite"]
    xs = np.linspace(0.07, 0.93, len(boxes))
    for i, (x, t) in enumerate(zip(xs, boxes)):
        ax.text(x, 0.55, t, ha="center", va="center", fontsize=6.8, color=INK,
                bbox=dict(boxstyle="round,pad=0.4", fc="#f4f3ee" if i != 4 else "#cde2fb", ec=MUTED, lw=0.8))
        if i < len(boxes) - 1:
            ax.annotate("", xy=(xs[i + 1] - 0.068, 0.55), xytext=(x + 0.068, 0.55),
                        arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))
    ax.annotate("", xy=(xs[4], 0.12), xytext=(xs[5], 0.12),
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.8, connectionstyle="arc3,rad=-0.4"))
    ax.text((xs[4] + xs[5]) / 2, -0.26, "edit a control -> regenerate / re-render", ha="center", fontsize=6.8, color=MUTED)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.3, 1)
    save(fig, "fig1_pipeline")


# ------------------------------------------------------------------ fig 2: ESS ablation
def fig2():
    d = json.loads((DATA_OUT / "ess_long_range.json").read_text())
    dec = json.loads((DATA_OUT / "decoding_raw.json").read_text())
    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.5), gridspec_kw={"width_ratios": [5, 4, 3]})
    order = ["E0", "E1c", "E1", "E1-long", "E3b"]
    for ax, tag, title in ((axes[0], "T0.8", "training / representation, T=0.8\n(1 sample/song)"),
                           (axes[1], "T1.0", "training / representation, T=1.0\n(mean of 4 samples)")):
        names = [n for n in order if n in d[tag]]
        for i, n in enumerate(names):
            m, lo, hi, _ = d[tag][n]["metrics"]["section_plan_exact"]
            err(ax, i, m, lo, hi, COL[n])
            ax.text(i, lo - 0.035, f"{m:.3f}", ha="center", va="top", fontsize=7, color=INK)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, fontsize=8)
        ax.set_title(title, fontsize=8, color=INK)
        ax.set_ylim(0.3, 1.02)
    axes[0].set_ylabel("exact structure (labels + bars)")
    ax = axes[2]
    key = "E3b (legacy x4)"
    modes = [("single", "single"), ("selector", "selector@4"), ("oracle", "oracle@4")]
    for i, (k, lab) in enumerate(modes):
        v = dec.get(key, {}).get(k, {}).get("section_plan_exact")
        if v and v["mean"] is not None:
            err(ax, i, v["mean"], v["lo"], v["hi"], COL["E3b"])
            ax.text(i, v["lo"] - 0.035, f"{v['mean']:.3f}", ha="center", va="top", fontsize=7)
    ax.set_xticks(range(3))
    ax.set_xticklabels([m[1] for m in modes], fontsize=8)
    ax.set_title("E3b inference-time selection\n(separate from training)", fontsize=8)
    ax.set_ylim(0.3, 1.02)
    fig.tight_layout()
    save(fig, "fig2_ess_ablation")


# ------------------------------------------------------------------ fig 3: failure vs position
def fig3():
    d = json.loads((DATA_OUT / "ess_long_range.json").read_text())
    mets = [("fail", "section not exact"), ("lyric_omission", "lyric omission"), ("premature_eos", "premature EOS"),
            ("cram", "cramming")]
    fig, axes = plt.subplots(2, 4, figsize=(8.2, 4.0), sharex=True)
    xs = np.arange(5) * 20 + 10
    for r, tag in enumerate(("T0.8", "T1.0")):
        for c, (k, lab) in enumerate(mets):
            ax = axes[r, c]
            for n in ["E0", "E1c", "E1", "E1-long", "E3b"]:
                if n not in d[tag]:
                    continue
                b = d[tag][n]["buckets"][k]
                m = np.array([x[0] if x[0] is not None else np.nan for x in b], float)
                lo = np.array([x[1] if x[1] is not None else np.nan for x in b], float)
                hi = np.array([x[2] if x[2] is not None else np.nan for x in b], float)
                ax.plot(xs, m, "-o", color=COL[n], lw=2, ms=4, label=n)
                ax.fill_between(xs, lo, hi, color=COL[n], alpha=0.12, lw=0)
            ax.set_title(f"{lab} ({tag})", fontsize=8)
            ax.set_ylim(bottom=0)
            if r == 1:
                ax.set_xlabel("song position (%)")
    axes[0, 0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    save(fig, "fig3_failure_vs_position")


# ------------------------------------------------------------------ fig 4: intervention vs reseed
def fig4():
    s = json.loads((DATA_OUT / "interventions_summary.json").read_text())
    conds = ["bars_m4", "bars_m2", "bars_p2", "bars_p4", "label_bridge", "label_swap", "key_m3", "key_p2", "key_p5",
             "tempo_x0.8", "tempo_x1.25", "lyrics_sec", "lyrics_all"]
    panels = [("f_pc_js", "pitch-class distribution (JS)"), ("f_interval_js", "interval distribution (JS)"),
              ("d_melody", "melody sequence (1 − LCS)")]
    models = [m for m in ("qwen_e3b", "mupt", "midi_llm") if m in s]
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 3.6), sharey=True)
    for ax, (dk, lab) in zip(axes, panels):
        for j, model in enumerate(models):
            off = (j - (len(models) - 1) / 2) * 0.25
            for i, c in enumerate(conds):
                v = s[model].get(f"effect:{c}:whole:{dk}")
                if not v:
                    continue
                m, lo, hi = v[0], v[1], v[2]
                ax.errorbar(m, i + off, xerr=[[m - lo], [hi - m]], fmt="o", color=COL[model], ms=4, lw=1.2, capsize=2,
                            label=model if i == 0 or (c == conds[-1] and False) else None)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title(f"{lab}\ncross − within (> 0: beyond reseed)", fontsize=8)
    axes[0].set_yticks(range(len(conds)))
    axes[0].set_yticklabels(conds, fontsize=7.5)
    axes[0].invert_yaxis()
    h, l = [], []
    for model in models:
        h.append(plt.Line2D([], [], marker="o", color=COL[model], ls=""))
        l.append({"qwen_e3b": "Qwen-ABC (E3b)", "mupt": "MuPT", "midi_llm": "MIDI-LLM"}[model])
    fig.legend(h, l, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    save(fig, "fig4_intervention_vs_reseed")


# ------------------------------------------------------------------ fig 5: OOD
def fig5():
    import pandas as pd
    df = pd.read_parquet(DATA_OUT / "ood_samples.parquet")
    order = ["orig", "ood_chorus_first", "ood_end_on_verse", "ood_tempo", "ood_long20", "ood_long28",
             "ood_sections_p3", "ood_sections_p6"]
    labels = ["ID (own plan)", "chorus-first", "verse-final", "tempo 60/240", "section ≥20 bars", "section ≥28 bars",
              "+3 choruses", "+6 choruses"]
    models = [m for m in ("qwen_e3b", "qwen_e1", "qwen_e0", "mupt") if m in set(df.model)]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0), gridspec_kw={"width_ratios": [3, 2]})
    ax = axes[0]
    rng = np.random.default_rng(0)
    for j, model in enumerate(models):
        off = (j - (len(models) - 1) / 2) * 0.18
        for i, c in enumerate(order):
            v = df[(df.model == model) & (df.condition == c)]["own"].dropna().to_numpy()
            if not len(v):
                continue
            bs = rng.choice(v, (2000, len(v))).mean(1)
            m, lo, hi = v.mean(), *np.quantile(bs, [0.025, 0.975])
            err(ax, i + off, m, lo, hi, COL[model], label=None)
    ax.axvspan(0.5, 2.5, color="#f4f3ee", zorder=0)
    ax.text(1.5, 1.02, "compositional", ha="center", fontsize=7, color=MUTED)
    ax.text(5.0, 1.02, "extrapolative", ha="center", fontsize=7, color=MUTED)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7.5)
    ax.set_ylabel("condition adherence")
    ax.set_ylim(0, 1.08)
    ax = axes[1]
    for model in models:
        sub = df[(df.model == model) & df["own"].notna()]
        x = sub["s_n_sections_bits"].to_numpy()
        y = sub["own"].to_numpy()
        bins = np.quantile(x, np.linspace(0, 1, 7))
        idx = np.clip(np.digitize(x, bins[1:-1]), 0, 5)
        xm = [x[idx == k].mean() for k in range(6) if (idx == k).any()]
        ym = [y[idx == k].mean() for k in range(6) if (idx == k).any()]
        ax.plot(xm, ym, "-o", color=COL[model], lw=2, ms=4)
    ax.set_xlabel("section-count surprisal under the corpus (bits)")
    ax.set_title("adherence vs distance from training plans", fontsize=8)
    ax.set_ylim(0, 1.05)
    h = [plt.Line2D([], [], marker="o", color=COL[m], ls="") for m in models]
    fig.legend(h, [{"qwen_e3b": "Qwen-ABC (E3b)", "qwen_e1": "Qwen E1 (ESS, no recon.)", "qwen_e0": "Qwen E0 (no ESS)",
                    "mupt": "MuPT"}[m] for m in models], loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    save(fig, "fig5_ood_adherence")


FIGS = {"fig1": fig1, "fig2": fig2, "fig3": fig3, "fig4": fig4, "fig5": fig5}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    for k, f in FIGS.items():
        if a.only and k not in a.only.split(","):
            continue
        try:
            f()
        except FileNotFoundError as e:
            print(f"skip {k}: {e}")


if __name__ == "__main__":
    main()
