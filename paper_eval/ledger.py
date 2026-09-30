#!/usr/bin/env python
"""reports/paper_final/COMPUTE_LEDGER.csv from sacct, for every job listed in job_logs/SUBMISSIONS.txt.

  python -m paper_eval.ledger
"""

from __future__ import annotations

import csv
import re
import subprocess
from datetime import datetime

from .common import REPORT_DIR


def elapsed_s(s: str) -> int:
    d = 0
    if "-" in s:
        dd, s = s.split("-", 1)
        d = int(dd)
    parts = [int(x) for x in s.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return d * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--report-dir", default=str(REPORT_DIR))
    ap.add_argument("--submissions", default="job_logs/SUBMISSIONS.txt")
    a = ap.parse_args()
    from pathlib import Path
    rdir = Path(a.report_dir)
    subs = {}
    for line in (rdir / a.submissions).read_text().splitlines():
        m = re.match(r"(\d+)\s+(.*)", line.strip())
        if m:
            subs[m.group(1)] = m.group(2)
    rows = []
    fmt = "JobID,JobName%40,Partition,AllocTRES%80,Start,End,Elapsed,State"
    out = subprocess.run(["sacct", "-n", "-P", "-X", "-j", ",".join(subs), f"--format={fmt}"],
                         capture_output=True, text=True).stdout
    for line in out.splitlines():
        jid, name, part, tres, start, end, el, state = line.split("|")
        base = jid.split("_")[0]
        gpu = re.search(r"gres/gpu(?::(\w+))?=(\d+)", tres)
        n_gpu = int(gpu.group(2)) if gpu else 0
        gtype = (gpu.group(1) if gpu and gpu.group(1) else ("l40s" if "l40s" in part else "l40" if "l40" in part else "")) if gpu else "none"
        secs = elapsed_s(el) if el else 0
        desc = subs.get(base, "")
        # specific components first (e.g. "am2 csl_offc shard" is an AccoMontage2 job, "qwen chord gen mel:S1" a harmonizer job)
        model = next((m for m in ("qwen_e3b", "qwen_e1long", "qwen_e1", "qwen_e0", "mupt", "midi_llm", "midi-llm", "am2",
                                  "backing", "svs", "audio tools", "listening", "chord", "infill", "csl", "mel", "semantics")
                      if m in desc.lower() or m in name), "")
        model = {"audio tools": "audio_tools"}.get(model, model)
        rows.append({"job_id": jid, "job_name": name.strip(), "model": model or ("midi_llm" if "midillm" in name else ""),
                     "experiment": desc[:160], "partition": part, "gpu_type": gtype, "gpu_count": n_gpu,
                     "start": start, "end": end, "wall_time": el, "est_gpu_hours": round(secs * n_gpu / 3600, 3),
                     "state": state})
    rows.sort(key=lambda r: r["job_id"])
    cols = list(rows[0]) if rows else []
    with open(rdir / "COMPUTE_LEDGER.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    tot = sum(r["est_gpu_hours"] for r in rows)
    by = {}
    for r in rows:
        by.setdefault(r["gpu_type"], 0.0)
        by[r["gpu_type"]] += r["est_gpu_hours"]
    print(f"{len(rows)} job records, {tot:.1f} GPU-h total; by type {by}")


if __name__ == "__main__":
    main()
