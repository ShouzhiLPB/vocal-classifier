"""Summarise results/runs.jsonl: mean +- std over folds, and the accuracy-vs-length curve.

    python report.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

runs = [json.loads(line) for line in Path("results/runs.jsonl").read_text().splitlines() if line.strip()]
groups = defaultdict(list)
for r in runs:
    groups[(r["model"], r["seg"])].append(r)

rows = ["| model | segment (s) | folds | accuracy | macro-F1 | CPU latency model / feat+model (ms) | params | MACs |",
        "|---|---|---|---|---|---|---|---|"]
curves = defaultdict(list)
for (model, seg), rs in sorted(groups.items()):
    acc = np.array([r["accuracy"] for r in rs])
    f1 = np.array([r["macro_f1"] for r in rs])
    lat = (f"{rs[0]['cpu_model_ms']['median']:.2f} / {rs[0]['cpu_feature_plus_model_ms']['median']:.2f}"
           if "cpu_model_ms" in rs[0] else "-")
    rows.append(f"| {model} | {seg} | {len(rs)} | {acc.mean():.4f} ± {acc.std():.4f} | "
                f"{f1.mean():.4f} ± {f1.std():.4f} | {lat} | {rs[0].get('params', '-')} | {rs[0].get('macs', '-')} |")
    curves[model].append((seg, acc.mean(), acc.std()))

table = "\n".join(rows)
Path("results/summary.md").write_text(table + "\n")
print(table)

fig, ax = plt.subplots(figsize=(6, 4))
for model, pts in curves.items():
    s, m, e = map(np.array, zip(*sorted(pts)))
    ax.errorbar(s, m, yerr=e, marker="o", capsize=3, label=model)
ax.set(xscale="log", xlabel="segment length (s) = minimum decision delay",
       ylabel="test accuracy (mean ± std over folds)", title="Speech vs singing, unseen singers")
ax.set_xticks(sorted({p[0] for v in curves.values() for p in v}))
ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig("results/accuracy_vs_length.png", dpi=150)
print("saved results/summary.md and results/accuracy_vs_length.png")
