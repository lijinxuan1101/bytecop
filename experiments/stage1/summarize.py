"""Merge Stage 1 evaluation results into a single Markdown table.

Reads ``eval_results.json`` from every subdirectory of ``--runs`` and writes a
Markdown report with:
    - A summary table (one row per experiment)
    - A detailed per-condition table (one row per experiment × condition)

Usage
-----
    python experiments/stage1/summarize.py \
        --runs runs/stage1 \
        --output experiments/stage1/stage1_results.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# Column order for the per-condition table.
_CONDITION_ORDER = [
    "clean",
    "jpeg_q90", "jpeg_q70", "jpeg_q50", "jpeg_q30",
    "blur_s0.5", "blur_s1.0", "blur_s2.0",
    "resize_0.5", "resize_0.25",
    "noise_s0.02", "noise_s0.05", "noise_s0.10",
    "color_jitter", "center_crop_80",
]


def _load_runs(runs_root: Path) -> list[dict]:
    entries: list[dict] = []
    for sub in sorted(runs_root.iterdir()):
        if not sub.is_dir():
            continue
        eval_path = sub / "eval_results.json"
        if not eval_path.exists():
            print(f"  skip {sub.name}: no eval_results.json")
            continue
        with open(eval_path) as f:
            data = json.load(f)
        entries.append({"name": sub.name, "data": data})
    return entries


def _fmt(x: float | None) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "-"


def _write_markdown(entries: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Stage 1 — Results Summary")
    lines.append("")
    lines.append(
        "Every experiment was trained on **CIFAKE (10,000 sampled)** and "
        "evaluated on the same held-out `test/` split using the official "
        "15-condition robustness matrix."
    )
    lines.append("")

    # ---- Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Experiment | AUC_clean | AUC_robust | Final Score | Worst Condition AUC |")
    lines.append("|---|---:|---:|---:|---:|")
    for e in entries:
        d = e["data"]
        lines.append(
            f"| `{e['name']}` "
            f"| {_fmt(d.get('auc_clean'))} "
            f"| {_fmt(d.get('auc_robust'))} "
            f"| {_fmt(d.get('final_score'))} "
            f"| {_fmt(d.get('worst_condition_auc'))} |"
        )
    lines.append("")

    # ---- Per-condition table
    lines.append("## Per-Condition AUC")
    lines.append("")
    header = ["Condition"] + [f"`{e['name']}`" for e in entries]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] + ["---:"] * len(entries)) + "|")

    all_conditions = set()
    for e in entries:
        all_conditions.update(e["data"].get("conditions", {}).keys())
    ordered = [c for c in _CONDITION_ORDER if c in all_conditions] + \
              sorted(c for c in all_conditions if c not in _CONDITION_ORDER)

    for cond in ordered:
        row = [cond]
        for e in entries:
            auc = e["data"].get("conditions", {}).get(cond, {}).get("auc")
            row.append(_fmt(auc))
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # ---- Config snapshot
    lines.append("## Config Snapshot")
    lines.append("")
    lines.append(
        "Each run's exact config is copied to `runs/stage1/<name>/config.yaml`."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize Stage 1 results into a Markdown table.")
    p.add_argument("--runs",   required=True, help="Runs root (contains one folder per experiment).")
    p.add_argument("--output", required=True, help="Output Markdown path.")
    args = p.parse_args()

    entries = _load_runs(Path(args.runs))
    if not entries:
        print("No eval_results.json found; nothing to summarize.")
        return
    _write_markdown(entries, Path(args.output))


if __name__ == "__main__":
    main()
