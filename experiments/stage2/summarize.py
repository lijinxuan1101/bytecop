"""Merge Stage 2 evaluation results into a single Markdown table.

Usage
-----
    python experiments/stage2/summarize.py \
        --runs runs/stage2 \
        --output experiments/stage2/stage2_results.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


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
        agg_path = sub / "aggregation_stats.json"
        agg = None
        if agg_path.exists():
            with open(agg_path) as f:
                agg = json.load(f)
        entries.append({"name": sub.name, "data": data, "agg": agg})
    return entries


def _fmt(x: float | None) -> str:
    return f"{x:.4f}" if isinstance(x, (int, float)) else "-"


def _write_markdown(entries: list[dict], out_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Stage 2 — Results Summary")
    lines.append("")
    lines.append(
        "RGPA is the Stage 2 forensic branch. CIFAKE numbers are pipeline "
        "checks only. Formal evaluation uses SID-Set."
    )
    lines.append("")
    lines.append(
        "Exit: send RGPA into Stage 3 if it has independent value and the "
        "worst degradation condition does not clearly collapse. Otherwise drop "
        "the forensic branch and keep OpenCLIP-H."
    )
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append(
        "| Experiment | AUC_clean | AUC_robust | Final Score | "
        "Worst Condition AUC | L1(high,low) |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|")
    for e in entries:
        d = e["data"]
        l1 = e["agg"].get("mean_l1_high_low") if e["agg"] else None
        lines.append(
            f"| `{e['name']}` "
            f"| {_fmt(d.get('auc_clean'))} "
            f"| {_fmt(d.get('auc_robust'))} "
            f"| {_fmt(d.get('final_score'))} "
            f"| {_fmt(d.get('worst_condition_auc'))} "
            f"| {_fmt(l1)} |"
        )
    lines.append("")

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

    lines.append("## Config Snapshot")
    lines.append("")
    lines.append(
        "Each run's exact config is copied to `runs/stage2/<name>/config.yaml`. "
        "Per-sample forensic logits are in `val_predictions.json`."
    )
    lines.append("")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Summarize Stage 2 results.")
    p.add_argument("--runs", required=True, help="Runs root.")
    p.add_argument("--output", required=True, help="Output Markdown path.")
    args = p.parse_args()

    entries = _load_runs(Path(args.runs))
    if not entries:
        print("No eval_results.json found; nothing to summarize.")
        return
    _write_markdown(entries, Path(args.output))


if __name__ == "__main__":
    main()
