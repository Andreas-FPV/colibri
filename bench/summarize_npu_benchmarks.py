#!/usr/bin/env python3
"""Summarize bench/run_npu_benchmarks.ps1 results as a Markdown table.

    py -3 bench/summarize_npu_benchmarks.py [results-dir ...]

For every <label>.json in each directory: the number of measured runs, then the median and
the range (min-max) of TTFT, prompt tok/s and decode tok/s, and the peak working set.
Medians come from the raw per-run values; nothing is averaged or extrapolated.
"""
import json
import statistics
import sys
from pathlib import Path

ORDER = ["cpu", "vulkan", "cpu+npu", "cpu+npu-min2", "cpu+npu-min32", "vulkan+npu"]


def cell(values, fmt):
    med = statistics.median(values)
    lo, hi = min(values), max(values)
    return f"{med:{fmt}} ({lo:{fmt}}-{hi:{fmt}})" if lo != hi else f"{med:{fmt}}"


def summarize(directory):
    rows = []
    files = sorted(Path(directory).glob("*.json"),
                   key=lambda p: (ORDER.index(p.stem) if p.stem in ORDER else len(ORDER), p.stem))
    for path in files:
        data = json.loads(path.read_text(encoding="utf-8"))
        runs = data.get("runs") or []
        if not runs:
            continue
        rows.append("| {} | {} | {} | {} | {} | {:.2f} |".format(
            data.get("label", path.stem), len(runs),
            cell([r["ttft_s"] for r in runs], ".1f"),
            cell([r["prompt_tok_s"] for r in runs], ".2f"),
            cell([r["decode_tok_s"] for r in runs], ".2f"),
            max(r["peak_ws_gb"] for r in runs)))
    print(f"\n{directory}\n")
    print("| config | runs | TTFT s, median (range) | prompt tok/s | decode tok/s | peak RAM GB |")
    print("|---|---|---|---|---|---|")
    print("\n".join(rows))


def main(argv):
    dirs = argv or [str(Path(__file__).resolve().parent / "results")]
    for d in dirs:
        summarize(d)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
