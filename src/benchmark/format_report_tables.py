"""Render ``outputs/benchmark/table_a/summary_{raw,refined}.csv`` and
``outputs/benchmark/attention_ablations/ablation_results.json`` as markdown tables for
``MODEL_BENCHMARK_RESULTS.md``.

Usage
-----
    python -m src.benchmark.format_report_tables
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def format_table_a(summary_csv: Path) -> str:
    df = pd.read_csv(summary_csv)
    df = df.sort_values(["backend", "representation"])
    lines = ["| Representation | Backend | ARI (median) | ARI IQR | NMI (median) | NMI IQR | Accuracy (median) |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for _, row in df.iterrows():
        lines.append(
            f"| {row['representation']} | {row['backend']} "
            f"| {row['raw_ari_median' if 'raw_ari_median' in row else 'refined_ari_median']:.3f} "
            f"| [{row['raw_ari_iqr_low' if 'raw_ari_iqr_low' in row else 'refined_ari_iqr_low']:.3f}, "
            f"{row['raw_ari_iqr_high' if 'raw_ari_iqr_high' in row else 'refined_ari_iqr_high']:.3f}] "
            f"| {row['raw_nmi_median' if 'raw_nmi_median' in row else 'refined_nmi_median']:.3f} "
            f"| [{row['raw_nmi_iqr_low' if 'raw_nmi_iqr_low' in row else 'refined_nmi_iqr_low']:.3f}, "
            f"{row['raw_nmi_iqr_high' if 'raw_nmi_iqr_high' in row else 'refined_nmi_iqr_high']:.3f}] "
            f"| {row['raw_accuracy_median' if 'raw_accuracy_median' in row else 'refined_accuracy_median']:.3f} |"
        )
    return "\n".join(lines)


def format_attention_ablations(results_json: Path) -> str:
    with results_json.open() as f:
        results = json.load(f)
    lines = ["| Ablation | Accuracy | Loss | Params | n_test |", "|---|---:|---:|---:|---:|"]
    for name, metrics in results.items():
        lines.append(
            f"| {name} | {metrics['accuracy']:.4f} | {metrics['loss']:.4f} "
            f"| {metrics.get('n_params', '-')} | {metrics.get('n_test', '-')} |"
        )
    return "\n".join(lines)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--table-a-dir", type=Path, default=Path("outputs/benchmark/table_a"))
    parser.add_argument("--attention-dir", type=Path, default=Path("outputs/benchmark/attention_ablations"))
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmark/rendered_tables.md"))
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    parts = ["## Table A -- Per-Section Domain Discovery (raw)\n", format_table_a(args.table_a_dir / "summary_raw.csv")]
    parts += ["\n\n## Table A -- Per-Section Domain Discovery (spatially refined)\n", format_table_a(args.table_a_dir / "summary_refined.csv")]

    ablation_path = args.attention_dir / "ablation_results.json"
    if ablation_path.exists():
        parts += ["\n\n## Attention Ablations A1-A12\n", format_attention_ablations(ablation_path)]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(parts) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
