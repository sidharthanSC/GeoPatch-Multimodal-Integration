"""Published DLPFC values transcribed from STAIG Supplementary Figures S1-S2."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


STAIG_PUBLISHED_DLPFC = {
    "151507": (0.64, 0.72),
    "151508": (0.63, 0.67),
    "151509": (0.71, 0.71),
    "151510": (0.69, 0.69),
    "151669": (0.68, 0.69),
    "151670": (0.72, 0.65),
    "151671": (0.83, 0.78),
    "151672": (0.84, 0.77),
    "151673": (0.68, 0.74),
    "151674": (0.60, 0.68),
    "151675": (0.65, 0.72),
    "151676": (0.63, 0.72),
}

SPAGCN_PUBLISHED_DLPFC = {
    "151507": (0.51, 0.65),
    "151508": (0.38, 0.50),
    "151509": (0.47, 0.65),
    "151510": (0.48, 0.63),
    "151669": (0.33, 0.47),
    "151670": (0.40, 0.53),
    "151671": (0.55, 0.68),
    "151672": (0.61, 0.73),
    "151673": (0.51, 0.69),
    "151674": (0.40, 0.58),
    "151675": (0.30, 0.48),
    "151676": (0.32, 0.52),
}


def compare_with_published(metrics_path: Path, output_dir: Path) -> dict[str, object]:
    """Compare one repository run against values printed in STAIG Figs. S1-S2."""
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        observed = {row["section_id"]: row for row in csv.DictReader(handle)}
    rows: list[dict[str, object]] = []
    for section_id, (published_ari, published_nmi) in STAIG_PUBLISHED_DLPFC.items():
        row = observed[section_id]
        observed_ari = float(row["refined_ari"])
        observed_nmi = float(row["refined_nmi"])
        spa_ari, spa_nmi = SPAGCN_PUBLISHED_DLPFC[section_id]
        rows.append(
            {
                "section_id": section_id,
                "adapted_staig_ari": observed_ari,
                "adapted_staig_nmi": observed_nmi,
                "published_staig_ari": published_ari,
                "published_staig_nmi": published_nmi,
                "published_spagcn_ari": spa_ari,
                "published_spagcn_nmi": spa_nmi,
                "ari_gap_to_staig": observed_ari - published_ari,
                "nmi_gap_to_staig": observed_nmi - published_nmi,
            }
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    with (output_dir / "section_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "source": "STAIG Supplementary Figures S1-S2; values transcribed at two-decimal figure precision",
        "adapted_staig": {
            "mean_ari": float(np.mean([row["adapted_staig_ari"] for row in rows])),
            "median_ari": float(np.median([row["adapted_staig_ari"] for row in rows])),
            "mean_nmi": float(np.mean([row["adapted_staig_nmi"] for row in rows])),
            "median_nmi": float(np.median([row["adapted_staig_nmi"] for row in rows])),
        },
        "published_staig": {
            "mean_ari": float(np.mean([row["published_staig_ari"] for row in rows])),
            "median_ari": float(np.median([row["published_staig_ari"] for row in rows])),
            "mean_nmi": float(np.mean([row["published_staig_nmi"] for row in rows])),
            "median_nmi": float(np.median([row["published_staig_nmi"] for row in rows])),
        },
        "published_spagcn": {
            "mean_ari": float(np.mean([row["published_spagcn_ari"] for row in rows])),
            "median_ari": float(np.median([row["published_spagcn_ari"] for row in rows])),
            "mean_nmi": float(np.mean([row["published_spagcn_nmi"] for row in rows])),
            "median_nmi": float(np.median([row["published_spagcn_nmi"] for row in rows])),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_with_published(args.metrics_path, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
