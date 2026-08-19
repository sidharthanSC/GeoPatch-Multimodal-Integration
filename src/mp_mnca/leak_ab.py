"""Controlled A/B measuring the MP-MNCA Phase 1 label leak across DLPFC sections.

Phase 1 shipped with ``adata.obs["ground_truth"]`` as its contrastive pseudo-labels.
Those build the negative mask in ``staig/model.py``, so the loss never separated
spots sharing a cortical layer -- supervised contrastive learning on the evaluation
label.

The fix adopted from SM/geopatch drops the pseudo-label mask entirely: positives are
the same-spot diagonal plus spatial k-NN neighbours, and every other spot is a
negative (``model.contrastive_loss``). No labels of any kind are involved.

This runs both arms with *everything else identical* (seed, config, architecture,
epochs) so the difference isolates the leak.

    # one-time: cache small per-section arrays out of the 5 GB checkpoint
    python -m src.mp_mnca.leak_ab --prepare

    # then run any subset of sections (parallelize by launching several)
    python -m src.mp_mnca.leak_ab --sections 151507 151508 151509

Splitting preparation from execution matters on a 16 GB machine: loading
``dlpfc.pkl`` costs ~4 GB resident per process, so parallel workers would exhaust
memory. The cached arrays are ~50 MB per section instead.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from src.sparc_align.results_io import IncrementalCsv

from .config import MpMncaConfig
from .data import MpMncaSectionData

RUN_ROOT = Path("outputs/mp_mnca/leak_ab_all12_v1")

# (arm label, pseudo_label_source passed to fit_phase1)
#   "ground_truth" -> masked loss keyed on the cortical layer: the leak.
#   None           -> the production unmasked objective (model.contrastive_loss).
# NOTE: the committed results in this directory were produced when the unleaked arm
# used STAIG-style KMeans-on-image pseudo-labels. The unleaked arm is now the
# unmasked loss, so a re-run will not reproduce those exact numbers -- the leak's
# existence and rough magnitude carry over, the absolute values do not.
ARMS: tuple[tuple[str, str | None], ...] = (("ground_truth", "ground_truth"), ("unmasked", None))


def prepare_cache(checkpoint_path: Path, run_dir: Path, config: MpMncaConfig) -> list[str]:
    """Cache each section's prepared arrays so workers need not load the checkpoint."""
    from src.datasets.dlpfc import DlpfcDataset

    from .train_phase1 import prepare_section_phase1

    cache = run_dir / "prepared"
    cache.mkdir(parents=True, exist_ok=True)
    dataset = DlpfcDataset.from_checkpoint(checkpoint_path)
    section_ids = dataset.section_ids()
    for section_id in section_ids:
        target = cache / f"{section_id}.npz"
        if target.exists():
            print(f"{section_id}: cached", flush=True)
            continue
        data = prepare_section_phase1(dataset.get_section(section_id), section_id, config)
        np.savez_compressed(
            target,
            gene_expression=data.gene_expression,
            image_features=data.image_features,
            coordinates=data.coordinates,
            labels=data.labels,
            barcodes=data.barcodes,
            neighbor_indices=data.neighbor_indices,
        )
        print(f"{section_id}: cached {data.gene_expression.shape}", flush=True)
    return section_ids


def load_cached(run_dir: Path, section_id: str, n_neighbors: int = 6) -> MpMncaSectionData:
    """Rebuild a section from its cached arrays."""
    payload = np.load(run_dir / "prepared" / f"{section_id}.npz", allow_pickle=True)
    return MpMncaSectionData(
        section_id=section_id,
        gene_expression=payload["gene_expression"],
        image_features=payload["image_features"],
        coordinates=payload["coordinates"],
        labels=payload["labels"].astype(str),
        barcodes=payload["barcodes"].astype(str),
        neighbor_indices=payload["neighbor_indices"],
        gene_mask=None,
        n_neighbors=n_neighbors,
    )


def run_sections(
    section_ids: list[str], run_dir: Path, config: MpMncaConfig, device: str, tag: str
) -> list[dict]:
    """Run both arms on each section, flushing after every completed section."""
    from .train_phase1 import fit_phase1

    writer = IncrementalCsv(run_dir / f"leak_ab_{tag}.csv")
    rows: list[dict] = []
    for section_id in section_ids:
        data = load_cached(run_dir, section_id)
        row: dict = {"section_id": section_id, "epochs": config.epochs, "seed": config.seed}
        started = time.perf_counter()
        for arm, source in ARMS:
            result = fit_phase1(data, config, device=device, pseudo_label_source=source)
            row[f"{arm}_ari"] = result.metrics["refined_ari"]
            row[f"{arm}_nmi"] = result.metrics["refined_nmi"]
            row[f"{arm}_ari_unrefined"] = result.metrics["ari"]
            print(
                f"  {section_id} {arm:14s} refined_ARI={row[f'{arm}_ari']:.4f} "
                f"refined_NMI={row[f'{arm}_nmi']:.4f}",
                flush=True,
            )
        row["ari_leak"] = row["ground_truth_ari"] - row["unmasked_ari"]
        row["nmi_leak"] = row["ground_truth_nmi"] - row["unmasked_nmi"]
        row["elapsed_seconds"] = time.perf_counter() - started
        rows.append(row)
        writer.append(row)
        print(json.dumps(row, sort_keys=True, default=float), flush=True)
    return rows


def summarize(run_dir: Path) -> dict:
    """Merge every worker CSV and report the leak across sections.

    Workers write one CSV each so they never contend for a file; this stitches
    them back together into the single table the correction docs cite.
    """
    import csv

    merged = run_dir / "leak_ab_all_sections.csv"
    rows: list[dict] = []
    for path in sorted(run_dir.glob("leak_ab_*.csv")):
        # Skip our own output: it matches the same glob, and re-reading it would
        # double-count every section on the second invocation.
        if path == merged:
            continue
        with path.open(encoding="utf-8") as handle:
            rows.extend(dict(r) for r in csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"no worker CSVs under {run_dir}")

    rows.sort(key=lambda r: r["section_id"])

    # The committed 12-section results predate the switch to the unmasked objective
    # and name the unleaked arm "image_kmeans"; accept either so those CSVs -- the
    # evidence results/mp_mnca_label_leak_correction.md cites -- stay readable.
    clean_key = "unmasked_ari" if "unmasked_ari" in rows[0] else "image_kmeans_ari"
    clean_nmi_key = clean_key.replace("_ari", "_nmi")

    numeric = lambda key: np.array([float(r[key]) for r in rows])  # noqa: E731
    ari_leak, nmi_leak = numeric("ari_leak"), numeric("nmi_leak")
    leaked, clean = numeric("ground_truth_ari"), numeric(clean_key)

    summary = {
        "n_sections": len(rows),
        "sections": [r["section_id"] for r in rows],
        "ground_truth_ari": {"mean": float(leaked.mean()), "median": float(np.median(leaked))},
        clean_key: {"mean": float(clean.mean()), "median": float(np.median(clean))},
        "ari_leak": {
            "mean": float(ari_leak.mean()),
            "median": float(np.median(ari_leak)),
            "min": float(ari_leak.min()),
            "max": float(ari_leak.max()),
            "n_positive": int((ari_leak > 0).sum()),
        },
        "nmi_leak": {"mean": float(nmi_leak.mean()), "median": float(np.median(nmi_leak))},
    }

    with merged.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({k for r in rows for k in r}))
        writer.writeheader()
        writer.writerows(rows)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"{'section':>9} {'leaked ARI':>11} {'unsup ARI':>10} {'ARI leak':>9} {'NMI leak':>9}")
    print("-" * 54)
    for row in rows:
        print(
            f"{row['section_id']:>9} {float(row['ground_truth_ari']):>11.4f} "
            f"{float(row[clean_key]):>10.4f} {float(row['ari_leak']):>+9.4f} "
            f"{float(row['nmi_leak']):>+9.4f}"
        )
    print(
        f"\nmean ARI leak {summary['ari_leak']['mean']:+.4f}  "
        f"median {summary['ari_leak']['median']:+.4f}  "
        f"range [{summary['ari_leak']['min']:+.4f}, {summary['ari_leak']['max']:+.4f}]  "
        f"positive on {summary['ari_leak']['n_positive']}/{len(rows)} sections"
    )
    return summary


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=Path, default=Path("checkpoints/dlpfc.pkl"))
    parser.add_argument("--run-dir", type=Path, default=RUN_ROOT)
    parser.add_argument("--prepare", action="store_true", help="cache section arrays and exit")
    parser.add_argument("--summarize", action="store_true", help="merge worker CSVs and report")
    parser.add_argument("--sections", nargs="*", help="sections to run; default all cached")
    parser.add_argument("--tag", default="all", help="suffix for this worker's CSV")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = MpMncaConfig(
        epochs=args.epochs, seed=args.seed, batch_size=256, learning_rate=3e-4,
        mask_rate=0.1, num_heads=8, temperature=10.0, image_pseudo_clusters=40,
        image_pca_dim=16, refinement_neighbors=15,
    )
    args.run_dir.mkdir(parents=True, exist_ok=True)

    if args.summarize:
        summarize(args.run_dir)
        return

    if args.prepare:
        sections = prepare_cache(args.checkpoint_path, args.run_dir, config)
        (args.run_dir / "config.json").write_text(
            json.dumps({"config": config.to_dict(), "sections": sections, "arms": list(ARMS)}, indent=2),
            encoding="utf-8",
        )
        print(f"prepared {len(sections)} sections", flush=True)
        return

    sections = args.sections or sorted(
        p.stem for p in (args.run_dir / "prepared").glob("*.npz")
    )
    rows = run_sections(sections, args.run_dir, config, args.device, args.tag)
    if rows:
        leaks = [r["ari_leak"] for r in rows]
        print(
            f"\n{args.tag}: mean ARI leak {np.mean(leaks):+.4f} "
            f"median {np.median(leaks):+.4f} over {len(rows)} sections",
            flush=True,
        )


if __name__ == "__main__":
    main()
