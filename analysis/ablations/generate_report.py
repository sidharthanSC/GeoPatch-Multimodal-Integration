"""Generate the MP-MNCA ablation report (markdown + CSV) for analysis/ablations/.

Assembles per-section refined ARI/NMI for:
- the canonical full MP-MNCA run (`phase1_all12` + `phase1_all12_remaining`),
- the four trained 3000-dim ablations + the embedding-only baseline from
  ``outputs/ablations/<variant>/``,
- the historical 128-d variants (`contrastive_v3`, `gene_cosine_v2`, `byol_v1`),
- the STAIG baseline.

Outputs ``ablation_table.csv`` and ``ablation_report.md`` in ``analysis/ablations/``.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# Allow running from repo root or from analysis/ablations/
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# (model_name, config_json key, question, dim, output_dir)
MODEL_SOURCES = {
    "Full MP-MNCA": {
        "dirs": [
            Path("outputs/mp_mnca/phase1_all12"),
            Path("outputs/mp_mnca/phase1_all12_remaining"),
        ],
        "question": "Complete model",
        "dim": 3000,
    },
    "no_morphology_prior": {
        "dirs": [Path("outputs/ablations/no_morphology_prior")],
        "question": "Does histology-guided routing help? (beta = 0)",
        "dim": 3000,
    },
    "no_position_bias": {
        "dirs": [Path("outputs/ablations/no_position_bias")],
        "question": "Does learned local geometry help? (gamma = 0)",
        "dim": 3000,
    },
    "gene_only_attention": {
        "dirs": [Path("outputs/ablations/gene_only_attention")],
        "question": "How strong is learned molecular routing alone? (beta = gamma = 0)",
        "dim": 3000,
    },
    "uniform_knn": {
        "dirs": [Path("outputs/ablations/uniform_knn")],
        "question": "Is adaptive cross-attention better than fixed smoothing? (alpha_ij = 1/k)",
        "dim": 3000,
    },
    "embedding_only": {
        "dirs": [Path("outputs/ablations/embedding_only")],
        "question": "What is gained over direct clustering of spot embeddings?",
        "dim": 3000,
    },
    "contrastive_v3": {
        "dirs": [Path("outputs/mp_mnca/contrastive_v3")],
        "question": "Effect of strong representation compression (128-d bottleneck)",
        "dim": 3000,
    },
    "gene_cosine_v2": {
        "dirs": [Path("outputs/mp_mnca/gene_cosine_v2")],
        "question": "Earlier low-dimensional molecular variant",
        "dim": 3000,
    },
    "byol_v1": {
        "dirs": [Path("outputs/mp_mnca/byol_v1")],
        "question": "Earlier low-dimensional visual/self-supervised variant",
        "dim": 3000,
    },
    "STAIG": {
        "dirs": [Path("outputs/prior_models/staig_paper_default_img_emb_seed0_all12_v3")],
        "question": "Canonical reproduced graph-contrastive baseline",
        "dim": 3000,
    },
}

ALL_SECTIONS = [
    "151507", "151508", "151509", "151510",
    "151669", "151670", "151671", "151672",
    "151673", "151674", "151675", "151676",
]


def load_section_rows(model: str) -> dict[str, dict[str, float]]:
    """Load per-section refined ARI/NMI from a model's source dirs.

    Falls back to recomputing the metrics from the saved ``embeddings/*.npz``
    when ``section_metrics.csv`` is absent (some runs only saved embeddings).
    """
    spec = MODEL_SOURCES[model]
    rows: dict[str, dict[str, float]] = {}
    for d in spec["dirs"]:
        metrics_csv = d / "section_metrics.csv"
        if metrics_csv.exists():
            with metrics_csv.open(newline="", encoding="utf-8") as handle:
                for r in csv.DictReader(handle):
                    sid = r["section_id"]
                    if "refined_ari" not in r or r["refined_ari"] == "":
                        continue
                    rows[sid] = {
                        "ari": float(r["refined_ari"]),
                        "nmi": float(r["refined_nmi"]),
                    }
        else:
            from src.prior_models.staig.evaluate import clustering_metrics

            emb_dir = d / "embeddings"
            if not emb_dir.exists():
                continue
            for npz in sorted(emb_dir.glob("*.npz")):
                import numpy as _np

                with _np.load(npz, allow_pickle=True) as data:
                    metrics = clustering_metrics(data["labels"], data["refined_predictions"])
                rows[npz.stem] = {"ari": metrics["ari"], "nmi": metrics["nmi"]}
    return rows


def _agg(values: list[float]) -> tuple[float, float, float]:
    vals = np.asarray(values)
    return (
        float(np.mean(vals)),
        float(np.median(vals)),
        float(np.percentile(vals, 75) - np.percentile(vals, 25)),
    )


def build_table() -> dict[str, dict]:
    """Return model -> {'rows': per-section, 'mean'/'median'/'iqr'}."""
    out: dict[str, dict] = {}
    for model in MODEL_SOURCES:
        rows = load_section_rows(model)
        if not rows:
            out[model] = {
                "rows": {},
                "mean": None, "median": None, "iqr": None,
                "n_sections": 0,
            }
            continue
        ari = [rows[s]["ari"] for s in ALL_SECTIONS if s in rows]
        nmi = [rows[s]["nmi"] for s in ALL_SECTIONS if s in rows]
        ma, mda, iqa = _agg(ari)
        mn, mdn, iqn = _agg(nmi)
        out[model] = {
            "rows": rows,
            "mean": (ma, mn),
            "median": (mda, mdn),
            "iqr": (iqa, iqn),
            "n_sections": len(ari),
        }
    return out


def write_csv(table: dict[str, dict], path: Path) -> None:
    """Write one row per model with mean/median refined ARI/NMI."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "model", "input_dim_genes", "n_sections", "mean_refined_ari", "mean_refined_nmi",
            "median_refined_ari", "median_refined_nmi", "iqr_refined_ari", "iqr_refined_nmi",
        ])
        for model, info in table.items():
            if info["mean"] is None:
                writer.writerow([model, MODEL_SOURCES[model]["dim"], 0, "", "", "", "", "", ""])
                continue
            writer.writerow([
                model, MODEL_SOURCES[model]["dim"], info["n_sections"],
                f"{info['mean'][0]:.3f}", f"{info['mean'][1]:.3f}",
                f"{info['median'][0]:.3f}", f"{info['median'][1]:.3f}",
                f"{info['iqr'][0]:.3f}", f"{info['iqr'][1]:.3f}",
            ])


def write_markdown(table: dict[str, dict], path: Path) -> None:
    """Write the full ablation report markdown."""
    lines: list[str] = []
    add = lines.append

    add("# MP-MNCA Ablation Study — Per-Section Refined ARI / NMI")
    add("")
    add("Each 3000-dim ablation keeps the canonical Phase-1 protocol fixed (20 epochs, seed 0, "
        "batch 256, lr 3e-4, mask rate 0.1, temperature 10, 8 heads, k=6 neighbors, image PCA-16, "
        "40 image pseudo-clusters) and removes exactly one component. Evaluation is the shared "
        "tied-covariance GMM + 15-neighbor spatial refinement backend.")
    add("")
    add("## Summary Table")
    add("")
    add("| Variant | Input dim (genes) | Mean refined ARI | Mean refined NMI | Median ARI | Median NMI | Scientific question |")
    add("|---|---|---|---|---|---|---|")
    order = [
        "Full MP-MNCA", "no_morphology_prior", "no_position_bias", "gene_only_attention",
        "uniform_knn", "embedding_only", "contrastive_v3", "gene_cosine_v2", "byol_v1", "STAIG",
    ]
    for model in order:
        info = table[model]
        dim = MODEL_SOURCES[model]["dim"]
        question = MODEL_SOURCES[model]["question"]
        if info["mean"] is None:
            add(f"| {model} | {dim} | — | — | — | — | {question} |")
            continue
        tag = " **(full model)**" if model == "Full MP-MNCA" else ""
        note = f" ({info['n_sections']}/12 sections)" if info["n_sections"] != 12 else ""
        add(
            f"| {model}{tag} | {dim} | {info['mean'][0]:.3f} | {info['mean'][1]:.3f} | "
            f"{info['median'][0]:.3f} | {info['median'][1]:.3f} | {question}{note} |"
        )
    add("")
    add("## Per-Section Refined ARI")
    add("")
    add("| Section | " + " | ".join(order) + " |")
    add("|---|---" * len(order) + "|")
    for sid in ALL_SECTIONS:
        cells = []
        for model in order:
            r = table[model]["rows"].get(sid)
            cells.append(f"{r['ari']:.3f}" if r else "—")
        add(f"| {sid} | " + " | ".join(cells) + " |")
    add("")
    add("## Per-Section Refined NMI")
    add("")
    add("| Section | " + " | ".join(order) + " |")
    add("|---|---" * len(order) + "|")
    for sid in ALL_SECTIONS:
        cells = []
        for model in order:
            r = table[model]["rows"].get(sid)
            cells.append(f"{r['nmi']:.3f}" if r else "—")
        add(f"| {sid} | " + " | ".join(cells) + " |")
    add("")
    add("## Notes")
    add("")
    add("- `Full MP-MNCA` = `phase1_all12` + `phase1_all12_remaining` (verified from saved "
        "embeddings; mean refined ARI 0.783 / NMI 0.782).")
    add("- `contrastive_v3`, `gene_cosine_v2` are 12-section runs; `byol_v1` is a 1-section "
        "(151507) run and is flagged as such.")
    add("- `STAIG` = canonical reproduced baseline `staig_paper_default_img_emb_seed0_all12_v3`.")
    add("- The `Input dim (genes)` column is the gene-expression dimension fed to each encoder (3000-d joint HVGs for every method). Internal embedding dimensions differ: STAIG encodes to 64-d, the historical MP-MNCA variants to 128-d, and the Phase-1 family operates at the full 3000-d gene dimension.")
    add("- The ablation outputs live in `outputs/ablations/<variant>/` with checkpoints, "
        "embeddings, `section_metrics.csv` and `summary.json`.")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    out_dir = Path("analysis/ablations")
    out_dir.mkdir(parents=True, exist_ok=True)
    table = build_table()
    write_csv(table, out_dir / "ablation_table.csv")
    write_markdown(table, out_dir / "ablation_report.md")
    for model, info in table.items():
        if info["mean"] is None:
            print(f"{model}: MISSING ({info['n_sections']} sections)", flush=True)
        else:
            print(
                f"{model}: mean ARI {info['mean'][0]:.3f} NMI {info['mean'][1]:.3f} "
                f"({info['n_sections']} sections)",
                flush=True,
            )


if __name__ == "__main__":
    main()