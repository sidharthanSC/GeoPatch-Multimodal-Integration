"""Markdown report for the SPARC-align convergence and batch-size studies.

    python -m src.sparc_align.report

Writes ``outputs/sparc_align/SPARC_CONVERGENCE_REPORT.md``: per-section refined ARI at
each epoch checkpoint, the mean/median curve, where each section peaks, the batch-size
sweep, and the standing comparison against the already-run baselines.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# Baselines are not re-run here: these are the collaborator's published 12-section
# figures from outputs/README.md, at each method's own best/native epoch budget.
BASELINES = [
    ("MP-MNCA Phase 1", "20", 0.5195, 0.5185, "SM/geopatch, `20260819_mp_mnca_phase1_orig_20ep_all12`"),
    ("SpaGCN", "200", 0.4969, 0.4856, "matched-epoch table, outputs/README.md"),
    ("GraphST", "200", 0.4765, 0.4895, "`graphst_seed41_all12_200ep` (native 600)"),
    ("STAIG", "200", 0.4730, 0.4850, "matched-epoch table (native 400)"),
    ("MuCoST", "200", 0.4555, 0.4461, "`mucost_seed2023_all12_200ep` (native 1000)"),
    ("STAIG (this repo, 400 ep)", "400", 0.5092, 0.5374, "`staig_baseline_seed0_all12_regen_v1`"),
]


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _table(rows, epochs, metric="ari"):
    sections = sorted({r["section_id"] for r in rows})
    lookup = {(r["section_id"], int(r["epoch"])): float(r[metric]) for r in rows}
    lines = ["| Section | " + " | ".join(f"{e} ep" for e in epochs) + " | best | @ epoch |",
             "|---|" + "---|" * (len(epochs) + 2)]
    for section in sections:
        vals = [lookup.get((section, e)) for e in epochs]
        present = [(e, v) for e, v in zip(epochs, vals) if v is not None]
        best_e, best_v = max(present, key=lambda p: p[1])
        cells = []
        for e, v in zip(epochs, vals):
            if v is None:
                cells.append("—")
            elif e == best_e:
                cells.append(f"**{v:.4f}**")
            else:
                cells.append(f"{v:.4f}")
        lines.append(f"| {section} | " + " | ".join(cells) + f" | **{best_v:.4f}** | {best_e} |")

    agg = []
    for e in epochs:
        vals = [lookup[(s, e)] for s in sections if (s, e) in lookup]
        agg.append((np.mean(vals), np.median(vals)) if vals else (np.nan, np.nan))
    lines.append("| **mean** | " + " | ".join(f"{m:.4f}" for m, _ in agg) + " | | |")
    lines.append("| **median** | " + " | ".join(f"{md:.4f}" for _, md in agg) + " | | |")
    return "\n".join(lines), agg, sections


def build_report(conv_run: str, batch_run: str | None, out_path: Path) -> Path:
    base = Path("outputs/sparc_align")
    conv = _read(base / conv_run / "convergence_metrics.csv")
    epochs = sorted({int(r["epoch"]) for r in conv})

    ari_table, ari_agg, sections = _table(conv, epochs, "ari")
    nmi_table, nmi_agg, _ = _table(conv, epochs, "nmi")
    means = [m for m, _ in ari_agg]
    best_i = int(np.argmax(means))
    best_epoch, best_mean = epochs[best_i], means[best_i]
    best_median = ari_agg[best_i][1]

    two = np.array([float(r["ari"]) for r in conv])
    s1 = np.array([float(r["stage1_support_ari"]) for r in conv])
    s1_wins = int((s1 > two).sum())
    two_mean, s1_mean = float(two.mean()), float(s1.mean())
    _bt, _b1 = defaultdict(list), defaultdict(list)
    for r in conv:
        _bt[r["section_id"]].append(float(r["ari"]))
        _b1[r["section_id"]].append(float(r["stage1_support_ari"]))
    two_vol = float(np.mean([np.std(v) for v in _bt.values()]))
    s1_vol = float(np.mean([np.std(v) for v in _b1.values()]))
    two_oracle = float(np.mean([max(v) for v in _bt.values()]))
    s1_oracle = float(np.mean([max(v) for v in _b1.values()]))

    per_section_best = defaultdict(list)
    for r in conv:
        per_section_best[r["section_id"]].append((float(r["ari"]), int(r["epoch"])))
    peak_epochs = [max(v)[1] for v in per_section_best.values()]

    parts = [
        "# SPARC-align: convergence, batch size, and standing vs the baselines",
        "",
        f"Twelve DLPFC sections. Stage 1 (the SPARC sparse autoencoder) is evaluated at "
        f"epochs {', '.join(map(str, epochs))}; **stage 2 (cross-attention + neighbour "
        f"contrastive) is fixed at 20 epochs** at every checkpoint.",
        "",
        "All checkpoints come from a **single 200-epoch run per section**, not five separate "
        "runs — separate runs would each reinitialize and would not be points on one "
        "trajectory, mixing convergence with seed noise.",
        "",
        "Evaluation is the shared protocol used by MP-MNCA and STAIG: PCA → tied GMM "
        "(sklearn's approximation of R `mclust` EEE) → 15-NN spatial refinement → ARI/NMI. "
        "Fully unsupervised: contrastive pseudo-labels come from KMeans on image PCA, never "
        "`ground_truth`.",
        "",
        "## Headline",
        "",
        f"| | |", "|---|---|",
        f"| Best mean refined ARI | **{best_mean:.4f}** at **{best_epoch} epochs** |",
        f"| Median at that epoch | {best_median:.4f} |",
        f"| Sections peaking at {best_epoch} ep | {peak_epochs.count(best_epoch)}/12 |",
        f"| Mean at {epochs[-1]} epochs | {means[-1]:.4f} |",
        f"| Spread across the whole epoch range | {max(means) - min(means):.4f} |",
        f"| Distinct peak epochs across sections | {len(set(peak_epochs))} of {len(epochs)} tested |",
        "",
        f"**The epoch budget barely matters.** The mean moves only "
        f"{max(means) - min(means):.4f} ARI across a {epochs[-1] // epochs[0]}x range in "
        f"training ({means[0]:.4f} at {epochs[0]} to {means[-1]:.4f} at {epochs[-1]}), against "
        f"a between-section spread of roughly "
        f"{max(max(v)[0] for v in per_section_best.values()) - min(max(v)[0] for v in per_section_best.values()):.2f}. "
        f"There is a knee around 50 epochs and essentially nothing after it.",
        "",
        f"Peak epochs are scattered across sections "
        f"({', '.join(f'{e} ep: {peak_epochs.count(e)}' for e in epochs if peak_epochs.count(e))}), "
        "so no section-independent optimum exists — \"train for N epochs\" is not a setting that "
        "transfers here. The nominal best is "
        f"{best_epoch} epochs, but it beats 50 epochs by only "
        f"{best_mean - means[epochs.index(50)]:+.4f}, which is well inside noise.",
        "",
        "## Refined ARI per section, by stage-1 epochs",
        "", ari_table, "",
        "## Refined NMI per section, by stage-1 epochs",
        "", nmi_table, "",
        "## Which phase to use",
        "",
        "SPARC-align has two trainable stages, so the operational question is whether the "
        "cross-attention stage earns its cost. Over all 60 (section, epoch) checkpoints:",
        "",
        "| | two-stage | stage 1 only (`support`) |",
        "|---|---|---|",
        f"| Mean refined ARI | **{two_mean:.4f}** | {s1_mean:.4f} |",
        f"| Within-section volatility across epochs | **{two_vol:.4f}** | {s1_vol:.4f} |",
        f"| Head-to-head wins | **{60 - s1_wins}/60** | {s1_wins}/60 |",
        f"| Oracle best-epoch mean | **{two_oracle:.4f}** | {s1_oracle:.4f} |",
        "",
        "**Use both stages.** The two-stage pipeline is ahead on mean, is about half as "
        "volatile across epochs, and stays ahead even under oracle per-section epoch "
        "selection. Stage 1 alone does beat it at individual checkpoints "
        f"({s1_wins} of 60), but those wins are not predictable in advance — the peak epoch "
        "differs per section — so they are not exploitable.",
        "",
        "Stage 2's main contribution here is **stabilization**: it roughly halves the "
        "epoch-to-epoch swing. Stage 1's sparse code is volatile (151671 alone swings "
        "0.17 -> 0.15 -> 0.42 -> 0.16 -> 0.46 across checkpoints), and the contrastive "
        "stage smooths that into a usable representation.",
        "",
        "## Standing vs the baselines",
        "",
        "Baselines were run by the collaborator and are **not** re-run here; figures are their "
        "published 12-section values at each method's own best/native budget.",
        "",
        "| Method | Epochs | Mean refined ARI | Median | Source |",
        "|---|---|---|---|---|",
    ]
    ours = ("**SPARC-align (ours)**", str(best_epoch), best_mean, best_median, f"`{conv_run}`")
    for name, ep, mean, median, src in sorted(
        BASELINES + [ours], key=lambda r: -r[2]
    ):
        parts.append(f"| {name} | {ep} | {mean:.4f} | {median:.4f} | {src} |")

    parts += [
        "",
        f"SPARC-align does not beat the baselines: {best_mean:.4f} against MP-MNCA's 0.5195 "
        "and STAIG's 0.4730–0.5092. The gap is structural rather than a tuning shortfall — "
        "cross-reconstruction NMSE from image to gene never falls below ~0.92 in any "
        "configuration, so morphology cannot predict expression in this tissue and the shared "
        "concept space ends up largely an image code, while cortical layer identity lives in "
        "the gene channel.",
        "",
    ]

    if batch_run and (base / batch_run / "batch_metrics.csv").exists():
        brows = _read(base / batch_run / "batch_metrics.csv")
        sizes = sorted({int(r["batch_size"]) for r in brows})
        by_size = defaultdict(list)
        for r in brows:
            by_size[int(r["batch_size"])].append(float(r["ari"]))
        bmeans = [np.mean(by_size[s]) for s in sizes]
        bbest = int(np.argmax(bmeans))
        parts += [
            "## Batch-size effect",
            "",
            f"At the best epoch budget ({brows[0]['epochs']} stage-1 epochs), all 12 sections.",
            "",
            "| Batch size | Mean refined ARI | Median | vs 256 |",
            "|---|---|---|---|",
        ]
        ref = np.mean(by_size[256]) if 256 in by_size else bmeans[0]
        for s, m in zip(sizes, bmeans):
            mark = " **(best)**" if s == sizes[bbest] else ""
            parts.append(
                f"| {s} | {m:.4f}{mark} | {np.median(by_size[s]):.4f} | {m - ref:+.4f} |"
            )
        spread = max(bmeans) - min(bmeans)

        # Sections are the biological replicates, so test paired across sections
        # rather than pooling runs.
        from scipy import stats as _stats

        per = defaultdict(dict)
        for r in brows:
            per[r["section_id"]][int(r["batch_size"])] = float(r["ari"])
        secs = sorted(per)
        small = np.array([per[s][sizes[bbest]] for s in secs])
        ref256 = np.array([per[s][256] for s in secs])
        wil = _stats.wilcoxon(small, ref256).pvalue
        tt = _stats.ttest_rel(small, ref256).pvalue
        wins = int((small - ref256 > 0).sum())
        per_best = Counter(max(per[s], key=per[s].get) for s in secs)
        within = float(np.mean([np.std(list(per[s].values())) for s in secs]))
        between = float(np.std([np.mean(list(per[s].values())) for s in secs]))

        parts += [
            "",
            f"**No statistically detectable batch-size effect.** Batch {sizes[bbest]} is "
            f"nominally best (+{bmeans[bbest] - np.mean(by_size[256]):.4f} over the paper's "
            f"256, winning {wins}/12 sections), but paired across the 12 sections that gives "
            f"Wilcoxon p={wil:.2f} and paired-t p={tt:.2f} — nowhere near significance at "
            "n=12. The curve is also non-monotonic (32 best, 128 worst, 256 second-best), "
            "which is the signature of noise rather than a trend.",
            "",
            f"Total spread across the whole 16x batch range is {spread:.4f} ARI. For scale, "
            f"the mean within-section spread across batch sizes is {within:.4f} against a "
            f"between-section spread of {between:.4f} — **which section you run on matters "
            f"about twice as much as which batch size you pick.**",
            "",
            "Per-section best batch size: "
            + ", ".join(f"{k}: {v}" for k, v in sorted(per_best.items()))
            + ". The lean toward 32 is worth noting as a weak signal — if anything it "
            "suggests smaller batches, i.e. more gradient steps per epoch, help slightly at "
            "this dataset scale — but it should not be reported as a result on this evidence.",
            "",
        ]

    spatial = base / "spatial_init_all12" / "spatial_init_metrics.csv"
    if spatial.exists():
        from scipy import stats as _st

        srows = _read(spatial)
        sby = defaultdict(dict)
        for r in srows:
            sby[r["section_id"]][r["arm"]] = float(r["ari"])
        ssecs = sorted(sby)
        arr = {a: np.array([sby[s][a] for s in ssecs]) for a in ("baseline", "before", "after")}
        parts += [
            "## Spatial k-NN smoothing: before vs after alignment",
            "",
            "SPARC uses the spatial graph only to choose *which* latents activate. This tests "
            "pushing it further: smoothing the gene and image streams over the k=6 graph "
            "**before** encoding (what STAIG/GraphST do implicitly via graph convolution), "
            "versus smoothing the latent code **after** SPARC but before stage 2. Morphology "
            f"kernel `softmax(beta*log s_ij)`, alpha=0.5, {len(ssecs)} sections at 50 epochs.",
            "",
            "| Arm | Mean refined ARI | Median | vs baseline | Wins | Wilcoxon p |",
            "|---|---|---|---|---|---|",
        ]
        for a in ("baseline", "before", "after"):
            if a == "baseline":
                parts.append(f"| baseline | {arr[a].mean():.4f} | {np.median(arr[a]):.4f} | — | — | — |")
            else:
                d = arr[a] - arr["baseline"]
                pv = _st.wilcoxon(arr[a], arr["baseline"]).pvalue
                parts.append(
                    f"| {a} | {arr[a].mean():.4f} | {np.median(arr[a]):.4f} | "
                    f"{d.mean():+.4f} | {int((d > 0).sum())}/{len(ssecs)} | {pv:.2f} |"
                )
        db = arr["before"] - arr["baseline"]
        parts += [
            "",
            f"**Before beats after** — roughly triple the mean gain "
            f"({db.mean():+.4f} vs {(arr['after'] - arr['baseline']).mean():+.4f}) and the "
            "better median. That fits the mechanism: `after` is largely redundant because "
            "stage 2's cross-attention already aggregates over the same graph, whereas "
            "`before` changes what SPARC actually encodes.",
            "",
            f"**But it is not a significant improvement.** Wilcoxon "
            f"p={_st.wilcoxon(arr['before'], arr['baseline']).pvalue:.2f} at n={len(ssecs)}, and the "
            f"per-section spread is severe: 151671 gains {sby['151671']['before'] - sby['151671']['baseline']:+.4f} "
            f"while 151508 loses {sby['151508']['before'] - sby['151508']['baseline']:+.4f}, with "
            f"{int((db < 0).sum())} of {len(ssecs)} sections getting worse. The mean gain rests on "
            "three large winners.",
            "",
            f"It also does not close the gap: {arr['before'].mean():.4f} against MP-MNCA's 0.5195 "
            "and STAIG's 0.4730-0.5092. Recommended as the better of the two placements and "
            "worth keeping, but not reportable as an improvement on this evidence.",
            "",
        ]

    parts += [
        "## Figures",
        "",
        f"- `{conv_run}/figures/convergence_per_section_ari.png` — one panel per section",
        f"- `{conv_run}/figures/convergence_mean_ari.png` — mean with IQR band",
        f"- `{conv_run}/figures/annotations/DLPFC_151507_epochs.png` — spatial domains vs ground truth",
    ]
    if batch_run:
        parts.append(f"- `{batch_run}/figures/batch_size_ari.png` — batch-size effect")
    parts += [
        "", "## Reproduce", "",
        "Both studies build a prepared-section cache on first use (~219 MB, git-ignored) so "
        "no study process pays the ~4 GB `dlpfc.pkl` load.",
        "", "```bash",
        "# Convergence: one 200-epoch run per section, scored at each checkpoint (~55 min)",
        "python -m src.sparc_align.convergence --study convergence \\",
        "    --name convergence_all12 --save-predictions 151507",
        "",
        "# Batch sweep, one size per process. Splitting this way keeps peak memory low",
        "# enough to survive a 16 GB machine; a single process running all five was",
        "# OOM-killed. Results resume automatically, so an eviction costs only the",
        "# in-flight size rather than the whole sweep.",
        "for bs in 32 64 128 256 512; do",
        "    python -m src.sparc_align.convergence --study batch \\",
        f"        --name batch_size_all12 --epochs {epochs[epochs.index(50)]} --batch-sizes $bs",
        "done",
        "",
        "# Figures and report",
        "python -m src.sparc_align.plots --study convergence --metric ari",
        "python -m src.sparc_align.plots --study convergence --metric nmi",
        "python -m src.sparc_align.plots --study batch --metric ari",
        "python -m src.sparc_align.annotation_plots --section-id 151507",
        "python -m src.sparc_align.report",
        "```", ""]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conv-run", default="convergence_all12")
    parser.add_argument("--batch-run", default="batch_size_all12")
    parser.add_argument("--out", default="outputs/sparc_align/SPARC_CONVERGENCE_REPORT.md")
    args = parser.parse_args()
    print(build_report(args.conv_run, args.batch_run, Path(args.out)))


if __name__ == "__main__":
    main()
