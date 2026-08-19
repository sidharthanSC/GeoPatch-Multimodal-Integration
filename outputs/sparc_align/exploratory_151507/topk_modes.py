"""Validate the three new TopK selection modes against the paper baseline."""
from pathlib import Path
from dataclasses import replace
from src.datasets.dlpfc import DlpfcDataset
from src.sparc_align.config import SparcConfig
from src.sparc_align.data import prepare_section
from src.sparc_align.train import fit_sparc
from src.sparc_align.evaluate import cluster_sparc_latents

ds = DlpfcDataset.from_checkpoint(Path("checkpoints/dlpfc.pkl"))
adata = ds.get_section("151507")
print(f"{'mode':>14} {'selfG':>7} {'selfI':>7} {'i2g':>7} {'g2i':>7} "
      f"{'ARI:sum':>8} {'ARI:sup':>8} {'NMI:sup':>8} {'jacc':>6} {'alive':>6}")
for mode in ["global_sum", "quota", "rank_fusion", "partitioned"]:
    cfg = SparcConfig(n_latents=1024, k_active=128, epochs=50,
                      topk_mode=mode, spatial_topk_weight=0.75)
    d = prepare_section(adata, "151507", cfg)
    r = fit_sparc(d, cfg, device="mps", verbose=False)
    m = r.metrics
    sup = cluster_sparc_latents(r.latents, d.labels, d.coordinates,
                                replace(cfg, cluster_input="support"))["metrics"]
    print(f"{mode:>14} {m['self_nmse_gene']:>7.3f} {m['self_nmse_image']:>7.3f} "
          f"{m['cross_nmse_image_to_gene']:>7.3f} {m['cross_nmse_gene_to_image']:>7.3f} "
          f"{m['refined_ari']:>8.4f} {sup['refined_ari']:>8.4f} {sup['refined_nmi']:>8.4f} "
          f"{m['support_jaccard_gene_image']:>6.3f} {m['all_alive_fraction']:>6.3f}", flush=True)
