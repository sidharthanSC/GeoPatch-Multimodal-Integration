"""Ablation: spatially-aggregated Global TopK weight. w=0.0 is the paper exactly."""
from pathlib import Path
from dataclasses import replace
import numpy as np
from src.datasets.dlpfc import DlpfcDataset
from src.sparc_align.config import SparcConfig
from src.sparc_align.data import prepare_section
from src.sparc_align.train import fit_sparc
from src.sparc_align.evaluate import cluster_sparc_latents

ds = DlpfcDataset.from_checkpoint(Path("checkpoints/dlpfc.pkl"))
adata = ds.get_section("151507")
print(f"{'w':>5} {'selfG':>7} {'selfI':>7} {'i2g':>7} {'ARI:sum':>9} {'ARI:supp':>9} {'NMI:sum':>9}")
for w in [0.0, 0.25, 0.5, 0.75, 0.9]:
    cfg = SparcConfig(n_latents=1024, k_active=128, epochs=50, spatial_topk_weight=w)
    data = prepare_section(adata, "151507", cfg)
    r = fit_sparc(data, cfg, device="mps", verbose=False)
    m = r.metrics
    sup = cluster_sparc_latents(r.latents, data.labels, data.coordinates,
                                replace(cfg, cluster_input="support"))["metrics"]
    print(f"{w:>5.2f} {m['self_nmse_gene']:>7.3f} {m['self_nmse_image']:>7.3f} "
          f"{m['cross_nmse_image_to_gene']:>7.3f} {m['refined_ari']:>9.4f} "
          f"{sup['refined_ari']:>9.4f} {m['refined_nmi']:>9.4f}", flush=True)
