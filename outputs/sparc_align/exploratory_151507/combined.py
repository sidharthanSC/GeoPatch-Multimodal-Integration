"""Stage1 (spatial TopK) -> Stage2 (cross-attention + contrastive), 151507."""
from pathlib import Path
from dataclasses import replace
import numpy as np
from src.datasets.dlpfc import DlpfcDataset
from src.sparc_align.config import SparcConfig
from src.sparc_align.data import prepare_section
from src.sparc_align.train import fit_sparc
from src.sparc_align.evaluate import build_cluster_input
from src.sparc_align.attention_stage import fit_attention_stage

ds = DlpfcDataset.from_checkpoint(Path("checkpoints/dlpfc.pkl"))
adata = ds.get_section("151507")
print(f"{'w':>5} {'stage1_supp':>12} {'stage2_sum':>11} {'stage2_supp':>12} {'s2_NMI':>8}")
for w in [0.0, 0.75]:
    cfg = SparcConfig(n_latents=1024, k_active=128, epochs=50,
                      spatial_topk_weight=w, attention_epochs=20)
    data = prepare_section(adata, "151507", cfg)
    s1 = fit_sparc(data, cfg, device="mps", verbose=False)
    from src.sparc_align.evaluate import cluster_sparc_latents
    s1s = cluster_sparc_latents(s1.latents, data.labels, data.coordinates,
                                replace(cfg, cluster_input="support"))["metrics"]["refined_ari"]
    out = {}
    for mode in ["sum", "support"]:
        mat = build_cluster_input(s1.latents, mode).astype(np.float32)
        r = fit_attention_stage(data, mat, cfg, device="cpu", verbose=False)
        out[mode] = (r.metrics["refined_ari"], r.metrics["refined_nmi"])
    print(f"{w:>5.2f} {s1s:>12.4f} {out['sum'][0]:>11.4f} {out['support'][0]:>12.4f} "
          f"{out['sum'][1]:>8.4f}", flush=True)
