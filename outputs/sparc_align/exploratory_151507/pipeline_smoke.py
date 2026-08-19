"""Smoke test: SPARC stage 1 -> cross-attention stage 2 on one section."""
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
cfg = SparcConfig(n_latents=1024, k_active=128, epochs=50, attention_epochs=20)
data = prepare_section(ds.get_section("151507"), "151507", cfg)

s1 = fit_sparc(data, cfg, device="mps", verbose=False)
print(f"stage1 SPARC        refined_ARI={s1.metrics['refined_ari']:.4f} "
      f"refined_NMI={s1.metrics['refined_nmi']:.4f}")

for mode in ["support", "sum"]:
    matrix = build_cluster_input(s1.latents, mode).astype(np.float32)
    s2 = fit_attention_stage(data, matrix, replace(cfg, cluster_input="sum"),
                             device="cpu", verbose=False)
    print(f"stage2 attn[{mode:>7}] refined_ARI={s2.metrics['refined_ari']:.4f} "
          f"refined_NMI={s2.metrics['refined_nmi']:.4f}")
