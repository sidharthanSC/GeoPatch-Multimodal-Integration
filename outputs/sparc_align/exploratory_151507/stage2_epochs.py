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
cfg = SparcConfig(n_latents=1024, k_active=128, epochs=50)
data = prepare_section(ds.get_section("151507"), "151507", cfg)
s1 = fit_sparc(data, cfg, device="mps", verbose=False)
matrix = build_cluster_input(s1.latents, "sum").astype(np.float32)
print(f"{'attn_epochs':>12} {'refined_ARI':>12} {'refined_NMI':>12}")
for e in [20, 50, 100, 200]:
    r = fit_attention_stage(data, matrix, replace(cfg, attention_epochs=e), device="cpu", verbose=False)
    print(f"{e:>12} {r.metrics['refined_ari']:>12.4f} {r.metrics['refined_nmi']:>12.4f}", flush=True)
