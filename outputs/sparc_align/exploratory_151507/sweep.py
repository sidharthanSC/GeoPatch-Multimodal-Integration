"""Capacity + cluster-input sweep for SPARC on one DLPFC section."""
from pathlib import Path
import json, numpy as np, torch
from src.datasets.dlpfc import DlpfcDataset
from src.sparc_align.config import SparcConfig
from src.sparc_align.data import prepare_section
from src.sparc_align.train import fit_sparc
from src.sparc_align.evaluate import cluster_sparc_latents
from dataclasses import replace

ds = DlpfcDataset.from_checkpoint(Path("checkpoints/dlpfc.pkl"))
adata = ds.get_section("151507")

rows = []
for n_latents, k in [(1024, 32), (1024, 128), (2048, 256), (4096, 512)]:
    cfg = SparcConfig(n_latents=n_latents, k_active=k, epochs=50)
    data = prepare_section(adata, "151507", cfg)
    res = fit_sparc(data, cfg, device="mps", verbose=False)
    m = res.metrics
    base = dict(n_latents=n_latents, k=k,
                self_gene=m["self_nmse_gene"], self_img=m["self_nmse_image"],
                cross_g2i=m["cross_nmse_gene_to_image"], cross_i2g=m["cross_nmse_image_to_gene"],
                alive=m["all_alive_fraction"])
    # re-cluster the same latents under every cluster_input mode
    for mode in ["sum", "gene", "image", "concat", "support"]:
        c = cluster_sparc_latents(res.latents, data.labels, data.coordinates,
                                  replace(cfg, cluster_input=mode))
        base[f"ari_{mode}"] = c["metrics"]["refined_ari"]
    rows.append(base)
    print(json.dumps(base, default=float), flush=True)

print()
hdr = f"{'L':>5} {'k':>4} {'selfG':>6} {'selfI':>6} {'g2i':>6} {'i2g':>6} " + " ".join(f"{'ARI:'+m:>11}" for m in ["sum","gene","image","concat","support"])
print(hdr)
for r in rows:
    print(f"{r['n_latents']:>5} {r['k']:>4} {r['self_gene']:>6.3f} {r['self_img']:>6.3f} "
          f"{r['cross_g2i']:>6.3f} {r['cross_i2g']:>6.3f} " +
          " ".join(f"{r['ari_'+m]:>11.4f}" for m in ["sum","gene","image","concat","support"]))
