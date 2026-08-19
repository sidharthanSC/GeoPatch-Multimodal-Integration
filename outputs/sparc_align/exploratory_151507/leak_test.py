"""Controlled A/B: MP-MNCA Phase 1 with ground-truth vs image-KMeans pseudo-labels.

Everything else -- seed, config, section, epochs, architecture -- is identical.
The only variable is what defines the contrastive negative mask.
"""
from pathlib import Path
import json, sys
from src.datasets.dlpfc import DlpfcDataset
from src.mp_mnca.config import MpMncaConfig
from src.mp_mnca.train_phase1 import prepare_section_phase1, fit_phase1

sections = sys.argv[1].split(",") if len(sys.argv) > 1 else ["151507"]
epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 20

ds = DlpfcDataset.from_checkpoint(Path("checkpoints/dlpfc.pkl"))
cfg = MpMncaConfig(epochs=epochs, seed=0, batch_size=256, learning_rate=3e-4,
                   mask_rate=0.1, num_heads=8, temperature=10.0,
                   image_pseudo_clusters=40, image_pca_dim=16, refinement_neighbors=15)

rows = []
for sid in sections:
    data = prepare_section_phase1(ds.get_section(sid), sid, cfg)
    row = {"section": sid}
    for source in ["ground_truth", "image_kmeans"]:
        res = fit_phase1(data, cfg, device="cpu", pseudo_label_source=source)
        row[f"{source}_ari"] = res.metrics["refined_ari"]
        row[f"{source}_nmi"] = res.metrics["refined_nmi"]
        print(f"  {sid} {source:14s} refined_ARI={res.metrics['refined_ari']:.4f} "
              f"refined_NMI={res.metrics['refined_nmi']:.4f}", flush=True)
    rows.append(row)

print("\n" + "="*78)
print(f"{'section':>9} {'GT-leak ARI':>12} {'KMeans ARI':>12} {'drop':>8} "
      f"{'GT-leak NMI':>12} {'KMeans NMI':>12}")
print("="*78)
for r in rows:
    print(f"{r['section']:>9} {r['ground_truth_ari']:>12.4f} {r['image_kmeans_ari']:>12.4f} "
          f"{r['ground_truth_ari']-r['image_kmeans_ari']:>8.4f} "
          f"{r['ground_truth_nmi']:>12.4f} {r['image_kmeans_nmi']:>12.4f}")
json.dump(rows, open("/private/tmp/claude-501/-Users-sidharthansc-Documents-Sidharthan/"
                     "6ac23b36-e479-45ed-9535-f52cf3d0ed31/scratchpad/leak_test.json", "w"), indent=2)
