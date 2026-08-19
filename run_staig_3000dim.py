import sys, time, json
from pathlib import Path

sys.path.insert(0, ".")

from src.prior_models.staig.train import run_dlpfc
from src.prior_models.staig.config import StaigConfig
from src.prior_models.staig.data import load_feature_bundle

config = StaigConfig(
    n_neighbors=5,
    hidden_dim=3000,
    projection_dim=3000,
    n_layers=1,
    temperature=10.0,
    epochs=400,
    learning_rate=5e-4,
    weight_decay=1e-5,
    feature_mask_rate_1=0.1,
    feature_mask_rate_2=0.1,
    image_pseudo_clusters=40,
    image_pca_dim=16,
    refinement_neighbors=15,
    seed=0,
    dtype="float32",
)
node_features = load_feature_bundle(
    Path("outputs/prior_models/staig_node_features_feat.npz"), "feat"
)
t0 = time.time()
summary = run_dlpfc(
    Path("checkpoints/dlpfc.pkl"),
    Path("outputs/prior_models/staig_3000dim_seed0_all12_v1"),
    config,
    sections=None,
    device=None,
    node_features=node_features,
    image_features=None,
)
print(json.dumps(summary["metrics"], indent=2), flush=True)
print(f"TOTAL {time.time()-t0:.0f}s", flush=True)