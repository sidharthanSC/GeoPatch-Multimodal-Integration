import numpy as np
from pathlib import Path
from sklearn.neighbors import NearestNeighbors

d = Path('outputs/ablations_nomask_smoke/full/embeddings/151507.npz')
z = np.load(d, allow_pickle=True)
emb = z['embeddings']
coords = z['coordinates']
labels = z['labels']
emb_n = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

nn = NearestNeighbors(n_neighbors=7).fit(coords)
_, idx = nn.kneighbors(coords)
nb = emb_n[idx[:, 1:]]
sim = (emb_n[:, None, :] * nb).sum(-1)
print('mean cosine sim to spatial neighbors: %.4f' % sim.mean())

rng = np.random.default_rng(0)
ri = rng.integers(0, len(emb), (len(emb), 6))
sim_r = (emb_n[:, None, :] * emb_n[ri]).sum(-1)
print('mean cosine sim to random spots:     %.4f' % sim_r.mean())

same = labels[:, None] == labels[idx[:, 1:]]
print('same-layer neighbor sim:    %.4f' % sim[same].mean())
print('cross-layer neighbor sim:   %.4f' % sim[~same].mean())

# also: how correlated is the RAW gene expression of neighbors vs random?
feat = np.load(Path('outputs/prior_models/staig_node_features_feat.npz'), allow_pickle=True)['feat']
feat_n = feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-12)
sim_f = (feat_n[:, None, :] * feat_n[idx[:, 1:]]).sum(-1)
sim_fr = (feat_n[:, None, :] * feat_n[ri]).sum(-1)
print('raw gene: neighbor sim %.4f vs random %.4f' % (sim_f.mean(), sim_fr.mean()))