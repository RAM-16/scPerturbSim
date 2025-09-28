import scanpy as sc
import numpy as np
import scipy.sparse as sp

def load_pbmc3k_synth(hvg=2000, min_cells=3, n_perturb=5, ctrl_label="ctrl", seed=42):
    rng = np.random.RandomState(seed)

    adata = sc.datasets.pbmc3k()
    sc.pp.filter_genes(adata, min_cells=min_cells)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=hvg, subset=True, flavor="seurat_v3")

    G = adata.n_vars
    perts = [ctrl_label] + [f"pert{i+1}" for i in range(n_perturb)]
    signatures = {}

    for p in perts[1:]:
        idx = rng.choice(G, size=rng.randint(40, 120), replace=False)
        delta = rng.normal(0.3, 0.15, size=idx.size).astype(np.float32)
        flip = rng.rand(idx.size) < 0.25
        delta[flip] *= -1
        sig = np.zeros(G, dtype=np.float32)
        sig[idx] = delta
        signatures[p] = sig

    labels = rng.choice(perts, size=adata.n_obs, p=[0.5] + [0.5/len(signatures)]*len(signatures))
    adata.obs["perturb"] = labels

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
    Y = X.copy()
    for p, sig in signatures.items():
        sel = (labels == p)
        if sel.any():
            Y[sel] = X[sel] + sig

    Y = np.clip(Y, a_min=0.0, a_max=np.percentile(Y, 99.9))
    adata.layers["control"] = sp.csr_matrix(X)
    adata.layers["target"]  = sp.csr_matrix(Y)
    adata.uns["perturbations"] = perts
    adata.uns["ctrl_label"] = ctrl_label
    return adata
