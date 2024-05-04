import copy
import gc
import os
import sys
import warnings
import importlib
from pathlib import Path
import itertools

latent_dir = sys.argv[1]
print(latent_dir)
latent_subset_identifiers = sys.argv[2]
print(latent_subset_identifiers)
latent_subsets = sys.argv[3]
print(latent_subsets)
n_neighbors = sys.argv[4]
print(n_neighbors)

import anndata as ad
import cupy as cp
import networkx as nx
import numpy as np
import pandas as pd
import rapids_singlecell as rsc
import scanpy as sc
import scib
import scib_metrics
import scipy.stats
import statsmodels.api as sm
import torch
from sklearn.neighbors import kneighbors_graph
from statsmodels.formula.api import ols

sys.path.append("/home/icb/kemal.inecik/work/codes/tardis")
import tardis

tardis.config = tardis.config_server

sc.settings.verbosity = 3

print(f"CUDA used: {torch.cuda.is_available()}", flush=True)


latent_subset_identifiers = latent_subset_identifiers.strip().split(".")
latent_subsets = [list(map(int, i.strip().split(','))) for i in latent_subsets.strip().split('.')]
n_neighbors = int(n_neighbors)

subsets = {k: v for k,v in zip(latent_subset_identifiers, latent_subsets)}
combinations = [c for n_combination in range(1, len(subsets)) for c in sorted(itertools.combinations(subsets.keys(), n_combination))]

print("data_loading", flush=True)
adata = ad.read_h5ad(latent_dir)

# butun permutasyonlar icin umap hesaplat..
to_main_latent = []
for ind, c in enumerate(combinations):
    subset_latent = sorted([i for j in c for i in subsets[j]])
    print(c)
    print(subset_latent)

    adata_subset = ad.AnnData(X=adata[:, subset_latent].X, obs=adata.obs)
    
    rsc.utils.anndata_to_GPU(adata_subset)
    rsc.pp.neighbors(adata=adata_subset, n_neighbors=n_neighbors)
    rsc.tl.umap(adata=adata_subset)

    p1, p2 = os.path.splitext(latent_dir)
    p = p1 + "_subset-" + "-".join(c) + p2
    print(p)

    h = dict(
        subset = np.array(list(c)),
        subset_indices = np.array(subset_latent),
        main_latent = latent_dir,
        subset_latent = p
    )
    
    adata_subset.uns["tardis_subset"] = h
    to_main_latent.append(p)
    
    print("writing", flush=True)
    adata_subset.write_h5ad(p)
    
    del adata_subset
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

print("writing the main", flush=True)
adata.uns["tardis_subsets"] = to_main_latent
adata.write_h5ad(latent_dir)


