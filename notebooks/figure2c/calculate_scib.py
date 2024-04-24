import copy
import gc
import os
import sys
import warnings
import importlib
from pathlib import Path

latent_dir = sys.argv[1]
model_type = sys.argv[2]
scib_metric = sys.argv[3]

latent_subset_identifiers = sys.argv[4]
latent_subsets = sys.argv[5]

scib_metrics = dict(
    "bioconservation": [1,2,3,4]
    "batchcorrection": [1,2,3,4]
)
metric_list = scib_metrics[scib_metric]

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from scib_metrics.benchmark import Benchmarker, BioConservation, BatchCorrection

sys.path.append("/home/icb/kemal.inecik/work/codes/tardis")
import tardis
tardis.config = tardis.config_server
sc.settings.verbosity = 3

print("data_loading", flush=True)
latent = ad.read_h5ad(latent_dir)

if model_type == "tardis":
    latent_subset_identifiers = latent_subset_identifiers.strip().split(".")
    latent_subsets = [list(map(int, i.strip().split(','))) for i in latent_subsets.strip().split('.')]
    subsets = {k: v for k,v in zip(latent_subset_identifiers, latent_subsets)}
    
    latent = latent[:, subsets["unreserved"]].copy()
    # calculate only for this one
elif model_type == "scvi":
    # get unintegrated
    # get scvi latent
    # get harmony
    # calculate for all three

latent
adata = ad.read_h5ad(latent_dir)

for a_metric in metric_list:
    
    print(a_metric)

    bm = Benchmarker(
        adata,
        batch_key="concatenated_integration_covariates",
        label_key="cell_type",
        embedding_obsm_keys=["X"],
        pre_integrated_embedding_obsm_key="X_pca",
        bio_conservation_metrics=biocons,
        n_jobs=-1,
    )
    
    
    # adata_subset = ad.AnnData(X=adata[:, subset_latent].X, obs=adata.obs)
    
    # rsc.utils.anndata_to_GPU(adata_subset)
    # rsc.pp.neighbors(adata=adata_subset, n_neighbors=n_neighbors)
    # rsc.tl.umap(adata=adata_subset)

    # p1, p2 = os.path.splitext(latent_dir)
    # p = p1 + "_subset-" + "-".join(c) + p2
    # print(p)

    # h = dict(
    #     subset = c,
    #     subset_indices = subset_latent,
    #     main_latent = latent_dir,
    #     subset_latent = p
    # )
    
    # adata_subset.uns["tardis_subset"] = h
    # to_main_latent.append(copy.deepcopy(h))
    
    # print("writing", flush=True)
    # adata_subset.write_h5ad(latent_dir)
    
    # del adata_subset
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    gc.collect()

adata.uns["tardis_subsets"] = to_main_latent
adata.write_h5ad(latent_dir)


