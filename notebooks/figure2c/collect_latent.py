import copy
import gc
import os
import sys
import warnings
import importlib
from pathlib import Path

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


data_dir = sys.argv[1]
model_dir = sys.argv[2]
output_dir = sys.argv[3]
n_neighbors = int(sys.argv[4])

print("data_loading", flush=True)

updated_obs = pd.read_csv(
    os.path.join(
        "/lustre/groups/ml01/workspace/kemal.inecik/hdca/data", "metadata", "combined_with_hierarchy", "anno_V1.csv"
    ),
    index_col="Unnamed: 0",
)
new_cols = ["LVL3", "LVL2", "LVL1", "LVL0"]

adata = ad.read_h5ad(data_dir)
adata.obs = pd.concat([adata.obs.copy(), updated_obs.copy().loc[adata.obs.index][new_cols]], axis=1)
adata.obs["age"] = adata.obs["age"].astype("str").astype("category")

print("model_loading", flush=True)
vae = tardis.MyModel.load(dir_path=model_dir, adata=adata)

print("metrics_calculating", flush=True)
batch_size = 512

metrics = {
    "reconstruction_error": vae.get_reconstruction_error(batch_size=batch_size)["reconstruction_loss"],
    "elbo": vae.get_elbo(batch_size=batch_size).item(),
    "r2_train": vae.get_reconstruction_r2(batch_size=batch_size, indices=vae.train_indices),
    "r2_train_deg_20": vae.get_reconstruction_r2(
        top_n_differentially_expressed_genes=20, batch_size=batch_size, indices=vae.train_indices
    ),
    "r2_train_deg_50": vae.get_reconstruction_r2(
        top_n_differentially_expressed_genes=50, batch_size=batch_size, indices=vae.train_indices
    ),
    "r2_validation": vae.get_reconstruction_r2(batch_size=batch_size, indices=vae.validation_indices),
    "r2_validation_deg_20": vae.get_reconstruction_r2(
        top_n_differentially_expressed_genes=20, batch_size=batch_size, indices=vae.validation_indices
    ),
    "r2_validation_deg_50": vae.get_reconstruction_r2(
        top_n_differentially_expressed_genes=50, batch_size=batch_size, indices=vae.validation_indices
    ),
}
print(metrics, flush=True)

print("get_latent, calculate graph", flush=True)
latent = ad.AnnData(X=vae.get_latent_representation(), obs=adata.obs.copy())
rsc.utils.anndata_to_GPU(latent)
rsc.pp.neighbors(adata=latent, n_neighbors=n_neighbors)
rsc.tl.umap(adata=latent)

print("writing", flush=True)
latent.uns["metrics"] = metrics
latent.write_h5ad(output_dir)







