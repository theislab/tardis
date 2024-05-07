from inVAE import FinVAE, NFinVAE

import warnings
import os
import sys
import gc
import warnings

import anndata as ad
import scanpy as sc
import copy
import torch
from pathlib import Path
import networkx as nx
from sklearn.neighbors import kneighbors_graph
import numpy as np
import scanpy as sc
import pandas as pd
import numpy as np
import scipy.stats

adata_file_path = os.path.join("/home/icb/kemal.inecik/lustre_workspace/tardis_data/processed", "dataset_complete_Suo.h5ad")
assert os.path.isfile(adata_file_path), f"File not already exist: `{adata_file_path}`"
adata = ad.read_h5ad(adata_file_path)
adata.X = adata.X.astype(np.float32)
print(adata_file_path, flush=True)
print(adata, flush=True)

def create_random_mask(shape, ratio_true, seed=None):
    rng = np.random.default_rng(seed)
    random_floats = rng.random(shape)
    mask = random_floats < ratio_true
    return mask

print(f"CUDA: {torch.cuda.is_available()}", flush=True)

adata_train_bool = create_random_mask(adata.shape[0], ratio_true=0.8, seed=0)
adata_train = adata[adata_train_bool].copy()
adata_valid = adata[~adata_train_bool].copy()

inv_covar_keys = {
    'cont': [],
    'cat': []
}

spur_covar_keys = {
    'cont': [],
    'cat': ['organ', 'integration_donor', 'integration_library_platform_coarse'] #set to the keys in the adata
}

model = FinVAE(
    adata = adata_train,
    layer = None,
    inv_covar_keys = inv_covar_keys,
    spur_covar_keys = spur_covar_keys,
    device="cuda",
    latent_dim_inv = 20, 
    latent_dim_spur = 5,
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    model.train(n_epochs=300, lr_train=0.001, weight_decay=0.0001)

model.save("/home/icb/kemal.inecik/lustre_workspace/tardis_data/models/invae_7.pt")

invariant = model.get_latent_representation(adata, latent_type='invariant')
spurious = model.get_latent_representation(adata, latent_type='spurious')

joblib.dump(invariant, "/home/icb/kemal.inecik/lustre_workspace/tardis_data/processed/inveriant_7.joblib")
joblib.dump(spurious, "/home/icb/kemal.inecik/lustre_workspace/tardis_data/processed/spurious_7.joblib")
