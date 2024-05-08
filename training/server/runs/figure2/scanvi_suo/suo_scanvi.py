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
import scvi
import scipy.stats

sys.path.append("/home/icb/kemal.inecik/work/codes/tardis")
import tardis
tardis.config = tardis.config_server
print(f"CUDA used: {torch.cuda.is_available()}")

adata_file_path = os.path.join(tardis.config.io_directories["processed"], "dataset_complete_Suo.h5ad")
assert os.path.isfile(adata_file_path), f"File not already exist: `{adata_file_path}`"
adata = ad.read_h5ad(adata_file_path)
adata.obs["age"] = adata.obs["age"].astype("str").astype("category")

n_epochs_kl_warmup = 400
model_params = dict(
    n_latent=24, 
    gene_likelihood = "nb",
)

train_params = dict(
    train_size=0.8,
    batch_size=512,
    limit_train_batches=0.25, 
    limit_val_batches=0.25,
    max_epochs=400,
)

dataset_params = dict(
    layer=None, 
    labels_key="cell_type",
    unlabeled_category="NA",
    batch_key="concatenated_integration_covariates",
    categorical_covariate_keys=None,
)


scvi.model.SCANVI.setup_anndata(adata, **dataset_params)

vae = scvi.model.SCANVI(adata, **model_params)
vae.train(**train_params)

dir_path = os.path.join(
    tardis.config.io_directories["models"],
    "suo_scanvi_v2"
)

vae.save(
    dir_path,
    overwrite=True,
)