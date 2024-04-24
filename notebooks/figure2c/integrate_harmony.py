import copy
import gc
import os
import sys
import warnings
import importlib
from pathlib import Path

adata_dir = sys.argv[1]


import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from harmony import harmonize

print(adata_dir)
adata = ad.read_h5ad(adata_dir)

sc.pp.normalize_total(adata, target_sum=1e6)
sc.pp.log1p(adata)
sc.pp.pca(adata)

harmony = harmonize(adata.obsm["X_pca"], adata.obs, batch_key="concatenated_integration_covariates")
x_pca = adata.obsm["X_pca"].copy()

del adata
gc.collect()

adata = ad.read_h5ad(adata_dir)
adata.obsm["X_pca"] = x_pca.copy()
adata.obsm["Unintegrated"] = x_pca.copy()
adata.obsm["harmony"] = harmony

print(adata)
adata.write_h5ad(adata_dir)
