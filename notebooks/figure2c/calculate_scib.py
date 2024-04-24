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
adata_dir = sys.argv[6]

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
adata = ad.read_h5ad(adata_dir)

if scib_metric == "batchcorrection":
    biocons = BioConservation(
        isolated_labels=False,
        nmi_ari_cluster_labels_leiden=False,
        nmi_ari_cluster_labels_kmeans=False,
        silhouette_label=False,
        clisi_knn=False
    )
    batchcor = BatchCorrection()
elif scib_metric == "bioconservation":
    biocons = BioConservation()
    batchcor = BatchCorrection(
        silhouette_batch=False,
        ilisi_knn=False,
        kbet_per_label=False,
        graph_connectivity=False,
        pcr_comparison=False
    )
else:
    raise ValueError


def get_results(self, min_max_scale: bool = True) -> pd.DataFrame:
    _LABELS = "labels"
    _BATCH = "batch"
    _X_PRE = "X_pre"
    _METRIC_TYPE = "Metric Type"
    _AGGREGATE_SCORE = "Aggregate score"
    
    df = self._results.transpose()
    df.index.name = "Embedding"
    df = df.loc[df.index != _METRIC_TYPE]
    if min_max_scale:
        # Use sklearn to min max scale
        df = pd.DataFrame(
            MinMaxScaler().fit_transform(df),
            columns=df.columns,
            index=df.index,
        )
    df = df.transpose()
    df[_METRIC_TYPE] = self._results[_METRIC_TYPE].values

    # Compute scores
    per_class_score = df.groupby(_METRIC_TYPE).mean().transpose()
    # This is the default scIB weighting from the manuscript
    # per_class_score["Total"] = 0.4 * per_class_score["Batch correction"] + 0.6 * per_class_score["Bio conservation"]
    df = pd.concat([df.transpose(), per_class_score], axis=1)
    df.loc[_METRIC_TYPE, per_class_score.columns] = _AGGREGATE_SCORE
    return df

if model_type == "tardis":
    latent_subset_identifiers = latent_subset_identifiers.strip().split(".")
    latent_subsets = [list(map(int, i.strip().split(','))) for i in latent_subsets.strip().split('.')]
    subsets = {k: v for k,v in zip(latent_subset_identifiers, latent_subsets)}

    # calculate only for this one
    latent.obsm["tardis"] = latent[:, subsets["unreserved"]].X
    latent.obsm["Unintegrated"] = adata.obsm["Unintegrated"]
    print(latent, flush=True)
    
    print("starting tardis scib", flush=True)
    bm = Benchmarker(
        adata=latent,
        batch_key="concatenated_integration_covariates",
        label_key="cell_type",
        embedding_obsm_keys=["tardis"],
        pre_integrated_embedding_obsm_key="Unintegrated",  # equals to X_pca
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcor,
        n_jobs=-1,
    )
    bm.benchmark()
    print("done", flush=True)
    df = get_results(bm, min_max_scale=False)
    
elif model_type == "scvi":
    
    adata.obsm["scvi"] = latent.X
    adata = adata.copy()
    print(adata, flush=True)
    print("starting scvi scib", flush=True)
    bm = Benchmarker(
        adata=adata,
        batch_key="concatenated_integration_covariates",
        label_key="cell_type",
        embedding_obsm_keys=["X_pca", "harmony", "scvi"],
        pre_integrated_embedding_obsm_key="Unintegrated",  # equals to X_pca
        bio_conservation_metrics=biocons,
        batch_correction_metrics=batchcor,
        n_jobs=-1,
    )
    bm.benchmark()
    df = get_results(bm, min_max_scale=False)
else:
    raise ValueError


print(df, flush=True)

p1, p2 = os.path.splitext(latent_dir)
p = p1 + f"_scib_{scib_metric}.pickle"
print(p, flush=True)

df.to_pickle(p)


