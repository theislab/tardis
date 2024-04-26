# as in the original datasets are mistakenly removes var_names. re-creating them will cause the trained models malfunction. 
# just an alternative approach..

import copy
import gc
import os
import sys
import warnings

import anndata as ad
import scanpy as sc
import pandas as pd
from scanpy.preprocessing._highly_variable_genes import highly_variable_genes

sys.path.append("/home/icb/kemal.inecik/work/codes/tardis")
from tardis._utils.preprocessing import (  # noqa
    NA_CELL_TYPE_PLACEHOLDER,
    RANK_GENES_GROUPS_KEY,
    calculate_de_genes,
    deep_memory_usage,
    select_hvgs,
)

sc.settings.verbosity = 3

print("read adata", flush=True)
adata_file_path = (
    "/lustre/groups/ml01/workspace/kemal.inecik/hdca/" "temp/preprocessing/unification_union_20240330_hvg.h5ad"
)
assert os.path.isfile(adata_file_path), f"File not already exist: `{adata_file_path}`"
adata = ad.read_h5ad(adata_file_path)
print(adata, flush=True)

GENE_NAME_COLUMN = "hgnc"
print("starting..")

for handle in adata.obs["handle_anndata"].value_counts().index:
    print("###", flush=True)
    print(handle, flush=True)

    adata_dataset_1 = adata[adata.obs["handle_anndata"] == handle]
    var_column = f"{GENE_NAME_COLUMN}_{handle}"
    adata_dataset_1 = adata_dataset_1[:, adata_dataset_1.var[var_column] != NA_CELL_TYPE_PLACEHOLDER].copy()

    # remove cell-cycle, bcr, tcr. Also, scanpy filter genes
    adata_dataset_1 = adata_dataset_1[:, adata_dataset_1.var["highly_variable_omitted_reason"] == "NA"].copy()
    del adata_dataset_1.var
    gc.collect()
    print("hvg", flush=True)
    adata_dataset_1.layers["counts"] = adata_dataset_1.X.copy()
    sc.pp.normalize_total(adata_dataset_1, target_sum=1e6)
    sc.pp.log1p(adata_dataset_1)

    # Calculate dispersion and mean with scanpy function to determine max and min means.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        highly_variable_genes(
            adata=adata_dataset_1,
            layer=None,
            batch_key="concatenated_integration_covariates",
            subset=False,
            inplace=True,
        )
        adata_dataset_1.var.drop(columns=["highly_variable"], inplace=True)
    adata_dataset_1_var_before = adata_dataset_1.var.copy()
    gc.collect()

    hvg_dict = dict(hvg_number=2**13, min_mean=0.05, max_mean=3)

    _, final_hvg_selection, _ = select_hvgs(
        adata_var=adata_dataset_1.var.copy(),
        top_gene_number=hvg_dict["hvg_number"],
        min_mean=hvg_dict["min_mean"],
        max_mean=hvg_dict["max_mean"],
    )

    adata_dataset_1 = adata_dataset_1[:, adata_dataset_1.var.index.isin(final_hvg_selection)]

    write_path = "/lustre/groups/ml01/workspace/kemal.inecik/tardis_data/processed/" f"dataset_complete_{handle}_varnames.pickle"
    print(write_path, flush=True)
    df = adata_dataset_1.var.copy()
    df.to_pickle(write_path)
    print("ok", flush=True)
    
    del df, adata_dataset_1
    gc.collect()


