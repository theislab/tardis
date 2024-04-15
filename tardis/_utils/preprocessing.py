#!/usr/bin/env python3

import sys
import warnings

import numpy as np
import pandas as pd
import scipy
from scanpy.tools import rank_genes_groups as sc_rank_genes_groups

NA_CELL_TYPE_PLACEHOLDER = "NA"
RANK_GENES_GROUPS_KEY = "rank_genes_groups"


def deep_memory_usage(obj, seen=None):
    """Recursively estimate memory usage of objects, including sparse arrays."""
    if seen is None:
        seen = set()
    # Avoid duplicate references
    if id(obj) in seen:
        return 0
    seen.add(id(obj))
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum([deep_memory_usage(v, seen) for v in obj.values()])
        size += sum([deep_memory_usage(k, seen) for k in obj.keys()])
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum([deep_memory_usage(i, seen) for i in obj])
    elif isinstance(obj, (pd.DataFrame, pd.Series)):
        size += obj.memory_usage(deep=True).sum()
    elif isinstance(obj, np.ndarray):
        size += obj.nbytes
    elif scipy.sparse.issparse(obj):
        # For sparse matrices, consider the data, indices, and indptr arrays
        size += obj.data.nbytes + obj.indices.nbytes + obj.indptr.nbytes
    # Add other data types if necessary
    return round(size / (1024**2), 2)


def select_hvgs(adata_var, top_gene_number, min_mean, max_mean):
    assert not np.isnan(adata_var["means"]).any(), "There are `nan`s in the HVG."

    count_removed = 0
    if isinstance(min_mean, int) or isinstance(min_mean, float):
        _mim = min_mean < adata_var["means"]
        count_removed += np.sum(~_mim)
        adata_var = adata_var[_mim]
    else:
        assert min_mean is None

    if isinstance(max_mean, int) or isinstance(max_mean, float):
        _mam = adata_var["means"] < max_mean
        count_removed += np.sum(~_mam)
        adata_var = adata_var[_mam]
    else:
        assert max_mean is None

    passed_thresholding = adata_var.index.copy()

    # sort genes by how often they selected as hvg within each batch and
    # break ties with normalized dispersion across batches. as the original function does.
    adata_var["dispersions_norm"] = adata_var["dispersions_norm"].astype("float32")
    if "highly_variable_nbatches" in adata_var.columns:
        adata_var.sort_values(
            ["highly_variable_nbatches", "dispersions_norm"],
            ascending=False,
            na_position="last",
            inplace=True,
            ignore_index=False,
        )
    else:
        warnings.warn("`highly_variable_nbatches` is not found. Only `dispersions_norm` will be used.")
        adata_var.sort_values(
            ["dispersions_norm"], ascending=False, na_position="last", inplace=True, ignore_index=False
        )

    if top_gene_number is None:
        top_gene_number = len(adata_var)
    elif len(adata_var) < top_gene_number:
        warnings.warn(
            f"Queried HVG gene count is lower than `{len(adata_var)}`. "
            f"Note: min-max (and np.nan values) filtered out: {count_removed}"
        )
        top_gene_number = len(adata_var)

    adata_var_sorted_index = adata_var.index.copy()

    adata_var = adata_var[:top_gene_number]
    assert np.sum(np.isnan(adata_var["dispersions_norm"])) == 0, "Unexpected error."
    final_hvg_selection = adata_var.index.copy()

    return set(passed_thresholding), set(final_hvg_selection), adata_var_sorted_index


def calculate_de_genes(adata, na_cell_type_placeholder=None):
    """Robust to NA cell type. ignores it and does not produce DE genes for NA."""
    if na_cell_type_placeholder is None:
        na_cell_type_placeholder = NA_CELL_TYPE_PLACEHOLDER

    valid_groups = [i for i in adata.obs["cell_type"].cat.categories if i != na_cell_type_placeholder]

    sc_rank_genes_groups(
        adata=adata,
        groupby="cell_type",
        method="t-test",
        groups=valid_groups,
        pts=True,
        # rankby_abs=True,  # get upregulated and downregulated genes.
        key_added=RANK_GENES_GROUPS_KEY,
    )
