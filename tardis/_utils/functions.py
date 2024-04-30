#!/usr/bin/env python3

import anndata as ad
import numpy as np
import pandas as pd
from scanpy.tools import Ingest as sc_Ingest


def isnumeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def create_random_mask(shape, ratio_true, seed=None):
    rng = np.random.default_rng(seed)
    random_floats = rng.random(shape)
    mask = random_floats < ratio_true
    return mask


def categorical_covariate_validator(n1: list[str] | None, n2: list[str] | None):
    n1 = n1 if n1 is not None else []
    n2 = n2 if n2 is not None else []
    i = set(n1).intersection(set(n2))
    if len(i) != 0:
        raise ValueError(
            f"If a key is defined as a disentenglement target, do not put the same ket as categorical key: {i}"
        )


def label_transfer(
    reference_embeddings: ad.AnnData,
    query_embeddings: ad.AnnData,
    cluster_list: list[str],
    neighbors_count: int,
):
    # Compute a joint neighbor graph for reference and query embeddings
    joint_graph = sc_Ingest(reference_embeddings)
    joint_graph.fit(query_embeddings)
    joint_graph.neighbors(k=neighbors_count)

    # Calculate distances to top neighbors_count neighbors for each cell and store indices
    top_neighbor_distances, top_neighbor_indices = joint_graph._distances, joint_graph._indices

    # Transform distances with Gaussian kernel
    distance_stds = np.std(top_neighbor_distances, axis=1)
    distance_stds = (2.0 / distance_stds) ** 2
    distance_stds = distance_stds.reshape(-1, 1)
    transformed_distances = np.exp(-np.true_divide(top_neighbor_distances, distance_stds))

    # Normalize transformed distances so that they sum to 1
    normalized_weights = transformed_distances / np.sum(transformed_distances, axis=1, keepdims=True)

    # Initialize empty series to store predicted labels and uncertainties for each query cell
    cell_uncertainties = pd.Series(index=query_embeddings.obs_names, dtype="float64")
    predicted_labels = pd.Series(index=query_embeddings.obs_names, dtype="object")

    # Iterate through query cells
    for cluster in cluster_list:
        train_labels = reference_embeddings.obs[cluster].values
        for index in range(len(normalized_weights)):
            # Store cell types present among neighbors in reference
            unique_cell_types = np.unique(train_labels[top_neighbor_indices[index]])

            # Store best label and corresponding probability
            best_cell_type, best_probability = None, 0.0

            # Iterate through all cell types present among the cell's neighbors
            for label in unique_cell_types:
                prob = normalized_weights[index, train_labels[top_neighbor_indices[index]] == label].sum()
                if best_probability < prob:
                    best_probability = prob
                    best_cell_type = label
            else:
                final_label = best_cell_type

            # Store best label and corresponding uncertainty
            cell_uncertainties.iloc[index] = max(1 - best_probability, 0)
            predicted_labels.iloc[index] = final_label

        query_embeddings.obs[f"transf_{cluster}"] = predicted_labels
        query_embeddings.obs[f"transf_{cluster}_unc"] = cell_uncertainties
