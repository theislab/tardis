#!/usr/bin/env python3

import logging
from collections.abc import Sequence
from typing import Dict, Literal, Optional, Union

import numpy as np
import pandas as pd
import sklearn.metrics
import torch
import tqdm
from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.utils import unsupported_if_adata_minified
import sklearn
import scipy

from ._myconstants import NA_CELL_TYPE_PLACEHOLDER, RANK_GENES_GROUPS_KEY

logger = logging.getLogger(__name__)


class ModuleMetricsMixin:

    @torch.inference_mode()
    def my_sample(
        self,
        tensors: Optional[Dict[str, torch.Tensor]] = None,
        generative_outputs: Optional[Dict[str, torch.Tensor]] = None,
        inference_outputs: Optional[Dict[str, torch.Tensor]] = None,
        n_samples: Optional[int] = 1,
        latent_space: bool = False,
        return_dist: bool = False,
    ) -> torch.Tensor:
        """Alternative `sample` method, more efficient during training."""

        # Check the parameters are in correct format.
        if generative_outputs is not None:
            assert tensors is None and inference_outputs is None
        if inference_outputs is not None:
            assert tensors is None and generative_outputs is None
        if tensors is not None:
            assert inference_outputs is None and generative_outputs is None

        # Calculate latent or output distributions
        if not latent_space and tensors is not None:
            _, generative_outputs = self.forward(tensors=tensors, compute_loss=False)
            dist = generative_outputs["px"]
        elif not latent_space:
            dist = generative_outputs["px"]
        elif tensors is not None:
            inference_outputs, _ = self.forward(tensors, compute_loss=False)
            dist = inference_outputs["qz"]
        else:
            dist = inference_outputs["qz"]

        if return_dist:  # Return distributions of latent or output
            return dist
        else:  # Return samples of latent or output
            # The output is with there dimension [n_samples, batch_size, feature_size]
            return dist.sample([n_samples])


class MetricsMixin:

    @staticmethod
    def _create_de_genes_mask(
        top_n_genes: int,
        de_genes_dataframe: Optional[pd.DataFrame] = None,
        cell_type_list: Optional[Union[np.ndarray, list]] = None,
        adata_var_names: Optional[pd.Series] = None,
        cell_type_is_not_na: Optional[Union[str, Union[np.ndarray, list]]] = None,
    ):
        # Initialize an empty list to store the mask arrays
        mask_arrays = []

        if cell_type_is_not_na is None or isinstance(cell_type_is_not_na, str):
            # Create `cell_type_is_not_na`. either use default or defined key as the NA

            if cell_type_is_not_na is None:
                # NA element is defined, use default `NA`
                _na = NA_CELL_TYPE_PLACEHOLDER
            else:
                _na = cell_type_is_not_na

            assert isinstance(cell_type_list, np.ndarray)
            assert _na not in de_genes_dataframe.columns, f"Excepted cell-type is in DE dataframe: {_na}"
            cell_type_is_not_na = cell_type_list != _na

        for v, is_v_not_na in zip(cell_type_list, cell_type_is_not_na):
            # skip `na` cell types
            if is_v_not_na:
                # Get the associated column
                try:
                    j = de_genes_dataframe[v]
                except KeyError as e:
                    raise KeyError(
                        f"The key '{v}' does not exist in the DataFrame. "
                        "It is likely because it is excluded in DE genes calculation previously (as it is given "
                        "as a `NA` placeholder). If you used `src.preprocessing.calculate_de_genes` for de calculation "
                        "use the same `na_cell_type_placeholder` here for `cell_type_is_not_na` parameter."
                    ) from e
                # Get first top_n_genes elements
                k = j[:top_n_genes]
                # Create the mask array
                mask = adata_var_names.isin(k)
                mask_arrays.append(mask.values)

            else:
                # a mock mask. which will not take effect in the calculations.
                mask = np.zeros(len(adata_var_names), dtype=bool)
                mask[:top_n_genes] = True
                np.random.shuffle(mask)
                mask_arrays.append(mask)

        mask_arrays = np.vstack(mask_arrays)

        return mask_arrays.astype(bool)

    @staticmethod
    def _de_genes_metrics_variables_creator(model, adata):

        labels_full_name = model.adata_manager.get_state_registry(REGISTRY_KEYS.LABELS_KEY)["categorical_mapping"]
        labels_full_name = {j: i for i, j in zip(range(len(labels_full_name)), labels_full_name)}
        try:
            de_genes_dataframe = pd.DataFrame(adata.uns[RANK_GENES_GROUPS_KEY]["names"])
        except KeyError as e:
            raise KeyError("DEG key `{RANK_GENES_GROUPS_KEY}` are not found in `anndata.uns`.") from e
        de_genes_dataframe.rename(columns=labels_full_name, inplace=True)
        return labels_full_name, de_genes_dataframe

    @staticmethod
    def _calculate_r2_reconstruction_de_genes_masking_helper(np_arr, the_mask, is_not_na_arr):
        n = the_mask.shape[0]
        np_arr = np_arr[the_mask]
        np_arr = np_arr.reshape(n, int(np_arr.size / n))

        if isinstance(is_not_na_arr, np.ndarray):
            return np_arr[is_not_na_arr, :]

        elif isinstance(is_not_na_arr, torch.Tensor):
            return np_arr[is_not_na_arr.view(-1), :]

        else:
            raise ValueError

    @staticmethod
    def _calculate_r2_reconstruction_de_genes_calculator(
        true, pred, masking_tensor, cell_type_is_not_na, min_item_for_calculation
    ):
        # Perform the sum operation
        if isinstance(cell_type_is_not_na, np.ndarray):
            cell_type_is_not_na_sum = np.sum(cell_type_is_not_na)
        elif isinstance(cell_type_is_not_na, torch.Tensor):
            cell_type_is_not_na_sum = torch.sum(cell_type_is_not_na).item()
        else:
            raise TypeError("Input data type not supported.")

        if cell_type_is_not_na_sum > min_item_for_calculation:
            pred = MetricsMixin._calculate_r2_reconstruction_de_genes_masking_helper(
                pred, masking_tensor, cell_type_is_not_na
            )
            true = MetricsMixin._calculate_r2_reconstruction_de_genes_masking_helper(
                true, masking_tensor, cell_type_is_not_na
            )

            return true, pred

        else:
            raise ValueError("Error during the calculation of r2 scores for `de_genes`.")

    @torch.inference_mode()
    @unsupported_if_adata_minified
    def get_knn_purity(
        self, 
        labels_key: str, 
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_neighbors: int = 30,
    ) -> float:
        adata = self._validate_anndata(adata if adata is not None else self.adata_manager.adata)
        data = self.get_latent_representation(adata=adata, indices=indices)
        labels = adata.obs[labels_key].values.flatten()
        return MetricsMixin.get_knn_purity_precalculated(data=data, labels=labels, n_neighbors=n_neighbors)        
    
    @staticmethod
    def get_knn_purity_precalculated(
        data: np.ndarray, 
        labels: np.ndarray, 
        # Number of nearest neighbors.
        n_neighbors: int = 30 
    ) -> float:  # between zero and one.
        labels = sklearn.preprocessing.LabelEncoder().fit_transform(labels.ravel())
        nbrs = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors + 1).fit(data)
        indices = nbrs.kneighbors(data, return_distance=False)[:, 1:]
        neighbors_labels = np.vectorize(lambda i: labels[i])(indices)
        scores = ((neighbors_labels - labels.reshape(-1, 1)) == 0).mean(axis=1)
        res = [np.mean(scores[labels == i]) for i in np.unique(labels)]

        return np.mean(res)
    
    @torch.inference_mode()
    @unsupported_if_adata_minified
    def get_entropy_batch_mixing_precalculated(
        self, 
        labels_key: str, 
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        n_neighbors: int = 50, 
        n_pools: int = 50, 
        n_samples_per_pool: int = 100
    ) -> float:
        adata = self._validate_anndata(adata if adata is not None else self.adata_manager.adata)
        data = self.get_latent_representation(adata=adata, indices=indices)
        labels = adata.obs[labels_key].values.flatten()
        return MetricsMixin.get_entropy_batch_mixing_precalculated(
            data=data, labels=labels, n_neighbors=n_neighbors, n_pools=n_pools, n_samples_per_pool=n_samples_per_pool)    
    
    @staticmethod
    def get_entropy_batch_mixing_precalculated(
        data: np.ndarray, 
        labels: np.ndarray,
        # Number of nearest neighbors.
        n_neighbors: int = 50, 
        # Number of EBM computation which will be averaged.
        n_pools: int = 50, 
        # Number of samples to be used in each pool of execution.
        n_samples_per_pool: int = 100
    ) -> float:  # between zero and one.

        def __entropy_from_indices(indices, n_cat):
            return scipy.stats.entropy(np.array(np.unique(indices, return_counts=True)[1].astype(np.int32)), base=n_cat)
        n_cat = len(np.unique(labels))
        # print(f'Calculating EBM with n_cat = {n_cat}')
        neighbors = sklearn.neighbors.NearestNeighbors(n_neighbors=n_neighbors + 1).fit(data)
        indices = neighbors.kneighbors(data, return_distance=False)[:, 1:]
        batch_indices = np.vectorize(lambda i: labels[i])(indices)
        entropies = np.apply_along_axis(__entropy_from_indices, axis=1, arr=batch_indices, n_cat=n_cat)

        # average n_pools entropy results where each result is an average of n_samples_per_pool random samples.
        if n_pools == 1:
            score = np.mean(entropies)
        else:
            score = np.mean([
                np.mean(entropies[np.random.choice(len(entropies), size=n_samples_per_pool)])
                for _ in range(n_pools)
            ])

        return score


    @torch.inference_mode()
    @unsupported_if_adata_minified
    def get_reconstruction_r2(
        self,
        adata: Optional[AnnData] = None,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        # 0 if all genes should be considered
        top_n_differentially_expressed_genes: int = 0,
        min_item_for_calculation: int = 10,
        aggregate_method_datapoints: Literal["variance_weighted", "uniform_average", "flatten", "mean"] = "mean",
        verbose: bool = False,
        debug: bool = False,
    ) -> float:
        """Get reconstruction performance by R2 score."""
        adata = self._validate_anndata(adata if adata is not None else self.adata_manager.adata)
        
        batch_size = (
            (adata.n_obs if indices is None else min(adata.n_obs, len(indices))) if batch_size is None else batch_size
        )
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)
        values = []

        if top_n_differentially_expressed_genes < 0:
            raise ValueError
        elif top_n_differentially_expressed_genes != 0:
            labels_full_name, de_genes_dataframe = MetricsMixin._de_genes_metrics_variables_creator(
                model=self, adata=adata
            )

        for tensors in tqdm.tqdm(scdl) if verbose else scdl:
            pred = (
                self.module.my_sample(tensors=tensors, latent_space=False, return_dist=False, n_samples=1)
                .cpu()
                .numpy()
                .mean(axis=0)  # aggregate 'n_sample's
            )
            true = tensors[REGISTRY_KEYS.X_KEY].cpu().numpy()

            if top_n_differentially_expressed_genes != 0:
                # TODO: remove cell_type_list to deg_category, and put it in setup anndata
                # TODO: accept mask 
                # TODO: write a simpler method that does not require this tensor, just 
                #   accepts categorty, data, that is all. batch size and reduction etc as well of course
                # TODO: put all these functions into a class
                cell_type_list = tensors[REGISTRY_KEYS.LABELS_KEY].view(-1).cpu().numpy()
                if NA_CELL_TYPE_PLACEHOLDER in labels_full_name:
                    cell_type_is_not_na = cell_type_list != labels_full_name[NA_CELL_TYPE_PLACEHOLDER]
                else:
                    cell_type_is_not_na = np.ones_like(cell_type_list).astype(bool)

                masking_tensor = MetricsMixin._create_de_genes_mask(
                    top_n_genes=top_n_differentially_expressed_genes,
                    de_genes_dataframe=de_genes_dataframe,
                    cell_type_list=cell_type_list,
                    cell_type_is_not_na=cell_type_is_not_na,
                    adata_var_names=pd.Series(list(adata.var.index)),
                )

                true, pred = MetricsMixin._calculate_r2_reconstruction_de_genes_calculator(
                    true, pred, masking_tensor, cell_type_is_not_na, min_item_for_calculation
                )

            if aggregate_method_datapoints not in ["flatten", "mean"]:
                r2 = sklearn.metrics.r2_score(y_true=true, y_pred=pred, multioutput=aggregate_method_datapoints)
            elif aggregate_method_datapoints == "flatten":
                r2 = sklearn.metrics.r2_score(y_true=true.flatten(), y_pred=pred.flatten())
            elif aggregate_method_datapoints == "mean":  # not recommended
                r2 = sklearn.metrics.r2_score(y_true=true.mean(axis=0), y_pred=pred.mean(axis=0))
            else:
                raise ValueError

            values.append(r2)

            if debug:
                return true, pred, r2

        return np.mean(values)

    # TODO: implement distentenglement measures
    # How do you define disentenglement. Look at papers.
    # Intuitive idea: run sklearn classifiers on the data, test their accuracy, with and without tardis loss
    #     for reserved and unreserved latent spaces
    # AWS, LISI etc for reserved and unreserved latent given label is the target metadata
