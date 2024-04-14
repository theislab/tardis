#!/usr/bin/env python3

import logging
from collections.abc import Sequence
from typing import Optional

import torch
from anndata import AnnData
from scvi.utils import unsupported_if_adata_minified

logger = logging.getLogger(__name__)


class MetricsMixin:

    @torch.inference_mode()
    @unsupported_if_adata_minified
    def get_reconstruction_r2(
        self,
        adata: AnnData,
        indices: Optional[Sequence[int]] = None,
        batch_size: Optional[int] = None,
        with_differentially_expressed_genes: bool = False,
    ) -> float:
        # TODO: Calculate DEG during preprocessing.
        # TODO: filter low quality genes during preprocessing.
        # sc.pp.filter_cells(adata, min_genes=10, inplace=True)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(adata=adata, indices=indices, batch_size=batch_size)  # noqa
        raise NotImplementedError

    # How do you define disentenglement. Look at papers.

    # Intuitive idea: run sklearn classifiers on the data, test their accuracy, with and without tardis loss
    #     for reserved and unreserved latent spaces

    # AWS, LISI etc for reserved and unreserved latent given label is the target metadata
