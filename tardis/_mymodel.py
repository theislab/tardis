#!/usr/bin/env python3

from __future__ import annotations

import copy
import logging
import os
from typing import Literal

from anndata import AnnData
from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model._utils import _init_library_size
from scvi.model.base import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin

from ._disentenglementtargetconfigurations import DisentenglementTargetConfigurations
from ._disentenglementtargetmanager import DisentenglementTargetManager
from ._myconstants import MODEL_NAME, REGISTRY_KEY_DISENTENGLEMENT_TARGETS
from ._mymodule import MyModule
from ._mytrainingmixin import MyUnsupervisedTrainingMixin
from ._utils.wandb import check_wandb_configurations
from ._utils.warnings import ignore_predetermined_warnings

logger = logging.getLogger(__name__)


class MyModel(
    RNASeqMixin,
    VAEMixin,
    ArchesMixin,
    MyUnsupervisedTrainingMixin,
    BaseModelClass,
):
    """single-cell Variational Inference :cite:p:`Lopez18`.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`. If
        ``None``, then the underlying module will not be initialized until training, and a
        :class:`~lightning.pytorch.core.LightningDataModule` must be passed in during training
        (``EXPERIMENTAL``).
    n_hidden
        Number of nodes per hidden layer.
    n_latent
        Dimensionality of the latent space.
    n_layers
        Number of hidden layers used for encoder and decoder NNs.
    dropout_rate
        Dropout rate for neural networks.
    dispersion
        One of the following:

        * ``'gene'`` - dispersion parameter of NB is constant per gene across cells
        * ``'gene-batch'`` - dispersion can differ between different batches
        * ``'gene-label'`` - dispersion can differ between different labels
        * ``'gene-cell'`` - dispersion can differ for every gene in every cell
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    **kwargs
        Additional keyword arguments for :class:`~_mymodule.MyModule`.
    """

    _module_cls = MyModule

    def __init__(
        self,
        adata: AnnData | None = None,
        n_hidden: int = 128,
        n_latent: int = 10,
        n_layers: int = 1,
        dropout_rate: float = 0.1,
        dispersion: Literal["gene", "gene-batch", "gene-label", "gene-cell"] = "gene",
        gene_likelihood: Literal["zinb", "nb", "poisson"] = "nb",
        latent_distribution: Literal["normal", "ln"] = "normal",
        **kwargs,
    ):
        super().__init__(adata)
        assert not self._module_init_on_train, "The model currently does not support initialization without data."

        self._module_kwargs = {
            "n_hidden": n_hidden,
            "n_latent": n_latent,
            "n_layers": n_layers,
            "dropout_rate": dropout_rate,
            "dispersion": dispersion,
            "gene_likelihood": gene_likelihood,
            "latent_distribution": latent_distribution,
            **kwargs,
        }
        self._model_summary_string = (
            f"{MODEL_NAME} model with the following parameters: \n"
            f"n_hidden: {n_hidden}, n_latent: {n_latent}, n_layers: {n_layers}, "
            f"dropout_rate: {dropout_rate}, dispersion: {dispersion}, "
            f"gene_likelihood: {gene_likelihood}, latent_distribution: {latent_distribution}."
        )

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_batch = self.summary_stats.n_batch
        use_size_factor_key = REGISTRY_KEYS.SIZE_FACTOR_KEY in self.adata_manager.data_registry
        library_log_means, library_log_vars = None, None
        if not use_size_factor_key:
            library_log_means, library_log_vars = _init_library_size(self.adata_manager, n_batch)

        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_batch=n_batch,
            n_labels=self.summary_stats.n_labels,
            n_continuous_cov=self.summary_stats.get("n_extra_continuous_covs", 0),
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent=n_latent,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            dispersion=dispersion,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            **kwargs,
        )

        self.init_params_ = self._get_init_params(locals())

    @classmethod
    def setup_anndata(
        cls,
        adata: AnnData,
        layer: str | None = None,
        batch_key: str | None = None,
        labels_key: str | None = None,
        size_factor_key: str | None = None,
        categorical_covariate_keys: list[str] | None = None,
        continuous_covariate_keys: list[str] | None = None,
        disentenglement_targets_configurations: list[dict] | None = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())

        if disentenglement_targets_configurations is None:
            disentenglement_targets_configurations = []
        # This also checks whether the dict follows the format required.
        disentenglement_targets_configurations = DisentenglementTargetConfigurations(
            items=disentenglement_targets_configurations
        )

        _dtsak = disentenglement_targets_configurations.get_ordered_obs_key()
        disentenglement_targets_setup_anndata_keys = _dtsak if len(_dtsak) > 0 else None

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            CategoricalJointObsField(REGISTRY_KEY_DISENTENGLEMENT_TARGETS, disentenglement_targets_setup_anndata_keys),
        ]
        adata_minify_type = _get_adata_minify_type(adata)
        assert adata_minify_type is None, f"{MODEL_NAME} model currently does not support minified data."
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

        DisentenglementTargetManager.set_configurations(value=disentenglement_targets_configurations)
        DisentenglementTargetManager.set_anndata_manager_state_registry(
            value={
                registry_key: adata_manager.registry["field_registries"][registry_key]["state_registry"]
                for registry_key in adata_manager.registry["field_registries"]
                if registry_key != REGISTRY_KEYS.X_KEY
            }
        )

    @classmethod
    def setup_wandb(cls, wandb_configurations: dict, hyperparams: dict | None = None, check_credientials: bool = False):
        if hasattr(cls, "wandb_logger"):
            assert (
                cls.wandb_logger.experiment._is_finished
            ), "Multiple W&B initialization may cause unexpected behaviours!"

        # Import necessary libraries
        from pytorch_lightning.loggers import WandbLogger

        if check_credientials:
            check_wandb_configurations(wandb_configurations=wandb_configurations)

        # add these values to the environment
        if wandb_configurations["environment_variables"] is not None:
            for k, v in wandb_configurations["environment_variables"].items():
                os.environ[k] = v

        # save wandb_configurations into the hyperparameters list
        hyperparams = dict() if hyperparams is None else hyperparams
        assert "wandb_configurations" not in hyperparams.keys()
        hyperparams["wandb_configurations"] = copy.deepcopy(wandb_configurations)

        # Create a WandbLogger using the run object
        with ignore_predetermined_warnings():
            cls.wandb_logger = WandbLogger(
                config=hyperparams, **wandb_configurations["wandblogger_kwargs"]
            )  # cls.wandb_logger.experiment.id
