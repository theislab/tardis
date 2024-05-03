#!/usr/bin/env python3

from __future__ import annotations

import copy
import logging
import os
import warnings

from anndata import AnnData
from scvi import REGISTRY_KEYS, settings
from scvi.data import AnnDataManager
from scvi.data._utils import _get_adata_minify_type
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from scvi.model._utils import _init_library_size
from scvi.model.base import ArchesMixin, BaseModelClass, RNASeqMixin, VAEMixin

from ._counteractivegenerator import CachedPossibleGroupDefinitionIndices
from ._disentanglement import Disentanglements
from ._disentanglementmanager import DisentanglementManager
from ._metricsmixin import MetricsMixin
from ._modelplotting import ModelPlotting
from ._myconstants import MODEL_NAME, REGISTRY_KEY_DISENTANGLEMENT_TARGETS, REGISTRY_KEY_METRICS_COVARIATES_HELPER
from ._mymodule import MyModule
from ._mymonitor import (
    AuxillaryLossWarmupManager,
    ModelLevelMetrics,
    ProgressBarManager,
    TrainingEpochLogger,
    TrainingStepLogger,
)
from ._mytrainingmixin import MyUnsupervisedTrainingMixin
from ._utils.functions import categorical_covariate_validator
from ._utils.wandb import check_wandb_configurations
from ._utils.warnings import ignore_predetermined_warnings

logger = logging.getLogger(__name__)


class MyModel(
    RNASeqMixin, VAEMixin, ArchesMixin, MyUnsupervisedTrainingMixin, BaseModelClass, ModelPlotting, MetricsMixin
):
    """Tardis model"""

    _module_cls = MyModule
    # Keep the original AnndataLoader for everything else other than training.
    # This causes to miss the counteractive minibatch generation for other things.
    _data_loader_cls = AnnDataLoader

    def __init__(self, adata: AnnData, **kwargs):
        super().__init__(adata)

        if self._module_init_on_train:
            raise ValueError("The model currently does not support initialization without data.")

        self._module_kwargs = {**kwargs}
        self._model_summary_string = f"{MODEL_NAME} model"

        n_cats_per_cov = (
            self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        n_cats_per_disentenglement_covariates = (
            self.adata_manager.get_state_registry(REGISTRY_KEY_DISENTANGLEMENT_TARGETS).n_cats_per_key
            if REGISTRY_KEY_DISENTANGLEMENT_TARGETS in self.adata_manager.data_registry
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
            use_size_factor_key=use_size_factor_key,
            library_log_means=library_log_means,
            library_log_vars=library_log_vars,
            n_cats_per_disentenglement_covariates=n_cats_per_disentenglement_covariates,
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
        model_level_metrics: list[dict] | None = None,
        model_level_metrics_helper_covariates: list[str] | None = None,
        **kwargs,
    ):
        setup_method_args = cls._get_setup_method_args(**locals())

        TrainingStepLogger.reset()
        TrainingEpochLogger.reset()
        AuxillaryLossWarmupManager.reset()
        ProgressBarManager.reset()
        DisentanglementManager.reset()
        CachedPossibleGroupDefinitionIndices.reset()
        ModelLevelMetrics.reset()

        ModelLevelMetrics.add(model_level_metrics if model_level_metrics is not None else [])
        if disentenglement_targets_configurations is None:
            disentenglement_targets_configurations = []
        # This also checks whether the dict follows the format required.
        disentenglement_targets_configurations = Disentanglements(items=disentenglement_targets_configurations)

        _dtsak = disentenglement_targets_configurations.get_ordered_obs_key()
        disentenglement_targets_setup_anndata_keys = _dtsak if len(_dtsak) > 0 else None
        categorical_covariate_validator(categorical_covariate_keys, disentenglement_targets_setup_anndata_keys)

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False),
            CategoricalJointObsField(REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys),
            NumericalJointObsField(REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys),
            CategoricalJointObsField(REGISTRY_KEY_DISENTANGLEMENT_TARGETS, disentenglement_targets_setup_anndata_keys),
            CategoricalJointObsField(REGISTRY_KEY_METRICS_COVARIATES_HELPER, model_level_metrics_helper_covariates),
        ]
        adata_minify_type = _get_adata_minify_type(adata)
        assert adata_minify_type is None, f"{MODEL_NAME} model currently does not support minified data."
        adata_manager = AnnDataManager(fields=anndata_fields, setup_method_args=setup_method_args)
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

        DisentanglementManager.set_configurations(value=disentenglement_targets_configurations)
        DisentanglementManager.set_anndata_manager_state_registry(
            value={
                registry_key: adata_manager.registry["field_registries"][registry_key]["state_registry"]
                for registry_key in adata_manager.registry["field_registries"]
                if registry_key != REGISTRY_KEYS.X_KEY
            }
        )

        if disentenglement_targets_setup_anndata_keys is not None and (
            batch_key is not None or categorical_covariate_keys is not None
        ):
            warnings.warn(
                message=(
                    f"`{batch_key, categorical_covariate_keys}` is defined as `batch_key` or "
                    "`categorical_covariate`, it will be given to both encoder and decoder. "
                    "Make sure this does not contain information of any of your disentenglement targets. For "
                    "example, if `donor_id` is chosen as a batch key, do not disentengle donor level information "
                    "such as `sex` or `age`. The decoder should not use the disentengled latent spaces, simply "
                    "ignores, if it is already given a batch_key. "
                ),
                category=UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )

    @classmethod
    def setup_wandb(
        cls,
        wandb_configurations: dict,
        hyperparams: dict | None = None,
        check_credientials: bool = False,
        verbose: bool = True,
    ):
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
            cls.wandb_logger_verbose = verbose
