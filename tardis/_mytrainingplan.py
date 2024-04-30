#!/usr/bin/env python3

import copy

import numpy as np
import sklearn
import torch
from scvi import REGISTRY_KEYS
from scvi.module.base import LossOutput
from scvi.train import TrainingPlan
from scvi.train._metrics import ElboMetric

from ._disentanglementmanager import DisentanglementManager as DM
from ._metricsmixin import MetricsMixin
from ._myconstants import REGISTRY_KEY_METRICS_COVARIATES_HELPER
from ._mymonitor import ModelLevelMetrics, ProgressBarManager, TrainingEpochLogger, TrainingStepLogger
from ._utils.functions import create_random_mask


class MyTrainingPlan(TrainingPlan):

    _metrics_tensors = [
        "z",
        REGISTRY_KEYS.X_KEY,
        REGISTRY_KEYS.BATCH_KEY,
        REGISTRY_KEYS.LABELS_KEY,
        REGISTRY_KEY_METRICS_COVARIATES_HELPER,
    ]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.train_step_outputs: dict
        self.validation_step_outputs: dict
        self.train_model_level_metric_switch: bool
        self.validation_model_level_metric_switch: bool
        self._reset_step_outputs(mode="train")
        self._reset_step_outputs(mode="validation")

    def forward(self, *args, **kwargs):
        TrainingStepLogger.set_step(key="gglobal", value=copy.deepcopy(self.global_step))
        TrainingEpochLogger.set_epoch(key="current", value=self.current_epoch)
        TrainingStepLogger.increment_step(key="forward")
        return self.module(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        TrainingStepLogger.increment_step(key="test")
        return super().test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        TrainingStepLogger.increment_step(key="predict")
        return super().predict_step(*args, **kwargs)

    def _combine_step_outputs(self, mode):
        if mode == "train":
            return {i: torch.vstack(v).detach().cpu().numpy() for i, v in self.train_step_outputs.items()}
        elif mode == "validation":
            return {i: torch.vstack(v).detach().cpu().numpy() for i, v in self.validation_step_outputs.items()}

    def _reset_step_outputs(self, mode):
        if mode == "train":
            self.train_step_outputs = dict()
        elif mode == "validation":
            self.validation_step_outputs = dict()
        else:
            raise ValueError

    def _calculate_model_level_metrics(self, mode):
        step_outputs = self._combine_step_outputs(mode=mode)

        for metric_identifier, metric_settings in ModelLevelMetrics.items[mode].items():
            # subsample using metric_settings["subsample"]
            # if else statement for the metric, using staticmethos in the metricsmixin

            if metric_identifier == "demo":
                metric = MetricsMixin.get_demo_metric(
                    t1=step_outputs[REGISTRY_KEYS.BATCH_KEY], t2=step_outputs[REGISTRY_KEYS.LABELS_KEY]
                )

            elif metric_identifier.startswith("metric_mi|"):

                obs_key = metric_identifier.split("metric_mi|")[1]
                err_message = (
                    f"To calculate `{metric_identifier}`, `{obs_key}` key should be but "
                    "in `model_level_metrics_helper_covariates` in `setup_anndata`."
                )
                if REGISTRY_KEY_METRICS_COVARIATES_HELPER not in step_outputs:
                    raise ValueError(err_message)
                field_keys = DM.anndata_manager_state_registry[REGISTRY_KEY_METRICS_COVARIATES_HELPER]["field_keys"]
                if obs_key not in field_keys:
                    raise ValueError(err_message)
                obs_ind = field_keys.index(obs_key)

                if (
                    "latent_subset" not in metric_settings["metric_kwargs"]
                    or "reduce" not in metric_settings["metric_kwargs"]
                ):
                    raise ValueError(
                        "`latent_subset` should be one of the parameter of `metric_kwargs` "
                        f"for `{metric_identifier}` metric."
                    )
                metric_kwargs = copy.deepcopy(metric_settings["metric_kwargs"])
                latent_subset = metric_kwargs.pop("latent_subset")
                reduce = metric_kwargs.pop("reduce")
                obs_labels = step_outputs[REGISTRY_KEY_METRICS_COVARIATES_HELPER][:, obs_ind]

                data = step_outputs["z"][:, latent_subset] if latent_subset is not None else step_outputs["z"]
                factors = sklearn.preprocessing.LabelEncoder().fit_transform(obs_labels.flatten())
                factors = np.broadcast_to(np.expand_dims(factors, axis=1), data.shape)

                mask = create_random_mask(
                    shape=data.shape[0], ratio_true=metric_settings["subsample"], seed=TrainingStepLogger.forward
                )

                metric = MetricsMixin.get_MI_precalculated(
                    data=data[mask, :], factors=factors[mask, :], **metric_kwargs
                )
                metric = reduce(metric)

            else:
                raise NotImplementedError

            self.log(
                name=f"{metric_identifier}_{mode}",
                value=metric,
                on_step=False,
                on_epoch=True,
                prog_bar=self.is_key_should_be_in_progress_bar(f"{metric_identifier}_{mode}", mode),
                sync_dist=self.use_sync_dist,
            )

    def on_train_epoch_end(self) -> None:
        if not self.train_model_level_metric_switch:
            return
        self._calculate_model_level_metrics(mode="train")
        self._reset_step_outputs(mode="train")

    def on_validation_epoch_end(self) -> None:
        if not self.validation_model_level_metric_switch:
            return
        self._calculate_model_level_metrics(mode="validation")
        self._reset_step_outputs(mode="validation")

    def on_train_epoch_start(self) -> None:
        self.train_model_level_metric_switch = False
        for _, metric_settings in ModelLevelMetrics.items["train"].items():
            if (self.current_epoch + 1) % metric_settings["every_n_epoch"] == 0:
                self.train_model_level_metric_switch = True

    def on_validation_epoch_start(self) -> None:
        self.validation_model_level_metric_switch = False
        for _, metric_settings in ModelLevelMetrics.items["validation"].items():
            if (self.current_epoch + 1) % metric_settings["every_n_epoch"] == 0:
                self.validation_model_level_metric_switch = True

    def training_step(self, batch, batch_idx):  # noqa
        if "kl_weight" in self.loss_kwargs:
            kl_weight = self.kl_weight
            self.loss_kwargs.update({"kl_weight": kl_weight})
            self.log("kl_weight", kl_weight, on_step=True, on_epoch=False)
        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, mode="train")
        TrainingStepLogger.increment_step(key="training")

        if self.train_model_level_metric_switch:
            for d in [batch, inference_outputs]:
                for k, v in d.items():
                    if k not in self._metrics_tensors:
                        continue
                    if k not in self.train_step_outputs:
                        self.train_step_outputs[k] = list()
                    self.train_step_outputs[k].append(v)

        return scvi_loss.loss

    def validation_step(self, batch, batch_idx):  # noqa
        inference_outputs, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, mode="validation")
        TrainingStepLogger.increment_step(key="validation")

        if self.validation_model_level_metric_switch:
            for d in [batch, inference_outputs]:
                for k, v in d.items():
                    if k not in self._metrics_tensors:
                        continue
                    if k not in self.validation_step_outputs:
                        self.validation_step_outputs[k] = list()
                    self.validation_step_outputs[k].append(v)

    @torch.inference_mode()
    def compute_and_log_metrics(
        self, loss_output: LossOutput, metrics: dict[str, ElboMetric], mode: str, report_step_total_loss: bool = False
    ):
        rec_loss = loss_output.reconstruction_loss_sum
        n_obs_minibatch = loss_output.n_obs_minibatch
        kl_local = loss_output.kl_local_sum
        kl_global = loss_output.kl_global_sum

        self.log(
            f"total_loss_{mode}",
            loss_output.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=self.is_key_should_be_in_progress_bar(f"total_loss_{mode}", mode),
            sync_dist=self.use_sync_dist,
        )

        if report_step_total_loss:
            self.log(
                f"total_loss_step_{mode}",
                loss_output.loss,
                on_step=True,
                on_epoch=False,
                prog_bar=False,
                sync_dist=self.use_sync_dist,
            )

        metrics[f"elbo_{mode}"].update(
            reconstruction_loss=rec_loss,
            kl_local=kl_local,
            kl_global=kl_global,
            n_obs_minibatch=n_obs_minibatch,
        )
        # pytorch lightning handles everything with the torchmetric object
        self.log_dict(
            {k: v for k, v in metrics.items() if not self.is_key_should_be_in_progress_bar(k, mode)},
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            batch_size=n_obs_minibatch,
            sync_dist=self.use_sync_dist,
        )
        self.log_dict(
            {k: v for k, v in metrics.items() if self.is_key_should_be_in_progress_bar(k, mode)},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=n_obs_minibatch,
            sync_dist=self.use_sync_dist,
        )

        # accumlate extra metrics passed to loss recorder
        for key in loss_output.extra_metrics_keys:
            met = loss_output.extra_metrics[key]
            if isinstance(met, torch.Tensor):
                if met.shape != torch.Size([]):
                    raise ValueError("Extra tracked metrics should be 0-d tensors.")
                met = met.detach()
            self.log(
                f"{key}_{mode}",
                met,
                on_step=False,
                on_epoch=True,
                prog_bar=self.is_key_should_be_in_progress_bar(f"{key}_{mode}", mode),
                batch_size=n_obs_minibatch,
                sync_dist=self.use_sync_dist,
            )

    def is_key_should_be_in_progress_bar(self, key_with_mode, mode):
        key, _ = key_with_mode.rsplit(f"_{mode}", 1)
        if len(key) == 0:
            raise ValueError("Key cannot be empty")

        if key in ProgressBarManager.keys and mode in ProgressBarManager.modes:
            return True

        return False
