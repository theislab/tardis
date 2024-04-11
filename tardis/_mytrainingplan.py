#!/usr/bin/env python3

import copy

import torch
from scvi.module.base import LossOutput
from scvi.train import TrainingPlan
from scvi.train._metrics import ElboMetric

from ._progressbarmanager import ProgressBarManager
from ._trainingsteplogger import TrainingStepLogger


class MyTrainingPlan(TrainingPlan):

    def forward(self, *args, **kwargs):
        TrainingStepLogger.set_step(key="gglobal", value=copy.deepcopy(self.global_step))
        TrainingStepLogger.increment_step(key="forward")
        return self.module(*args, **kwargs)

    def training_step(self, *args, **kwargs):
        TrainingStepLogger.increment_step(key="training")
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args, **kwargs):
        TrainingStepLogger.increment_step(key="validation")
        return super().validation_step(*args, **kwargs)

    def test_step(self, *args, **kwargs):
        TrainingStepLogger.increment_step(key="test")
        return super().test_step(*args, **kwargs)

    def predict_step(self, *args, **kwargs):
        TrainingStepLogger.increment_step(key="predict")
        return super().predict_step(*args, **kwargs)

    def is_key_should_be_in_progress_bar(self, key_with_mode, mode):
        key, _ = key_with_mode.rsplit(f"_{mode}", 1)
        if len(key) == 0:
            raise ValueError("Key cannot be empty")

        if key in ProgressBarManager.keys and mode in ProgressBarManager.modes:
            return True

        return False

    def training_step(self, batch, batch_idx):  # noqa
        if "kl_weight" in self.loss_kwargs:
            kl_weight = self.kl_weight
            self.loss_kwargs.update({"kl_weight": kl_weight})
            self.log("kl_weight", kl_weight, on_step=True, on_epoch=False)
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.compute_and_log_metrics(scvi_loss, self.train_metrics, "train")
        return scvi_loss.loss

    def validation_step(self, batch, batch_idx):  # noqa
        _, _, scvi_loss = self.forward(batch, loss_kwargs=self.loss_kwargs)
        self.compute_and_log_metrics(scvi_loss, self.val_metrics, "validation")

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
