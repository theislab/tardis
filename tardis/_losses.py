from collections import defaultdict
import torch

from .losses import LOSSES
from ._myconstants import LOSS_NAMING_DELIMITER
from ._myconstants import LOSS_NAMING_PREFIX
from ._myconstants import LATENT_INDEX_GROUP_NAMES
from ._progressbarmanager import ProgressBarManager
from ._trainingsteplogger import TrainingEpochLogger


class LossesBase:
    def __init__(self, loss_configs, obs_key):
        self._warmup_periods = defaultdict(list)

        self.reserved = []
        self.unreserved = []
        self.complete = []

        for loss_config in loss_configs:
            loss_config = self._validate_loss_config(loss_config)
            loss_cls = LOSSES[loss_config["method"]]
            loss_obj = loss_cls(
                weight=loss_config["weight"],
                is_minimized=loss_config["is_minimized"],
                transformation=loss_config["transformation"],
                method_kwargs=loss_config["method_kwargs"],
            )

            loss_type = loss_config["loss_type"]

            if loss_type == "reserved":
                self.reserved.append(loss_obj)
            elif loss_type == "unreserved":
                self.unreserved.append(loss_obj)
            else:
                self.complete.append(loss_obj)

            identifier = LOSS_NAMING_DELIMITER.join(
                [
                    LOSS_NAMING_PREFIX,
                    obs_key,
                    loss_type,
                    loss_cls.__name__.lower(),
                ]
            )

            self.obs_key = obs_key

            if loss_config["progress_bar"]:
                ProgressBarManager.add(identifier)

            self._warmup_periods[loss_type].append(loss_config["warmup_period"])

    def _validate_loss_config(self, loss_config):

        if not isinstance(loss_config, dict):
            raise ValueError("loss_config should be a dictionary.")

        method = loss_config.get("method", None)
        weight = loss_config.get("weight", 1.0)
        transformation = loss_config.get("transformation", "identity")
        progress_bar = loss_config.get("progress_bar", False)
        method_kwargs = loss_config.get("method_kwargs", {})
        loss_type = loss_config.get("type", "complete")
        is_minimized = loss_config.get("is_minimized", True)
        warmup_period = loss_config.get("warmup_period", [0, 0])

        if method is None:
            raise ValueError("loss_config should have a key 'method'.")

        if not isinstance(method, str):
            raise ValueError("loss_config['method'] should be a string.")
        if not isinstance(weight, (int, float)):
            raise ValueError("loss_config['weight'] should be an int or a float")
        if not isinstance(transformation, str):
            raise ValueError("loss_config['transformation'] should be a string.")
        if not isinstance(progress_bar, bool):
            raise ValueError("loss_config['progress_bar'] should be a boolean.")
        if not isinstance(method_kwargs, dict):
            raise ValueError("loss_config['method_kwargs'] should be a dictionary.")
        if not isinstance(loss_type, str):
            raise ValueError("loss_config['type'] should be a string.")
        if loss_type not in set(LATENT_INDEX_GROUP_NAMES):
            raise ValueError(
                f"loss_config['type'] should be one of {LATENT_INDEX_GROUP_NAMES}"
            )

        return dict(
            method=method,
            weight=weight,
            transformation=transformation,
            progress_bar=progress_bar,
            method_kwargs=method_kwargs,
            loss_type=loss_type,
            is_minimized=is_minimized,
            warmup_period=warmup_period,
        )

    def get_warmup_weight(self, warmup_period):
        epoch = TrainingEpochLogger.current
        start, end = warmup_period
        if epoch < start:
            return 0.0
        elif epoch > end:
            return 1.0
        else:
            return (epoch - start) / (end + 1 - start)

    def get_total_loss(
        self,
        outputs,
        counteractive_positive_outputs,
        counteractive_negative_outputs,
        indices,
        *coefficients,
    ):
        total_loss = [{}, {}]

        for loss_type in LATENT_INDEX_GROUP_NAMES:
            loss_fns = getattr(self, loss_type)
            relavant_indices = getattr(indices, loss_type)
            warmup_periods = self._warmup_periods[loss_type]

            for warmup_period, loss_fn in zip(warmup_periods, loss_fns):
                identifier = LOSS_NAMING_DELIMITER.join(
                    [
                        LOSS_NAMING_PREFIX,
                        self.obs_key,
                        loss_type,
                        type(loss_fn).__name__.lower(),
                    ]
                )
                if loss_fn is loss_fn.is_minimized:
                    counteractive_outputs = counteractive_positive_outputs
                else:
                    counteractive_outputs = counteractive_negative_outputs

                index = int(loss_fn.is_minimized)
                warmup_weight = self.get_warmup_weight(warmup_period)
                loss = loss_fn.forward(outputs, counteractive_outputs, relavant_indices)
                total_loss[index][identifier] = (
                    warmup_weight * coefficients[index] * loss
                )

        return total_loss


class Losses(LossesBase):

    def get_total_loss(
        self,
        outputs,
        counteractive_positive_outputs,
        counteractive_negative_outputs,
        indices,
        *coefficients,
    ):
        positive_losses, negative_losses = super().get_total_loss(
            outputs,
            counteractive_positive_outputs,
            counteractive_negative_outputs,
            indices,
            coefficients,
        )
        positive_losses.update(negative_losses)
        return positive_losses


class Triplets(LossesBase):

    def get_total_loss(
        self,
        outputs,
        counteractive_positive_outputs,
        counteractive_negative_outputs,
        indices,
        *coefficients,
    ):
        positive_losses, negative_losses = super().get_total_loss(
            outputs,
            counteractive_positive_outputs,
            counteractive_negative_outputs,
            indices,
            coefficients,
        )

        positive = torch.mean(torch.cat([v for v in positive_losses.values()]))
        negative = torch.mean(torch.cat([v for v in negative_losses.values()]))
        loss = torch.max(positive - negative, torch.zeros_like(positive))
        concatenated_keys = [
            k.split(LOSS_NAMING_DELIMITER)[-1]
            for k in set(positive_losses.keys()).union(negative_losses.keys())
        ].join(LOSS_NAMING_DELIMITER)
        identifier = LOSS_NAMING_DELIMITER.join(
            [
                LOSS_NAMING_PREFIX,
                self.obs_key,
                "triplet",
                *concatenated_keys,
            ]
        )
        return {identifier: loss}
