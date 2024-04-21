#!/usr/bin/env python3

from collections import defaultdict

import torch
from scvi import REGISTRY_KEYS

from ._myconstants import (
    LATENT_INDEX_GROUP_NAMES,
    LOSS_NAMING_DELIMITER,
    LOSS_NAMING_PREFIX,
    REGISTRY_KEY_DISENTANGLEMENT_TARGETS,
)
from ._progressbarmanager import ProgressBarManager
from ._trainingsteplogger import TrainingEpochLogger
from .losses import LOSSES


def isnumeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


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

            latent_group = loss_config["latent_group"]

            if latent_group == "reserved":
                self.reserved.append(loss_obj)
            elif latent_group == "unreserved":
                self.unreserved.append(loss_obj)
            else:
                self.complete.append(loss_obj)

            identifier = LOSS_NAMING_DELIMITER.join(
                [
                    LOSS_NAMING_PREFIX,
                    obs_key,
                    latent_group,
                    loss_cls.__name__.lower(),
                ]
            )

            self.obs_key = obs_key

            if loss_config["progress_bar"]:
                ProgressBarManager.add(identifier)

            self._warmup_periods[latent_group].append(loss_config["warmup_period"])

    def _validate_loss_config(self, loss_config):

        if not isinstance(loss_config, dict):
            raise ValueError("loss_config should be a dictionary.")

        method = loss_config.get("method", None)
        weight = loss_config.get("weight", 1.0)
        transformation = loss_config.get("transformation", "identity")
        progress_bar = loss_config.get("progress_bar", False)
        method_kwargs = loss_config.get("method_kwargs", {})
        latent_group = loss_config.get("latent_group", "complete")
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
        if not isinstance(latent_group, str):
            raise ValueError("loss_config['type'] should be a string.")
        if latent_group not in set(LATENT_INDEX_GROUP_NAMES):
            raise ValueError(f"loss_config['type'] should be one of {LATENT_INDEX_GROUP_NAMES}")

        return dict(
            method=method,
            weight=weight,
            transformation=transformation,
            progress_bar=progress_bar,
            method_kwargs=method_kwargs,
            latent_group=latent_group,
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

    def get_inputs(self, inputs, positive_inputs, negative_inputs, target_type, pseudo_categories):

        device = inputs[REGISTRY_KEYS.X_KEY].device

        if target_type == "pseudo_categorical":

            _inputs = inputs[REGISTRY_KEY_DISENTANGLEMENT_TARGETS][:, self.index].detach().cpu().numpy()
            _positive_inputs = None
            _negative_inputs = (
                negative_inputs[REGISTRY_KEY_DISENTANGLEMENT_TARGETS][:, self.index].detach().cpu().numpy()
            )

            _inputs = torch.tensor(pseudo_categories(_inputs), device=device)
            _negative_inputs = torch.tensor(pseudo_categories(_negative_inputs), device=device)
        else:
            _inputs = inputs[REGISTRY_KEYS.X_KEY]
            _positive_inputs = positive_inputs[REGISTRY_KEYS.X_KEY]
            _negative_inputs = negative_inputs[REGISTRY_KEYS.X_KEY]

        return _inputs, _positive_inputs, _negative_inputs

    def get_total_loss(
        self,
        inputs,
        positive_inputs,
        negative_inputs,
        outputs,
        counteractive_positive_outputs,
        counteractive_negative_outputs,
        indices,
        pseudo_categories,
    ):
        weighted_total_loss = [{}, {}]
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

                _inputs, _positive_inputs, _negative_inputs = self.get_inputs(
                    inputs,
                    positive_inputs,
                    negative_inputs,
                    loss_fn.target_type,
                    pseudo_categories,
                )
                if loss_fn is loss_fn.is_minimized:
                    counteractive_outputs = counteractive_positive_outputs
                    coefficients = loss_fn.get_coefficients(_inputs, _positive_inputs)
                else:
                    counteractive_outputs = counteractive_negative_outputs
                    coefficients = loss_fn.get_coefficients(_inputs, _negative_inputs)

                index = int(loss_fn.is_minimized)
                warmup_weight = self.get_warmup_weight(warmup_period)
                loss = loss_fn.forward(outputs, counteractive_outputs, relavant_indices)
                total_loss[index][identifier] = coefficients * loss
                weighted_total_loss[index][identifier] = total_loss[index][identifier] * warmup_weight
        return weighted_total_loss, total_loss


class Losses(LossesBase):

    def get_total_loss(
        self,
        inputs,
        positive_inputs,
        negative_inputs,
        outputs,
        counteractive_positive_outputs,
        counteractive_negative_outputs,
        indices,
        pseudo_categories,
    ):
        (weighted_positive_losses, weighted_negative_losses), (
            positive_losses,
            negative_losses,
        ) = super().get_total_loss(
            inputs,
            positive_inputs,
            negative_inputs,
            outputs,
            counteractive_positive_outputs,
            counteractive_negative_outputs,
            indices,
            pseudo_categories,
        )
        positive_losses.update(negative_losses)
        weighted_positive_losses.update(weighted_negative_losses)

        return weighted_positive_losses, positive_losses


class Triplets(LossesBase):

    def get_total_loss(
        self,
        inputs,
        positive_inputs,
        negative_inputs,
        outputs,
        counteractive_positive_outputs,
        counteractive_negative_outputs,
        indices,
        pseudo_categories,
    ):
        (weighted_positive_losses, weighted_negative_losses), (
            positive_losses,
            negative_losses,
        ) = super().get_total_loss(
            inputs,
            positive_inputs,
            negative_inputs,
            outputs,
            counteractive_positive_outputs,
            counteractive_negative_outputs,
            indices,
            pseudo_categories,
        )

        positive = torch.mean(torch.cat([v for v in positive_losses.values()]))
        negative = torch.mean(torch.cat([v for v in negative_losses.values()]))

        weighted_positive = torch.mean(torch.cat([v for v in weighted_positive_losses.values()]))
        weighted_negative = torch.mean(torch.cat([v for v in weighted_negative_losses.values()]))

        loss = torch.max(positive - negative, torch.zeros_like(positive))
        weighted_loss = torch.max(weighted_positive - weighted_negative, torch.zeros_like(weighted_positive))

        concatenated_keys = [
            k.split(LOSS_NAMING_DELIMITER)[-1] for k in set(positive_losses.keys()).union(negative_losses.keys())
        ].join(LOSS_NAMING_DELIMITER)

        identifier = LOSS_NAMING_DELIMITER.join(
            [
                LOSS_NAMING_PREFIX,
                self.obs_key,
                "triplet",
                *concatenated_keys,
            ]
        )

        return {identifier: weighted_loss}, {identifier: loss}
