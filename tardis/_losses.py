#!/usr/bin/env python3

from collections import defaultdict

import torch
from scvi import REGISTRY_KEYS

from ._loss import LOSS
from ._myconstants import (
    LATENT_INDEX_GROUP_COMPLETE,
    LATENT_INDEX_GROUP_NAMES,
    LATENT_INDEX_GROUP_RESERVED,
    LATENT_INDEX_GROUP_UNRESERVED,
    LATENT_INDEX_GROUP_UNRESERVED_COMPLETE,
    LOSS_NAMING_DELIMITER,
    LOSS_NAMING_PREFIX,
    REGISTRY_KEY_DISENTANGLEMENT_TARGETS,
    TARGET_TYPES,
    TARGET_TYPE_PSEUDO_CATEGORICAL,
    TARGET_TYPE_CATEGORICAL
)
from ._mymonitors import ProgressBarManager, TrainingEpochLogger


class LossesBase:
    
    def __init__(self, loss_configs, obs_key):

        for loss_config_index, loss_config in enumerate(loss_configs):
            
            loss_config = self._validate_loss_config(loss_config)
            loss_cls = LOSS[loss_config["method"]]
            loss_obj = loss_cls(
                weight=loss_config["weight"],
                is_minimized=loss_config["is_minimized"],
                transformation=loss_config["transformation"],
                method_kwargs=loss_config["method_kwargs"],
            )
            

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

    def _validate_loss_config(self, loss_config):

        if not isinstance(loss_config, dict):
            raise ValueError("loss_config should be a dictionary.")

        # Define the required keys and their expected types
        required_keys = {
            "method": str,
            "weight": (int, float),
            "transformation": str,
            "progress_bar": bool,
            "method_kwargs": dict,
            "latent_group": str,  # Assuming LATENT_INDEX_GROUP_COMPLETE is an integer constant
            "is_minimized": bool,
            "warmup_period": (list, type(None)),
            "target_type": str,
            "non_categorical_coefficient": (str, type(None))
        }

        # Check for the presence and type of each required key
        for key, expected_type in required_keys.items():
            if key not in loss_config:
                raise KeyError(f"The required key '{key}' is missing from loss_config.")
            if not isinstance(loss_config[key], expected_type):
                raise TypeError(f"The key '{key}' in loss_config must be of type {expected_type.__name__}.")
        
        # Assert that there are no extra keys in the dictionary
        extra_keys = set(loss_config.keys()) - set(required_keys.keys())
        if extra_keys:
            raise KeyError(f"Unexpected keys found in loss_config: {extra_keys}")
        
        if loss_config['warmup_period'] is None:
            loss_config['warmup_period'] = [-1, -2]
        if loss_config['non_categorical_coefficient'] is None:
            loss_config['non_categorical_coefficient'] = "none"
        
        # Additional value checks can be added here if necessary
        if loss_config['weight'] < 0:
            raise ValueError("The 'weight' value should be non-negative.")
        elif len(loss_config['warmup_period']) != 2 or not all(isinstance(n, int) for n in loss_config['warmup_period']):
            raise ValueError("The 'warmup_period' must be a list of two integers.")        
        elif loss_config['latent_group'] not in LATENT_INDEX_GROUP_NAMES:
            raise ValueError(f"loss_config['latent_group'] should be one of {LATENT_INDEX_GROUP_NAMES}")
        elif loss_config['target_type'] not in TARGET_TYPES:
            raise ValueError(f"loss_config['target_type'] should be one of {TARGET_TYPES}")
        elif loss_config['method'] not in LOSS.keys():
            raise ValueError(f"loss_config['method'] should be one of {list(LOSS.keys())}")
        elif loss_config["is_minimized"] and loss_config["target_type"] == TARGET_TYPE_PSEUDO_CATEGORICAL:
            raise ValueError(
                f"The {TARGET_TYPE_PSEUDO_CATEGORICAL} coefficient calculation gets two same vector if counteractive "
                "example is not negative. This coefficients makes sense only for negative counteractive examples."
            )
        elif loss_config["target_type"] == TARGET_TYPE_CATEGORICAL and loss_config["non_categorical_coefficient"] != "none":
            raise ValueError(f"`non_categorical_coefficient` should be `none` for `target_type` `{TARGET_TYPE_CATEGORICAL}`")
        elif self.target_type != TARGET_TYPE_PSEUDO_CATEGORICAL and loss_config['non_categorical_coefficient'] == "none":
            raise ValueError(f"`non_categorical_coefficient` should not `none` for `target_type` `{TARGET_TYPE_PSEUDO_CATEGORICAL}`")
        
        return loss_config
    

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
                    # TODO: ?
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


class SimpleLosses(LossesBase):

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


class TripletLosses(LossesBase):

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
        # TODO: branch out for different strategies
        positive = torch.mean(torch.cat([v for v in positive_losses.values()]))
        negative = torch.mean(torch.cat([v for v in negative_losses.values()]))

        weighted_positive = torch.mean(torch.cat([v for v in weighted_positive_losses.values()]))
        weighted_negative = torch.mean(torch.cat([v for v in weighted_negative_losses.values()]))

        loss = torch.max(positive - negative, torch.zeros_like(positive))
        # TODO: device!
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
