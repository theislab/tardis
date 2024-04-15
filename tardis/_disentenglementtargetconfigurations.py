#!/usr/bin/env python3
import torch

from dataclasses import dataclass
from typing import List

from pydantic import (
    BaseModel,
    StrictStr,
)

from .losses import LOSSES
from ._myconstants import LOSS_NAMING_DELIMITER, LOSS_NAMING_PREFIX, LOSS_TYPES
from ._progressbarmanager import ProgressBarManager

from ._myconstants import LOSS_TYPES


class CounteractiveMinibatchSettings(BaseModel):
    method: StrictStr
    # Accepts any dict without specific type checking.
    method_kwargs: dict
    # for now only `random` implemented for counteractive_minibatch_method, raise ValueError in method selection.
    # seed should be in method_kwargs: str or `global_seed` etc


class Losses:
    def __init__(self, loss_configs, obs_key):
        self.reserved = []
        self.unreserved = []
        self.complete = []

        for loss_config in loss_configs:
            loss_config = self._validate_loss_config(loss_config)
            loss_cls = LOSSES[loss_config["method"]]
            loss_obj = loss_cls(
                weight=loss_config["weight"],
                method_kwargs=loss_config["method_kwargs"],
                transformation=loss_config["transformation"],
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
            ProgressBarManager.add(identifier)

    def _validate_loss_config(self, loss_config):

        if not isinstance(loss_config, dict):
            raise ValueError("loss_config should be a dictionary.")

        method = loss_config.get("method", None)
        weight = loss_config.get("weight", 1.0)
        transformation = loss_config.get("transformation", "identity")
        progress_bar = loss_config.get("progress_bar", True)
        method_kwargs = loss_config.get("method_kwargs", {})
        loss_type = loss_config.get("type", "complete")

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
        if loss_type not in set(LOSS_TYPES):
            raise ValueError(f"loss_config['type'] should be one of {LOSS_TYPES}")

        return dict(
            method=method,
            weight=weight,
            transformation=transformation,
            progress_bar=progress_bar,
            method_kwargs=method_kwargs,
            loss_type=loss_type,
        )


@dataclass
class Indices:
    reserved: torch.Tensor = torch.tensor([], dtype=torch.int)
    unreserved: torch.Tensor = torch.tensor([], dtype=torch.int)
    complete: torch.Tensor = torch.tensor([], dtype=torch.int)


class Disentanglement:

    def __init__(
        self,
        obs_key: str,
        n_reserved_latent: int,
        counteractive_minibatch_settings: CounteractiveMinibatchSettings,
        losses: List[dict],
    ):
        self.obs_key = obs_key
        self.n_reserved_latent = n_reserved_latent
        self.counteractive_minibatch_settings = CounteractiveMinibatchSettings(
            **counteractive_minibatch_settings
        )
        self._losses = Losses(losses, obs_key)
        self._indices = Indices()

    @property
    def reserved_indices(self):
        return self._indices.reserved

    @property
    def unreserved_indices(self):
        return self._indices.unreserved

    @property
    def complete_indices(self):
        return self._indices.complete

    @reserved_indices.setter
    def reserved_indices(self, value):
        self._indices.reserved = value

    @unreserved_indices.setter
    def unreserved_indices(self, value):
        self._indices.unreserved = value

    @complete_indices.setter
    def complete_indices(self, value):
        self._indices.complete = value

    def get_total_loss(self, outputs, counteractive_outputs):
        total_loss = {}
        for loss_type in LOSS_TYPES:
            loss_fns = getattr(self._losses, loss_type)
            indices = getattr(self._indices, loss_type)
            for loss_fn in loss_fns:
                identifier = LOSS_NAMING_DELIMITER.join(
                    [
                        LOSS_NAMING_PREFIX,
                        self.obs_key,
                        loss_type,
                        type(loss_fn).__name__.lower(),
                    ]
                )
                total_loss[identifier] = loss_fn.forward(
                    outputs, counteractive_outputs, indices
                )
        return total_loss
