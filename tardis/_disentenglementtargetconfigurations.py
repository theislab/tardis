#!/usr/bin/env python3
import numpy as np
import torch

from dataclasses import dataclass
from typing import List, Union

from pydantic import (
    BaseModel,
    StrictStr,
)

from ._losses import Losses, Triplets
from scvi import REGISTRY_KEYS
from ._myconstants import REGISTRY_KEY_DISENTANGLEMENT_TARGETS


def isnumeric(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


class CounteractiveMinibatchSettings(BaseModel):
    method: StrictStr
    # Accepts any dict without specific type checking.
    method_kwargs: dict
    # for now only `random` implemented for counteractive_minibatch_method, raise ValueError in method selection.
    # seed should be in method_kwargs: str or `global_seed` etc


@dataclass
class Indices:
    reserved: torch.Tensor = torch.tensor([], dtype=torch.int)
    unreserved: torch.Tensor = torch.tensor([], dtype=torch.int)
    complete: torch.Tensor = torch.tensor([], dtype=torch.int)


class CoefficientFunction:
    def __call__(self, x, other):
        pass


COEFFICIENT_FUNCTIONS = {
    "none": lambda x, other: torch.ones(x.shape[0]).device(x.device),
    "abs": lambda x, other: torch.abs(x - other),
    "square": lambda x, other: torch.square(x - other),
    "categorical": lambda x, other: torch.ones(x.shape[0]).device(x.device),
}


class Disentanglement:

    def __init__(
        self,
        obs_key: str,
        n_reserved_latent: int,
        counteractive_minibatch_settings: CounteractiveMinibatchSettings,
        losses: Union[List[dict], dict] = [],
        triplets: Union[List[dict], dict] = [],
        positive_coefficient: Union[float, CoefficientFunction] = 1.0,
        negative_coefficient: Union[float, CoefficientFunction] = 1.0,
        target_type: str = "categorical",
    ):
        self.obs_key = obs_key
        self.target_type = target_type
        self.n_reserved_latent = n_reserved_latent

        self.counteractive_minibatch_settings = CounteractiveMinibatchSettings(
            **counteractive_minibatch_settings
        )
        if isinstance(losses, dict):
            losses = [losses]

        self._losses = Losses(losses, obs_key)

        if isinstance(triplets, dict):
            triplets = [triplets]

        self._triplets = []
        for triplet in triplets:
            self._triplets.append(Triplets(triplet, obs_key))

        self._indices = Indices()

        if isinstance(positive_coefficient, float):
            self._positive_coefficient = (
                lambda x, other: positive_coefficient
                * torch.ones(x.shape[0], device=x.device)
            )
        elif isinstance(positive_coefficient, str):
            if positive_coefficient not in COEFFICIENT_FUNCTIONS:
                raise ValueError(
                    f"Unknown coefficient function: {positive_coefficient}"
                )
            self._positive_coefficient = COEFFICIENT_FUNCTIONS[positive_coefficient]
        else:
            self._positive_coefficient = positive_coefficient

        if isinstance(negative_coefficient, float):
            self._negative_coefficient = (
                lambda x, other: negative_coefficient
                * torch.ones(x.shape[0], device=x.device)
            )
        elif isinstance(negative_coefficient, str):
            if negative_coefficient not in COEFFICIENT_FUNCTIONS:
                raise ValueError(
                    f"Unknown coefficient function: {negative_coefficient}"
                )
            self._negative_coefficient = COEFFICIENT_FUNCTIONS[negative_coefficient]
        else:
            self._negative_coefficient = negative_coefficient

        if self.target_type == "pseudo_categorical":
            self._positive_coefficient = lambda x, other: torch.ones(x.shape[0]).device(
                x.device
            )

        self._category_to_values = []
        self.index = None

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

    def set_mappings(self, mappings):
        is_numeric = all([isnumeric(mapping) for mapping in mappings])
        _category_to_values = [float(j) if is_numeric else j for j in mappings]
        self._category_to_values = np.vectorize(lambda x: _category_to_values[x])

    def convert_array_categorical_to_value(self, array):
        return self._category_to_values(array)

    def get_total_loss(
        self,
        inputs,
        positive_inputs,
        negative_inputs,
        outputs,
        counteractive_positive_outputs,
        counteractive_negative_outputs,
    ):

        device = inputs[REGISTRY_KEYS.X_KEY].device

        if self.target_type == "pseudo_categorical":

            _inputs = (
                inputs[REGISTRY_KEY_DISENTANGLEMENT_TARGETS][:, self.index]
                .detach()
                .cpu()
                .numpy()
            )
            _positive_inputs = None
            _negative_inputs = (
                negative_inputs[REGISTRY_KEY_DISENTANGLEMENT_TARGETS][:, self.index]
                .detach()
                .cpu()
                .numpy()
            )

            _inputs = torch.tensor(
                self.convert_array_categorical_to_value(_inputs), device=device
            )
            _negative_inputs = torch.tensor(
                self.convert_array_categorical_to_value(_negative_inputs), device=device
            )
        else:
            _inputs = inputs[REGISTRY_KEYS.X_KEY]
            _positive_inputs = positive_inputs[REGISTRY_KEYS.X_KEY]
            _negative_inputs = negative_inputs[REGISTRY_KEYS.X_KEY]

        positive_coefficient = self._positive_coefficient(_inputs, _positive_inputs)
        negative_coefficient = self._negative_coefficient(_inputs, _negative_inputs)

        total_loss = self._losses.get_total_loss(
            outputs,
            counteractive_positive_outputs,
            counteractive_negative_outputs,
            self._indices,
            positive_coefficient,
            negative_coefficient,
        )
        for triplet_losses in self._triplets:
            triplet_loss = triplet_losses.get_total_loss(
                outputs,
                counteractive_positive_outputs,
                counteractive_negative_outputs,
                self._indices,
                positive_coefficient,
                negative_coefficient,
            )
            total_loss.update(triplet_loss)
        return total_loss
