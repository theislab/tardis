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


class Disentanglement:

    def __init__(
        self,
        obs_key: str,
        n_reserved_latent: int,
        counteractive_minibatch_settings: CounteractiveMinibatchSettings,
        losses: Union[List[dict], dict] = [],
        triplets: Union[List[dict], dict] = [],
    ):
        self.obs_key = obs_key
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
        self._pseudo_categories = np.vectorize(lambda x: _category_to_values[x])

    def get_total_loss(
        self,
        inputs,
        positive_inputs,
        negative_inputs,
        outputs,
        counteractive_positive_outputs,
        counteractive_negative_outputs,
    ):

        weighted_loss, total_loss = self._losses.get_total_loss(
            inputs,
            positive_inputs,
            negative_inputs,
            outputs,
            counteractive_positive_outputs,
            counteractive_negative_outputs,
            self._indices,
            self._pseudo_categories,
        )
        for triplet_losses in self._triplets:
            cur_weighted_loss, cur_loss = triplet_losses.get_total_loss(
                outputs,
                counteractive_positive_outputs,
                counteractive_negative_outputs,
                self._indices,
                self._pseudo_categories,
            )
            total_loss.update(cur_loss)
            weighted_loss.update(cur_weighted_loss)

        return weighted_loss, total_loss
