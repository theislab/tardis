#!/usr/bin/env python3

from dataclasses import dataclass
from typing import List, Union

import numpy as np

from ._losses import SimpleLosses, TripletLosses
from ._utils.functions import isnumeric
from ._counteractivegenerator import CounteractiveGeneratorSettings



@dataclass
class Indices:
    reserved: List[int] = []
    unreserved: List[int] = []
    complete: List[int] = []
    complete_unreserved: List[int] = []


class Disentanglement:

    def __init__(
        self,
        obs_key: str,
        n_reserved_latent: int,
        counteractive_generator_settings: CounteractiveGeneratorSettings,
        simple_losses: List[dict] = [],
        triplet_losses: List[dict] = [],
    ):
        self.obs_key = obs_key
        self.n_reserved_latent = n_reserved_latent
        self.counteractive_generator_settings = CounteractiveGeneratorSettings(**counteractive_generator_settings)

        self._simple_losses = SimpleLosses(simple_losses, obs_key)    
        # TODO: enforce only one loss at least!

        # TODO: why like below?
        self._triplet_losses = []
        for triplet in triplet_losses:
            self._triplet_losses.append(TripletLosses(triplet, obs_key))

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
    
    @property
    def complete_unreserved_indices(self):
        return self._indices.complete_unreserved

    @reserved_indices.setter
    def reserved_indices(self, value):
        self._indices.reserved = value

    @unreserved_indices.setter
    def unreserved_indices(self, value):
        self._indices.unreserved = value

    @complete_indices.setter
    def complete_indices(self, value):
        self._indices.complete = value
        
    @complete_unreserved_indices.setter
    def complete_unreserved_indices(self, value):
        self._indices.complete_unreserved = value

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

        weighted_loss, total_loss = self._simple_losses.get_total_loss(
            inputs,
            positive_inputs,
            negative_inputs,
            outputs,
            counteractive_positive_outputs,
            counteractive_negative_outputs,
            self._indices,
            self._pseudo_categories,
        )
        # TODO: didnt understand why not just above
        for triplet_losses in self._triplet_losses:
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
