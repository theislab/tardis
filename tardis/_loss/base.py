#!/usr/bin/env python3

from abc import ABC, abstractmethod
from typing import Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F


TRANSFORMATIONS = {
    "identity": lambda x: x,
    "negative": lambda loss: -loss,
    "inverse": lambda loss: 1 / (loss + 1),
    "reverse": lambda loss: 1 - loss,
    "sigmoid": lambda loss: F.sigmoid(loss),
    "reverse_sigmoid": lambda loss: 1 - F.sigmoid(loss),
    "tanh": lambda loss: F.tanh(loss),
    "reverse_tanh": lambda loss: 1 - F.tanh(loss),
    "exponential_decay": lambda loss: torch.exp(-loss),
}

COEFFICIENT_FUNCTIONS = {
    "none": lambda x, other: torch.ones(x.shape[0]).to(device=x.device),
    "abs": lambda x, other: torch.abs(x - other),
    "square": lambda x, other: torch.square(x - other),
}


class TardisLoss(nn.Module, ABC):

    def __init__(
        self,
        weight: float,
        transformation: str,
        method_kwargs: Dict[str, any],
        latent_group: str,
        is_minimized: bool,
        target_type: str,
        warmup_periods: List[int],
        non_categorical_coefficient_method: str,
    ) -> None:

        super().__init__()
        self.method = method 
        self.weight = weight
        self.transformation = TRANSFORMATIONS[transformation]
        self.is_minimized = is_minimized
        self.target_type = target_type
        # TODO: index needed!
        # TODO: loss name set here

        
        
        self.non_categorical_coefficient_fn = COEFFICIENT_FUNCTIONS[pseudocategorical_coefficient]    
        self.method_kwargs = method_kwargs

    def get_coefficients(self, x, other):
        return self.coefficient_fn(x, other)

    @abstractmethod
    def _forward(
        self,
        outputs: Dict[str, torch.Tensor],
        counteractive_outputs: Dict[str, torch.Tensor],
        relevant_latent_indices: torch.Tensor,
    ) -> torch.Tensor:
        pass

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        counteractive_outputs: Dict[str, torch.Tensor],
        relevant_latent_indices: torch.Tensor,
    ) -> torch.Tensor:
        loss = (
            self.transformation(
                self._forward(
                    outputs, 
                    counteractive_outputs, 
                    relevant_latent_indices, 
                    **self.method_kwargs
                )  # get pure loss
            )  # transform e.g. reverse, inverse, sigmoid
            * self.weight  # multiply with a predefined constant
            # TODO: multiply with coefficient here
        )
        return loss
