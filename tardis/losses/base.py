from abc import ABC, abstractmethod
from typing import Dict


import torch
import torch.nn.functional as F
import torch.nn as nn


TRANSFORMATIONS = {
    "none": lambda x: x,
    "identity": lambda x: x,
    "negative": torch.neg,
    "sigmoid": F.sigmoid,
    "inverse": lambda x: 1 / (x + 1),
    "exponential_decay": lambda x: torch.exp(-x),
    "tanh": F.tanh,
}


class TardisLoss(nn.Module, ABC):

    def __init__(
        self, weight: float, method_kwargs: dict, transformation: str = "none"
    ):
        super(TardisLoss, self).__init__()
        self.weight = weight
        self.method_kwargs = method_kwargs
        self.transformation = TRANSFORMATIONS[transformation]

    def _validate_forward_inputs(
        self, outputs, counteractive_outputs, relevant_latent_indices
    ):
        if not (
            isinstance(outputs, dict)
            and "z" in outputs.keys()
            and "qz" in outputs.keys()
        ):
            raise ValueError("outputs should be a dictionary with key 'z'")
        if not (
            isinstance(counteractive_outputs, dict)
            and "z" in counteractive_outputs.keys()
            and "qz" in counteractive_outputs.keys()
        ):
            raise ValueError(
                "counteractive_outputs should be a dictionary with key 'z'"
            )
        if not (isinstance(relevant_latent_indices, torch.Tensor)):
            raise ValueError("relevant_latent_indices should be a torch.Tensor")
        if not relevant_latent_indices.shape[-1] <= outputs["z"].shape[-1]:
            raise ValueError(
                "relevant_latent_indices should be the subset of indices of outputs['z']"
            )
        if not counteractive_outputs["z"].shape == outputs["z"].shape:
            raise ValueError(
                "counteractive_outputs['z'] should have the same shape as outputs['z']"
            )
        if not len(outputs["z"].shape) == 2:
            raise ValueError("outputs['z'] should be a 2D tensor")
        if not len(counteractive_outputs["z"].shape) == 2:
            raise ValueError("counteractive_outputs['z'] should be a 2D tensor")
        if not len(relevant_latent_indices.shape) == 1:
            if len(relevant_latent_indices.shape) != 2:
                raise ValueError(
                    "relevant_latent_indices should be a 1D tensor or a 2D tensor with shape (1, n)"
                )
            relevant_latent_indices = relevant_latent_indices.squeeze()
        if not relevant_latent_indices.dtype == torch.int:
            raise ValueError("relevant_latent_indices should be of type torch.int")

    @abstractmethod
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        counteractive_outputs: Dict[str, torch.Tensor],
        relevant_latent_indices: torch.Tensor,
    ):
        pass
