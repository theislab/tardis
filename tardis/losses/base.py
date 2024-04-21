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


class CoefficientFunction:
    def __call__(self, x: Dict[str, torch.Tensor], other: Dict[str, torch.Tensor]):
        pass


COEFFICIENT_FUNCTIONS = {
    "none": lambda x, other: torch.ones(x.shape[0]).to(device=x.device),
    "abs": lambda x, other: torch.abs(x - other),
    "square": lambda x, other: torch.square(x - other),
}


class TardisLoss(nn.Module, ABC):

    def __init__(
        self,
        weight: float,
        is_minimized: bool = True,
        transformation: str = "none",
        coefficient: str = "none",
        target_type: str = "categorical",
        method_kwargs: Dict[str, any] = {},
    ) -> None:

        super(TardisLoss, self).__init__()
        self._weight = weight
        self.transformation = TRANSFORMATIONS[transformation]
        self._is_minimized = is_minimized

        if isinstance(coefficient, float):
            self.coefficient_fn = lambda x, other: coefficient * torch.ones(
                x.shape[0], device=x.device
            )
        elif isinstance(coefficient, str):
            if coefficient not in COEFFICIENT_FUNCTIONS:
                raise ValueError(f"Unknown coefficient function: {coefficient}")
            self.coefficient_fn = COEFFICIENT_FUNCTIONS[coefficient]
        else:
            self.coefficient_fn = coefficient

        if self.is_minimized and target_type == "pseudo_categorical":
            raise ValueError(
                "The pseudo-categorical coefficient calculation will get two same vector if counteractive "
                "example is not negative. This coefficients makes sense only for negative counteractive examples."
            )
        self.target_type = target_type
        self.method_kwargs = method_kwargs

    def get_coefficients(self, x, other):
        return self.coefficient_fn(x, other)

    @property
    def is_minimized(self):
        return self._is_minimized

    @property
    def weight(self):
        if self._is_minimized:
            return self._weight
        else:
            return -self._weight

    @weight.setter
    def weight(self, value: float):
        if value > 0:
            self._weight = value
        else:
            raise ValueError("weight should be positive")

    def _validate_forward_inputs(
        self,
        outputs: Dict[str, torch.Tensor],
        counteractive_outputs: Dict[str, torch.Tensor],
        relevant_latent_indices: torch.Tensor,
    ) -> torch.Tensor:
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
        self._validate_forward_inputs(
            outputs, counteractive_outputs, relevant_latent_indices
        )
        loss = self.weight * self.transformation(
            self._forward(outputs, counteractive_outputs, relevant_latent_indices)
        )
        return loss
