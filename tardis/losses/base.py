from abc import ABC, abstractmethod
from typing import Optional
from pydantic import (
    BaseModel,
    StrictBool,
    StrictFloat,
    StrictStr,
    field_validator,
)  # ValidationError

import torch.nn as nn


class TardisLoss(nn.Module):

    def __init__(
        method: str,
        weight: float,
        transformation: str,
        progress_bar: bool,
        # Accepts any dict without specific type checking.
        method_kwargs: dict,
        loss_identifier_string: Optional[str] = "",
    ):
        self.method = method
        self.weight = weight
        self.transformation = transformation
        self.progress_bar = progress_bar
        self.method_kwargs = method_kwargs
        self.loss_identifier_string = loss_identifier_string

    @abstractmethod
    def forward(self, outputs, counteractive_outputs, relevant_latent_indices):
        pass
