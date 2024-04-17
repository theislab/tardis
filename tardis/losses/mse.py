from typing import Any
import torch.nn.functional as F

from .base import TardisLoss


class MSE(TardisLoss):

    def _forward(self, outputs, counteractive_outputs, relevant_latent_indices) -> Any:
        self._validate_forward_inputs(
            outputs, counteractive_outputs, relevant_latent_indices
        )
        return F.mse_loss(
            input=outputs["z"][:, relevant_latent_indices].clone(),
            target=counteractive_outputs["z"][:, relevant_latent_indices].clone(),
            reduction="none",
        ).mean(dim=-1)
