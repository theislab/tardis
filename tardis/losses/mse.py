from typing import Any
import torch.nn.functional as F

from .base import TardisLoss


class MSE(TardisLoss):

    def forward(self, outputs, counteractive_outputs, relevant_latent_indices) -> Any:
        return F.mse_loss(
            input=outputs["z"][:, relevant_latent_indices].clone(),
            target=counteractive_outputs["z"][:, relevant_latent_indices].clone(),
            reduction="none",
        ).mean(dim=-1)
