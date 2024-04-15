import torch.nn.functional as F
from typing import Any
from .base import TardisLoss


class MAE(TardisLoss):

    def forward(self, outputs, counteractive_outputs, relevant_latent_indices) -> Any:
        self._validate_forward_inputs(
            outputs, counteractive_outputs, relevant_latent_indices
        )
        return self.weight * self.transformation(
            F.l1_loss(
                input=outputs["z"][:, relevant_latent_indices].clone(),
                target=counteractive_outputs["z"][:, relevant_latent_indices].clone(),
                reduction="none",
            ).mean(dim=-1)
        )
