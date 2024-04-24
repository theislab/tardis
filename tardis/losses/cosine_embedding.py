from typing import Dict

import torch
import torch.nn.functional as F

from .base import TardisLoss


class CosineEmbeddingLoss(TardisLoss):

    def _forward(
        self,
        outputs: Dict[str, torch.Tensor],
        counteractive_outputs: Dict[str, torch.Tensor],
        relevant_latent_indices: torch.Tensor,
    ) -> torch.Tensor:
        if self._is_minimized:
            targets = torch.ones(outputs["z"].shape[0])
        else:
            targets = -1 * torch.ones(outputs["z"].shape[0])

        return F.cosine_embedding_loss(
            outputs["z"][:, relevant_latent_indices],
            counteractive_outputs["z"].clone()[:, relevant_latent_indices],
            targets,
        )
