#!/usr/bin/env python3

from typing import Dict

import torch
import torch.nn.functional as F

from .base import TardisLoss


class CosineSimilarity(TardisLoss):

    def _forward(
        self,
        outputs: Dict[str, torch.Tensor],
        counteractive_outputs: Dict[str, torch.Tensor],
        relevant_latent_indices: torch.Tensor,
    ) -> torch.Tensor:

        return F.cosine_similarity(
            x1=outputs["z"][:, relevant_latent_indices],
            x2=counteractive_outputs["z"].clone()[:, relevant_latent_indices],
        )
