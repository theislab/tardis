#!/usr/bin/env python3

from typing import Dict

import torch
import torch.nn.functional as F

from .base import TardisLoss


class MAE(TardisLoss):

    def _forward(
        self,
        outputs: Dict[str, torch.Tensor],
        counteractive_outputs: Dict[str, torch.Tensor],
        relevant_latent_indices: torch.Tensor,
    ) -> torch.Tensor:

        return F.l1_loss(
            input=outputs["z"][:, relevant_latent_indices].clone(),
            target=counteractive_outputs["z"][:, relevant_latent_indices].clone(),
            reduction="none",
        ).mean(dim=-1)
