import torch
from typing import Any
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence
from .base import TardisLoss


class JSDNormal(TardisLoss):

    def forward(self, outputs, counteractive_outputs, relevant_latent_indices) -> Any:
        self._validate_forward_inputs(
            outputs, counteractive_outputs, relevant_latent_indices
        )
        qz_inference = outputs["qz"]
        qz_counteractive = counteractive_outputs["qz"]

        dist_p = Normal(
            qz_inference.loc[:, relevant_latent_indices],
            qz_inference.scale[:, relevant_latent_indices],
        )
        dist_q = Normal(
            qz_counteractive.loc[:, relevant_latent_indices],
            qz_counteractive.scale[:, relevant_latent_indices],
        )

        # Calculate the midpoint distribution using loc and scale
        mean_m = 0.5 * (dist_p.loc + dist_q.loc)
        std_m = torch.sqrt(0.5 * (dist_p.scale**2 + dist_q.scale**2))
        dist_m = Normal(mean_m, std_m)

        # Compute the KL divergences
        kl_pm = kl_divergence(dist_p, dist_m)
        kl_qm = kl_divergence(dist_q, dist_m)

        # Jensen-Shannon Divergence
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd


class JSD(TardisLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.latent_distribution = kwargs.get("latent_distribution", "normal")

        if self.latent_distribution == "normal":
            self._jsd = JSDNormal(*args, **kwargs)
        else:
            raise NotImplementedError(
                f"JSD with {self.latent_distribution} latent distribution is not implemented yet."
            )

    def _forward(self, outputs, counteractive_outputs, relevant_latent_indices) -> Any:
        self._validate_forward_inputs(
            outputs, counteractive_outputs, relevant_latent_indices
        )
        return self._jsd.forward(
            outputs, counteractive_outputs, relevant_latent_indices
        )