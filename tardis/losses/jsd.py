import torch
from typing import Any
from torch.distributions import Normal
from torch.nn.functional import kl_div
from .base import TardisLoss


def _jensen_shannon_divergence_with_normal_parameters(dist_p, dist_q):
    # Calculate the midpoint distribution using loc and scale
    mean_m = 0.5 * (dist_p.loc + dist_q.loc)
    std_m = torch.sqrt(0.5 * (dist_p.scale**2 + dist_q.scale**2))
    dist_m = Normal(mean_m, std_m)

    # Compute the KL divergences
    kl_pm = kl_div(dist_p, dist_m)
    kl_qm = kl_div(dist_q, dist_m)

    # Jensen-Shannon Divergence
    jsd = 0.5 * (kl_pm + kl_qm)
    return jsd


class JSD(TardisLoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.latent_distribution is None:
            self.latent_distribution = "normal"

        if self.latent_distribution == "normal":
            self._jsd = JSD
        else:
            raise NotImplementedError(
                f"JSD with {self.latent_distribution} latent distribution is not implemented yet."
            )

    def forward(self, outputs, counteractive_outputs, relevant_latent_indices) -> Any:
        qz_inference = outputs["qz"].clone()
        qz_counteractive = counteractive_outputs["qz"]

        dist_p = Normal(
            qz_inference.loc[:, relevant_latent_indices],
            qz_inference.scale[:, relevant_latent_indices],
        )
        dist_q = Normal(
            qz_counteractive.loc[:, relevant_latent_indices],
            qz_counteractive.scale[:, relevant_latent_indices],
        )

        return _jensen_shannon_divergence_with_normal_parameters(dist_p, dist_q)
