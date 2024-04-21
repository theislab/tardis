#!/usr/bin/env python3

from typing import Dict, Optional

import torch

from .base import TardisLoss


def _wasserstein_loss_with_normal_latent_distribution(
    outputs: Dict[str, torch.Tensor],
    counteractive_outputs: Dict[str, torch.Tensor],
    relevant_latent_indices: torch.Tensor,
    epsilon: Optional[float] = 1e-8,
) -> torch.Tensor:
    # The Wasserstein distance between two normal distributions is given by the following formula:
    # W_{2,i}^2 = (\mu_{1,i} - \mu_{2,i})^2 + (\sigma_{1,i}^2 + \sigma_{2,i}^2 - 2 \sigma_{1,i} \sigma_{2,i})

    # This formula assumes that the covariance matrices are diagonal, allowing for an element-wise computation.
    # A covariance matrix is said to be diagonal if all off-diagonal elements are zero. This means that there's
    # no covariance between different dimensions—each dimension varies independently of the others.

    qz_inference = outputs["qz"]
    qz_counteractive = counteractive_outputs["qz"]

    loc_inference = qz_inference.loc[:, relevant_latent_indices]
    loc_counteractive = qz_counteractive.loc[:, relevant_latent_indices]

    scale_inference = qz_inference.scale[:, relevant_latent_indices]
    scale_counteractive = qz_counteractive.scale[:, relevant_latent_indices]

    # epsilon is for numerical stability
    scale_inference_squared = torch.pow(scale_inference + epsilon, 2)
    scale_counteractive_squared = torch.pow(scale_counteractive + epsilon, 2)

    # The squared difference of means, element-wise.
    mean_diff_sq = (loc_inference - loc_counteractive).pow(2)
    # For diagonal covariances, the trace term simplifies to an element-wise operation.
    trace_term = scale_inference_squared + scale_counteractive_squared - 2 * (scale_inference * scale_counteractive)

    # Just mean to get the total loss for each datapoint.
    # The loss should not be scaled up or down based on number of relevant latents, so not sum but mean.
    return (mean_diff_sq + trace_term).mean(dim=-1)


class WassersteinLoss(TardisLoss):
    def __init__(
        self,
        weight: float,
        is_minimized: bool = True,
        transformation: str = "none",
        coefficient: str = "none",
        target_type: str = "categorical",
        method_kwargs: Dict[str, any] = {},
    ) -> None:
        super().__init__(
            weight=weight,
            is_minimized=is_minimized,
            transformation=transformation,
            coefficient=coefficient,
            target_type=target_type,
            method_kwargs=method_kwargs,
        )

        latent_distribution = method_kwargs.get("latent_distribution", "normal")

        if latent_distribution == "normal":
            self.latent_distribution = "normal"
        else:
            raise NotImplementedError(
                f"Wasserstein loss with {self.latent_distribution} latent distribution is not implemented yet."
            )

        self.epsilon = method_kwargs.get("epsilon", 1e-8)

    @property
    def loss_fn(self):
        if self.latent_distribution == "normal":
            return _wasserstein_loss_with_normal_latent_distribution
        else:
            raise NotImplementedError(
                f"Wasserstein loss with {self.latent_distribution} latent distribution is not implemented yet."
            )

    def _forward(
        self,
        outputs: Dict[str, torch.Tensor],
        counteractive_outputs: Dict[str, torch.Tensor],
        relevant_latent_indices: torch.Tensor,
    ) -> torch.Tensor:

        return self.loss_fn(
            outputs,
            counteractive_outputs,
            relevant_latent_indices,
            epsilon=self.epsilon,
        )
