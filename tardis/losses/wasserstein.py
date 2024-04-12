import torch
from typing import Any

from tardis.losses.base import TardisLoss


def _wasserstein_loss_with_normal_latent_distribution(
    outputs, counteractive_outputs, relevant_latent_indices, epsilon=1e-8
) -> Any:
    # The Wasserstein distance between two normal distributions is given by the following formula:
    # W_{2,i}^2 = (\mu_{1,i} - \mu_{2,i})^2 + (\sigma_{1,i}^2 + \sigma_{2,i}^2 - 2 \sigma_{1,i} \sigma_{2,i})

    # This formula assumes that the covariance matrices are diagonal, allowing for an element-wise computation.
    # A covariance matrix is said to be diagonal if all off-diagonal elements are zero. This means that there's
    # no covariance between different dimensions—each dimension varies independently of the others.

    qz_inference = outputs["qz"].clone()
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
    trace_term = (
        scale_inference_squared
        + scale_counteractive_squared
        - 2 * (scale_inference * scale_counteractive)
    )

    # Just mean to get the total loss for each datapoint.
    # The loss should not be scaled up or down based on number of relevant latents, so not sum but mean.
    return (mean_diff_sq + trace_term).mean(dim=-1)


class WassersteinLoss(TardisLoss):
    def __init__(
        self,
        weight: float,
        transformation: str,
        method: str,
        progress_bar: bool | None = False,
        latent_distribution: str | None = None,
        kwargs: dict | None = None,
    ):
        super().__init__(
            weight, transformation, method, progress_bar, latent_distribution, kwargs
        )

        if self.latent_distribution is None:
            self.latent_distribution = "normal"

        if self.latent_distribution == "normal":
            self._loss_fn = _wasserstein_loss_with_normal_latent_distribution
        else:
            raise NotImplementedError(
                f"Wasserstein loss with {self.latent_distribution} latent distribution is not implemented yet."
            )

        if "epsilon" in kwargs:
            self.epsilon = kwargs["epsilon"]
        else:
            self.epsilon = 1e-8

    def forward(self, outputs, counteractive_outputs, relevant_latent_indices) -> Any:

        return self.loss_fn(
            outputs,
            counteractive_outputs,
            relevant_latent_indices,
            epsilon=self.epsilon,
        )
