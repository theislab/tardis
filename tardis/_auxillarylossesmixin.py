#!/usr/bin/env python3

from typing import Dict, List

import torch
import torch.nn.functional as F
from scvi.module.base import auto_move_data
from torch.distributions import Normal, kl_divergence

from ._DEBUG import DEBUG
from ._disentenglementtargetmanager import DisentenglementTargetManager
from ._myconstants import (
    BOTH_EXAMPLE_KEY,
    LATENT_INDEX_GROUP_COMPLETE,
    LATENT_INDEX_GROUP_RESERVED,
    LATENT_INDEX_GROUP_UNRESERVED,
    NEGATIVE_EXAMPLE_KEY,
    POSITIVE_EXAMPLE_KEY,
    REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS,
)


class FinalTransformation:
    items = {
        "none": lambda loss: loss,
        "negative": lambda loss: -loss,
        "inverse": lambda loss: 1 / (loss + 1),
        "reverse": lambda loss: 1 - loss,
        "sigmoid": lambda loss: F.sigmoid(loss),
        "reverse_sigmoid": lambda loss: 1 - F.sigmoid(loss),
        "tanh": lambda loss: F.tanh(loss),
        "reverse_tanh": lambda loss: 1 - F.tanh(loss),
        "exponential_decay": lambda loss: torch.exp(-loss),
    }

    @classmethod
    def get(cls, key):
        try:
            return cls.items[key]
        except KeyError as e:
            raise KeyError(f"Transformation key should be a string and one of the following: {cls.items.keys()}") from e


class AuxillaryLossesMixin:

    @torch.inference_mode()
    @auto_move_data
    def inference_counteractive_minibatch(self, counteractive_minibatch_tensors):
        counteractive_inference_inputs = self._get_inference_input(counteractive_minibatch_tensors)
        counteractive_inference_outputs = self.inference(**counteractive_inference_inputs)
        return counteractive_inference_outputs

    def calculate_auxillary_losses(self, tensors, inference_outputs):

        result = dict()
        for target_obs_key_ind, target_obs_key in enumerate(
            DisentenglementTargetManager.configurations.get_ordered_obs_key()
        ):
            relevant_latent_indices = AuxillaryLossesMixin.relevant_latent_indices(target_obs_key_ind)

            # Although sometimes `inference_counteractive_positive` or `inference_counteractive_negative` are not
            # used at all, calculate for coding simplicity. Needs optimization for deployment.
            inference_counteractive_positive = self.inference_counteractive_minibatch(
                tensors[REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS][target_obs_key][POSITIVE_EXAMPLE_KEY]
            )
            inference_counteractive_negative = self.inference_counteractive_minibatch(
                tensors[REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS][target_obs_key][NEGATIVE_EXAMPLE_KEY]
            )

            config_main = DisentenglementTargetManager.configurations.get_by_index(target_obs_key_ind)
            for config_individual in config_main.auxillary_losses:
                loss = self.calculate_auxillary_loss(
                    t=tensors,
                    i=target_obs_key_ind,
                    io=inference_outputs,
                    iop=inference_counteractive_positive,
                    ion=inference_counteractive_negative,
                    c=config_individual,
                    rli=relevant_latent_indices,
                )
                result[config_individual.loss_identifier_string] = loss

        return result

    @staticmethod
    def relevant_latent_indices(target_obs_key_ind) -> Dict[str, List[int]]:
        return {
            LATENT_INDEX_GROUP_COMPLETE: DisentenglementTargetManager.configurations.latent_indices,
            LATENT_INDEX_GROUP_RESERVED: DisentenglementTargetManager.configurations.get_by_index(
                target_obs_key_ind
            ).reserved_latent_indices,
            LATENT_INDEX_GROUP_UNRESERVED: DisentenglementTargetManager.configurations.get_by_index(
                target_obs_key_ind
            ).unreserved_latent_indices,
        }

    def calculate_auxillary_loss(self, t, i, io, iop, ion, c, rli):
        if not c.apply:  # TODO: include warm-up epoch period
            return torch.zeros(io["z"].shape[0]).to(io["z"].device)

        if c.method == "wasserstein_qz" and self.latent_distribution == "normal":
            func = self.wasserstein_with_normal_parameters

        elif c.method == "wasserstein_qz" and self.latent_distribution == "ln":
            raise NotImplementedError

        elif c.method == "mse_z":
            func = self.mse_with_reparametrized_z

        elif c.method == "mae_z":
            func = self.mae_with_reparametrized_z

        elif c.method == "kl_qz" and self.latent_distribution == "normal":
            raise NotImplementedError

        elif c.method == "kl_qz" and self.latent_distribution == "ln":
            raise NotImplementedError

        elif c.method == "cosine_similarity_z":
            func = self.cosine_similarity_with_reparametrized_z

        elif c.method == "cosine_embedding_z":
            func = self.cosine_embedding_with_reparametrized_z

        else:
            raise ValueError("Unknown auxillary loss method.")

        loss = func(
            t=t, i=i, io=io, iop=iop, ion=ion, rli=rli, lg=c.latent_group, ce=c.counteractive_example, **c.method_kwargs
        )

        return FinalTransformation.get(key=c.transformation)(loss=loss) * c.weight

    def mse_with_reparametrized_z(self, t, i, io, iop, ion, rli, ce, lg):
        if ce in [NEGATIVE_EXAMPLE_KEY, POSITIVE_EXAMPLE_KEY]:
            return F.mse_loss(
                input=(ion if ce == NEGATIVE_EXAMPLE_KEY else iop)["z"][:, rli[lg]],  # true
                target=io["z"][:, rli[lg]],  # pred
                reduction="none",
            ).mean(dim=1)
        elif ce == BOTH_EXAMPLE_KEY:
            raise ValueError(f"MSE does not work with {BOTH_EXAMPLE_KEY} example datapoints.")
        else:
            raise ValueError("Undefined counteractive example key.")

    def mae_with_reparametrized_z(self, t, i, io, iop, ion, rli, ce, lg):
        if ce in [NEGATIVE_EXAMPLE_KEY, POSITIVE_EXAMPLE_KEY]:
            return F.l1_loss(
                input=(ion if ce == NEGATIVE_EXAMPLE_KEY else iop)["z"][:, rli[lg]],  # true
                target=io["z"][:, rli[lg]],  # pred
                reduction="none",
            ).mean(dim=1)
        elif ce == BOTH_EXAMPLE_KEY:
            raise ValueError(f"MAE does not work with {BOTH_EXAMPLE_KEY} example datapoints.")
        else:
            raise ValueError("Undefined counteractive example key.")

    def cosine_similarity_with_reparametrized_z(self, t, i, io, iop, ion, rli, ce, lg):
        if ce in [NEGATIVE_EXAMPLE_KEY, POSITIVE_EXAMPLE_KEY]:
            return F.cosine_similarity(
                x1=(ion if ce == NEGATIVE_EXAMPLE_KEY else iop)["z"][:, rli[lg]],  # true
                x2=io["z"][:, rli[lg]],  # pred
                dim=1,
            )
        elif ce == BOTH_EXAMPLE_KEY:
            raise ValueError(f"Cosine similarity does not work with {BOTH_EXAMPLE_KEY} example datapoints.")
        else:
            raise ValueError("Undefined counteractive example key.")

    def cosine_embedding_with_reparametrized_z(self, t, i, io, iop, ion, rli, ce, lg):
        DEBUG.x1 = ion["z"][:, rli[lg]]
        DEBUG.x2 = io["z"][:, rli[lg]]

        # use t and i to get original labels..

        raise NotImplementedError  # F.cosine_embedding_loss

    def wasserstein_with_normal_parameters(self, t, i, io, iop, ion, rli, ce, lg, epsilon=1e-8):

        # W_{2,i}^2 = (\mu_{1,i} - \mu_{2,i})^2 + (\sigma_{1,i}^2 + \sigma_{2,i}^2 - 2 \sigma_{1,i} \sigma_{2,i})

        # This formula assumes that the covariance matrices are diagonal, allowing for an element-wise computation.
        # A covariance matrix is said to be diagonal if all off-diagonal elements are zero. This means that there's
        # no covariance between different dimensions—each dimension varies independently of the others.

        qz_inference = io["qz"]
        qz_counteractive = (ion if ce == NEGATIVE_EXAMPLE_KEY else iop)["qz"]
        loc_inference = qz_inference.loc[:, rli[lg]]
        loc_counteractive = qz_counteractive.loc[:, rli[lg]]
        scale_inference = qz_inference.scale[:, rli[lg]]
        scale_counteractive = qz_counteractive.scale[:, rli[lg]]
        # epsilon is for numerical stability
        scale_inference_squared = torch.pow(scale_inference + epsilon, 2)
        scale_counteractive_squared = torch.pow(scale_counteractive + epsilon, 2)

        # The squared difference of means, element-wise.
        mean_diff_sq = (loc_inference - loc_counteractive).pow(2)
        # For diagonal covariances, the trace term simplifies to an element-wise operation.
        trace_term = scale_inference_squared + scale_counteractive_squared - 2 * (scale_inference * scale_counteractive)

        # Just mean to get the total loss for each datapoint.
        # The loss should not be scaled up or down based on number of relevant latents, so not sum but mean.
        return (mean_diff_sq + trace_term).mean(dim=1)

    def kl_with_normal_parameters(self, t, io, iop, ion, rli, ce, lg):
        raise NotImplementedError

    def _jensen_shannon_divergence_with_normal_parameters(self, dist_p, dist_q):
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

    # TODO: minimize mutual information
    # TODO: kl-loss
    # TODO: Bhattacharyya distance
    # TODO: cosine similarity and cosine embedding loss
    # TODO: some other similarity based loss
    # TODO: constrastive loss: F.triplet_margin_loss, F.triplet_margin_with_distance_loss
    # TODO: making reserved variables as multivariate normal, calculating kl accordingly?

    # Notes:
    # - When only reserved is active for `sex`, you have two blob at the end.
    # - When unreserved loss is also active you have one blob.
    # - The second makes sure the same cell types are put together as we have `within_label` option activated.
