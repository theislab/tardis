#!/usr/bin/env python3

import torch
import torch.nn.functional as F
from scvi.module.base import auto_move_data
from torch.distributions import Normal, kl_divergence
from torch.nn.functional import cosine_similarity

from ._disentenglementtargetmanager import DisentenglementTargetManager
from ._myconstants import REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS


class MyModuleAuxillaryLosses:

    @torch.inference_mode()
    @auto_move_data
    def inference_counteractive_minibatch(self, counteractive_minibatch_tensors):
        inference_inputs = self._get_inference_input(counteractive_minibatch_tensors)
        inference_outputs = self.inference(**inference_inputs)
        return inference_outputs

    def calculate_auxillary_losses(self, tensors, inference_outputs):

        result = dict()
        for target_obs_key_ind, target_obs_key in enumerate(
            DisentenglementTargetManager.configurations.get_ordered_obs_key()
        ):
            counteractive_minibatch_tensors = tensors[REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS][target_obs_key]
            inference_outputs_counteractive = self.inference_counteractive_minibatch(counteractive_minibatch_tensors)
            config_main = DisentenglementTargetManager.configurations.get_by_index(target_obs_key_ind)

            for auxillary_loss_key in config_main.auxillary_losses.items:
                config_individual = getattr(config_main.auxillary_losses, auxillary_loss_key)
                relevant_latent_indices = MyModuleAuxillaryLosses.relevant_latent_indices(
                    auxillary_loss_key=auxillary_loss_key, target_obs_key_ind=target_obs_key_ind
                )
                loss = self.calculate_auxillary_loss(
                    inference_outputs=inference_outputs,
                    inference_outputs_counteractive=inference_outputs_counteractive,
                    config=config_individual,
                    relevant_latent_indices=relevant_latent_indices,
                )
                result[config_individual.loss_identifier_string] = loss

        return result

    @staticmethod
    def relevant_latent_indices(auxillary_loss_key, target_obs_key_ind):
        if auxillary_loss_key == "complete_latent":
            return DisentenglementTargetManager.configurations.latent_indices

        elif auxillary_loss_key == "reserved_subset":
            return DisentenglementTargetManager.configurations.get_by_index(target_obs_key_ind).reserved_latent_indices

        elif auxillary_loss_key == "unreserved_subset":
            return DisentenglementTargetManager.configurations.get_by_index(
                target_obs_key_ind
            ).unreserved_latent_indices

        else:
            raise ValueError("Unknown auxillary loss.")

    def calculate_auxillary_loss(
        self, inference_outputs, inference_outputs_counteractive, config, relevant_latent_indices
    ):
        if not config.apply:  # TODO: include warm-up epoch period
            return torch.zeros(inference_outputs["z"].shape[0]).to(inference_outputs["z"].device)

        latent_distribution = self.latent_distribution

        if config.method == "wasserstein_qz" and latent_distribution == "normal":
            func = self.wasserstein_with_normal_parameters

        elif config.method == "wasserstein_qz" and latent_distribution == "ln":
            raise NotImplementedError("`wasserstein` method with `ln` latent distribution is not implemented yet.")

        elif config.method == "mse_z":
            func = self.mse_with_reparametrized_z

        elif config.method == "mae_z":
            func = self.mae_with_reparametrized_z

        elif config.method == "kl_qz" and latent_distribution == "normal":
            raise NotImplementedError("`kl` method with `normal` latent distribution is not implemented yet.")

        elif config.method == "kl_qz" and latent_distribution == "ln":
            raise NotImplementedError("`kl` method with `ln` latent distribution is not implemented yet.")
        elif config.method == "triplet":
            func = self.triplet_loss
        else:
            raise ValueError("Unknown auxillary loss method.")

        loss = func(inference_outputs, inference_outputs_counteractive, relevant_latent_indices, **config.method_kwargs)

        return self.final_transformation(loss=loss, transformation_key=config.transformation) * config.weight

    def final_transformation(self, loss: torch.tensor, transformation_key: str):
        if transformation_key == "none":
            return loss

        elif transformation_key == "negative":
            return loss * -1

        elif transformation_key == "sigmoid":
            return F.sigmoid(loss)

        elif transformation_key == "inverse":
            return 1 / (loss + 1)

        elif transformation_key == "exponential_decay":
            return torch.exp(-loss)

        elif transformation_key == "tanh":
            return F.tanh(loss)

        else:
            raise ValueError(f"Unknown transformation key in auxillary loss definition `{transformation_key}`.")

    def mse_with_reparametrized_z(self, inference_outputs, inference_outputs_counteractive, relevant_latent_indices):
        return F.mse_loss(
            input=inference_outputs_counteractive["z"][:, relevant_latent_indices].clone(),  # true
            target=inference_outputs["z"][:, relevant_latent_indices].clone(),  # pred
            reduction="none",
        ).mean(dim=1)

    def mae_with_reparametrized_z(self, inference_outputs, inference_outputs_counteractive, relevant_latent_indices):
        return F.l1_loss(
            input=inference_outputs_counteractive["z"][:, relevant_latent_indices].clone(),  # true
            target=inference_outputs["z"][:, relevant_latent_indices].clone(),  # pred
            reduction="none",
        ).mean(dim=1)
        # TODO: CHECK IF `clone()` NEEDED.

    def cosine_similarity_with_reparametrized_z(
        self, inference_outputs, inference_outputs_counteractive, relevant_latent_indices
    ):
        return F.cosine_similarity(
            x1=inference_outputs_counteractive["z"][:, relevant_latent_indices].clone(),
            x2=inference_outputs["z"][:, relevant_latent_indices].clone(),
        )

    def cosine_embedding_with_reparametrized_z(
        self, inference_outputs, inference_outputs_counteractive, relevant_latent_indices
    ):
        raise NotImplementedError
        # return F.cosine_embedding_loss
        # reduction="none"

    def wasserstein_with_normal_parameters(
        self, inference_outputs, inference_outputs_counteractive, relevant_latent_indices, epsilon=1e-8
    ):

        # W_{2,i}^2 = (\mu_{1,i} - \mu_{2,i})^2 + (\sigma_{1,i}^2 + \sigma_{2,i}^2 - 2 \sigma_{1,i} \sigma_{2,i})

        # This formula assumes that the covariance matrices are diagonal, allowing for an element-wise computation.
        # A covariance matrix is said to be diagonal if all off-diagonal elements are zero. This means that there's
        # no covariance between different dimensions—each dimension varies independently of the others.

        qz_inference = inference_outputs["qz"]
        qz_counteractive = inference_outputs_counteractive["qz"]
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
        return (mean_diff_sq + trace_term).mean(dim=1)

    def kl_with_normal_parameters(
        self, inference_outputs, inference_outputs_counteractive, relevant_latent_indices, epsilon=1e-8
    ):
        raise NotImplementedError

    
    def triplet_loss(self, inference_outputs, inference_outputs_counteractive, relevant_latent_indices, margin=2.):
        
        positive_indices = [i for i in range(inference_outputs["z"].shape[1]) if i not in relevant_latent_indices]
    
        positive = F.mse_loss(
            input=inference_outputs_counteractive["z"][:, positive_indices].clone(),  # true
            target=inference_outputs["z"][:, positive_indices].clone(),  # pred
            reduction="none",
        ).mean(dim=1)
        
        negative = F.mse_loss(
            input=inference_outputs_counteractive["z"][:, relevant_latent_indices].clone(),  # true
            target=inference_outputs["z"][:, relevant_latent_indices].clone(),  # pred
            reduction="none",
        ).mean(dim=1)

        return torch.relu(positive - negative + margin)
    
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
