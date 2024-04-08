#!/usr/bin/env python3

import torch
from scvi.module.base import auto_move_data
from torch.distributions import kl_divergence as kl

from ._DEBUG import DEBUG
from ._disentenglementtargetmanager import DisentenglementTargetManager
from ._myconstants import LOSS_NAMING_DELIMITER, REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS


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
                relevant_latent_indices = MyModuleAuxillaryLosses._relevant_latent_indices(
                    auxillary_loss_key=auxillary_loss_key, target_obs_key_ind=target_obs_key_ind
                )
                loss = self.calculate_auxillary_loss(
                    inference_outputs=inference_outputs,
                    inference_outputs_counteractive=inference_outputs_counteractive,
                    config=config_individual,
                    relevant_latent_indices=relevant_latent_indices,
                )
                result[LOSS_NAMING_DELIMITER.join([target_obs_key, auxillary_loss_key])] = loss

        return result

    @staticmethod
    def _relevant_latent_indices(auxillary_loss_key, target_obs_key_ind):
        if auxillary_loss_key == "complete_latent":
            pass  # TODO
        elif auxillary_loss_key == "reserved_subset":
            pass  # TODO
        elif auxillary_loss_key == "unreserved_subset":
            pass  # TODO
        else:
            raise ValueError("Unknown auxillary loss.")

    def calculate_auxillary_loss(
        self, inference_outputs, inference_outputs_counteractive, config, relevant_latent_indices
    ):
        if not config.apply:
            return torch.tensor(-212.0)

        latent_distribution = self.latent_distribution
        if config.method == "wasserstein" and latent_distribution == "normal":
            func = self.wasserstein_with_normal_parameters

        elif config.method == "wasserstein" and latent_distribution == "ln":
            raise NotImplementedError("`wasserstein` method with `ln` latent distribution is not implemented yet.")
        elif config.method == "kl" and latent_distribution == "normal":
            # TODO: Also implement this one.
            raise NotImplementedError("`kl` method with `normal` latent distribution is not implemented yet.")
        elif config.method == "kl" and latent_distribution == "ln":
            raise NotImplementedError("`kl` method with `ln` latent distribution is not implemented yet.")
        else:
            raise ValueError("Unknown auxillary loss method and latent distribution combination.")

        # print(inference_outputs["qz"])
        DEBUG.inference_outputs = inference_outputs["qz"]

        # TODO: subset the latent with the provided reserved_latent list
        loss = func(inference_outputs["qz"], inference_outputs_counteractive["qz"], **config.method_kwargs)
        return loss * config.weight * (-1 if config.negative_sign is True else +1)

    def wasserstein_with_normal_parameters(self, qz_minibatch, qz_counteractive):
        # TODO: calculate wasserstein loss.
        return torch.tensor(1.0)

    def wasserstein_with_ln_parameters(self):
        pass

    def kl_with_normal_parameters(self):
        pass

    def kl_with_ln_parameters(self):
        pass
