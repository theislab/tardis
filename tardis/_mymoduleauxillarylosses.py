#!/usr/bin/env python3

import torch
from scvi import REGISTRY_KEYS

from scvi.module.base import auto_move_data
from torch.distributions import kl_divergence as kl

from ._disentenglementtargetmanager import DisentenglementTargetManager
from ._myconstants import REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS, LOSS_DELIMITER


class MyModuleAuxillaryLosses:

    @torch.inference_mode()
    @auto_move_data
    def inference_counteractive_minibatch(self, counteractive_minibatch_tensors):
        # look at forward pass and do the same
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
            config_main = DisentenglementTargetManager.configurations.get_by_index(target_obs_key_ind).auxillary_losses
            
            for auxillary_loss_key in config_main.items:
                config_individual = getattr(config_main, auxillary_loss_key)
                loss = self.calculate_auxillary_loss(
                    inference_outputs, inference_outputs_counteractive, config_individual,
                    # TODO: pass the relevant reserved_latent list
                )
                result[LOSS_DELIMITER.join([target_obs_key, auxillary_loss_key])] = loss
        
        return result

    def calculate_auxillary_loss(self, inference_outputs, inference_outputs_counteractive, config):
        if not config.apply:
            return torch.tensor(0.0)
        
        latent_distribution = self.latent_distribution
        if config.method == "wasserstein" and latent_distribution == "normal":
            func = self.wasserstein_with_normal_parameters
            
        elif config.method == "wasserstein" and latent_distribution == "ln":
            raise NotImplementedError(f"`wasserstein` method with `ln` latent distribution is not implemented yet.")
        elif config.method == "kl" and latent_distribution == "normal":
            # TODO: Also implement this one.
            raise NotImplementedError(f"`kl` method with `normal` latent distribution is not implemented yet.")
        elif config.method == "kl" and latent_distribution == "ln":
            raise NotImplementedError(f"`kl` method with `ln` latent distribution is not implemented yet.")
        else:
            raise ValueError("Unknown auxillary loss method and latent distribution combination.")
        
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
    
    