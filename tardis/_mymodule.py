#!/usr/bin/env python3

import copy
import logging

import torch
from scvi import REGISTRY_KEYS
from scvi.module import VAE
from scvi.module.base import LossOutput, auto_move_data
from torch.distributions import kl_divergence as kl

from ._DEBUG import DEBUG  # noqa
from ._disentenglementtargetmanager import DisentenglementTargetManager
from ._myconstants import minified_method_not_supported_message

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


class MyModule(VAE):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Modify `DisentenglementTargetManager` to define reserved latents in loss calculations.
        self.n_total_reserved_latent = 0
        for dtconfig in DisentenglementTargetManager.configurations.items:
            dtconfig.reserved_latent_indices = list(
                range(self.n_total_reserved_latent, self.n_total_reserved_latent + dtconfig.n_reserved_latent)
            )
            self.n_total_reserved_latent += dtconfig.n_reserved_latent
        if self.n_latent - self.n_total_reserved_latent < 1:
            raise ValueError("Not enough latent space variables to reserve for targets.")
        self.n_total_unreserved_latent = self.n_latent - self.n_total_reserved_latent
        DisentenglementTargetManager.configurations.unreserved_latent_indices = list(
            range(self.n_total_reserved_latent, self.n_latent)
        )  # If no target is defined, this list will contain all latent space variables.

        del self._minified_data_type

    @property
    def minified_data_type(self, *args, **kwargs):
        raise NotImplementedError(minified_method_not_supported_message)

    @minified_data_type.setter
    def minified_data_type(self, *args, **kwargs):
        raise NotImplementedError(minified_method_not_supported_message)

    def _cached_inference(self, *args, **kwargs):
        raise NotImplementedError(minified_method_not_supported_message)

    def _regular_inference(self, *args, **kwargs):
        raise NotImplementedError(minified_method_not_supported_message)

    def _get_inference_input(
        self,
        tensors,
    ):
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        x = tensors[REGISTRY_KEYS.X_KEY]
        input_dict = {
            "x": x,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": cat_covs,
        }

        return input_dict

    @auto_move_data
    def inference(
        self,
        x,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        n_samples=1,
    ):
        """High level inference method.

        Runs the inference (encoder) model.
        """
        x_ = x
        if self.use_observed_lib_size:
            library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        if cont_covs is not None and self.encode_covariates:
            encoder_input = torch.cat((x_, cont_covs), dim=-1)
        else:
            encoder_input = x_
        if cat_covs is not None and self.encode_covariates:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()
        qz, z = self.z_encoder(encoder_input, batch_index, *categorical_input)
        ql = None
        if not self.use_observed_lib_size:
            ql, library_encoded = self.l_encoder(encoder_input, batch_index, *categorical_input)
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand((n_samples, library.size(0), library.size(1)))
            else:
                library = ql.sample((n_samples,))
        outputs = {"z": z, "qz": qz, "ql": ql, "library": library}
        return outputs

    def _debug(self, tensors):

        a = copy.deepcopy(locals())
        for i in a:
            if i not in ["cls", "mro"]:
                exec(f"DEBUG.{i} = a[i]")

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        self._debug(tensors)

        x = tensors[REGISTRY_KEYS.X_KEY]
        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(dim=-1)
        if not self.use_observed_lib_size:
            kl_divergence_l = kl(
                inference_outputs["ql"],
                generative_outputs["pl"],
            ).sum(dim=1)
        else:
            kl_divergence_l = torch.tensor(0.0, device=x.device)

        reconst_loss = -generative_outputs["px"].log_prob(x).sum(-1)

        kl_local_for_warmup = kl_divergence_z
        kl_local_no_warmup = kl_divergence_l

        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z,
        }
        return LossOutput(loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local)

    @torch.inference_mode()
    @auto_move_data
    def calculate_r2_reconstruction(self):
        raise NotImplementedError
