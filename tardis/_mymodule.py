#!/usr/bin/env python3

import logging
import warnings

import torch
from scvi import REGISTRY_KEYS, settings
from scvi.module import VAE
from scvi.module.base import LossOutput, auto_move_data
from torch.distributions import kl_divergence as kl

from ._auxillarylossesmixin import AuxillaryLossesMixin
from ._disentanglementmanager import DisentanglementManager
from ._metricsmixin import ModuleMetricsMixin
from ._myconstants import (
    AUXILLARY_LOSS_MEAN,
    LOSS_NAMING_DELIMITER,
    REGISTRY_KEY_DISENTANGLEMENT_TARGETS_TENSORS,
    WEIGHTED_LOSS_SUFFIX,
    minified_method_not_supported_message,
)
from scvi.nn import one_hot
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from torch.distributions import Normal
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


class MyModule(VAE, AuxillaryLossesMixin, ModuleMetricsMixin):

    def __init__(self, *args, include_auxillary_loss: bool = True, **kwargs):
        super().__init__(*args, **kwargs)

        # Modify `DisentenglementTargetManager` to define reserved latents in loss calculations.
        self.n_total_reserved_latent = 0
        for dtconfig in DisentanglementManager.configurations.items:
            dtconfig.reserved_latent_indices = list(
                range(self.n_total_reserved_latent, self.n_total_reserved_latent + dtconfig.n_reserved_latent)
            )
            dtconfig.unreserved_latent_indices = [
                i for i in range(self.n_latent) if i not in dtconfig.reserved_latent_indices
            ]
            self.n_total_reserved_latent += dtconfig.n_reserved_latent
        if self.n_latent - self.n_total_reserved_latent < 1:
            raise ValueError("Not enough latent space variables to reserve for targets.")
        self.n_total_unreserved_latent = self.n_latent - self.n_total_reserved_latent
        DisentanglementManager.configurations.unreserved_latent_indices = list(
            range(self.n_total_reserved_latent, self.n_latent)
        )  # If no target is defined, this list will contain all latent space variables.
        DisentanglementManager.configurations.latent_indices = list(range(self.n_latent))

        self.auxillary_losses_keys: list[str] | None = None
        self.include_auxillary_loss = include_auxillary_loss

        # Remove the variable got from VAE initialization due to it is inherited from BaseMinifiedModeModuleClass.
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

    @auto_move_data
    def generative(
        self,
        z,
        library,
        batch_index,
        cont_covs=None,
        cat_covs=None,
        size_factor=None,
        y=None,
        transform_batch=None,
    ):
        if cont_covs is None:
            decoder_input = z
        elif z.dim() != cont_covs.dim():
            decoder_input = torch.cat(
                [z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1
            )
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if cat_covs is not None:
            ### TARDIS: TEMPORARY SOLUTION
            raise NotImplementedError("Tardis temp solution didn't consider this case yet.")
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch
            ### TARDIS: TEMPORARY SOLUTION
            batch_index = torch.zeros_like(batch_index)

        if not self.use_size_factor_key:
            size_factor = library

        px_scale, px_r, px_rate, px_dropout = self.decoder(
            self.dispersion,
            decoder_input,
            size_factor,
            ### TARDIS: TEMPORARY SOLUTION
            # batch_index,
            torch.zeros_like(batch_index),
            *categorical_input,
            y,
        )
        
        if self.dispersion == "gene-label":
            px_r = F.linear(
                one_hot(y, self.n_labels), self.px_r
            )  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
            raise NotImplementedError("Tardis temp solution didn't consider this case yet.")
        elif self.dispersion == "gene":
            px_r = self.px_r

        px_r = torch.exp(px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)

        # Priors
        if self.use_observed_lib_size:
            pl = None
        else:
            (
                local_library_log_means,
                local_library_log_vars,
            ) = self._compute_local_library_params(batch_index)
            pl = Normal(local_library_log_means, local_library_log_vars.sqrt())
        pz = Normal(torch.zeros_like(z), torch.ones_like(z))
        return {
            "px": px,
            "pl": pl,
            "pz": pz,
        }

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

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
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

        # Note that `kl_weight` is determined dynamically depending on `n_epochs_kl_warmup` parameter.
        weighted_kl_local = kl_weight * kl_local_for_warmup + kl_local_no_warmup

        # `weighted_auxillary_losses` is also determined dynamically, depending on `warmup_epoch_range` parameter.

        if REGISTRY_KEY_DISENTANGLEMENT_TARGETS_TENSORS in tensors:
            # Then we are using the MyAnnDataLoader, which includes the counteractive minibatches. This happens
            # when the model is in training phase. Have a look at the `_data_loader_cls` in Model.
            # Removing this if-else loop causes model not loadable from drive.
            weighted_auxillary_losses, auxillary_losses = self.calculate_auxillary_losses(tensors, inference_outputs)
        else:
            weighted_auxillary_losses, auxillary_losses = dict(), dict()

        if len(weighted_auxillary_losses) > 0:
            total_weighted_auxillary_losses = torch.sum(torch.stack(list(weighted_auxillary_losses.values())), dim=0)
            total_auxillary_losses = torch.sum(torch.stack(list(auxillary_losses.values())), dim=0)
        else:
            total_weighted_auxillary_losses = torch.zeros(reconst_loss.shape[0]).to(reconst_loss.device)
            total_auxillary_losses = torch.zeros(reconst_loss.shape[0]).to(reconst_loss.device)

        # Report the losses
        report_auxillary_losses = dict()
        report_auxillary_losses.update({i: torch.mean(auxillary_losses[i]) for i in auxillary_losses})
        report_auxillary_losses[AUXILLARY_LOSS_MEAN] = torch.mean(total_auxillary_losses)
        # Also report weighted losses. Note that this is not used in progress bar etc, as like `kl_local`.
        report_auxillary_losses.update(
            {
                LOSS_NAMING_DELIMITER.join([i, WEIGHTED_LOSS_SUFFIX]): torch.mean(weighted_auxillary_losses[i])
                for i in weighted_auxillary_losses
            }
        )
        report_auxillary_losses[LOSS_NAMING_DELIMITER.join([AUXILLARY_LOSS_MEAN, WEIGHTED_LOSS_SUFFIX])] = torch.mean(
            total_weighted_auxillary_losses
        )
        # Also report kl after weighting
        report_auxillary_losses[LOSS_NAMING_DELIMITER.join(["kl_local", WEIGHTED_LOSS_SUFFIX])] = torch.mean(
            weighted_kl_local
        )

        # Add to the total loss, or do not add for debugging etc purposes
        if self.include_auxillary_loss:
            loss = torch.mean(reconst_loss + weighted_kl_local + total_weighted_auxillary_losses)
        else:
            warnings.warn(
                message="Auxillary loss is not added to the total loss. (include_auxillary_loss=False)",
                category=UserWarning,
                stacklevel=settings.warnings_stacklevel,
            )
            loss = torch.mean(reconst_loss + weighted_kl_local)

        kl_local = {
            "kl_divergence_l": kl_divergence_l,
            "kl_divergence_z": kl_divergence_z,
        }

        if self.auxillary_losses_keys is None and len(weighted_auxillary_losses) > 0:
            self.auxillary_losses_keys = list(weighted_auxillary_losses.keys())

        return LossOutput(
            loss=loss, reconstruction_loss=reconst_loss, kl_local=kl_local, extra_metrics=report_auxillary_losses
        )
