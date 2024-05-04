#!/usr/bin/env python3

import logging
import warnings
from collections.abc import Iterable
from typing import Callable, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F
from scvi import REGISTRY_KEYS, settings
from scvi._types import Tunable
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module import VAE
from scvi.module.base import BaseMinifiedModeModuleClass, LossOutput, auto_move_data
from scvi.nn import DecoderSCVI, Encoder, one_hot
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl

from ._auxillarylossesmixin import AuxillaryLossesMixin
from ._disentanglementmanager import DisentanglementManager
from ._metricsmixin import ModuleMetricsMixin
from ._mycomponents import DecoderSCVIReservedLatentInjection
from ._myconstants import (
    AUXILLARY_LOSS_MEAN,
    LOSS_NAMING_DELIMITER,
    REGISTRY_KEY_DISENTANGLEMENT_TARGETS,
    REGISTRY_KEY_DISENTANGLEMENT_TARGETS_TENSORS,
    WEIGHTED_LOSS_SUFFIX,
    minified_method_not_supported_message,
)

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


class MyModule(VAE, BaseMinifiedModeModuleClass, AuxillaryLossesMixin, ModuleMetricsMixin):

    def __init__(
        self,
        n_input: int,
        n_batch: int = 0,
        n_labels: int = 0,
        n_hidden: Tunable[int] = 128,
        n_latent: Tunable[int] = 10,
        n_layers: Tunable[int] = 1,
        n_continuous_cov: int = 0,
        n_cats_per_cov: Optional[Iterable[int]] = None,
        dropout_rate: Tunable[float] = 0.1,
        dispersion: Tunable[Literal["gene", "gene-batch", "gene-label", "gene-cell"]] = "gene",
        log_variational: Tunable[bool] = True,
        gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
        latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
        deeply_inject_covariates: Tunable[bool] = True,
        use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
        use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
        use_size_factor_key: bool = False,
        use_observed_lib_size: Tunable[bool] = True,
        library_log_means: Optional[np.ndarray] = None,
        library_log_vars: Optional[np.ndarray] = None,
        var_activation: Tunable[Callable] = None,
        extra_encoder_kwargs: Optional[dict] = None,
        extra_decoder_kwargs: Optional[dict] = None,
        # changed default: `encode_covariates`: `False` in scVI
        # We want encoder to use covariates so it is set to True. encode covariates only used in encoder in scVI.
        encode_covariates: Tunable[bool] = True,
        # extra parameters:
        n_cats_per_disentenglement_covariates: Optional[Iterable[int]] = None,
        deeply_inject_disentengled_latents: bool = True,
        include_auxillary_loss: bool = True,
        beta_kl_weight: float = 1.0,
    ):
        BaseMinifiedModeModuleClass.__init__(self)
        self.dispersion = dispersion
        self.n_latent = n_latent
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.n_batch = n_batch
        self.n_labels = n_labels
        self.latent_distribution = latent_distribution
        self.encode_covariates = encode_covariates

        self.use_size_factor_key = use_size_factor_key
        self.use_observed_lib_size = use_size_factor_key or use_observed_lib_size
        if not self.use_observed_lib_size:
            if library_log_means is None or library_log_vars is None:
                raise ValueError(
                    "If not using observed_lib_size, " "must provide library_log_means and library_log_vars."
                )

            self.register_buffer("library_log_means", torch.from_numpy(library_log_means).float())
            self.register_buffer("library_log_vars", torch.from_numpy(library_log_vars).float())

        if self.dispersion == "gene":
            self.px_r = torch.nn.Parameter(torch.randn(n_input))
        elif self.dispersion == "gene-batch":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_batch))
        elif self.dispersion == "gene-label":
            self.px_r = torch.nn.Parameter(torch.randn(n_input, n_labels))
        elif self.dispersion == "gene-cell":
            pass
        else:
            raise ValueError(
                "dispersion must be one of ['gene', 'gene-batch',"
                " 'gene-label', 'gene-cell'], but input was "
                "{}.format(self.dispersion)"
            )

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # z encoder goes from the n_input-dimensional data to an n_latent-d
        # latent space representation
        n_input_encoder = n_input + n_continuous_cov * encode_covariates
        cat_list = [n_batch] + list([] if n_cats_per_cov is None else n_cats_per_cov)
        encoder_cat_list = cat_list + list(
            [] if n_cats_per_disentenglement_covariates is None else n_cats_per_disentenglement_covariates
        )
        encoder_cat_list = encoder_cat_list if encode_covariates else None
        _extra_encoder_kwargs = extra_encoder_kwargs or {}
        self.z_encoder = Encoder(
            n_input_encoder,
            n_latent,
            n_cat_list=encoder_cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            distribution=latent_distribution,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # l encoder goes from n_input-dimensional data to 1-d library size
        self.l_encoder = Encoder(
            n_input_encoder,
            1,
            n_layers=1,
            n_cat_list=encoder_cat_list,
            n_hidden=n_hidden,
            dropout_rate=dropout_rate,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_encoder,
            use_layer_norm=use_layer_norm_encoder,
            var_activation=var_activation,
            return_dist=True,
            **_extra_encoder_kwargs,
        )
        # decoder goes from n_latent-dimensional space to n_input-d data
        n_input_decoder = n_latent + n_continuous_cov
        _extra_decoder_kwargs = extra_decoder_kwargs or {}
        self.decoder = DecoderSCVI(
            n_input_decoder,
            n_input,
            n_cat_list=cat_list,
            n_layers=n_layers,
            n_hidden=n_hidden,
            inject_covariates=deeply_inject_covariates,
            use_batch_norm=use_batch_norm_decoder,
            use_layer_norm=use_layer_norm_decoder,
            scale_activation="softplus" if use_size_factor_key else "softmax",
            **_extra_decoder_kwargs,
        )

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
        DisentanglementManager.configurations.reserved_latent_indices = list(range(self.n_total_reserved_latent))
        DisentanglementManager.configurations.unreserved_latent_indices = list(
            range(self.n_total_reserved_latent, self.n_latent)
        )  # If no target is defined, this list will contain all latent space variables.
        DisentanglementManager.configurations.latent_indices = list(range(self.n_latent))

        self.auxillary_losses_keys: list[str] | None = None
        self.include_auxillary_loss = include_auxillary_loss
        self.beta_kl_weight = beta_kl_weight
        self.n_cats_per_disentenglement_covariates = n_cats_per_disentenglement_covariates
        self.deeply_inject_covariates = deeply_inject_covariates
        self.deeply_inject_disentengled_latents = deeply_inject_disentengled_latents
        self.deeply_inject_disentengled_latents_activated = (
            self.deeply_inject_disentengled_latents and len(DisentanglementManager.configurations) != 0
        )

        if self.deeply_inject_disentengled_latents_activated:
            if not self.deeply_inject_covariates:
                raise ValueError(
                    "`deeply_inject_covariates` should be `True` if "
                    "`deeply_inject_disentengled_latents` is set to `True`."
                )
            # redefine decoder
            n_input_decoder = n_latent - self.n_total_reserved_latent + n_continuous_cov
            self.decoder = DecoderSCVIReservedLatentInjection(
                n_input=n_input_decoder,
                n_output=n_input,
                n_reserved=self.n_total_reserved_latent,
                n_cat_list=cat_list,
                n_layers=n_layers,
                n_hidden=n_hidden,
                inject_covariates=deeply_inject_covariates,
                use_batch_norm=use_batch_norm_decoder,
                use_layer_norm=use_layer_norm_decoder,
                scale_activation="softplus" if use_size_factor_key else "softmax",
                **_extra_decoder_kwargs,
            )

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

    def _get_inference_input(
        self,
        tensors,
    ):
        batch_index = tensors[REGISTRY_KEYS.BATCH_KEY]

        cont_key = REGISTRY_KEYS.CONT_COVS_KEY
        cont_covs = tensors[cont_key] if cont_key in tensors.keys() else None

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key] if cat_key in tensors.keys() else None

        dist_key = REGISTRY_KEY_DISENTANGLEMENT_TARGETS
        dist_covs = tensors[dist_key] if dist_key in tensors.keys() else None

        if cat_covs is None and dist_covs is None:
            stacked_cat_covs = None
        elif cat_covs is None:
            stacked_cat_covs = dist_covs
        elif dist_covs is None:
            stacked_cat_covs = cat_covs
        else:
            stacked_cat_covs = torch.hstack([cat_covs, dist_covs])

        x = tensors[REGISTRY_KEYS.X_KEY]
        input_dict = {
            "x": x,
            "batch_index": batch_index,
            "cont_covs": cont_covs,
            "cat_covs": stacked_cat_covs,
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
            decoder_input = torch.cat([z, cont_covs.unsqueeze(0).expand(z.size(0), -1, -1)], dim=-1)
        else:
            decoder_input = torch.cat([z, cont_covs], dim=-1)

        if transform_batch is not None:
            batch_index = torch.ones_like(batch_index) * transform_batch

        if not self.use_size_factor_key:
            size_factor = library

        if cat_covs is not None:
            categorical_input = torch.split(cat_covs, 1, dim=1)
        else:
            categorical_input = ()

        if self.deeply_inject_disentengled_latents_activated:
            r = self.n_total_reserved_latent
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion,
                decoder_input[:, r:],  # as the cont_covs are added at the end
                size_factor,
                batch_index,
                *categorical_input,
                y,
                reserved_latent_injection=decoder_input[:, :r].detach()  # TODO: detach or not!
            )
        else:
            px_scale, px_r, px_rate, px_dropout = self.decoder(
                self.dispersion, decoder_input, size_factor, batch_index, *categorical_input, y
            )

        if self.dispersion == "gene-label":
            px_r = F.linear(one_hot(y, self.n_labels), self.px_r)  # px_r gets transposed - last dimension is nb genes
        elif self.dispersion == "gene-batch":
            px_r = F.linear(one_hot(batch_index, self.n_batch), self.px_r)
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
        weighted_kl_local = kl_weight * kl_local_for_warmup * self.beta_kl_weight + kl_local_no_warmup

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
