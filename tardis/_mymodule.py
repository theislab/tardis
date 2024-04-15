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

import logging
import warnings

import torch
from scvi import REGISTRY_KEYS, settings
from scvi.module import VAE
from scvi.module.base import LossOutput, auto_move_data
from torch.distributions import kl_divergence as kl

from ._disentenglementtargetmanager import DisentanglementTargetManager
from ._myconstants import (
    LOSS_MEAN_BEFORE_WEIGHT,
    minified_method_not_supported_message,
)

torch.backends.cudnn.benchmark = True

logger = logging.getLogger(__name__)


class MyModule(VAE):

    def __init__(self, *args, include_auxillary_loss: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.auxillary_losses_keys: list[str] | None = None
        self.include_auxillary_loss = include_auxillary_loss

        # Modify `DisentenglementTargetManager` to define reserved latents in loss calculations.

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
            ql, library_encoded = self.l_encoder(
                encoder_input, batch_index, *categorical_input
            )
            library = library_encoded

        if n_samples > 1:
            untran_z = qz.sample((n_samples,))
            z = self.z_encoder.z_transformation(untran_z)
            if self.use_observed_lib_size:
                library = library.unsqueeze(0).expand(
                    (n_samples, library.size(0), library.size(1))
                )
            else:
                library = ql.sample((n_samples,))
        outputs = {"z": z, "qz": qz, "ql": ql, "library": library}
        return outputs

    @torch.inference_mode()
    @auto_move_data
    def inference_counteractive_minibatch(self, counteractive_minibatch_tensors):
        counteractive_inference_inputs = self._get_inference_input(
            counteractive_minibatch_tensors
        )
        counteractive_inference_outputs = self.inference(
            **counteractive_inference_inputs
        )
        return counteractive_inference_outputs

    def loss(
        self,
        tensors,
        inference_outputs,
        generative_outputs,
        kl_weight: float = 1.0,
    ):
        x = tensors[REGISTRY_KEYS.X_KEY]
        kl_divergence_z = kl(inference_outputs["qz"], generative_outputs["pz"]).sum(
            dim=-1
        )
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

        if len(DisentanglementTargetManager.disentanglements) > 0:
            counteractive_inference_outputs = self.inference_counteractive_minibatch(
                tensors
            )
            auxillary_losses = DisentanglementTargetManager.get_loss(
                inference_outputs, counteractive_inference_outputs
            )
        else:
            total_auxillary_losses = {}

        if self.auxillary_losses_keys is None and len(auxillary_losses) > 0:
            self.auxillary_losses_keys = list(auxillary_losses.keys())

        if len(auxillary_losses) > 0:
            total_auxillary_losses = torch.sum(
                torch.stack(list(auxillary_losses.values())), dim=0
            )
        else:
            total_auxillary_losses = torch.zeros(reconst_loss.shape[0]).to(
                reconst_loss.device
            )
        report_auxillary_losses = {
            k: torch.mean(v) for k, v in auxillary_losses.items()
        }
        report_auxillary_losses[LOSS_MEAN_BEFORE_WEIGHT] = torch.mean(
            total_auxillary_losses
        )

        # TODO: Test below line with different options.
        # Note that `kl_weight` is determined dynamically depending on `n_epochs_kl_warmup` parameter.
        total_auxillary_losses = kl_weight * total_auxillary_losses

        if self.include_auxillary_loss:
            loss = torch.mean(reconst_loss + weighted_kl_local + total_auxillary_losses)
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

        return LossOutput(
            loss=loss,
            reconstruction_loss=reconst_loss,
            kl_local=kl_local,
            extra_metrics=report_auxillary_losses,
        )
