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

import torch
import torch.nn.functional as F
from scvi.module.base import auto_move_data

from ._disentenglementtargetmanager import DisentenglementTargetManager
from ._myconstants import REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS
from .losses import LOSSES

TRANSFORMATIONS = {
    "none": lambda x: x,
    "negative": torch.neg,
    "sigmoid": F.sigmoid,
    "inverse": lambda x: 1 / (x + 1),
    "exponential_decay": lambda x: torch.exp(-x),
    "tanh": F.tanh,
}


class AuxillaryLossesMixin:

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

    def calculate_auxillary_losses(self, tensors, inference_outputs):

        result = dict()
        for target_obs_key_ind, target_obs_key in enumerate(
            DisentenglementTargetManager.configurations.get_ordered_obs_key()
        ):
            counteractive_minibatch_tensors = tensors[
                REGISTRY_KEY_DISENTENGLEMENT_TARGETS_TENSORS
            ][target_obs_key]
            inference_outputs_counteractive = self.inference_counteractive_minibatch(
                counteractive_minibatch_tensors
            )
            config = DisentenglementTargetManager.configurations.get_by_index(
                target_obs_key_ind
            )

            for auxillary_loss_key in config.auxillary_losses.items:
                loss_config = getattr(config.auxillary_losses, auxillary_loss_key)
                relevant_latent_indices = AuxillaryLossesMixin.relevant_latent_indices(
                    auxillary_loss_key=auxillary_loss_key,
                    target_obs_key_ind=target_obs_key_ind,
                )

                if (
                    not isinstance(loss_config["transformation"], str)
                    or not loss_config["transformation"] in TRANSFORMATIONS.keys()
                ):
                    raise ValueError(
                        "Transformation key should be a string and one of the following: {}".format(
                            ", ".join(TRANSFORMATIONS.keys())
                        )
                    )

                method = loss_config["method"].lower().strip().split("_")[0]

                if not isinstance(method, str) or not method in LOSSES.keys():
                    raise ValueError(
                        "Transformation key should be a string and one of the following: {}".format(
                            ", ".join(LOSSES.keys())
                        )
                    )

                loss_fn = LOSSES[method](
                    weight=loss_config["weight"],
                    # method=loss_config["method"],
                    method_kwargs=loss_config["method_kwargs"],
                    # transformation=loss_config["transformation"],
                    # progress_bar=loss_config["progress_bar"],
                    loss_identifier_string=loss_config.get(
                        "loss_identifier_string", ""
                    ),
                )

                loss = TRANSFORMATIONS[loss_config["transformation"]](
                    loss_fn.forward(
                        outputs=inference_outputs,
                        # Note that always clone the the tensor of interest in
                        # `tensors` or `inference_outputs` before calculating an auxillary loss.
                        counteractive_outputs=inference_outputs_counteractive,
                        # config=loss_config,
                        relevant_latent_indices=relevant_latent_indices,
                    )
                )
                result[method] = loss

        return result

    @staticmethod
    def relevant_latent_indices(auxillary_loss_key, target_obs_key_ind):
        if auxillary_loss_key == "complete_latent":
            return torch.tensor(
                DisentenglementTargetManager.configurations.latent_indices,
                dtype=torch.int,
            )

        elif auxillary_loss_key == "reserved_subset":
            return torch.tensor(
                DisentenglementTargetManager.configurations.get_by_index(
                    target_obs_key_ind
                ).reserved_latent_indices,
                dtype=torch.int,
            )

        elif auxillary_loss_key == "unreserved_subset":
            return torch.tensor(
                DisentenglementTargetManager.configurations.get_by_index(
                    target_obs_key_ind
                ).unreserved_latent_indices,
                dtype=torch.int,
            )

        else:
            raise ValueError("Unknown auxillary loss.")
