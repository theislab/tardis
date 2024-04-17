"TODO: We are calculating cosine similarity not cosine distance. The loss should be 1 - cosine similarity. Fix the test cases accordingly."
import torch.nn.functional as F
from .base import TardisLoss


class CosineSimilarity(TardisLoss):

    def _forward(self, outputs, counteractive_outputs, relevant_latent_indices):
        self._validate_forward_inputs(
            outputs, counteractive_outputs, relevant_latent_indices
        )
        return F.cosine_similarity(
            x1=outputs["z"][:, relevant_latent_indices],
            x2=counteractive_outputs["z"].clone()[:, relevant_latent_indices],
        )
