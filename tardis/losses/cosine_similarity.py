import torch.nn.functional as F
from .base import TardisLoss


class CosineSimilarity(TardisLoss):
    def __init__(
        self,
        method: str,
        weight: float,
        transformation: str,
        progress_bar: bool,
        method_kwargs: dict,
        loss_identifier_string: str = "",
    ):
        super().__init__(
            method=method,
            weight=weight,
            transformation=transformation,
            progress_bar=progress_bar,
            method_kwargs=method_kwargs,
            loss_identifier_string=loss_identifier_string,
        )
    def forward(self, outputs, counteractive_outputs, relevant_latent_indices):
        return F.cosine_similarity(
            x1=outputs["z"][:, relevant_latent_indices],
            x2=counteractive_outputs["z"].clone()[:, relevant_latent_indices],
        )
