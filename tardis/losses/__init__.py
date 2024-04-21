#!/usr/bin/env python3

from .cosine_similarity import CosineSimilarity
from .jsd import JSD
from .mae import MAE
from .mse import MSE
from .wasserstein import WassersteinLoss

LOSSES = {
    "cosine_similarity": CosineSimilarity,
    "wasserstein": WassersteinLoss,
    "jsd": JSD,
    "mae": MAE,
    "mse": MSE,
}
