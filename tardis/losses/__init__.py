from .cosine_similarity import CosineSimilarity
from .wasserstein import WassersteinLoss
from .jsd import JSD
from .mae import MAE
from .mse import MSE


LOSSES = {
    "cosine_similarity": CosineSimilarity,
    "wasserstein": WassersteinLoss,
    "jsd": JSD,
    "mae": MAE,
    "mse": MSE,
}
