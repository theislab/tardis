import pytest
import torch

from .mse import MSE


@pytest.fixture
def mse():
    return MSE(weight=1.0, method_kwargs={})


def test_perfect_predictions(mse):
    # Perfect predictions in torch arrays format
    outputs = {"z": torch.Tensor([[1, 2, 3]]), "qz": torch.Tensor([[1, 2, 3]])}
    counteractive_outputs = {
        "z": torch.Tensor([[1, 2, 3]]),
        "qz": torch.Tensor([[1, 2, 3]]),
    }
    relevant_latent_indices = torch.tensor([0, 1, 2], dtype=torch.int)

    assert mse(outputs, counteractive_outputs, relevant_latent_indices) == 0.0


def test_noisy_predictions(mse):
    # Noisy predictions in torch arrays format
    outputs = {"z": torch.Tensor([[1, 2, 3]]), "qz": torch.Tensor([[1, 2, 3]])}
    counteractive_outputs = {
        "z": torch.Tensor([[3, 4, 5]]),
        "qz": torch.Tensor([[2, 3, 4]]),
    }
    relevant_latent_indices = torch.tensor([0, 1, 2], dtype=torch.int)

    assert mse(outputs, counteractive_outputs, relevant_latent_indices) == 4.0
