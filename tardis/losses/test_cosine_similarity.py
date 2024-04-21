import torch
import pytest
from .cosine_similarity import CosineSimilarity


@pytest.fixture
def cosine_similarity_loss():
    return CosineSimilarity(weight=1.0, method_kwargs={})


def test_cosine_similarity_loss(cosine_similarity_loss):
    # Test case 1: Same vectors, expect loss to be close to 0
    dict1 = {
        "z": torch.tensor([[1.0, 0.0, 0.0]]),
        "qz": torch.tensor([[1.0, 0.0, 0.0]]),
    }
    dict2 = {
        "z": torch.tensor([[1.0, 0.0, 0.0]]),
        "qz": torch.tensor([[1.0, 0.0, 0.0]]),
    }
    relevant_latent_indices = torch.tensor([0, 1, 2], dtype=torch.int)
    loss = cosine_similarity_loss(dict1, dict2, relevant_latent_indices)
    assert torch.isclose(loss, torch.tensor(1.0), atol=1e-5)


def test_cosine_similarity_loss_with_orthogonal_vectors(cosine_similarity_loss):
    # Test case 2: Orthogonal vectors, expect loss to be close to 1
    dict1 = {
        "z": torch.tensor([[1.0, 0.0, 0.0]]),
        "qz": torch.tensor([[1.0, 0.0, 0.0]]),
    }
    dict2 = {
        "z": torch.tensor([[0.0, 1.0, 0.0]]),
        "qz": torch.tensor([[0.0, 1.0, 0.0]]),
    }
    relevant_latent_indices = torch.tensor([0, 1, 2], dtype=torch.int)
    loss = cosine_similarity_loss(dict1, dict2, relevant_latent_indices)
    assert torch.isclose(loss, torch.tensor(0.0), atol=1e-5)


def test_cosine_similarity_loss_with_opposite_vectors(cosine_similarity_loss):
    # Test case 3: Opposite vectors, expect loss to be close to 2
    dict1 = {
        "z": torch.tensor([[1.0, 0.0, 0.0]]),
        "qz": torch.tensor([[1.0, 0.0, 0.0]]),
    }
    dict2 = {
        "z": torch.tensor([[-1.0, 0.0, 0.0]]),
        "qz": torch.tensor([[-1.0, 0.0, 0.0]]),
    }
    relevant_laten_indices = torch.tensor([0, 1, 2], dtype=torch.int)
    loss = cosine_similarity_loss(dict1, dict2, relevant_laten_indices)
    assert torch.isclose(loss, torch.tensor(-1.0), atol=1e-5)


def test_cosine_similarity_loss_with_non_unit_vectors(cosine_similarity_loss):
    # Test case 4: Non-unit vectors, expect loss to be scaled accordingly
    dict1 = {
        "z": torch.tensor([[2.0, 0.0, 0.0]]),
        "qz": torch.tensor([[2.0, 0.0, 0.0]]),
    }
    dict2 = {
        "z": torch.tensor([[0.0, 3.0, 0.0]]),
        "qz": torch.tensor([[0.0, 3.0, 0.0]]),
    }
    relevant_latent_indices = torch.tensor([0, 1, 2], dtype=torch.int)
    loss = cosine_similarity_loss(dict1, dict2, relevant_latent_indices)
    expected_loss = (2.0 * 0.0 + 0.0 * 3.0) / (
        torch.norm(dict1["z"]) * torch.norm(dict2["z"])
    )
    assert torch.isclose(loss, expected_loss, atol=1e-5)


if __name__ == "__main__":
    pytest.main()
