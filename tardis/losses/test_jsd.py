import pytest
import torch

from .jsd import JSD
from torch.distributions import Normal


@pytest.fixture
def jsd():
    return JSD(weight=1.0, method_kwargs=dict(latent_distribution="normal"))


@pytest.mark.parametrize(
    "mu1, sigma1, mu2, sigma2, expected",
    [
        (
            torch.zeros((1, 1)),
            torch.ones((1, 1)),
            torch.zeros((1, 1)),
            torch.ones((1, 1)),
            0.0,
        ),  # Same distribution
    ],
)
def test_jensen_shannon_divergence(mu1, sigma1, mu2, sigma2, expected, jsd):
    outputs = {"qz": Normal(mu1, sigma1), "z": torch.randn(1, 1)}
    counteractive_outputs = {"qz": Normal(mu2, sigma2), "z": torch.randn(1, 1)}
    relevant_latent_indices = torch.tensor([0], dtype=torch.int)

    if expected is not None:
        res = jsd.forward(outputs, counteractive_outputs, relevant_latent_indices)
        assert torch.isclose(res, torch.tensor(expected), atol=1e-5)
    else:
        with pytest.raises(ValueError):
            jsd.forward(outputs, counteractive_outputs, relevant_latent_indices)


def test_symmetric_property(jsd):
    mu1, sigma1 = torch.zeros((1, 1)), torch.ones((1, 1))
    mu2, sigma2 = 2 * torch.ones((1, 1)), torch.ones((1, 1))

    dist1 = torch.distributions.Normal(mu1, sigma1)
    dist2 = torch.distributions.Normal(mu2, sigma2)

    outputs = {"qz": dist1, "z": torch.randn(1, 1)}
    counteractive_outputs = {"qz": dist2, "z": dist2.sample((1,)).reshape((1, 1))}
    relevant_latent_indices = torch.tensor([0], dtype=torch.int)

    jsd1 = jsd.forward(outputs, counteractive_outputs, relevant_latent_indices)
    jsd2 = jsd.forward(counteractive_outputs, outputs, relevant_latent_indices)
    assert torch.isclose(jsd1, jsd2, atol=1e-5)
