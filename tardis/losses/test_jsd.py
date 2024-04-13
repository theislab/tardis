import pytest
import torch

from .jsd import _jensen_shannon_divergence_with_normal_parameters


@pytest.mark.parametrize(
    "mu1, sigma1, mu2, sigma2, expected",
    [
        (0, 1, 0, 1, 0.0),  # Same distribution
    ],
)
def test_jensen_shannon_divergence(mu1, sigma1, mu2, sigma2, expected):
    dist1 = torch.distributions.Normal(mu1, sigma1)
    dist2 = torch.distributions.Normal(mu2, sigma2)

    if expected is not None:
        jsd = _jensen_shannon_divergence_with_normal_parameters(dist1, dist2)
        assert torch.isclose(jsd, torch.tensor(expected), atol=1e-5)
    else:
        with pytest.raises(ValueError):
            _jensen_shannon_divergence_with_normal_parameters(dist1, dist2)


def test_symmetric_property():
    mu1, sigma1 = 0, 1
    mu2, sigma2 = 2, 1

    dist1 = torch.distributions.Normal(mu1, sigma1)
    dist2 = torch.distributions.Normal(mu2, sigma2)

    jsd1 = _jensen_shannon_divergence_with_normal_parameters(dist1, dist2)
    jsd2 = _jensen_shannon_divergence_with_normal_parameters(dist2, dist1)
    assert torch.isclose(jsd1, jsd2, atol=1e-5)
