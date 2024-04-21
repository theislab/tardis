import torch
import pytest
from .wasserstein import WassersteinLoss


class TestWassersteinLoss:
    @pytest.fixture
    def loss_instance(self):
        method_kwargs = {"latent_distribution": "normal", "epsilon": 1e-8}
        return WassersteinLoss(weight=1.0, method_kwargs=method_kwargs)

    def test_loss_initialization(self, loss_instance):
        assert loss_instance.weight == 1.0
        assert loss_instance.latent_distribution == "normal"
        assert hasattr(loss_instance, "epsilon")

    def test_loss_forward_normal_latent_distribution(self, loss_instance):
        qz_mean = torch.randn(10, 20)
        qz_stddev = torch.rand(10, 20)
        z_samples = torch.randn(10, 20)
        outputs = {"qz": torch.distributions.Normal(qz_mean, qz_stddev), "z": z_samples}
        counteractive_outputs = {
            "qz": torch.distributions.Normal(qz_mean, qz_stddev),
            "z": z_samples,
        }
        relevant_latent_indices = torch.tensor([0, 1, 2], dtype=torch.int)

        # Ensure the forward method returns a tensor
        loss = loss_instance.forward(
            outputs, counteractive_outputs, relevant_latent_indices
        )
        assert isinstance(loss, torch.Tensor)

        # Ensure the loss value is reasonable (non-negative)
        assert torch.all(loss >= 0)

        # Ensure the loss value is scaled by weight
        assert torch.all(
            torch.isclose(
                loss,
                loss_instance.weight
                * loss_instance.loss_fn(
                    outputs,
                    counteractive_outputs,
                    relevant_latent_indices,
                    epsilon=1e-8,
                ),
            )
        )


if __name__ == "__main__":
    pytest.main()
