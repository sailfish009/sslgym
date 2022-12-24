"""Test custom pytorch optimizers."""
import torch

from src.pytorch.optimizers import LARS


def test_lars() -> None:
    """Test LARS."""
    params = torch.nn.Parameter(torch.rand(10, 10))
    constant = torch.rand(10, 10)

    original = params.data.sum().item()

    adamw = LARS([params])

    out = constant * params
    loss = out.sum()
    loss.backward()
    adamw.step()

    modified = params.data.sum().item()

    assert original != modified
