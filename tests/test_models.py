import torch

import fishyrl.models


def test_ChannelNorm():
    """Test the ChannelNorm layer."""
    # Create a random tensor
    torch.manual_seed(42)
    x = 2 * torch.randn(2, 3, 4, 4) + 1
    # Apply the normalization
    norm = fishyrl.models.ChannelNorm(3, elementwise_affine=False, bias=False, eps=1e-5)
    output = norm(x)
    output_mean = output.mean(dim=1)
    output_std = output.std(dim=1, correction=0)
    # Check that the output is the correct shape and distribution
    assert output.shape == x.shape
    assert torch.allclose(output_mean, torch.zeros_like(output_mean), atol=1e-3)
    assert torch.allclose(output_std, torch.ones_like(output_std), atol=1e-3)


def test_RecurrentModel():
    """Test RecurrentModel."""
    # Create a random input tensor
    torch.manual_seed(42)
    s = torch.randn(2, 7)
    a = torch.randn(2, 3)
    # Create the recurrent model
    model = fishyrl.models.BlockRecurrentModel(stoch_dim=7, act_dim=3, hidden_dim=5, deter_dim=22, num_blocks=11)
    # Test the forward pass with no initial hidden state
    h0 = model(s, a)
    assert h0.shape == (2, 22)
    # Test the forward pass with an initial hidden state
    h1 = model(s, a, h0)
    assert h1.shape == (2, 22)
