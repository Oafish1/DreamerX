"""Utility distributions for reinforcement learning agents."""

import torch


def uniform_mix(logits: torch.Tensor, ratio: float = .01) -> tuple[torch.Tensor, torch.Tensor]:
    """Mix the input logits with a uniform distribution on the final dimension.

    :param logits: The input logits to mix, of shape (..., num_classes).
    :type logits: torch.Tensor
    :param ratio: The ratio of uniform distribution to mix with the input logits.
    :type ratio: float
    :return: The mixed logits, of shape (..., num_classes).
    :rtype: torch.Tensor

    """
    # Compute probabilities from logits and mix with uniform distribution
    probs = torch.softmax(logits, dim=-1)
    probs = (1 - ratio) * probs + ratio * (1 / probs.shape[-1])

    # Return probabilities and logits
    # NOTE: `torch.distributions.utils.probs_to_logits` is just log with clamping
    return torch.distributions.utils.probs_to_logits(probs), probs


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Apply the symmetric logarithm transformation to the input tensor.

    :param x: The input tensor to transform.
    :type x: torch.Tensor
    :return: The transformed tensor.
    :rtype: torch.Tensor

    """
    return torch.sign(x) * torch.log1p(torch.abs(x))
