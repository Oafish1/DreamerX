"""Utility models and layers for construction of reinforcement learning agents."""

import math
from typing import Any

import torch
import torch.nn as nn

from . import distributions


class MLP(nn.Module):
    """MLP for processing vector inputs."""
    def __init__(self, input_dim: int, output_dim: int | None = None, hidden_dims: list[int] | None = None) -> None:
        """Initialize the MLP.

        :param input_dim: The dimension of the input vector.
        :type input_dim: int
        :param output_dim: The dimension of the output vector. If provided,
            a final hidden layer will be added with no normalization or activation
        :type output_dim: int | None
        :param hidden_dims: The dimensions of the hidden layers.
        :type hidden_dims: list[int] | None

        """
        super().__init__()

        # Create model in blocks
        layers = []
        hidden_dims = [input_dim] + hidden_dims
        for i in range(len(hidden_dims) - 1):
            layers += [
                # Layer, dropout, norm, activation
                nn.Linear(hidden_dims[i], hidden_dims[i + 1], bias=False),
                nn.LayerNorm(hidden_dims[i + 1], eps=1e-3),
                nn.SiLU(),
            ]

        # Add final hidden layer
        if output_dim is not None:
            layers += [nn.Linear(hidden_dims[-1], output_dim)]

        # Assemble model
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the MLP.

        :param x: The input tensor of shape (batch_size, input_dim).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_size, hidden_dims[-1]).
        :rtype: torch.Tensor

        """
        return self._model(x)


class MLPEncoder(nn.Module):
    """MLP encoder for processing vector observations."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512) -> None:
        """Initialize the MLP encoder.

        :param input_dim: The dimension of the input vector observation.
        :type input_dim: int
        :param output_dim: The dimension of the output vector.
        :type output_dim: int
        :param hidden_dim: The dimension of the hidden layers. (Default: ``512``)
        :type hidden_dim: int

        """
        super().__init__()

        # Create model
        self._model = MLP(
            input_dim=input_dim,
            hidden_dims=3 * [hidden_dim] + [output_dim],
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the MLP encoder.

        :param x: The input tensor of shape (batch_size, input_dim).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_size, hidden_dim).
        :rtype: torch.Tensor

        """
        return self._model(distributions.symlog(x))


class MLPDecoder(nn.Module):
    """MLP decoder for reconstructing vector observations from latent representations."""
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512) -> None:
        """Initialize the MLP decoder.

        :param input_dim: The dimension of the input tensor.
        :type input_dim: int
        :param output_dim: The dimension of the output vector observations.
        :type output_dim: int
        :param hidden_dim: The dimension of the hidden layers. (Default: ``512``)
        :type hidden_dim: int

        """
        super().__init__()

        # Create model
        self._model = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dims=3 * [hidden_dim],
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor] | torch.Tensor:
        """Perform a forward pass through the MLP decoder.

        :param x: The input tensor of shape (batch_size, hidden_dim).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_size, output_dim).
        :rtype: torch.Tensor

        """
        return self._model(x)


class ChannelNorm(nn.LayerNorm):
    """Layer normalization across the channel dimension."""
    def __init__(self, *args: list[Any], **kwargs: dict[str, Any]) -> None:
        """Initialize the ChannelNorm layer.

        :param args: Positional arguments for the LayerNorm.
        :type args: list[Any]
        :param kwargs: Keyword arguments for the LayerNorm.
        :type kwargs: dict[str, Any]

        """
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization across the channel dimension.

        Swaps the channel dimension to the end, applies layer normalization, then swaps it back.

        :param x: The input tensor of shape (batch_size, channels, height, width).
        :type x: torch.Tensor
        :return: The normalized tensor of the same shape.
        :rtype: torch.Tensor

        """
        # Permute channels to last dimension
        x = x.permute(0, 2, 3, 1)
        # Apply layer normalization
        x = super().forward(x)
        # Reverse permutation
        return x.permute(0, 3, 1, 2)


class CNNEncoder(nn.Module):
    """CNN encoder for processing image observations.

    Processes a (generally) 64x64 image to 4x4 using [2, 4, 8, 16] channels and interleaved
    layer normalization and SiLU activations.

    """
    def __init__(self, input_channels: int, image_size: tuple[int, int] = (64, 64)) -> None:
        """Initialize the CNN encoder.

        :param input_channels: The number of input channels in the image.
        :type input_channels: int
        :param image_size: The size of the input image. (Default: ``(64, 64)``)
        :type image_size: tuple[int, int]

        """
        super().__init__()

        # Create model in blocks
        layers = []
        num_blocks = 4
        hidden_channels = [input_channels] + [2 * 2 ** i for i in range(num_blocks)]
        for i in range(num_blocks):
            layers += [
                # Layer, dropout, norm, activation
                nn.Conv2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=4, stride=2, padding=1, bias=False),
                ChannelNorm(hidden_channels[i + 1], eps=1e-3),
                nn.SiLU(),
            ]
        layers += [nn.Flatten()]
        self._output_dim = hidden_channels[-1] * math.prod(image_size) // 4 ** num_blocks
        self._model = nn.Sequential(*layers)

    @property
    def output_dim(self) -> int:
        """Output dimension of the CNN encoder.

        :type: int

        """
        return self._output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the CNN encoder.

        :param x: The input tensor of shape (batch_size, channels, height, width).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_size, output_dim, height, width).
        :rtype: torch.Tensor

        """
        # NOTE: Difference here with SheepRL, where batch dims are flattened during forward
        return self._model(x)


class CNNDecoder(nn.Module):
    """CNN decoder for reconstructing image observations from latent representations.

    Uses transposed convolutions to upsample from 4x4 to 64x64, with interleaved layer normalization
    and SiLU activations.

    """
    def __init__(self, output_channels: int, latent_size: int) -> None:
        """Initialize the CNN decoder.

        :param output_channels: The number of output channels in the reconstructed image.
        :type output_channels: int
        :param latent_size: The size of the latent representation.
        :type latent_size: int

        """
        super().__init__()

        # Create model in blocks
        layers = [
            nn.Linear(latent_size, 16 * 4 * 4),
            nn.Unflatten(-1, (-1, 4, 4)),
        ]
        num_blocks = 4
        hidden_channels = [2 * 2 ** (num_blocks - i - 1) for i in range(num_blocks)] + [output_channels]
        for i in range(num_blocks):
            # Add blocks if not last
            if i != num_blocks - 1:
                layers += [
                    # Layer, dropout, norm, activation
                    nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=4, stride=2, padding=1, bias=False),
                    ChannelNorm(hidden_channels[i + 1], eps=1e-3),
                    nn.SiLU(),
                ]
            # Otherwise, add final block with bias and no norm/activation
            else:
                layers += [
                    nn.ConvTranspose2d(hidden_channels[i], hidden_channels[i + 1], kernel_size=4, stride=2, padding=1, bias=True),
                ]
        self._model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the CNN decoder.

        :param x: The input tensor of shape (batch_size, latent_dim).
        :type x: torch.Tensor
        :return: The output tensor of shape (batch_size, channels, height, width).
        :rtype: torch.Tensor

        """
        return self._model(x)


class LayerNormGRU(nn.Module):
    """GRU layer with internal layer normalization."""
    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Initialize the LayerNormGRU.

        :param input_size: The size of the input tensor.
        :type input_size: int
        :param hidden_size: The size of the hidden state.
        :type hidden_size: int

        """
        super().__init__()

        # Record parameters
        self._hidden_size = hidden_size

        # Initialize layers
        self._mlp = nn.Sequential(
            nn.Linear(input_size + hidden_size, 3 * hidden_size, bias=True),
            nn.LayerNorm(3 * hidden_size, eps=1e-3),
        )

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        """Perform a forward pass through the LayerNormGRU.

        Applies layer normalization to the hidden state after the GRU computation.

        :param x: The input tensor of shape (batch_size, input_dim).
        :type x: torch.Tensor
        :param h: The initial hidden state of shape (batch_size, hidden_dim), or None to use zeros. (Default: ``None``)
        :type h: torch.Tensor | None
        :return: The final hidden state of shape (batch_size, hidden_dim).
        :rtype: torch.Tensor

        """
        # If h is None, initialize to zeros
        if h is None:
            h = torch.zeros(*x.shape[:-1], self._hidden_size, device=x.device, dtype=x.dtype)

        # Process input and hidden state through MLP
        x = torch.cat((x, h), dim=-1)
        x = self._mlp(x)

        # Get reset, update, and candidate gates
        reset, update, candidate = x.chunk(3, dim=-1)
        reset = torch.sigmoid(reset)
        update = torch.sigmoid(update - 1)  # Bias update gate towards copying
        candidate = torch.tanh(reset * candidate)
        # NOTE: This deviates from the standard GRU formulation, but is a common variant that works well in practice

        # Update hidden state
        return (1 - update) * h + update * candidate


class RecurrentModel(nn.Module):
    """Recurrent model for processing sequences of latent representations.

    Uses a GRU to process sequences of latent representations, with layer normalization and SiLU
    activations on the output.

    """
    def __init__(self, input_size: int, hidden_size: int) -> None:
        """Initialize the recurrent model.

        :param input_size: The size of the input tensor.
        :type input_size: int
        :param hidden_size: The size of the hidden state in the GRU.
        :type hidden_size: int

        """
        super().__init__()

        # Store variables
        self._hidden_size = hidden_size

        # Create MLP and GRU layers
        self._mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size, eps=1e-3),
            nn.SiLU(),
        )
        self._gru = LayerNormGRU(hidden_size, hidden_size)

    @property
    def hidden_size(self) -> int:
        """Hidden size of the recurrent model.

        :type: int

        """
        return self._hidden_size

    def forward(self, x: torch.Tensor, h: torch.Tensor | None = None) -> torch.Tensor:
        """Perform a forward pass through the recurrent model.

        :param x: The input tensor of shape (batch_size, seq_len, latent_dim).
        :type x: torch.Tensor
        :param h: The initial hidden state of shape (1, batch_size, hidden_dim), or None to use zeros. (Default: ``None``)
        :type h: torch.Tensor | None
        :return: The final hidden state of shape (batch_size, hidden_dim).
        :rtype: torch.Tensor

        """
        # Pass through the MLP and GRU
        x = self._mlp(x)
        h = self._gru(x, h)
        return h


class RSSM(nn.Module):
    """Recurrent state-space model for modeling environment dynamics.

    The RSSM consists of a recurrent model for tracking the hidden state, a representation model for inferring the stochastic
    state from the hidden state and observation, and a transition model for predicting the stochastic state from the
    hidden state.

    """
    def __init__(
        self,
        recurrent_model: nn.Module,
        representation_model: nn.Module,
        transition_model: nn.Module,
        bins: int = 32,
    ) -> None:
        """Initialize the RSSM.

        :param recurrent_model: The recurrent model for keeping track of the environment dynamics.
        :type recurrent_model: nn.Module
        :param representation_model: The model for inferring the stochastic state from the hidden state and observation.
        :type representation_model: nn.Module
        :param transition_model: The model for predicting the stochastic state from the hidden state.
        :type transition_model: nn.Module
        :param bins: The number of bins for the stochastic state. (Default: ``32``)
        :type bins: int

        """
        super().__init__()

        # Record variables
        self._recurrent_model = recurrent_model
        self._representation_model = representation_model
        self._transition_model = transition_model
        self._bins = bins

        # Initialize recurrent model hidden state
        # NOTE: Making this initial state learnable allows the model to learn an initial state
        #       rather than always starting from zeros
        self._initial_hidden_state = nn.Parameter(torch.zeros(recurrent_model.hidden_size))

    @property
    def initial_hidden_state(self) -> torch.Tensor:
        """Trainable initial hidden state of the recurrent model.

        :type: torch.Tensor

        """
        return self._initial_hidden_state

    def infer_stochastic(self, hidden_state: torch.Tensor, embedded_obs: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Infer the stochastic state.

        Infer the stochastic state from the hidden state (using the transition model) or from the hidden state and the
        embedded observation (using the representation model).

        :param hidden_state: The hidden state from the recurrent model, of shape (batch_size, hidden_dim).
        :type hidden_state: torch.Tensor
        :param embedded_obs: The embedded observation, of shape (batch_size, obs_dim). (Default: ``None``)
        :type embedded_obs: torch.Tensor | None
        :return: The logits and sampled stochastic state.
        :rtype: tuple[torch.Tensor, torch.Tensor]

        """
        # Get the prior/posterior distribution
        # Use the transition model if not using an observation
        if embedded_obs is None:
            logits = self._transition_model(hidden_state)
        # Otherwise, use the representation model
        else:
            logits = self._representation_model(torch.cat((hidden_state, embedded_obs), dim=-1))

        # Reshape logits into bins
        logits = logits.view(*logits.shape[:-1], -1, self._bins)

        # Mix with uniform distribution
        logits, probs = distributions.uniform_mix(logits)

        # Sample
        # NOTE: SheepRL uses `torch.distributions.Independent`, but this is unneeded since we never
        #       compute log probabilities
        dist = torch.distributions.OneHotCategoricalStraightThrough(probs=probs)
        sample = dist.rsample()

        # Don't sample, and instead take the mode - Might be useful for inference
        # maxprobs = probs.argmax(dim=-1)
        # mode = torch.nn.functional.one_hot(maxprobs, num_classes=probs.size(-1))

        # Flatten both logits and sample
        # NOTE: This assumes no continuity between bins. It may be more faithful to theory if these were
        #       made into continuous values after categorical sampling. However, this is what is done in
        #       the actual implementation.
        #       https://github.com/danijar/dreamerv3/blob/b65cf81a6fb13625af8722127459283f899a35d9/dreamerv3/rssm.py#L136
        #       https://github.com/danijar/embodied/blob/f460cf117bb6ca30e16d22876601845a3fc41082/embodied/jax/outs.py#L243
        #       https://github.com/danijar/embodied/blob/f460cf117bb6ca30e16d22876601845a3fc41082/embodied/jax/outs.py#L40
        logits = logits.view(*logits.shape[:-2], -1)
        sample = sample.view(*sample.shape[:-2], -1)

        return logits, sample

    def forward(
        self,
        action: torch.Tensor,
        posterior: torch.Tensor,
        hidden_state: torch.Tensor | None = None,
        embedded_obs: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Perform one step of the RSSM.

        :param action: The action taken, of shape (batch_size, action_dim).
        :type action: torch.Tensor
        :param posterior: The posterior stochastic state from the previous step, of shape (batch_size, latent_dim).
        :type posterior: torch.Tensor
        :param hidden_state: The initial hidden state of the recurrent model, of shape (batch_size, hidden_dim), or None to use the initial hidden state.
        :type hidden_state: torch.Tensor | None
        :param embedded_obs: The embedded observation, of shape (batch_size, obs_dim), or None if imagining.
        :type embedded_obs: torch.Tensor | None
        :return: A dictionary containing the prior and posterior logits and samples, and the updated hidden state.
        :rtype: dict[str, torch.Tensor]

        """
        # Get initial hidden state if not provided
        # NOTE: SheepRL performs one step of the recurrent model to get to the initial hidden state.
        #       We do not perform this, as it appears to be unnecessary
        if hidden_state is None:
            hidden_state = torch.tanh(self._initial_hidden_state).expand(action.shape[:-1], -1)

        # If not providing a posterior, infer the prior from the hidden state
        # if posterior is None:
        #     posterior = self.infer_stochastic(hidden_state)[1]
        #     posterior = self.infer_stochastic(hidden_state, embedded_obs)[1]

        # Initialize returns
        return_dict = {}

        # Update the hidden state using the recurrent model
        hidden_state = self._recurrent_model(torch.cat((posterior, action), dim=-1), hidden_state)
        return_dict['hidden_state'] = hidden_state

        # Get the prior distribution from the transition model
        prior_logits, prior = self.infer_stochastic(hidden_state)
        return_dict['prior_logits'] = prior_logits
        return_dict['prior'] = prior

        # Get the posterior distribution from the representation model if not imagining
        if embedded_obs is not None:
            posterior_logits, posterior = self.infer_stochastic(hidden_state, embedded_obs)
            return_dict['posterior_logits'] = posterior_logits
            return_dict['posterior'] = posterior

        return return_dict
