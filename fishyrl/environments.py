"""Environment definitions for FishyRL."""

import enum
from abc import abstractmethod
from typing import Any

import gymnasium as gym
import numpy as np

from . import utilities as frl_utilities


class VectorizedEnvironment:
    """Abstract base class for vectorized environments."""
    @property
    @abstractmethod
    def num_envs(self) -> int:
        """The number of environments.

        :type: int

        """
        pass

    @property
    @abstractmethod
    def render_fps(self) -> int:
        """The number of frames per second while rendering.

        :type: int

        """
        pass

    @property
    @abstractmethod
    def obs_shape(self) -> np.ndarray:
        """The shape of the observation space.

        :type: np.ndarray

        """
        pass

    @abstractmethod
    def action_sample(self) -> np.ndarray:
        """Sample an action from the action space.

        :rtype: np.ndarray

        """
        pass

    @abstractmethod
    def reset(self, seed: int = None, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment.

        :param seed: The random seed for the environment. (Default: ``None``)
        :type seed: int
        :param kwargs: Additional keyword arguments for resetting the environment.
        :type kwargs: dict[str, Any]
        :return: A tuple containing the initial observations and additional info.
        :rtype: tuple[np.ndarray, dict[str, Any]]

        """
        pass

    @abstractmethod
    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Perform a step in the environment.

        :param actions: The actions to perform in the environment.
        :type actions: np.ndarray
        :return: A tuple containing the next observations, rewards, terminations, truncations, and additional info.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]

        """
        pass

    @abstractmethod
    def render(self) -> np.ndarray:
        """Render the environment and return a frame.

        :return: The rendered frame.
        :rtype: np.ndarray

        """
        pass

    @abstractmethod
    def copy(self, **kwargs: dict[str, Any]) -> 'VectorizedEnvironment':
        """Copy the environment, overriding parameters in `kwargs`.

        :param kwargs: Additional keyword arguments for the new environment instance.
        :type kwargs: dict[str, Any]
        :return: A copy of the environment with overridden parameters.
        :rtype: VectorizedEnvironment

        """
        pass


class VectorizedGymEnvironment(VectorizedEnvironment):
    """A vectorized environment wrapper for Gymnasium environments."""
    def __init__(self, env_name: str, num_envs: int = 1, allow_rendering: bool = False, **init_kwargs: dict[str, Any]) -> None:
        """Initialize the vectorized Gymnasium environment.

        :param env_name: The name of the Gymnasium environment to create.
        :type env_name: str
        :param num_envs: The number of parallel environments to create. (Default: ``1``)
        :type num_envs: int
        :param allow_rendering: Whether to allow rendering of the environment. (Default: ``False``)
        :type allow_rendering: bool
        :param init_kwargs: Additional keyword arguments for initializing the Gymnasium environment.
        :type init_kwargs: dict[str, Any]

        """
        # Parameters
        self._env_name = env_name
        self._num_envs = num_envs
        self._allow_rendering = allow_rendering
        self._init_kwargs = init_kwargs

        # Initialize environments
        self._envs = gym.vector.AsyncVectorEnv([
            lambda: gym.make(
                self._env_name,
                render_mode='rgb_array' if self._allow_rendering else None,
                **self._init_kwargs) for _ in range(self._num_envs)])

    @property
    def num_envs(self) -> int:
        """The number of environments.

        :type: int

        """
        return self._envs.num_envs

    @property
    def render_fps(self) -> int:
        """The number of frames per second while rendering.

        :type: int

        """
        return self._envs.metadata['render_fps']

    @property
    def obs_shape(self) -> np.ndarray:
        """The shape of the observation space.

        :type: np.ndarray

        """
        return self._envs.observation_space.shape

    def action_sample(self) -> np.ndarray:
        """Sample an action from the action space.

        :return: A sampled action.
        :rtype: np.ndarray

        """
        return self._envs.action_space.sample()

    def reset(self, seed: int = None, **kwargs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the environment.

        :param seed: The random seed for the environment. (Default: ``None``)
        :type seed: int
        :param kwargs: Additional keyword arguments for resetting the environment.
        :type kwargs: dict[str, Any]
        :return: A tuple containing the initial observations and additional info.
        :rtype: tuple[np.ndarray, dict[str, Any]]

        """
        return self._envs.reset(seed=seed, **kwargs)

    def step(self, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Take a step in the environment.

        :param actions: The actions to take in the environment.
        :type actions: np.ndarray
        :return: A tuple containing the next observations, rewards, dones, truncations, and additional info.
        :rtype: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]

        """
        return self._envs.step(actions)

    def render(self) -> np.ndarray:
        """Render the environment and return a frame.

        :return: A rendered frame.
        :rtype: np.ndarray

        """
        if not self._allow_rendering:
            raise ValueError('Parameter `allow_rendering` is set to `False`.')
        return np.stack(self._envs.render(), axis=0)

    def copy(self, **kwargs: dict[str, Any]) -> 'VectorizedGymEnvironment':
        """Copy the environment, overriding parameters in `kwargs`.

        :param kwargs: Additional keyword arguments for the new environment instance.
        :type kwargs: dict[str, Any]
        :return: A copy of the environment with overridden parameters.
        :rtype: VectorizedGymEnvironment

        """
        new_kwargs = {
            'env_name': self._env_name,
            'num_envs': self._num_envs,
            'allow_rendering': self._allow_rendering,
            **self._init_kwargs}
        new_kwargs.update(kwargs)
        return VectorizedGymEnvironment(**new_kwargs)


class ENVIRONMENT_IDENTIFIERS(enum.Enum, metaclass=frl_utilities.CaseInsensitiveEnumMeta):
    """String identifiers for environment definitions, mapped to their corresponding classes."""
    GYMNASIUM = VectorizedGymEnvironment
