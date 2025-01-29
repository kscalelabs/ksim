"""Defines the base environment class.

An environment defines some interaction between the agent and the world.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import xax
from omegaconf import MISSING

from ksim.action.base import Action
from ksim.state.base import State

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class EnvironmentConfig:
    """Master configuration class for the base environment."""

    num_environments: int = xax.field(
        value=MISSING,
        help="Number of parallel environments to simulate.",
    )


Tconfig = TypeVar("Tconfig", bound=EnvironmentConfig)
Tstate = TypeVar("Tstate", bound=State)
Taction = TypeVar("Taction", bound=Action)


class Environment(ABC, Generic[Tconfig, Tstate, Taction]):
    def __init__(
        self,
        config: Tconfig,
        *,
        model: Callable[[Tstate], Taction] | None = None,
    ) -> None:
        self.config = config
        self.model = model

    @property
    def num_envs(self) -> int:
        return self.config.num_environments

    @abstractmethod
    def get_initial_state(self) -> Tstate: ...

    @abstractmethod
    def step(self, actions: Taction, state: Tstate) -> Tstate: ...

    @abstractmethod
    def reset(self, state: Tstate) -> Tstate: ...

    def get_test_action(self, state: Tstate) -> Taction:
        if self.model is not None:
            return self.model(state)

        raise NotImplementedError(
            "If you want to use the `test_run` method to test your environment, you must either "
            "implement the `get_test_action` method to return a sample action for a given state, or "
            "pass a model to the environment constructor."
        )

    def test_run(self, num_steps: int) -> None:
        """Runs an environment for a given number of steps.

        Args:
            num_steps: The number of steps to run the environment for.
            action_fn: A function that generates actions for the environment.
        """
        state = self.get_initial_state()
        state = self.reset(state)
        logger.info("Running %d steps", num_steps)
        for step in range(num_steps):
            logger.info("Step %d / %d", step, num_steps)
            actions = self.get_test_action(state)
            state = self.step(actions, state)
