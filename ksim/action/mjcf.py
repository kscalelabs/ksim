"""Defines actions for MuJoCo environments."""

from dataclasses import dataclass

import jax.numpy as jnp

from ksim.action.base import Action


@dataclass
class MjcfAction(Action):
    """Defines an action for a MuJoCo environment."""

    # The actions to be executed.
    actions_nj: jnp.ndarray
