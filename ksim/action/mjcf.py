"""Defines actions for MuJoCo environments."""

import jax.numpy as jnp
from flax import struct

from ksim.action.base import Action


@struct.dataclass
class MjcfAction(Action):
    """Defines an action for a MuJoCo environment."""

    # The actions to be executed.
    actions_nj: jnp.ndarray
