"""Defines the base action class."""

from dataclasses import dataclass

import jax


@jax.tree_util.register_dataclass
@dataclass
class Action:
    pass
