"""Defines utility functions for adding motion priors, to learn specific actions."""

from abc import abstractmethod
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Generic, TypeVar

import attrs
import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
import xax
from jaxtyping import PRNGKeyArray

import bvhio
import glm
from bvhio.lib.hierarchy import Joint as BvhioJoint
