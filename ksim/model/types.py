"""Types for ML modeling."""

from typing import Callable, Tuple

from flax.core import FrozenDict
from jaxtyping import Array, PRNGKeyArray

ActionLogProbFn = Callable[
    [FrozenDict[str, Array], FrozenDict[str, Array], PRNGKeyArray], Tuple[Array, Array]
]
