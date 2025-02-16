"""Common interface for different termination conditions."""

from .base import Termination, TerminationBuilder
from .mjcf import (
    IllegalContactTermination,
    IllegalContactTerminationBuilder,
    MinimumHeightTermination,
    PitchTooGreatTermination,
    RollTooGreatTermination,
)
