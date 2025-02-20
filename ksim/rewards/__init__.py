"""Common interface for different reward functions."""

from .base import Reward, RewardBuilder
from .mjcf import (
    AngularVelocityXYPenalty,
    FootContactPenalty,
    FootContactPenaltyBuilder,
    FootSlipPenalty,
    FootSlipPenaltyBuilder,
    LinearVelocityZPenalty,
    TrackAngularVelocityZReward,
    TrackLinearVelocityXYReward,
)
