"""Defines K-Sim dataset types."""

__all__ = [
    "TrajectoryDataset",
]

import json
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import TypeVar

import numpy as np
from dpshdl.dataset import Dataset

from ksim.types import Rewards, Trajectory

T = TypeVar("T")


@dataclass
class TrajectoryDatasetWriter:
    def __init__(self, path: str | Path, num_samples: int) -> None:
        super().__init__()

        self.path = Path(path)
        self.num_samples = num_samples
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fp: np.memmap | None = None
        self.index = 0

    def __enter__(self) -> "TrajectoryDatasetWriter":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self.fp is not None:
            self.fp.flush()
            self.fp = None

    def write(self, trajectory: Trajectory, rewards: Rewards) -> None:
        if self.index >= self.num_samples:
            raise ValueError("Dataset is full")

        sample = {
            "done": trajectory.done,
            "xquat": trajectory.xquat,
            "xpos": trajectory.xpos,
            "qpos": trajectory.qpos,
            "qvel": trajectory.qvel,
            "action": trajectory.action,
            "timestep": trajectory.timestep,
            **trajectory.obs,
            **trajectory.command,
            **trajectory.termination_components,
            "reward": rewards.total,
            **rewards.components,
        }

        # Lazily create the memmap file using the sample sizes.
        if self.fp is None:
            shapes = {k: v.shape for k, v in sample.items()}
            with open(self.path.with_suffix(".meta.json"), "w") as f:
                json.dump(shapes, f)
            total_size = sum(np.prod(v.shape) for v in sample.values())
            self.fp = np.memmap(self.path, dtype=np.float32, mode="w+", shape=(self.num_samples, total_size))
            self.index = 0

        arr = np.concatenate([v.flatten() for v in sample.values()])
        self.fp[self.index] = arr
        self.index += 1


@dataclass
class TrajectoryDataset(Dataset[tuple[Trajectory, Rewards], tuple[Trajectory, Rewards]]):
    def next(self) -> tuple[Trajectory, Rewards]:
        raise NotImplementedError

    @classmethod
    def writer(cls, path: str | Path, num_samples: int) -> TrajectoryDatasetWriter:
        return TrajectoryDatasetWriter(path, num_samples)
