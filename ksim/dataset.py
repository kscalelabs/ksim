"""Defines K-Sim dataset types."""

__all__ = [
    "TrajectoryDataset",
]

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Mapping, TypeVar

import jax.numpy as jnp
import numpy as np
import xax
from dpshdl.dataset import Dataset
from jaxtyping import Array

from ksim.types import Rewards, Trajectory

T = TypeVar("T")


def recursive_flatten(arr: Mapping[str, Array | Mapping[str, Array]], join: str = ".") -> list[tuple[str, Array]]:
    flat = []
    for k, v in arr.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat.append((f"{k}{join}{kk}", vv))
        else:
            flat.append((k, v))
    return flat


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

        sample: Mapping[str, Array | Mapping[str, Array]] = {
            "xquat": trajectory.xquat,
            "xpos": trajectory.xpos,
            "qpos": trajectory.qpos,
            "qvel": trajectory.qvel,
            "action": trajectory.action,
            "done": trajectory.done,
            "success": trajectory.success,
            "timestep": trajectory.timestep,
            "obs": trajectory.obs,
            "command": trajectory.command,
            "termination_components": trajectory.termination_components,
            "reward": rewards.total,
            "reward_components": rewards.components,
        }

        # FrozenDict is not JSON serializable.
        sample = {k: v._dict if isinstance(v, xax.FrozenDict) else v for k, v in sample.items()}

        flat_sample = recursive_flatten(sample)

        # Lazily create the memmap file using the sample sizes.
        if self.fp is None:
            total_size = sum(int(np.prod(v.shape)) for _, v in flat_sample)
            meta = {
                "names": [name for name, _ in flat_sample],
                "shapes": [v.shape for _, v in flat_sample],
                "num_samples": self.num_samples,
                "total_size": total_size,
            }
            with open(self.path.with_suffix(".meta.json"), "w") as f:
                json.dump(meta, f, indent=2)
            self.fp = np.memmap(self.path, dtype=np.float32, mode="w+", shape=(self.num_samples, total_size))
            self.index = 0

        arr = np.concatenate([v.flatten() for _, v in flat_sample])
        self.fp[self.index] = arr
        self.index += 1


@dataclass
class TrajectoryDataset(Dataset[tuple[Trajectory, Rewards], tuple[Trajectory, Rewards]]):
    def __init__(
        self,
        path: str | Path,
        meta_path: str | Path | None = None,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__()

        # Stores the inputs.
        self.path = Path(path)
        self.meta_path = self.path.with_suffix(".meta.json") if meta_path is None else Path(meta_path)
        self.rng = random.Random(1337) if rng is None else rng

        # Loads metadata.
        with open(self.meta_path, "r") as f:
            self.meta = json.load(f)

        # Opens the memory-mapped dataset.
        shape = (self.meta["num_samples"], self.meta["total_size"])
        self.ds = np.memmap(self.path, dtype=np.float32, mode="r", shape=shape)

        # Sets the current index.
        self.index = self.rng.randint(0, self.meta["num_samples"] - 1)

    def next(self) -> tuple[Trajectory, Rewards]:
        sample = jnp.array(self.ds[self.index])
        self.index = self.rng.randint(0, self.meta["num_samples"] - 1)

        arrs: dict[str, Array] = {}
        offset = 0
        for name, shape in zip(self.meta["names"], self.meta["shapes"]):
            nelem = np.prod(shape)
            arrs[name] = sample[offset : offset + nelem].reshape(shape)
            offset += nelem

        def _dict(prefix: str) -> xax.FrozenDict[str, Array]:
            return xax.FrozenDict({k.split(".", 1)[1]: v for k, v in arrs.items() if k.startswith(prefix)})

        return (
            Trajectory(
                qpos=arrs["qpos"],
                qvel=arrs["qvel"],
                xpos=arrs["xpos"],
                xquat=arrs["xquat"],
                obs=_dict("obs"),
                command=_dict("command"),
                event_state=_dict("event_state"),
                action=arrs["action"],
                done=arrs["done"],
                success=arrs["success"],
                timestep=arrs["timestep"],
                termination_components=_dict("termination_components"),
                aux_outputs=_dict("aux_outputs"),
            ),
            Rewards(
                total=arrs["reward"],
                components=_dict("reward_components"),
                carry=_dict("reward_carry"),
            ),
        )

    @classmethod
    def writer(cls, path: str | Path, num_samples: int) -> TrajectoryDatasetWriter:
        return TrajectoryDatasetWriter(path, num_samples)


def show_trajectory_dataset_adhoc() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=str)
    parser.add_argument("-m", "--meta", type=str)
    parser.add_argument("-n", "--max-samples", type=int, default=10)
    parser.add_argument("-e", "--handle-errors", action="store_true")
    parser.add_argument("-l", "--log-interval", type=int, default=1)
    args = parser.parse_args()

    dataset = TrajectoryDataset(args.path, args.meta)
    dataset.test(
        max_samples=args.max_samples,
        handle_errors=args.handle_errors,
        log_interval=args.log_interval,
    )


if __name__ == "__main__":
    # python -m ksim.dataset <path>
    show_trajectory_dataset_adhoc()
