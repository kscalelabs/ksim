"""Defines a simple task for training a walking policy with teacher-student forcing.

To train this task, first you should train a walking policy using the vanilla
`walking.py` example script, to get a checkpoint.
"""

from dataclasses import dataclass
from pathlib import Path

import equinox as eqx
import xax
from jaxtyping import PRNGKeyArray
from omegaconf import MISSING

import ksim

from .walking import DefaultHumanoidModel, HumanoidWalkingTask, HumanoidWalkingTaskConfig


class TeacherStudentModel(DefaultHumanoidModel):
    teacher: DefaultHumanoidModel

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        ckpt_path: str | Path,
        hidden_size: int,
        depth: int,
        num_mixtures: int,
    ) -> None:
        super().__init__(
            key=key,
            hidden_size=hidden_size,
            depth=depth,
            num_mixtures=num_mixtures,
        )

        ckpt_path = Path(ckpt_path)

        # Loads the model spec using the checkpoint config.
        config = xax.load_ckpt(ckpt_path, part="config")
        model_spec = eqx.filter_eval_shape(
            DefaultHumanoidModel,
            key=key,
            hidden_size=config.hidden_size,
            depth=config.depth,
            num_mixtures=config.num_mixtures,
        )

        # Loads the checkpoint model weights using the model spec.
        self.teacher = xax.load_ckpt(
            ckpt_path,
            part="model",
            model_template=model_spec,
        )


@dataclass
class HumanoidWalkingTeacherStudentTaskConfig(HumanoidWalkingTaskConfig, ksim.TeacherStudentConfig):
    ckpt_path: str = xax.field(
        value=MISSING,
        help="Path to the checkpoint to load the teacher policy from.",
    )


class HumanoidWalkingTeacherStudentTask(
    HumanoidWalkingTask[HumanoidWalkingTeacherStudentTaskConfig],
    ksim.TeacherStudentTask[HumanoidWalkingTeacherStudentTaskConfig],
):
    def get_model(self, key: PRNGKeyArray) -> TeacherStudentModel:
        return TeacherStudentModel(
            key,
            ckpt_path=self.config.ckpt_path,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_mixtures=self.config.num_mixtures,
        )
