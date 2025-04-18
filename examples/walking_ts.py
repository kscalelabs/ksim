"""Defines a simple task for training a walking policy with teacher-student forcing.

To train this task, first you should train a walking policy using the vanilla
`walking.py` example script, to get a checkpoint.
"""

from dataclasses import dataclass

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

        self.teacher = DefaultHumanoidModel(
            key=key,
            hidden_size=hidden_size,
            depth=depth,
            num_mixtures=num_mixtures,
        )


@dataclass
class HumanoidWalkingTeacherStudentTaskConfig(HumanoidWalkingTaskConfig):
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
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
            num_mixtures=self.config.num_mixtures,
        )
