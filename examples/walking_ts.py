"""Defines a simple task for training a walking policy with teacher-student forcing.

To train this task, first you should train a walking policy using the vanilla
`walking.py` example script, to get a checkpoint.
"""

from dataclasses import dataclass

import ksim

from .walking import HumanoidWalkingTask, HumanoidWalkingTaskConfig


@dataclass
class HumanoidWalkingTeacherStudentTaskConfig(HumanoidWalkingTaskConfig):
    pass


class HumanoidWalkingTeacherStudentTask(
    HumanoidWalkingTask[HumanoidWalkingTeacherStudentTaskConfig],
    ksim.TeacherStudentTask[HumanoidWalkingTeacherStudentTaskConfig],
):
    pass
