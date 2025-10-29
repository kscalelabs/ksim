"""Converts a checkpoint to a deployable model.

The expected command structure for K-Bot is:

- [0] x linear velocity [m/s]
- [1] y linear velocity [m/s]
- [2] z angular velocity [rad/s]
- [3] base height offset [m]
- [4] base roll [rad]
- [5] base pitch [rad]
- [6] right shoulder pitch [rad]
- [7] right shoulder roll [rad]
- [8] right elbow pitch [rad]
- [9] right elbow roll [rad]
- [10] right wrist pitch [rad]
- [11] left shoulder pitch [rad]
- [12] left shoulder roll [rad]
- [13] left elbow pitch [rad]
- [14] left elbow roll [rad]
- [15] left wrist pitch [rad]

"""  # NOTE the above is not true anymore, new kinfer allows specifying the command names

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from jaxtyping import Array
from kinfer.export.jax import export_fn
from kinfer.export.serialize import pack
from kinfer.rust_bindings import PyModelMetadata

import ksim

from .train import HumanoidWalkingTask, Model


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Carry:
    actor_carry: Array
    lpf_params: ksim.LowPassFilterParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()

    if not (ckpt_path := Path(args.checkpoint_path)).exists():
        raise FileNotFoundError(f"Checkpoint path {ckpt_path} does not exist")

    task: HumanoidWalkingTask = HumanoidWalkingTask.load_task(ckpt_path)
    mujoco_model = task.get_mujoco_model()
    init_params = ksim.InitParams(key=jax.random.PRNGKey(0), physics_model=mujoco_model)
    model = cast(Model, task.load_ckpt(ckpt_path, init_params=init_params, part="model")[0])

    joint_names_policy = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Removes the root joint.
    joint_names_passthrough = [
        "dof_right_wrist_gripper_05",
        "dof_left_wrist_gripper_05",
    ]
    joint_names = joint_names_policy + joint_names_passthrough

    # Constant values.
    depth = task.config.depth
    hidden_size = task.config.hidden_size
    carry_shape = (depth, 2, hidden_size)

    command_names = [
        "xvel",
        "yvel",
        "yawrate",
        "baseheight",
        "baseroll",
        "basepitch",
        "rshoulderpitch",
        "rshoulderroll",
        "rshoulderyaw",
        "relbowpitch",
        "rwristroll",
        "lshoulderpitch",
        "lshoulderroll",
        "lshoulderyaw",
        "lelbowpitch",
        "lwristroll",
        # cmds that are passed through directly:
        "rgripper",
        "lgripper",
    ]  # len 16 + 2

    metadata = PyModelMetadata(
        joint_names=joint_names,
        command_names=command_names,
        carry_size=(depth * 2 * hidden_size + len(joint_names_policy),),
    )

    init_carry = Carry(
        actor_carry=jnp.zeros(carry_shape),
        lpf_params=ksim.LowPassFilterParams.initialize(num_joints=len(joint_names_policy)),
    )
    flat, unravel = ravel_pytree(init_carry)

    @jax.jit
    def init_fn() -> Array:
        return flat

    @jax.jit
    def step_fn(
        joint_angles: Array,
        joint_angular_velocities: Array,
        projected_gravity: Array,
        gyroscope: Array,
        command: Array,
        carry: Array,
    ) -> tuple[Array, Array]:
        cmd_policy = command[:16]
        cmd_passthrough = command[16:]
        cmd_zero = (jnp.linalg.norm(command[..., :3], axis=-1) < 1e-3)[..., None]

        obs = jnp.concatenate(
            [
                task.normalize_joint_pos(joint_angles[: len(joint_names_policy)]),
                task.normalize_joint_vel(joint_angular_velocities[: len(joint_names_policy)]),
                task.encode_projected_gravity(projected_gravity),
                gyroscope,
                cmd_zero,
                cmd_policy,
            ],
            axis=-1,
        )

        # Converts the carry array to the PyTree.
        carry_pt: Carry = unravel(carry)
        dist, new_carry, lpf_params = model.actor.forward(
            obs_n=obs,
            carry=carry_pt.actor_carry,
            lpf_params=carry_pt.lpf_params,
        )
        action = jnp.concatenate([dist.mode(), cmd_passthrough])

        # Flattens the new carry.
        new_carry_pt = Carry(actor_carry=new_carry, lpf_params=lpf_params)
        new_carry_flat, _ = ravel_pytree(new_carry_pt)

        return action, new_carry_flat

    init_onnx = export_fn(
        model=init_fn,
        metadata=metadata,
    )

    step_onnx = export_fn(
        model=step_fn,
        metadata=metadata,
    )

    kinfer_model = pack(
        init_fn=init_onnx,
        step_fn=step_onnx,
        metadata=metadata,
    )

    # Saves the resulting model.
    (output_path := Path(args.output_path)).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(kinfer_model)


if __name__ == "__main__":
    main()
