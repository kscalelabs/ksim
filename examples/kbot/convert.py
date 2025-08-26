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

Since the training commands differ from this, we need to correct them here.
"""

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
from examples.kbot.train import HumanoidWalkingTask, Model


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
    key = jax.random.PRNGKey(0)
    init_params = ksim.InitParams(key=key, physics_model=mujoco_model)
    model = cast(Model, task.load_ckpt(ckpt_path, init_params=init_params, part="model")[0])

    # Loads the Mujoco model and gets the joint names.
    joint_names = ksim.get_joint_names_in_order(mujoco_model)[1:]  # Removes the root joint.

    # Constant values.
    depth = task.config.depth
    hidden_size = task.config.hidden_size
    carry_shape = (2, depth, hidden_size)

    metadata = PyModelMetadata(
        joint_names=joint_names,
        num_commands=16,
        carry_size=(2 * depth * hidden_size + len(joint_names),),
    )

    init_carry = Carry(
        actor_carry=jnp.zeros(carry_shape),
        lpf_params=ksim.LowPassFilterParams.initialize(num_joints=len(joint_names)),
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
        # Call the model.
        obs = jnp.concatenate(
            [
                joint_angles,
                joint_angular_velocities / 10.0,
                projected_gravity,
                gyroscope,
            ],
            axis=-1,
        )

        # Converts the carry array to the PyTree.
        carry_pt: Carry = unravel(carry)
        curriculum_level = jnp.ones(1)
        dist, new_carry, lpf_params = model.actor.forward(
            obs_n=obs,
            carry=carry_pt.actor_carry,
            curriculum_level=curriculum_level,
            lpf_params=carry_pt.lpf_params,
        )

        # Flattens the new carry.
        new_carry_pt = Carry(
            actor_carry=new_carry,
            lpf_params=lpf_params,
        )
        new_carry_flat, _ = ravel_pytree(new_carry_pt)

        return dist.mode(), new_carry_flat

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
