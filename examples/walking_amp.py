# mypy: disable-error-code="override"
"""Example walking task using Adversarial Motion Priors."""

import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, Self, TypeVar

import attrs
import distrax
import equinox as eqx
import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import optax
import xax
from jaxtyping import Array, PRNGKeyArray

import ksim

logger = logging.getLogger(__name__)


NUM_JOINTS = 21

NUM_INPUTS = 2 + 1 + 2 + NUM_JOINTS + NUM_JOINTS + 160 + 96 + 3 + NUM_JOINTS + 3 + 4 + 3 + 3


class DefaultHumanoidActor(eqx.Module):
    """MLP-based actor for the walking task."""

    mlp: eqx.nn.MLP
    num_inputs: int = eqx.static_field()
    num_outputs: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
        num_outputs: int,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=num_outputs * 2,
            width_size=hidden_size,
            depth=depth,
            activation=jax.nn.relu,
            key=key,
        )

        self.num_outputs = num_outputs
        self.num_inputs = num_inputs

    def forward(self, obs_n: Array) -> distrax.Distribution:
        prediction_n = self.mlp(obs_n)

        # Converts the output to a distribution.
        mean_nm = prediction_n[: self.num_outputs]
        std_nm = prediction_n[self.num_outputs :]

        # Softplus and clip to ensure positive standard deviations.
        std_nm = jnp.clip(jax.nn.softplus(std_nm), max=10.0)
        dist_n = distrax.Normal(mean_nm, std_nm)
        return dist_n


class DefaultHumanoidCritic(eqx.Module):
    """MLP-based critic for the walking task."""

    mlp: eqx.nn.MLP
    num_inputs: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=1,
            width_size=hidden_size,
            depth=depth,
            activation=jax.nn.relu,
            key=key,
        )

        self.num_inputs = num_inputs

    def forward(self, obs_n: Array) -> Array:
        return self.mlp(obs_n)


class DefaultHumanoidModel(eqx.Module):
    actor: DefaultHumanoidActor
    critic: DefaultHumanoidCritic

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        min_std: float,
        max_std: float,
        num_inputs: int,
        num_joints: int,
        num_mixtures: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        self.actor = DefaultHumanoidActor(
            key,
            num_inputs=num_inputs,
            num_outputs=num_joints,
            hidden_size=hidden_size,
            depth=depth,
        )
        self.critic = DefaultHumanoidCritic(
            key,
            num_inputs=num_inputs,
            hidden_size=hidden_size,
            depth=depth,
        )


class DefaultHumanoidDiscriminator(eqx.Module):
    """MLP that scores concatenated features from multiple frames.

    Input : concat ( features(t_n-num_frames+1), ..., features(t_n) )
    Output : 1 logit
    """

    mlp: eqx.nn.MLP
    num_inputs: int = eqx.static_field()

    def __init__(
        self,
        key: PRNGKeyArray,
        *,
        num_inputs: int,
        hidden_size: int,
        depth: int,
    ) -> None:
        self.mlp = eqx.nn.MLP(
            in_size=num_inputs,
            out_size=1,
            width_size=hidden_size,
            depth=depth,
            activation=jax.nn.relu,
            key=key,
        )
        self.num_inputs = num_inputs

    def forward(self, x: Array) -> Array:
        """Compute discriminator probabilities."""
        return self.mlp(x)


@attrs.define(frozen=True, kw_only=True)
class RandomMotionFrameReset(ksim.Reset):
    """Reset that initialises the robot state from a random clip / frame of the reference motion."""

    # Store reference motion positions/velocities as immutable (nested) tuples for better JIT friendliness.
    positions: tuple[tuple[float, ...], ...]
    velocities: tuple[tuple[float, ...], ...]
    reset_pos: bool = attrs.field(default=False)

    def __call__(self, data: ksim.PhysicsData, curriculum_level: Array, rng: PRNGKeyArray) -> ksim.PhysicsData:
        # Convert the stored tuples to JAX arrays for computation.
        positions_arr = jnp.asarray(self.positions)
        velocities_arr = jnp.asarray(self.velocities)

        # Figures out how many frames we have.
        num_frames = positions_arr.shape[0]

        key_frame, rng = jax.random.split(rng)
        frame_idx = jax.random.randint(key_frame, (), 0, num_frames)

        # Extract the qpos and qvel for the selected frame.
        qpos_ref = positions_arr[frame_idx]
        qvel_ref = velocities_arr[frame_idx]

        # Replace qpos and qvel in physics data.
        new_qvel = qvel_ref
        if self.reset_pos:
            new_qpos = qpos_ref
        else:
            new_qpos = jnp.concatenate([data.qpos[:7], qpos_ref[7:]])

        data = ksim.update_data_field(data, "qpos", new_qpos)
        data = ksim.update_data_field(data, "qvel", new_qvel)
        return data

    @classmethod
    def create(
        cls,
        positions: Array,  # (T, Nq)
        velocities: Array,  # (T, Nq)
        reset_pos: bool = False,
    ) -> Self:
        """Factory converting Arrays to nested tuples to avoid repeated hashing inside JIT."""
        # Convert to Python nested tuples so they are static/immutable.
        pos_tuple = tuple(tuple(map(float, frame)) for frame in np.asarray(positions).tolist())
        vel_tuple = tuple(tuple(map(float, frame)) for frame in np.asarray(velocities).tolist())
        return cls(positions=pos_tuple, velocities=vel_tuple, reset_pos=reset_pos)


@dataclass
class HumanoidWalkingAMPTaskConfig(ksim.AMPConfig):
    # Policy parameters.
    num_mixtures: int = xax.field(
        value=5,
        help="The number of mixtures for the actor.",
    )
    hidden_size: int = xax.field(
        value=512,
        help="The hidden size for the MLPs.",
    )
    depth: int = xax.field(
        value=2,
        help="The depth for the MLPs.",
    )

    # Disciminator parameters.
    discriminator_hidden_size: int = xax.field(
        value=512,
        help="The hidden size for the discriminator.",
    )
    discriminator_depth: int = xax.field(  # This affects the perceptive field i think
        value=3,
        help="The depth for the discriminator.",
    )
    num_frames: int = xax.field(
        value=2,
        help="The number of frames to use for the discriminator.",
    )

    # Optimizer parameters.
    learning_rate: float = xax.field(
        value=1e-3,
        help="Learning rate for PPO.",
    )
    adam_weight_decay: float = xax.field(
        value=0.0,
        help="Weight decay for the Adam optimizer.",
    )
    max_discriminator_grad_norm: float = xax.field(
        value=2.0,
        help="Maximum gradient norm for clipping.",
    )
    discriminator_learning_rate: float = xax.field(
        value=1e-4,
        help="Learning rate for the discriminator.",
    )

    # Curriculum parameters.
    num_curriculum_levels: int = xax.field(
        value=10,
        help="The number of curriculum levels to use.",
    )
    increase_threshold: float = xax.field(
        value=3.0,
        help="Increase the curriculum level when the mean trajectory length is above this threshold.",
    )
    decrease_threshold: float = xax.field(
        value=1.0,
        help="Decrease the curriculum level when the mean trajectory length is below this threshold.",
    )
    min_level_steps: int = xax.field(
        value=50,
        help="The minimum number of steps to wait before changing the curriculum level.",
    )

    # Rendering parameters.
    render_track_body_id: int | None = xax.field(
        value=0,
        help="The body id to track with the render camera.",
    )


Config = TypeVar("Config", bound=HumanoidWalkingAMPTaskConfig)


class HumanoidWalkingAMPTask(ksim.AMPTask[Config], Generic[Config]):
    """Adversarial Motion Prior task."""

    def __init__(self, config: Config) -> None:
        super().__init__(config)

        mj_model = self.get_mujoco_model()

        self.hand_left_id = ksim.get_body_data_idx_from_name(mj_model, "hand_left")
        self.hand_right_id = ksim.get_body_data_idx_from_name(mj_model, "hand_right")

        self.foot_left_id = ksim.get_body_data_idx_from_name(mj_model, "foot_left")
        self.foot_right_id = ksim.get_body_data_idx_from_name(mj_model, "foot_right")

    def get_mujoco_model(self) -> mujoco.MjModel:
        mjcf_path = (Path(__file__).parent / "data" / "scene.mjcf").resolve().as_posix()
        return mujoco.MjModel.from_xml_path(mjcf_path)

    def get_mujoco_model_metadata(self, mj_model: mujoco.MjModel) -> ksim.Metadata:
        return ksim.Metadata.from_model(
            mj_model,
            kp=1.0,
            kd=0.1,
            armature=1e-2,
            friction=1e-6,
        )

    def get_actuators(
        self,
        physics_model: ksim.PhysicsModel,
        metadata: ksim.Metadata | None = None,
    ) -> ksim.Actuators:
        assert metadata is not None, "Metadata is required"
        return ksim.PositionActuators(
            physics_model=physics_model,
            metadata=metadata,
        )

    def get_physics_randomizers(self, physics_model: ksim.PhysicsModel) -> list[ksim.PhysicsRandomizer]:
        return [
            ksim.StaticFrictionRandomizer(),
            ksim.ArmatureRandomizer(),
            ksim.MassMultiplicationRandomizer.from_body_name(physics_model, "torso"),
            ksim.JointDampingRandomizer(),
            ksim.JointZeroPositionRandomizer(),
        ]

    def get_events(self, physics_model: ksim.PhysicsModel) -> list[ksim.Event]:
        return [
            ksim.LinearPushEvent(
                linvel=0.3,
                interval_range=(0.25, 0.75),
            ),
        ]

    def get_resets(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reset]:
        ref_motion = self.get_real_motions(self.get_mujoco_model())
        ref_qpos = ref_motion["qpos"].squeeze(0)
        ref_qvel = ref_motion["qvel"].squeeze(0)
        return [
            RandomMotionFrameReset.create(ref_qpos, ref_qvel, reset_pos=True),
        ]

    def get_observations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Observation]:
        return [
            ksim.JointPositionObservation(),
            ksim.JointVelocityObservation(),
            ksim.ActuatorForceObservation(),
            ksim.CenterOfMassInertiaObservation(),
            ksim.CenterOfMassVelocityObservation(),
            ksim.BasePositionObservation(),
            ksim.BaseOrientationObservation(),
            ksim.BaseLinearVelocityObservation(),
            ksim.BaseAngularVelocityObservation(),
            ksim.BaseLinearAccelerationObservation(),
            ksim.BaseAngularAccelerationObservation(),
            ksim.ProjectedGravityObservation.create(
                physics_model=physics_model,
                framequat_name="orientation",
            ),
            ksim.ActuatorAccelerationObservation(),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_acc"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="imu_gyro"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="local_linvel"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="upvector"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="forwardvector"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="global_linvel"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="global_angvel"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="position"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="orientation"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_global_linvel"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_global_linvel"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_upvector"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_upvector"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="left_foot_pos"),
            ksim.SensorObservation.create(physics_model=physics_model, sensor_name="right_foot_pos"),
            ksim.FeetContactObservation.create(
                physics_model=physics_model,
                foot_left_geom_names=["foot_left"],
                foot_right_geom_names=["foot_right"],
                floor_geom_names=["floor"],
            ),
            ksim.FeetPositionObservation.create(
                physics_model=physics_model,
                foot_left_body_name="foot_left",
                foot_right_body_name="foot_right",
            ),
            ksim.FeetOrientationObservation.create_from_feet(
                physics_model=physics_model,
                foot_left_body_name="foot_left",
                foot_right_body_name="foot_right",
            ),
            ksim.TimestepObservation(),
        ]

    def get_commands(self, physics_model: ksim.PhysicsModel) -> list[ksim.Command]:
        return [
            ksim.LinearVelocityCommand(
                x_range=(-0.15, 1.0),
                y_range=(0.0, 0.0),
                switch_prob=self.config.ctrl_dt / 4.0,  # on avg 3 seconds between switches
            ),
            ksim.FloatVectorCommand(  # angular velocity
                ranges=((-0.3, 0.3),),
                zero_prob=0.8,  # 80% of the time the command is zero
                switch_prob=self.config.ctrl_dt / 4.0,  # on avg 3 seconds between switches
            ),
        ]

    def get_rewards(self, physics_model: ksim.PhysicsModel) -> list[ksim.Reward]:
        return [
            ksim.AMPReward(scale=20.0),
            ksim.StayAliveReward(scale=1000.0, balance=1000.0),
            ksim.XYAngularVelocityPenalty(scale=-0.01),
            ksim.AngularVelocityTrackingReward(
                index=("z"),
                command_name="float_vector_command",
                scale=5.0,
            ),
            # ksim.UprightReward(scale=2.0),
            # ksim.NaiveForwardReward(clip_max=1.0, scale=1.0),
            ksim.LinearVelocityTrackingReward(
                linvel_obs_name="base_linear_velocity_observation",
                scale=5.0,
            ),
        ]

    def get_terminations(self, physics_model: ksim.PhysicsModel) -> list[ksim.Termination]:
        return [
            ksim.BadZTermination(unhealthy_z_lower=0.9, unhealthy_z_upper=1.6),
            ksim.NotUprightTermination(max_radians=math.radians(80)),
            ksim.HighVelocityTermination(),
            ksim.FarFromOriginTermination(max_dist=10.0),
        ]

    def get_curriculum(self, physics_model: ksim.PhysicsModel) -> ksim.Curriculum:
        return ksim.EpisodeLengthCurriculum(
            num_levels=self.config.num_curriculum_levels,
            increase_threshold=self.config.increase_threshold,
            decrease_threshold=self.config.decrease_threshold,
            min_level_steps=self.config.min_level_steps,
        )

    def get_initial_model_carry(self, model: DefaultHumanoidModel, rng: PRNGKeyArray) -> None:
        return None

    def get_policy_model(self, key: PRNGKeyArray) -> DefaultHumanoidModel:
        return DefaultHumanoidModel(
            key,
            num_inputs=NUM_INPUTS,
            num_joints=NUM_JOINTS,
            num_mixtures=self.config.num_mixtures,
            min_std=0.01,
            max_std=1.0,
            hidden_size=self.config.hidden_size,
            depth=self.config.depth,
        )

    def get_discriminator_model(self, key: PRNGKeyArray) -> DefaultHumanoidDiscriminator:
        joint_inputs = NUM_JOINTS * 2  # sin, cos, velocity (temp no velocity)
        hand_inputs = 3 * 2  # xyz, left right
        feet_inputs = 3 * 2  # xyz, left right
        base_inputs = 6  # lin vel, ang vel
        features_per_frame = joint_inputs + hand_inputs + feet_inputs + base_inputs
        return DefaultHumanoidDiscriminator(
            key,
            num_inputs=features_per_frame * self.config.num_frames,
            hidden_size=self.config.discriminator_hidden_size,
            depth=self.config.discriminator_depth,
        )

    def get_policy_optimizer(self) -> optax.GradientTransformation:
        return (
            optax.adam(self.config.learning_rate)
            if self.config.adam_weight_decay == 0.0
            else optax.adamw(self.config.learning_rate, weight_decay=self.config.adam_weight_decay)
        )

    def get_discriminator_optimizer(self) -> optax.GradientTransformation:
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.config.max_discriminator_grad_norm),
            optax.adam(self.config.discriminator_learning_rate),
        )
        return optimizer

    def call_discriminator(
        self,
        model: DefaultHumanoidDiscriminator,
        motion: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
    ) -> Array:
        """Prepare input for the discriminator with multiple frames."""
        qpos = motion["qpos"]
        qvel = motion["qvel"]
        hand_pos = motion["hand_pos"]
        feet_pos = motion["feet_pos"]

        joints = qpos[..., 7:]

        # Similar to the 6d joint encoding but simplified because angles are 1D.
        # NOTE: I was experimenting with the relative encoding but idrt it helps at all
        # Delete this later
        sin_cos_joints = jnp.concatenate([joints, jnp.cos(joints)], axis=-1)

        # features_t = jnp.concatenate([sin_cos_joints, qvel, hand_pos, feet_pos], axis=-1)
        features_t = jnp.concatenate([sin_cos_joints, qvel[..., :6], hand_pos, feet_pos], axis=-1)

        num_frames = self.config.num_frames
        timesteps = features_t.shape[0]

        # Pad the beginning with the first frame repeated
        padded_features = jnp.concatenate([jnp.tile(features_t[:1], (num_frames - 1, 1)), features_t], axis=0)

        # Create sliding window of features
        multi_frame_features = []
        for t in range(timesteps):
            frame_features = padded_features[t : t + num_frames].flatten()
            multi_frame_features.append(frame_features)

        multi_frame_features = jnp.stack(multi_frame_features)  # (T, num_frames * features_per_frame)
        logits = jax.vmap(model.forward)(multi_frame_features)
        return logits.squeeze(-1)  # There is a shape check to match traj.done which is (T,)

    def get_real_motions(self, mj_model: mujoco.MjModel) -> xax.FrozenDict[str, Array]:
        """Loads a trajectory from a .npz file and converts it to the (batch, T, 20) tensor expected by AMP.

        Expected keys inside the .npz:
          • 'qpos'            –  (T, Nq)
          • optional 'frequency' –  sampling Hz (used for verification)
        """
        traj_path = Path(__file__).parent / "data" / "slow_fast_dh.npz"

        npz = np.load(traj_path, allow_pickle=True)

        qpos = jnp.array(npz["qpos"])[100:-70]

        if float(npz["frequency"]) != 1 / self.config.ctrl_dt:
            raise ValueError(f"Motion frequency {npz['frequency']} does not match ctrl_dt {self.config.ctrl_dt}")

        # Compute hand positions relative to the robot root for each frame.
        mj_data = mujoco.MjData(mj_model)
        t_frames = qpos.shape[0]
        hand_pos = np.zeros((t_frames, 6))  # left xyz, right xyz
        feet_pos = np.zeros((t_frames, 6))  # left xyz, right xyz

        for t_i in range(t_frames):
            mj_data.qpos = np.array(qpos[t_i])
            mujoco.mj_forward(mj_model, mj_data)

            # Root world position and rotation matrix (flattened 3x3 in mj_data.xmat)
            root_pos = mj_data.xpos[0]

            # World positions of hands
            left_hand_world = mj_data.xpos[self.hand_left_id]
            right_hand_world = mj_data.xpos[self.hand_right_id]

            # World positions of feet
            left_foot_world = mj_data.xpos[self.foot_left_id]
            right_foot_world = mj_data.xpos[self.foot_right_id]

            # Convert to root–relative coordinates
            left_hand_rel = left_hand_world - root_pos
            right_hand_rel = right_hand_world - root_pos

            left_feet_rel = left_foot_world - root_pos
            right_feet_rel = right_foot_world - root_pos

            hand_pos[t_i, 0:3] = left_hand_rel
            hand_pos[t_i, 3:6] = right_hand_rel

            feet_pos[t_i, 0:3] = left_feet_rel
            feet_pos[t_i, 3:6] = right_feet_rel

        hand_pos = jnp.array(hand_pos)
        feet_pos = jnp.array(feet_pos)

        joint_limits = ksim.get_position_limits(mj_model)
        joint_names = ksim.get_joint_names_in_order(mj_model)

        joint_mins = []
        joint_maxs = []
        for name in joint_names[1:]:  # skip freejoint
            if name not in joint_limits:
                raise KeyError(f"Joint '{name}' missing from joint limits dictionary")
            j_min, j_max = joint_limits[name]
            joint_mins.append(j_min)
            joint_maxs.append(j_max)

        joint_mins_arr = jnp.asarray(joint_mins)
        joint_maxs_arr = jnp.asarray(joint_maxs)

        # Separate freejoint (7) and articulated joints.
        qpos_root = qpos[..., :7]
        qpos_joints = qpos[..., 7:]

        # Bring each angle into range by shifting with multiples of 2π.
        two_pi = 2.0 * math.pi
        center_arr = (joint_mins_arr + joint_maxs_arr) / 2.0  # (J,)

        # Vectorised 2π-shifting about the joint-range centre.
        qpos_orig = qpos_joints  # keep a copy for statistics
        qpos_shifted = qpos_orig - jnp.round((qpos_orig - center_arr) / two_pi) * two_pi

        # Final clipping (handles ranges narrower than 2π or numerical drift).
        qpos_joints = jnp.clip(qpos_shifted, joint_mins_arr[None, :], joint_maxs_arr[None, :])

        # Re-assemble the full qpos.
        qpos = jnp.concatenate([qpos_root, qpos_joints], axis=-1)

        adjust_mask = jnp.any(jnp.abs(qpos_joints - qpos_orig) > 1e-6, axis=-1)
        num_adjusted = int(adjust_mask.sum())
        if num_adjusted:
            logger.info(
                "Reference motion sanitisation: adjusted %d/%d frames (%.1f%%) via wrap/clip.",
                num_adjusted,
                t_frames,
                100.0 * num_adjusted / t_frames,
            )

        # (batch, t, num_joints)

        # Get qvel
        joint_qvel = jnp.diff(qpos[:, 7:], prepend=qpos[:, 7:][:1], axis=0)

        base_qvel = jnp.diff(qpos[:, :3], prepend=qpos[:, :3][:1], axis=0)

        base_quat = qpos[..., 3:7]

        base_euler = xax.quat_to_euler(base_quat)
        base_ang_vel = jnp.diff(base_euler, prepend=base_euler[:1], axis=0)

        qvel = jnp.concatenate([base_qvel, base_ang_vel, joint_qvel], axis=-1)
        return xax.FrozenDict(
            {"qpos": qpos[None], "qvel": qvel[None], "hand_pos": hand_pos[None], "feet_pos": feet_pos[None]}
        )

    def trajectory_to_motion(self, trajectory: ksim.Trajectory) -> xax.FrozenDict[str, Array]:
        # Joint positions
        qpos = trajectory.qpos

        qvel = trajectory.qvel

        # Root world position and rotation matrix
        root_pos = trajectory.xpos[..., 0, :]

        # Hand world positions
        hand_left_world = trajectory.xpos[..., self.hand_left_id, :]
        hand_right_world = trajectory.xpos[..., self.hand_right_id, :]

        # Relative positions
        hand_left_rel = hand_left_world - root_pos
        hand_right_rel = hand_right_world - root_pos

        feet_left_world = trajectory.xpos[..., self.foot_left_id, :]
        feet_right_world = trajectory.xpos[..., self.foot_right_id, :]

        feet_left_rel = feet_left_world - root_pos
        feet_right_rel = feet_right_world - root_pos

        hand_pos = jnp.concatenate([hand_left_rel, hand_right_rel], axis=-1)
        feet_pos = jnp.concatenate([feet_left_rel, feet_right_rel], axis=-1)
        return xax.FrozenDict({"qpos": qpos, "qvel": qvel, "hand_pos": hand_pos, "feet_pos": feet_pos})

    def motion_to_qpos(self, motion: Array) -> Array:
        return motion["qpos"]

    def run_actor(
        self,
        model: DefaultHumanoidActor,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> distrax.Distribution:
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        # imu_acc_3 = observations["sensor_observation_imu_acc"]
        # imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd_1 = commands["float_vector_command"]

        obs_n = jnp.concatenate(
            [
                lin_vel_cmd_2,  # 2
                ang_vel_cmd_1,  # 1
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                proj_grav_3 / 8.0,  # 3
                act_frc_obs_n,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3 / 200.0,  # 3
            ],
            axis=-1,
        )

        return model.forward(obs_n)

    def run_critic(
        self,
        model: DefaultHumanoidCritic,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
    ) -> Array:
        timestep_1 = observations["timestep_observation"]
        dh_joint_pos_j = observations["joint_position_observation"]
        dh_joint_vel_j = observations["joint_velocity_observation"]
        com_inertia_n = observations["center_of_mass_inertia_observation"]
        com_vel_n = observations["center_of_mass_velocity_observation"]
        # imu_acc_3 = observations["sensor_observation_imu_acc"]
        # imu_gyro_3 = observations["sensor_observation_imu_gyro"]
        proj_grav_3 = observations["projected_gravity_observation"]
        act_frc_obs_n = observations["actuator_force_observation"]
        base_pos_3 = observations["base_position_observation"]
        base_quat_4 = observations["base_orientation_observation"]
        lin_vel_obs_3 = observations["base_linear_velocity_observation"]
        ang_vel_obs_3 = observations["base_angular_velocity_observation"]
        lin_vel_cmd_2 = commands["linear_velocity_command"]
        ang_vel_cmd_1 = commands["float_vector_command"]

        obs_n = jnp.concatenate(
            [
                lin_vel_cmd_2,  # 2
                ang_vel_cmd_1,  # 1
                jnp.cos(timestep_1),  # 1
                jnp.sin(timestep_1),  # 1
                dh_joint_pos_j,  # NUM_JOINTS
                dh_joint_vel_j / 10.0,  # NUM_JOINTS
                com_inertia_n,  # 160
                com_vel_n,  # 96
                proj_grav_3,  # 3
                act_frc_obs_n / 100.0,  # NUM_JOINTS
                base_pos_3,  # 3
                base_quat_4,  # 4
                lin_vel_obs_3,  # 3
                ang_vel_obs_3,  # 3
            ],
            axis=-1,
        )

        return model.forward(obs_n)

    def get_ppo_variables(
        self,
        model: DefaultHumanoidModel,
        trajectory: ksim.Trajectory,
        model_carry: None,
        rng: PRNGKeyArray,
    ) -> tuple[ksim.PPOVariables, None]:
        def get_log_prob(transition: ksim.Trajectory) -> Array:
            action_dist_t = self.run_actor(model.actor, transition.obs, transition.command)
            log_prob_t = action_dist_t.log_prob(transition.action)
            assert isinstance(log_prob_t, Array)
            return log_prob_t

        log_probs_tj = jax.vmap(get_log_prob)(trajectory)

        values_tj = jax.vmap(self.run_critic, in_axes=(None, 0, 0))(model.critic, trajectory.obs, trajectory.command)

        ppo_variables = ksim.PPOVariables(
            log_probs=log_probs_tj,
            values=values_tj.squeeze(-1),
        )

        return ppo_variables, None

    def sample_action(
        self,
        model: DefaultHumanoidModel,
        model_carry: None,
        physics_model: ksim.PhysicsModel,
        physics_state: ksim.PhysicsState,
        observations: xax.FrozenDict[str, Array],
        commands: xax.FrozenDict[str, Array],
        rng: PRNGKeyArray,
        argmax: bool,
    ) -> ksim.Action:
        action_dist_j = self.run_actor(
            model=model.actor,
            observations=observations,
            commands=commands,
        )

        action_j = action_dist_j.mode() if argmax else action_dist_j.sample(seed=rng)

        return ksim.Action(action=action_j, carry=None)


if __name__ == "__main__":
    # To run training, use the following command:
    #   python -m examples.walking_amp
    # To visualize the environment, use the following command:
    #   python -m examples.walking_amp run_mode=view
    # On MacOS or other devices with less memory, you can change the number
    # of environments and batch size to reduce memory usage. Here's an example
    # from the command line:
    #   python -m examples.walking_amp num_envs=8 num_batches=2
    HumanoidWalkingAMPTask.launch(
        HumanoidWalkingAMPTaskConfig(
            num_envs=2048,
            batch_size=256,
            num_passes=10,
            epochs_per_log_step=1,
            valid_every_n_steps=10,
            amp_grad_penalty_coef=1.0,
            # Simulation parameters.
            dt=0.005,
            ctrl_dt=0.02,
            iterations=3,
            ls_iterations=5,
            rollout_length_seconds=8.0,
            # PPO parameters
            # gamma=0.97,
            gamma=0.95,
            # lam=0.95,
            lam=0.98,
            entropy_coef=0.0001,
            learning_rate=3e-4,
            clip_param=0.3,
            global_grad_clip=1.0,
            render_markers=True,
        ),
    )
