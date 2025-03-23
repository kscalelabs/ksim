"""Example script to deploy a SavedModel in KOS-Sim."""

import argparse
import asyncio
import logging
import signal
import subprocess
import sys
import time
import types
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pykos

logger = logging.getLogger(__name__)

DT = 0.02  # Policy time step (50Hz)


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    Actuator(actuator_id=11, nn_id=0, kp=1.0, kd=5.0, max_torque=40.0, joint_name="left_shoulder_pitch_03"),
    Actuator(actuator_id=12, nn_id=1, kp=1.0, kd=5.0, max_torque=30.0, joint_name="left_shoulder_roll_03"),
    Actuator(actuator_id=13, nn_id=2, kp=1.0, kd=5.0, max_torque=30.0, joint_name="left_shoulder_yaw_02"),
    Actuator(actuator_id=14, nn_id=3, kp=1.0, kd=5.0, max_torque=40.0, joint_name="left_elbow_02"),
    Actuator(actuator_id=15, nn_id=4, kp=1.0, kd=5.0, max_torque=10.0, joint_name="left_wrist_02"),
    Actuator(actuator_id=21, nn_id=5, kp=1.0, kd=5.0, max_torque=40.0, joint_name="right_shoulder_pitch_03"),
    Actuator(actuator_id=22, nn_id=6, kp=1.0, kd=5.0, max_torque=30.0, joint_name="right_shoulder_roll_03"),
    Actuator(actuator_id=23, nn_id=7, kp=1.0, kd=5.0, max_torque=30.0, joint_name="right_shoulder_yaw_02"),
    Actuator(actuator_id=24, nn_id=8, kp=1.0, kd=5.0, max_torque=40.0, joint_name="right_elbow_02"),
    Actuator(actuator_id=25, nn_id=9, kp=1.0, kd=5.0, max_torque=10.0, joint_name="right_wrist_02"),
    Actuator(actuator_id=31, nn_id=10, kp=120.0, kd=5.0, max_torque=40.0, joint_name="left_hip_pitch_04"),
    Actuator(actuator_id=32, nn_id=11, kp=80.0, kd=5.0, max_torque=30.0, joint_name="left_hip_roll_03"),
    Actuator(actuator_id=33, nn_id=12, kp=80.0, kd=5.0, max_torque=30.0, joint_name="left_hip_yaw_03"),
    Actuator(actuator_id=34, nn_id=13, kp=120.0, kd=5.0, max_torque=40.0, joint_name="left_knee_04"),
    Actuator(actuator_id=35, nn_id=14, kp=20.0, kd=5.0, max_torque=10.0, joint_name="left_ankle_02"),
    Actuator(actuator_id=41, nn_id=15, kp=120.0, kd=5.0, max_torque=40.0, joint_name="right_hip_pitch_04"),
    Actuator(actuator_id=42, nn_id=16, kp=80.0, kd=5.0, max_torque=30.0, joint_name="right_hip_roll_03"),
    Actuator(actuator_id=43, nn_id=17, kp=80.0, kd=5.0, max_torque=30.0, joint_name="right_hip_yaw_03"),
    Actuator(actuator_id=44, nn_id=18, kp=120.0, kd=5.0, max_torque=40.0, joint_name="right_knee_04"),
    Actuator(actuator_id=45, nn_id=19, kp=20.0, kd=5.0, max_torque=10.0, joint_name="right_ankle_02"),
]


async def get_observation(kos: pykos.KOS) -> np.ndarray:
    (actuator_states,) = await asyncio.gather(
        kos.actuator.get_actuators_state([ac.actuator_id for ac in ACTUATOR_LIST]),
        # kos.imu.get_imu_values(),
    )

    state_dict = {state.actuator_id: state.position for state in actuator_states.states}

    pos_obs = np.array([state_dict[ac.actuator_id] for ac in sorted(ACTUATOR_LIST, key=lambda x: x.nn_id)])

    pos_obs = np.deg2rad(pos_obs)

    return pos_obs


async def send_actions(kos: pykos.KOS, actions: np.ndarray) -> None:
    actions = np.rad2deg(actions)
    actuator_commands: list[pykos.services.actuator.ActuatorCommand] = [
        {
            "actuator_id": ac.actuator_id,
            "position": actions[ac.nn_id],
        }
        for ac in ACTUATOR_LIST
    ]

    logger.debug(actuator_commands)
    await kos.actuator.command_actuators(actuator_commands)


async def configure_actuators(kos: pykos.KOS) -> None:
    for ac in ACTUATOR_LIST:
        await kos.actuator.configure_actuator(
            actuator_id=ac.actuator_id,
            kp=ac.kp,
            kd=ac.kd,
        )


async def reset(kos: pykos.KOS) -> None:
    await kos.sim.reset(
        pos={"x": 0.0, "y": 0.0, "z": 1.0},
        quat={"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
        joints=[
            {"name": actuator.joint_name, "pos": pos}
            for actuator, pos in zip(ACTUATOR_LIST, [0.0] * len(ACTUATOR_LIST))
        ],
    )


def spawn_kos_sim(no_render: bool) -> tuple[subprocess.Popen, Callable]:
    """Spawn the KOS-Sim default-humanoid process and return the process object."""
    logger.info("Starting KOS-Sim default-humanoid...")
    args = ["kos-sim", "kbot2-feet"]
    if no_render:
        args.append("--no-render")
    process = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    logger.info("Waiting for KOS-Sim to start...")
    time.sleep(5)

    def cleanup(sig: int | None = None, frame: types.FrameType | None = None) -> None:
        logger.info("Terminating KOS-Sim...")
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        if sig:
            sys.exit(0)

    signal.signal(signal.SIGTERM, cleanup)

    return process, cleanup


async def main(model_path: str, ip: str, no_render: bool, episode_length: int) -> None:
    # model = tf.saved_model.load(model_path)

    cmd = [0.0, 0.0]  # x, y

    cmd = np.array(cmd).reshape(1, -1)

    sim_process = None
    cleanup_fn = None

    try:
        # Try to connect to existing KOS-Sim
        logger.info("Attempting to connect to existing KOS-Sim...")
        kos = pykos.KOS(ip=ip)
        await kos.sim.get_parameters()
        logger.info("Connected to existing KOS-Sim instance.")
    except Exception as e:
        logger.info("Could not connect to existing KOS-Sim: %s", e)
        logger.info("Starting a new KOS-Sim instance locally...")
        sim_process, cleanup_fn = spawn_kos_sim(no_render)
        kos = pykos.KOS()
        attempts = 0
        while attempts < 5:
            try:
                await kos.sim.get_parameters()
                logger.info("Connected to new KOS-Sim instance.")
                break
            except Exception as connect_error:
                attempts += 1
                logger.info("Failed to connect to KOS-Sim: %s", connect_error)
                time.sleep(2)

        if attempts == 5:
            raise RuntimeError("Failed to connect to KOS-Sim")

    await configure_actuators(kos)
    await reset(kos)

    observation = (await get_observation(kos)).reshape(1, -1)

    if no_render:
        await kos.process_manager.start_kclip("deployment")

    # warm up model
    # model.infer(observation, cmd)

    target_time = time.time() + DT
    observation = await get_observation(kos)

    end_time = time.time() + episode_length

    try:
        while time.time() < end_time:
            observation = observation.reshape(1, -1)
            # action = np.array(model.infer(observation)).reshape(-1)

            action = np.random.uniform(-1.0, 1.0, size=observation[0].shape)

            observation, _ = await asyncio.gather(
                get_observation(kos),
                send_actions(kos, action),
            )

            if time.time() < target_time:
                await asyncio.sleep(max(0, target_time - time.time()))
            else:
                logger.info("Loop overran by %s seconds", time.time() - target_time)

            target_time += DT
    except asyncio.CancelledError:
        logger.info("Exiting...")
        if no_render:
            save_path = await kos.process_manager.stop_kclip("deployment")
            logger.info("KClip saved to %s", save_path)

        if cleanup_fn:
            cleanup_fn()

        raise KeyboardInterrupt

    logger.info("Episode finished!")

    if no_render:
        await kos.process_manager.stop_kclip("deployment")

    if cleanup_fn:
        cleanup_fn()


# (optionally) start the KOS-Sim server before running this script
# `kos-sim kbot2-feet`
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--ip", type=str, default="localhost")
    parser.add_argument("--episode_length", type=int, default=10)  # seconds
    parser.add_argument("--no-render", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO)
    asyncio.run(main(args.model_path, args.ip, args.no_render, args.episode_length))
