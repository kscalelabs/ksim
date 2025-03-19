"""Example script to deploy a SavedModel in KOS-Sim."""
import argparse
import asyncio
import time
from dataclasses import dataclass

import numpy as np
import pykos
import tensorflow as tf


@dataclass
class Actuator:
    actuator_id: int
    nn_id: int
    kp: float
    kd: float
    max_torque: float
    joint_name: str


ACTUATOR_LIST: list[Actuator] = [
    Actuator(actuator_id=1, nn_id=0, kp=50.0, kd=1.0, max_torque=30.0, joint_name="abdomen_y"),
    Actuator(actuator_id=2, nn_id=1, kp=50.0, kd=1.0, max_torque=30.0, joint_name="abdomen_z"),
    Actuator(actuator_id=3, nn_id=2, kp=50.0, kd=1.0, max_torque=30.0, joint_name="abdomen_x"),
    Actuator(actuator_id=4, nn_id=3, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_x_right"),
    Actuator(actuator_id=5, nn_id=4, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_z_right"),
    Actuator(actuator_id=6, nn_id=5, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_y_right"),
    Actuator(actuator_id=7, nn_id=6, kp=50.0, kd=1.0, max_torque=30.0, joint_name="knee_right"),
    Actuator(actuator_id=8, nn_id=7, kp=50.0, kd=1.0, max_torque=30.0, joint_name="ankle_x_right"),
    Actuator(actuator_id=9, nn_id=8, kp=50.0, kd=1.0, max_torque=30.0, joint_name="ankle_y_right"),
    Actuator(actuator_id=10, nn_id=9, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_x_left"),
    Actuator(actuator_id=11, nn_id=10, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_z_left"),
    Actuator(actuator_id=12, nn_id=11, kp=50.0, kd=1.0, max_torque=30.0, joint_name="hip_y_left"),
    Actuator(actuator_id=13, nn_id=12, kp=50.0, kd=1.0, max_torque=30.0, joint_name="knee_left"),
    Actuator(actuator_id=14, nn_id=13, kp=50.0, kd=1.0, max_torque=30.0, joint_name="ankle_x_left"),
    Actuator(actuator_id=15, nn_id=14, kp=50.0, kd=1.0, max_torque=30.0, joint_name="ankle_y_left"),
    Actuator(actuator_id=16, nn_id=15, kp=50.0, kd=1.0, max_torque=30.0, joint_name="shoulder1_right"),
    Actuator(actuator_id=17, nn_id=16, kp=50.0, kd=1.0, max_torque=30.0, joint_name="shoulder2_right"),
    Actuator(actuator_id=18, nn_id=17, kp=50.0, kd=1.0, max_torque=30.0, joint_name="elbow_right"),
    Actuator(actuator_id=19, nn_id=18, kp=50.0, kd=1.0, max_torque=30.0, joint_name="shoulder1_left"),
    Actuator(actuator_id=20, nn_id=19, kp=50.0, kd=1.0, max_torque=30.0, joint_name="shoulder2_left"),
    Actuator(actuator_id=21, nn_id=20, kp=50.0, kd=1.0, max_torque=30.0, joint_name="elbow_left"),
]

async def get_observation(kos: pykos.KOS) -> np.ndarray:
    actuator_states, imu_data = await asyncio.gather(
        kos.actuator.get_actuators_state([ac.actuator_id for ac in ACTUATOR_LIST]),
        kos.imu.get_imu_values(),
    )

    return np.array(actuator_states)

async def send_actions(kos: pykos.KOS, actions: np.ndarray) -> None:
    actuator_commands: list[pykos.ActuatorCommand] = [
        {
            "actuator_id": ac.actuator_id,
            "position": actions[ac.nn_id],

        }
        for ac in ACTUATOR_LIST
    ]

    await kos.actuator.command_actuators(actuator_commands)

async def configure_actuators(kos: pykos.KOS) -> None:
    for ac in ACTUATOR_LIST:
        await kos.actuator.configure_actuator(
            actuator_id=ac.actuator_id,
            kp=ac.kp,
            kd=ac.kd,
        )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    args = parser.parse_args()

    model = tf.saved_model.load(args.model_path)

    cmd = [0.5, 0.0] # x, y

    kos = pykos.KOS()

    dt = 0.02

    await configure_actuators(kos)

    target_time = time.time() + dt

    try:
        while True:
            observation = await get_observation(kos)
            print(observation)

            action = model.infer(observation, cmd)
            await send_actions(kos, action)

            await asyncio.sleep(time.time() - target_time)
            target_time += dt
    except KeyboardInterrupt:
        print("Exiting...")


# Start the KOS-Sim server before running this script
# `kos-sim default-humanoid`
if __name__ == "__main__":
    asyncio.run(main())
