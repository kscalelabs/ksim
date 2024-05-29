"""Visualization Script for KSim"""

import argparse
import os

import mujoco


def visualize(path_to_xml):
    """Visualize a MuJoCo XML file.

    Args:
        path_to_xml: Path to MuJoCo XML file.
    """
    m = mujoco.MjModel.from_xml_path(path_to_xml)
    d = mujoco.MjData(m)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        viewer.sync()

        while viewer.is_running():
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a MuJoCo XML file.")
    parser.add_argument("path_to_xml", type=str, help="Path to MuJoCo XML file.")
    args = parser.parse_args()

    visualize(args.path_to_xml)
