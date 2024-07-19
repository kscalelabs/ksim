# mypy: disable-error-code="operator,union-attr"
"""Defines common types and functions for exporting MJCF files.

Run:
    python ksim/scripts/process_mjcf.py /path/to/stompy.xml

Todo:
    0. Add IMU to the right position
    1. Armature damping setup for different parts of body
    2. Test control range limits?
    3. Add inertia in the first part of the body

Important: To define the collision type, and use collision meshes, add the following config
<default class="collision">
<geom type="capsule" mass="0" density="0" condim="3" contype="1" conaffinity="1" group="3" />
</default>

We can then add collision meshes to parts as such:
<geom class="collision" type="capsule" fromto="0 0.005 -.17 -.02 0.005 -.01" size="0.05" rgba="0.0 0.0 0.0 1" />
<geom class="collision" type="capsule" fromto="0 0.005 -.17 .02 0.005 -.01" size="0.05" rgba="0.0 0.0 0.0 1" />
</body>
"""

import argparse
import logging
import xml.dom.minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Union

from kol.formats import mjcf

logger = logging.getLogger(__name__)

STOMPY_HEIGHT = 1.0

# Joints to be added to final xml, with their default values
# Comment out the joints that are not needed
DEFAULT_STANDING = {
    # "torso roll": 2.58,
    # # arms
    # "left shoulder pitch": -0.534,
    # "left shoulder yaw": 2.54,
    # "left shoulder roll": -0.0314,
    # "right shoulder pitch": 2.45,
    # "right shoulder yaw": 3.77,
    # "right shoulder roll": -0.0314,
    # "left elbow pitch": 2.35,
    # "right elbow pitch": 2.65,
    # # hands
    # "left hand left finger": 0.0,
    # "left hand right finger": 0.0,
    # "right hand left finger": 0.0,
    # "right hand right finger": 0.0,
    # "left wrist roll": 1.79,
    # "left wrist pitch": 1.35,
    # "left wrist yaw": 1.07,
    # "right wrist roll": -2.13,
    # "right wrist pitch": 1.79,
    # "right wrist yaw": -0.251,
    # legs
    "left hip pitch": -1.6,
    "left hip roll": 1.41,
    "left hip yaw": -2.12,
    "left knee pitch": 2.01,
    "left ankle pitch": 0.238,
    "left ankle roll": 1.85,
    "right hip pitch": 1.76,
    "right hip roll": -1.54,
    "right hip yaw": 0.967,
    "right knee pitch": 2.07,
    "right ankle pitch": 0.377,
    "right ankle roll": 1.92,
}

# Joints to be added to final xml, with their default limits
# Comment out the joints that are not needed
DEFAULT_LIMITS = {
    # "torso roll": {
    #     "lower": -4.36332,
    #     "upper": 4.36332,
    # },
    # "left shoulder pitch": {
    #     "lower": -0.0,
    #     "upper": 0.2,
    # },
    # "left shoulder yaw": {
    #     "lower": 0.97738438,
    #     "upper": 5.3058009,
    # },
    # "left shoulder roll": {
    #     "lower": -4.71239,
    #     "upper": 4.71239,
    # },
    # "right shoulder pitch": {
    #     "lower": -4.71239,
    #     "upper": 4.71239,
    # },
    # "right shoulder yaw": {
    #     "lower": 0.97738438,
    #     "upper": 5.3058009,
    # },
    # "right shoulder roll": {
    #     "lower": -4.71239,
    #     "upper": 4.71239,
    # },
    # "left wrist roll": {
    #     "lower": -4.71239,
    #     "upper": 4.71239,
    # },
    # "left wrist pitch": {
    #     "lower": -3.66519,
    #     "upper": -1.39626,
    # },
    # "left wrist yaw": {
    #     "lower": 0,
    #     "upper": 1.5708,
    # },
    # "right wrist roll": {
    #     "lower": -4.71239,
    #     "upper": 4.71239,
    # },
    # "right wrist pitch": {
    #     "lower": -1.5708,
    #     "upper": 0.523599,
    # },
    # "right wrist yaw": {
    #     "lower": -1.5708,
    #     "upper": 0,
    # },
    "left hip pitch": {
        "lower": 0.5,
        "upper": 2.69,
    },
    "left hip roll": {
        "lower": -0.5,
        "upper": 0.5,
    },
    "left hip yaw": {
        "lower": 0.5,
        "upper": 1.19,
    },
    "left knee pitch": {
        "lower": 0.5,
        "upper": 2.5,
    },
    "left ankle pitch": {
        "lower": -0.8,
        "upper": 0.6,
    },
    "left ankle roll": {
        "lower": 1,
        "upper": 5,
    },
    "right hip pitch": {
        "lower": -1,
        "upper": 1,
    },
    "right hip roll": {
        "lower": -2.39,
        "upper": -1,
    },
    "right hip yaw": {
        "lower": -2.2,
        "upper": -1,
    },
    "right knee pitch": {
        "lower": 1.54,
        "upper": 3.5,
    },
    "right ankle pitch": {
        "lower": 0,
        "upper": 1.5,
    },
    "right ankle roll": {
        "lower": 1,
        "upper": 2.3,
    },
    # "left elbow pitch": {
    #     "lower": 1.4486233,
    #     "higher": 5.4454273,
    # },
    # "right elbow pitch": {
    #     "lower": 1.4486233,
    #     "higher": 5.4454273,
    # },
    # "left hand left finger": {
    #     "lower": -0.051,
    #     "upper": 0.0,
    # },
    # "left hand right finger": {
    #     "lower": 0,
    #     "upper": 0.051,
    # },
    # "right hand left finger": {
    #     "lower": -0.051,
    #     "upper": 0.0,
    # },
    # "right hand right finger": {
    #     "lower": 0,
    #     "upper": 0.051,
    # },
}

# Links that will have collision with the floor
COLLISION_LINKS = [
    "right_foot_1_rubber_grip_3_simple",
    "left_foot_1_rubber_grip_1_simple",
]

# How high the root should be set above the ground
ROOT_HEIGHT = "0.72"


def _pretty_print_xml(xml_string: str) -> str:
    """Formats the provided XML string into a pretty-printed version."""
    parsed_xml = xml.dom.minidom.parseString(xml_string)
    pretty_xml = parsed_xml.toprettyxml(indent="  ")

    # Split the pretty-printed XML into lines and filter out empty lines
    lines = pretty_xml.split("\n")
    non_empty_lines = [line for line in lines if line.strip() != ""]
    return "\n".join(non_empty_lines)


class Sim2SimRobot(mjcf.Robot):
    """A class to adapt the world in a Mujoco XML file."""

    def adapt_world(
        self, add_floor: bool = True, add_reference_position: bool = False, remove_frc_range: bool = False
    ) -> None:
        root: ET.Element = self.tree.getroot()

        if add_floor:
            asset = root.find("asset")
            asset.append(
                ET.Element(
                    "texture",
                    name="texplane",
                    type="2d",
                    builtin="checker",
                    rgb1=".0 .0 .0",
                    rgb2=".8 .8 .8",
                    width="100",
                    height="108",
                )
            )
            asset.append(
                ET.Element(
                    "material",
                    name="matplane",
                    reflectance="0.",
                    texture="texplane",
                    texrepeat="1 1",
                    texuniform="true",
                )
            )
            asset.append(ET.Element("material", name="visualgeom", rgba="0.5 0.9 0.2 1"))

        compiler = root.find("compiler")
        if self.compiler is not None:
            compiler = self.compiler.to_xml(compiler)

        worldbody = root.find("worldbody")
        worldbody.insert(
            0,
            mjcf.Light(
                directional=True,
                diffuse=(0.4, 0.4, 0.4),
                specular=(0.1, 0.1, 0.1),
                pos=(0, 0, 5.0),
                dir=(0, 0, -1),
                castshadow=False,
            ).to_xml(),
        )
        worldbody.insert(
            0,
            mjcf.Light(
                directional=True, diffuse=(0.6, 0.6, 0.6), specular=(0.2, 0.2, 0.2), pos=(0, 0, 4), dir=(0, 0, -1)
            ).to_xml(),
        )
        if add_floor:
            worldbody.insert(
                0,
                mjcf.Geom(
                    name="ground",
                    type="plane",
                    size=(0, 0, 1),
                    pos=(0.001, 0, 0),
                    quat=(1, 0, 0, 0),
                    material="matplane",
                    condim=1,
                    conaffinity=15,
                ).to_xml(),
            )

        motors: List[mjcf.Motor] = []
        sensor_pos: List[mjcf.Actuatorpos] = []
        sensor_vel: List[mjcf.Actuatorvel] = []
        sensor_frc: List[mjcf.Actuatorfrc] = []
        # Create motors and sensors for the joints
        joints = list(root.findall("joint"))
        for joint in DEFAULT_LIMITS.keys():
            if joint in DEFAULT_STANDING.keys():
                motors.append(
                    mjcf.Motor(
                        name=joint,
                        joint=joint,
                        gear=1,
                        ctrlrange=(-200, 200),
                        ctrllimited=True,
                    )
                )
                sensor_pos.append(mjcf.Actuatorpos(name=joint + "_p", actuator=joint, user="13"))
                sensor_vel.append(mjcf.Actuatorvel(name=joint + "_v", actuator=joint, user="13"))
                sensor_frc.append(mjcf.Actuatorfrc(name=joint + "_f", actuator=joint, user="13", noise=0.001))

        root = self.add_joint_limits(root, fixed=False)

        # Add motors and sensors
        root.append(mjcf.Actuator(motors).to_xml())
        root.append(mjcf.Sensor(sensor_pos, sensor_vel, sensor_frc).to_xml())

        # TODO: Add additional sensors when necessary
        sensors = root.find("sensor")
        sensors.extend(
            [
                ET.Element("framequat", name="orientation", objtype="site", noise="0.001", objname="imu"),
                ET.Element("gyro", name="angular-velocity", site="imu", noise="0.005", cutoff="34.9"),
                #         # ET.Element("framepos", name="position", objtype="site", noise="0.001", objname="imu"),
                #         # ET.Element("velocimeter", name="linear-velocity", site="imu", noise="0.001", cutoff="30"),
                #         # ET.Element("accelerometer", name="linear-acceleration", site="imu", noise="0.005", cutoff="157"),
                #         # ET.Element("magnetometer", name="magnetometer", site="imu"),
            ]
        )

        root.insert(
            1,
            mjcf.Option(
                timestep=0.001,
                iterations=50,
                solver="PGS",
                gravity=(0, 0, -9.81),
            ).to_xml(),
        )

        visual_geom = ET.Element("default", {"class": "visualgeom"})
        geom_attributes = {"material": "visualgeom", "condim": "1", "contype": "0", "conaffinity": "0"}
        ET.SubElement(visual_geom, "geom", geom_attributes)

        root.insert(
            1,
            mjcf.Default(
                joint=mjcf.Joint(armature=0.01, damping=0.01, limited=True, frictionloss=0.01),
                motor=mjcf.Motor(ctrllimited=True),
                equality=mjcf.Equality(solref=(0.001, 2)),
                geom=mjcf.Geom(
                    solref=(0.001, 2),
                    friction=(0.9, 0.2, 0.2),
                    condim=4,
                    contype=1,
                    conaffinity=15,
                ),
                visual_geom=visual_geom,
            ).to_xml(),
        )

        # Locate actual root body inside of worldbody
        root_body = worldbody.find(".//body")
        # Make position and orientation of the root body
        root_body.set("pos", "0 0 " + ROOT_HEIGHT)
        root_body.set("quat", "1 0 0 0")

        # Add cameras and imu
        root_body.insert(0, ET.Element("camera", name="front", pos="0 -3 1", xyaxes="1 0 0 0 1 2", mode="trackcom"))
        root_body.insert(
            1,
            ET.Element(
                "camera",
                name="side",
                pos="-2.893 -1.330 0.757",
                xyaxes="0.405 -0.914 0.000 0.419 0.186 0.889",
                mode="trackcom",
            ),
        )
        root_body.insert(2, ET.Element("site", name="imu", size="0.01", pos="0 0 0"))

        # # Add joints to all the movement of the base
        # Define the joints as individual elements
        joints = [
            ET.Element("joint", name="root_x", type="slide", axis="1 0 0", limited="false"),
            ET.Element("joint", name="root_y", type="slide", axis="0 1 0", limited="false"),
            ET.Element("joint", name="root_z", type="slide", axis="0 0 1", limited="false"),
            ET.Element("joint", name="root_ball", type="ball", limited="false"),
        ]

        # Insert each joint at the front of the root element
        for joint in reversed(joints):
            root_body.insert(0, joint)

        # if add_reference_position:
        #     root = self.add_reference_position(root)

        # add visual geom logic
        for body in root.findall(".//body"):
            original_geoms = list(body.findall("geom"))
            for geom in original_geoms:
                geom.set("class", "visualgeom")
                # Create a new geom element
                new_geom = ET.Element("geom")
                new_geom.set("type", geom.get("type"))
                new_geom.set("rgba", geom.get("rgba"))
                new_geom.set("mesh", geom.get("mesh"))
                if geom.get("pos"):
                    new_geom.set("pos", geom.get("pos"))
                if geom.get("quat"):
                    new_geom.set("quat", geom.get("quat"))
                # Exclude collision meshes
                if geom.get("mesh") not in COLLISION_LINKS:
                    new_geom.set("contype", "0")
                    new_geom.set("conaffinity", "0")
                    new_geom.set("group", "1")
                    new_geom.set("density", "0")

                # Append the new geom to the body
                index = list(body).index(geom)
                body.insert(index + 1, new_geom)

        if remove_frc_range:
            for body in root.findall(".//body"):
                joints = list(body.findall("joint"))
                for join in joints:
                    if "actuatorfrcrange" in join.attrib:
                        join.attrib.pop("actuatorfrcrange")

    def add_reference_position(self, root: ET.Element) -> None:
        # Find all 'joint' elements
        joints = root.findall(".//joint")

        default_standing = DEFAULT_STANDING
        for joint in joints:
            if joint.get("name") in default_standing.keys():
                joint.set("ref", str(default_standing[joint.get("name")]))

        return root

    def add_joint_limits(self, root: ET.Element, fixed: bool = False) -> None:
        joint_limits = DEFAULT_LIMITS

        for joint in root.findall(".//joint"):
            joint_name = joint.get("name")
            if joint_name in joint_limits:
                limits = joint_limits.get(joint_name)
                lower = str(limits.get("lower", 0.0))
                upper = str(limits.get("upper", 0.0))
                joint.set("range", f"{lower} {upper}")

        return root

    def save(self, path: Union[str, Path]) -> None:
        rough_string = ET.tostring(self.tree.getroot(), "utf-8")
        # Pretty print the XML
        formatted_xml = _pretty_print_xml(rough_string)
        logger.info("XML:\n%s", formatted_xml)
        with open(path, "w") as f:
            f.write(formatted_xml)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a MJCF file for the Stompy robot.")
    parser.add_argument("filepath", type=str, help="The path to load and save the MJCF file.")
    args = parser.parse_args()
    # Robot name is whatever string comes right before ".urdf" extension
    path = Path(args.filepath)
    robot_name = path.stem
    path = path.parent
    robot = Sim2SimRobot(
        robot_name,
        path,
        mjcf.Compiler(angle="radian", meshdir="meshes", autolimits=True, eulerseq="zyx"),
    )
    robot.adapt_world(add_reference_position=True)
    robot.save(path / f"{robot_name}_updated.xml")
