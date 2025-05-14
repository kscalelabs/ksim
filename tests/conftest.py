"""Defines PyTest configuration for the project."""

import random
from pathlib import Path

import mujoco
import pytest
from _pytest.fixtures import SubRequest
from _pytest.python import Function
from mujoco import mjx

import ksim


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    random.seed(1337)


def get_mujoco_humanoid_model() -> mujoco.MjModel:
    mjcf_path = (Path(__file__).parent / "assets" / "humanoid.mjcf").resolve().as_posix()
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    return mj_model


def get_mjx_humanoid_model() -> mjx.Model:
    """Get a dummy mjx.Model for testing."""
    mjcf_path = (Path(__file__).parent / "assets" / "humanoid.mjcf").resolve().as_posix()
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    mjx_model = mjx.put_model(mj_model)
    return mjx_model


@pytest.fixture(params=["mjx", "mujoco"])
def humanoid_model(request: SubRequest) -> ksim.PhysicsModel:
    """Get a humanoid model."""
    return get_mujoco_humanoid_model() if request.param == "mujoco" else get_mjx_humanoid_model()


def pytest_collection_modifyitems(items: list[Function]) -> None:
    items.sort(key=lambda x: x.get_closest_marker("slow") is not None)
