"""Defines PyTest configuration for the project."""

import random
from pathlib import Path

import mujoco
import pytest
from _pytest.python import Function


@pytest.fixture(autouse=True)
def set_random_seed() -> None:
    random.seed(1337)


@pytest.fixture()
def humanoid_model() -> mujoco.MjModel:
    mjcf_path = (Path(__file__).parent / "fixed_assets" / "default_humanoid_test.mjcf").resolve().as_posix()
    mj_model = mujoco.MjModel.from_xml_path(mjcf_path)
    return mj_model


def pytest_collection_modifyitems(items: list[Function]) -> None:
    items.sort(key=lambda x: x.get_closest_marker("slow") is not None)
