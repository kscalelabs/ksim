# mypy: disable-error-code="import-untyped"
#!/usr/bin/env python
"""Setup script for the project."""

import re

from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description: str = f.read()


with open("ksim/requirements.txt", "r", encoding="utf-8") as f:
    requirements: list[str] = f.read().splitlines()


with open("ksim/requirements-dev.txt", "r", encoding="utf-8") as f:
    requirements_dev: list[str] = f.read().splitlines()


with open("ksim/__init__.py", "r", encoding="utf-8") as fh:
    version_re = re.search(r"^__version__ = \"([^\"]*)\"", fh.read(), re.MULTILINE)
assert version_re is not None, "Could not find version in ksim/__init__.py"
version: str = version_re.group(1)


setup(
    name="ksim",
    version=version,
    description="A modular and easy-to-use framework for training policies in simulation.",
    author="K-Scale Labs",
    url="https://github.com/kscalelabs/ksim",
    long_description=long_description,
    long_description_content_type="text/markdown",
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={"dev": requirements_dev},
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "ksim-generate-reference=ksim.utils.priors:main",
            "ksim-visualize-reference=ksim.utils.priors:vis_entry_point",
        ],
    },
)
