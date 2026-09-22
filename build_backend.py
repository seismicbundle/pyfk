"""Request MPI build dependencies only for an upstream-style MPI build."""
import os
from setuptools.build_meta import *  # noqa: F403


def _requirements(config_settings=None):
    return ["mpi4py>=3.1"] if os.getenv("PYFK_USE_MPI", "0") == "1" else []


get_requires_for_build_wheel = _requirements
get_requires_for_build_editable = _requirements
get_requires_for_build_sdist = _requirements
