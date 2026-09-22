"""Build PyFK and its Cython extensions."""
import os
import sys
import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, find_packages, setup

windows = sys.platform == "win32"
mpi = os.getenv("PYFK_USE_MPI", "0") == "1"
include_dirs = [np.get_include()]
link_args = []
if mpi:
    if windows:
        raise RuntimeError("MPI builds require a Unix MPI compiler; use the CPU or CUDA build on Windows")
    import mpi4py
    os.environ.setdefault("CC", "mpicc")
    include_dirs.append(mpi4py.get_include())
    link_args.append("-lmpi")
compile_args = []
trace = int(os.getenv("PYFK_USE_CYTHON_TRACE", "0") == "1")
extensions = [
    Extension("pyfk.taup.taup", ["pyfk/taup/taup.pyx"], include_dirs=[np.get_include()],
              define_macros=[("CYTHON_TRACE", str(trace))]),
    Extension("pyfk.gf.waveform_integration", ["pyfk/gf/waveform_integration.pyx"],
              include_dirs=include_dirs, language="c++" if windows else "c",
              extra_compile_args=compile_args, extra_link_args=link_args),
]
setup(packages=find_packages(include=["pyfk", "pyfk.*"]),
      ext_modules=cythonize(extensions, language_level=3, compile_time_env={
          "PYFK_USE_MPI": "1" if mpi else "0", "PYFK_WINDOWS": windows}),
      include_package_data=True, zip_safe=False)
