.. _install:

Installing
==========

Python 3.10 to 3.13::

    git clone https://github.com/seismicbundle/pyfk.git
    cd pyfk
    python -m pip install .

A C compiler is required: Visual Studio Build Tools with the C++ workload on
Windows, GCC on Linux, the Xcode command line tools on macOS. On macOS, install
libomp (``brew install libomp``) for multithreading; ``PYFK_LIBOMP_PREFIX``
sets a non-default location.

GPU
---

::

    python -m pip install ".[cuda]"

This installs Numba 0.67.0, numba-cuda 0.30.4 and CuPy 14.2.0, the versions
the GPU path is tested with. Other versions are not supported. An NVIDIA driver
with CUDA 12 support is required.

MPI
---

On Linux or macOS, with an MPI compiler such as Open MPI::

    PYFK_USE_MPI=1 python -m pip install --no-cache-dir ".[mpi]"

``--no-cache-dir`` keeps pip from reusing a wheel built without MPI.

Conda
-----

To build a conda package from the recipe in ``conda/``::

    conda build conda --python 3.12 --override-channels -c conda-forge

Tests
-----

Run the tests from the directory that contains the checkout, so that the
installed package is imported instead of the source tree::

    python -m pip install pytest pytest-mpi
    python -m pytest --pyargs pyfk
    python -m pytest pyfk/tests --import-mode=importlib

The first command runs the upstream tests, the second the comparison with
upstream and the tests of the new options. GPU and MPI tests::

    PYFK_TEST_CUDA=1 python -m pytest pyfk/tests --import-mode=importlib -m cuda
    mpirun -np 3 python -m pytest --with-mpi --pyargs pyfk
