Validation
==========

With the default options, the results are byte-identical to upstream PyFK
(commit ``810e77d``) for the 13 cases in ``validation/upstream_cases.py``:
Green's functions, SAC timing headers and synthetics. The tests also check
that the results do not depend on the number of threads, that ``fmax`` gives
the full calculation with the integrated spectrum cut at ``fmax``, and that
the GPU agrees with the CPU within 1e-8 of the peak amplitude (1e-3 with mixed
precision).

By default the tests compare with ``tests/data/upstream-reference.npz``, built
from upstream on Windows. Upstream itself gives results on macOS that differ
from it by up to about 1e-8 of the peak amplitude, so this comparison allows
5e-8. For an exact comparison, build upstream on the same machine and set
``PYFK_TEST_UPSTREAM_REFERENCE`` to its output::

    python validation/prepare_upstream.py UPSTREAM_CHECKOUT UPSTREAM_SOURCE
    # install UPSTREAM_SOURCE in a separate environment, then:
    python validation/upstream_cases.py native-upstream.npz

``prepare_upstream.py`` applies the three changes needed to build upstream
with Cython 3 and MSVC: ``cpow=True``, C++ complex-number declarations, and
``temppp[kmat, jmat]`` in place of the equivalent ``temppp.T[jmat, kmat]``,
which Cython 3 rejects for the up and down wave selections.

Tested platforms
----------------

The GitHub Actions workflow runs the upstream tests and the comparison with
upstream built on the same runner:

.. list-table::
    :header-rows: 1

    * - Platform
      - Python
    * - Ubuntu 24.04 x86_64, GCC
      - 3.10, 3.12, 3.13
    * - macOS 26 ARM64, Apple clang, libomp
      - 3.10, 3.12, 3.13
    * - Windows Server 2025 x64, MSVC
      - 3.10, 3.12, 3.13
    * - Ubuntu 24.04, Open MPI 4.1.6, 3 processes
      - 3.12
    * - conda-build, Ubuntu 24.04 and macOS 26
      - 3.12

The GPU tests run outside CI, on Windows 11 with an RTX 5090 and Python 3.12.
Not tested: Intel macOS, Linux ARM, CUDA on Linux, conda packages on Windows.
``validation/unix.sh``, ``mpi.sh`` and ``conda.sh`` run the same tests on
Linux and macOS from a built sdist.
