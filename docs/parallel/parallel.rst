.. _parallel:

Parallelization
===============

Only the Green's function calculation is parallelized. The synthetics are a
convolution with the source time function and take little time.

OpenMP threads
--------------

With ``pmin=0`` in a single process, the CPU integration runs on OpenMP
threads. ``OMP_NUM_THREADS`` sets their number; the default is the number of
processors. OpenMP is available with MSVC and GCC, and with Apple clang when
libomp is installed. The results do not depend on the number of threads. With
MPI or a nonzero ``pmin``, each process runs on one thread.

MPI
---

Build with ``PYFK_USE_MPI=1`` (see :ref:`install`). A script that calls
``calculate_gf`` then splits the frequency loop across the processes when it
is started with ``mpirun``, without changes to the script::

    mpirun -np 4 python script.py

Every process receives the full result. See the
`mpi4py documentation <https://mpi4py.readthedocs.io/en/stable/>`__ for
restricting further processing to one rank.

GPU
---

``Config(cuda=True)`` or ``PYFK_USE_CUDA=1`` runs the integration on an NVIDIA
GPU. It requires the ``cuda`` extra (see :ref:`install`). Numba compiles the
kernels on the first call in a process, so that call is slower.

With ``pmin=0``, the wavenumber integration and the Bessel projection stay on
the GPU and only the final spectrum is copied back to the host. With a nonzero
``pmin``, the projection runs on the host as in upstream PyFK.

If the GPU runs out of memory, set ``CUDA_DIVIDE_NUM`` to split the
calculation into that many parts (default 1). ``CUDA_VISIBLE_DEVICES`` selects
the GPU.

``Config(mixed_precision=True)`` or ``PYFK_MIXED_FP32=1`` runs the layer
propagation in single precision for the frequencies where an overflow estimate
allows it. It is experimental and off by default.

Frequency ceiling
-----------------

``Config(fmax=...)`` or ``PYFK_FMAX_HZ`` stops the frequency loop at ``fmax``
Hz, on the CPU and the GPU. The number of wavenumbers grows with the
frequency, so the time saved is larger than the fraction of frequencies
removed.

The ceiling cuts the integrated spectrum without a taper. It changes the
waveform, including the onset, and can flip a first motion near a nodal plane.
The output is not band-limited: the exp(σt) correction after the inverse FFT
(``suppression_sigma``) spreads the change above and below ``fmax``, most near
the end of the trace. ``fmax=0`` means no ceiling.
