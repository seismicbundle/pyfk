PyFK, SeismicBundle version
===========================

.. placeholder-for-doc-index

This is a modified version of `PyFK <https://github.com/ziyixi/pyfk>`__ by Ziyi Xi,
the Python port of Lupei Zhu's `FK <http://www.eas.slu.edu/People/LZhu/home.html>`__
code for Green's functions and synthetic seismograms in layered 1D models.
The upstream API is unchanged; ``Config`` has two new keyword arguments.
The changes:

* CPU integration: with ``pmin=0`` the Bessel functions are tabulated once
  instead of being evaluated for every frequency, and the frequency loop runs
  on OpenMP threads. With the default options the results are byte-identical
  to upstream.
* GPU integration: the wavenumber integration and the Bessel projection stay
  on the GPU until the spectrum is complete.
* Optional frequency ceiling: ``Config(fmax=10.)`` integrates up to 10 Hz only.
* Optional mixed precision on the GPU (experimental).
* Windows support (MSVC), Python 3.10 to 3.13, a setuptools build, and no
  ``cysignals`` dependency.

Speed
-----

Time of one ``calculate_gf`` call on the CPU (AMD Ryzen 9 7900, 12 cores),
without a frequency ceiling: 9 layers over a halfspace, source depth 9 km,
23 receivers, ``dt=0.01``. The parameters are in ``benchmarks/timing.py``.

========================  ====================  ======================
\                         8192 points, 2-30 km  16384 points, 5-130 km
========================  ====================  ======================
upstream PyFK, 1 process  144 s                 1083 s
this version, 1 thread    47 s                  439 s
this version, 12 threads  12 s                  109 s
========================  ====================  ======================

Installation
------------

::

    git clone https://github.com/seismicbundle/pyfk.git
    cd pyfk
    python -m pip install .           # CPU
    python -m pip install ".[cuda]"   # with GPU support

A C compiler is required. On macOS, install ``libomp`` for multithreading.
MPI, conda and tests: ``docs/introduction/install.rst``.

New options
-----------

.. code-block:: python

    import numpy as np
    from pyfk import Config, SeisModel, SourceModel, calculate_gf

    # thickness (km), vs, vp (km/s), density (g/cm^3), Qs, Qp
    model = np.array([[10., 3.4, 5.8, 2.6, 600., 1200.],
                      [0., 3.8, 6.5, 2.8, 600., 1200.]])
    config = Config(model=SeisModel(model=model), source=SourceModel(sdep=8.),
                    receiver_distance=[10., 20.], npt=8192, dt=0.01,
                    fmax=10.)  # frequency ceiling in Hz
    gf = calculate_gf(config)

    # GPU, optionally with mixed precision
    config = Config(model=SeisModel(model=model), source=SourceModel(sdep=8.),
                    receiver_distance=[10., 20.], npt=8192, dt=0.01,
                    cuda=True, mixed_precision=True)
    gf = calculate_gf(config)

``fmax``
    Highest frequency integrated, in Hz. The sampling and the trace headers do
    not change. ``0`` means no ceiling; ``PYFK_FMAX_HZ`` sets the default.

``mixed_precision``
    Run the layer propagation on the GPU in single precision for the
    frequencies where an overflow estimate allows it; the others stay in double
    precision. ``PYFK_MIXED_FP32=1`` sets the default.

``OMP_NUM_THREADS``
    Number of CPU threads. The default is the number of processors.

Limitations
-----------

* The frequency ceiling cuts the integrated spectrum without a taper. It
  changes the waveform, including the onset, and can flip a first motion near
  a nodal plane. The output is not band-limited: the exp(σt) correction after
  the inverse FFT (``suppression_sigma``) spreads the change above and below
  ``fmax``, most near the end of the trace.
* Mixed precision is experimental. Its error is checked on the test cases
  only.
* The GPU path requires the versions pinned in the ``cuda`` extra, because it
  sets private Numba compiler options.
* Threads are used in a single process with ``pmin=0``. With MPI or a nonzero
  ``pmin``, each process runs on one thread.

Tested platforms and the comparison with upstream: ``docs/validation.rst``.

Citation
--------

Please cite the original PyFK:

Xi, Z., Li, J., Chen, M. and Wei, S., 2021, December. PyFK: A Fast MPI and CUDA
Accelerated Python Package for Calculating Synthetic Seismograms Based on the
Frequency-wavenumber Method. In AGU Fall Meeting Abstracts (Vol. 2021,
pp. S15E-0288).

If you used this version, please also give the repository URL,
https://github.com/seismicbundle/pyfk.

License
-------

Apache License 2.0 (``LICENSE``). See ``NOTICE``.
