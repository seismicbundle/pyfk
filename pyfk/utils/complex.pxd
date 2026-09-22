IF PYFK_WINDOWS:
    # MSVC's C complex ABI differs from Cython's. In C++ mode Cython uses
    # std::complex<double>, so the standard overloads preserve the same math.
    cdef extern from "<complex>" nogil:
        double complex clog "std::log"(double complex z)
        double complex csqrt "std::sqrt"(double complex z)
        double cimag "std::imag"(double complex z)
        double creal "std::real"(double complex z)
        double complex conj "std::conj"(double complex z)
ELSE:
    cdef extern from "<complex.h>" nogil:
        double complex clog(double complex z)
        double complex csqrt(double complex z)
        double cimag(double complex z)
        double creal(double complex z)
        double complex conj(double complex z)
