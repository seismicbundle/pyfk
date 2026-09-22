"""Validation shared by configuration and integration backends."""
import math
import os
from numbers import Real
from pyfk.utils.error_message import PyfkError


def resolve_fmax(value):
    """Frequency ceiling in Hz. None reads PYFK_FMAX_HZ; 0 means no ceiling."""
    if value is None:
        raw = os.getenv("PYFK_FMAX_HZ", "").strip()
        if not raw:
            return 0.0
        try:
            value = float(raw)
        except ValueError as exc:
            raise PyfkError("PYFK_FMAX_HZ must be a finite frequency in Hz") from exc
        if math.isfinite(value) and value < 0:
            return 0.0
    if isinstance(value, bool) or not isinstance(value, Real):
        raise PyfkError("fmax must be a finite non-negative frequency in Hz")
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise PyfkError("fmax must be a finite non-negative frequency in Hz")
    return value


def frequency_bin_ceiling(nfft2, dw, fmax=None):
    """Exclusive upper frequency bin for the ceiling (nfft2 when there is none)."""
    fmax = resolve_fmax(fmax)
    df = dw / (2 * math.pi)
    if fmax == 0 or fmax >= (nfft2 - 1) * df:
        return nfft2
    last = int(fmax / df)
    if (last + 1) * df <= fmax:
        last += 1
    elif last * df > fmax:
        last -= 1
    return min(nfft2, last + 1)


def resolve_mixed_precision(value):
    if value is None:
        return os.getenv("PYFK_MIXED_FP32", "0") == "1"
    if not isinstance(value, bool):
        raise PyfkError("mixed_precision must be True, False, or None")
    return value


def cuda_chunk_count():
    try:
        count = int(os.getenv("CUDA_DIVIDE_NUM", "1"))
    except ValueError as exc:
        raise PyfkError("CUDA_DIVIDE_NUM must be a positive integer") from exc
    if count < 1:
        raise PyfkError("CUDA_DIVIDE_NUM must be a positive integer")
    return count
