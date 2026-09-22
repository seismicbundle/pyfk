"""Validation shared by configuration and integration backends."""
import os
from pyfk.utils.error_message import PyfkError


def cuda_chunk_count():
    try:
        count = int(os.getenv("CUDA_DIVIDE_NUM", "1"))
    except ValueError as exc:
        raise PyfkError("CUDA_DIVIDE_NUM must be a positive integer") from exc
    if count < 1:
        raise PyfkError("CUDA_DIVIDE_NUM must be a positive integer")
    return count
