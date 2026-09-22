#!/bin/bash

if [ "$(uname)" = "Darwin" ]; then
    export PYFK_LIBOMP_PREFIX="${PREFIX}"
fi
"${PYTHON}" -m pip install . --no-deps --no-build-isolation
