#!/usr/bin/env bash
# Run the upstream and regression tests on three MPI ranks.
# Usage: validation/mpi.sh WORK_DIR SDIST, after validation/unix.sh with the same WORK_DIR.
set -eu
ROOT=$(cd "$1" && pwd)
SDIST=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYFK_USE_CUDA=0 PYFK_USE_MPI=1
export PYFK_TEST_UPSTREAM_REFERENCE="$ROOT/native-upstream.npz"
if [ "$(id -u)" = 0 ]; then
    # Open MPI refuses to run as root otherwise (for example in containers).
    export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
fi
uv venv --python 3.12 "$ROOT/env-mpi"
uv pip install --no-cache --python "$ROOT/env-mpi/bin/python" "$SDIST" mpi4py pytest pytest-mpi > "$ROOT/logs/mpi-install.log" 2>&1
cd "$ROOT/run"
mpirun -np 3 "$ROOT/env-mpi/bin/python" -m pytest --with-mpi --pyargs pyfk -q > "$ROOT/logs/mpi-upstream.log" 2>&1
mpirun -np 3 "$ROOT/env-mpi/bin/python" -m pytest "$ROOT/source/tests/test_regression.py" --import-mode=importlib --with-mpi -m mpi -q > "$ROOT/logs/mpi-regression.log" 2>&1
tail -8 "$ROOT/logs/mpi-upstream.log"
cat "$ROOT/logs/mpi-regression.log"
