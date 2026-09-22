#!/usr/bin/env bash
# Install an sdist with Python 3.10, 3.12 and 3.13 and run the tests outside the
# source tree, then build and test a wheel. Also builds unoptimized upstream PyFK
# as the reference for the strict comparison.
# Usage: validation/unix.sh WORK_DIR SDIST
set -eu
ROOT=$(cd "$1" && pwd)
SDIST=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYFK_USE_CUDA=0
unset PYFK_USE_MPI PYFK_FMAX_HZ PYFK_MIXED_FP32 PYFK_TEST_CUDA
mkdir -p "$ROOT/source" "$ROOT/logs" "$ROOT/run" "$ROOT/wheels"
tar -xzf "$SDIST" -C "$ROOT/source" --strip-components=1
SOURCE="$ROOT/source"
STATUS="$ROOT/status.tsv"
printf 'check\texit_code\n' > "$STATUS"
record() {
    label=$1
    shift
    set +e
    "$@" > "$ROOT/logs/$label.log" 2>&1
    result=$?
    set -e
    printf '%s\t%s\n' "$label" "$result" | tee -a "$STATUS"
    tail -n 8 "$ROOT/logs/$label.log"
    return "$result"
}
cd "$ROOT/run"
record reference-env uv venv --python 3.12 "$ROOT/env-reference"
if [ ! -d "$ROOT/upstream-git/.git" ]; then
    record reference-clone git clone --no-checkout https://github.com/ziyixi/pyfk.git "$ROOT/upstream-git"
fi
if [ ! -d "$ROOT/upstream-reference" ]; then
    record reference-prepare "$ROOT/env-reference/bin/python" "$SOURCE/validation/prepare_upstream.py" "$ROOT/upstream-git" "$ROOT/upstream-reference"
fi
record reference-install uv pip install --python "$ROOT/env-reference/bin/python" "$ROOT/upstream-reference"
record reference-generate "$ROOT/env-reference/bin/python" "$SOURCE/validation/upstream_cases.py" "$ROOT/native-upstream.npz"
export PYFK_TEST_UPSTREAM_REFERENCE="$ROOT/native-upstream.npz"
for version in 3.10 3.12 3.13; do
    ENV_DIR="$ROOT/env-$version"
    if record "env-$version" uv venv --python "$version" "$ENV_DIR" &&
       record "install-$version" uv pip install --python "$ENV_DIR/bin/python" "$SDIST" pytest pytest-mpi; then
        record "upstream-$version" "$ENV_DIR/bin/python" -m pytest --pyargs pyfk -q || true
        record "release-$version" "$ENV_DIR/bin/python" -m pytest "$SOURCE/tests" --import-mode=importlib -q || true
        "$ENV_DIR/bin/python" -m pip --version >/dev/null 2>&1 || true
        "$ENV_DIR/bin/python" -c 'import platform, numpy, scipy, obspy, pyfk; print(platform.platform()); print(platform.python_version(), numpy.__version__, scipy.__version__, obspy.__version__); print(pyfk.__file__)' > "$ROOT/logs/versions-$version.log"
    fi
done
if record wheel-build uv build --wheel --python "$ROOT/env-3.12/bin/python" --out-dir "$ROOT/wheels" "$SOURCE" &&
   record wheel-env uv venv --python 3.12 "$ROOT/env-wheel" &&
   record wheel-install uv pip install --python "$ROOT/env-wheel/bin/python" "$ROOT"/wheels/*.whl pytest pytest-mpi; then
    record wheel-upstream "$ROOT/env-wheel/bin/python" -m pytest --pyargs pyfk -q || true
    record wheel-release "$ROOT/env-wheel/bin/python" -m pytest "$SOURCE/tests" --import-mode=importlib -q || true
fi
cat "$STATUS"
awk -F '\t' 'NR > 1 && $2 != 0 {failed=1} END {exit failed}' "$STATUS"
