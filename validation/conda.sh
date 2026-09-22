#!/usr/bin/env bash
# Build and test the conda recipe with a local micromamba. Nothing is installed
# outside WORK_DIR and nothing is uploaded.
# Usage: validation/conda.sh WORK_DIR EXTRACTED_SDIST_DIR
set -eu
ROOT=$(cd "$1" && pwd)
SOURCE=$(cd "$2" && pwd)
case "$(uname -s)-$(uname -m)" in
    Darwin-arm64) platform=osx-arm64 ;;
    Darwin-x86_64) platform=osx-64 ;;
    Linux-x86_64) platform=linux-64 ;;
    *) echo 'Unsupported bootstrap platform' >&2; exit 2 ;;
esac
mkdir -p "$ROOT/conda-tools" "$ROOT/logs"
export MAMBA_ROOT_PREFIX="$ROOT/conda-tools/mamba-root"
export CONDA_BLD_PATH="$ROOT/conda-bld" CONDA_SOLVER=libmamba
# Use a fresh package cache: conda-build 26.7.1 fails to re-test cached packages
# whose version contains '+'.
export CONDA_PKGS_DIRS
CONDA_PKGS_DIRS=$(mktemp -d "$ROOT/conda-tools/package-cache-XXXXXX")
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYFK_USE_CUDA=0
unset PYFK_USE_MPI PYFK_FMAX_HZ PYFK_MIXED_FP32 PYFK_TEST_CUDA
if [ ! -x "$ROOT/conda-tools/bin/micromamba" ]; then
    curl -fLsS "https://micro.mamba.pm/api/micromamba/$platform/latest" -o "$ROOT/conda-tools/micromamba.tar.bz2"
    tar -xjf "$ROOT/conda-tools/micromamba.tar.bz2" -C "$ROOT/conda-tools" bin/micromamba
fi
MAMBA="$ROOT/conda-tools/bin/micromamba"
if [ ! -x "$ROOT/conda-tools/build-env/bin/conda" ]; then
    "$MAMBA" create -y -p "$ROOT/conda-tools/build-env" -c conda-forge --override-channels python=3.12 conda-build conda-libmamba-solver > "$ROOT/logs/conda-bootstrap.log" 2>&1
fi
"$MAMBA" run -p "$ROOT/conda-tools/build-env" conda build "$SOURCE/conda" --python 3.12 --override-channels -c conda-forge --no-anaconda-upload > "$ROOT/logs/conda-build.log" 2>&1
tail -25 "$ROOT/logs/conda-build.log"
