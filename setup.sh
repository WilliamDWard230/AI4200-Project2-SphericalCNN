#!/usr/bin/env bash
# Set up the Spherical CNN project on Linux (tested workflow: Python 3.13 on Warp1).
# Usage: bash setup.sh
set -Eeuo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
PROJECT_DIR="$PWD"

echo "[setup] Project: $PROJECT_DIR"

# Use the currently active virtual environment, or create a project-local one.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    PYTHON="$VIRTUAL_ENV/bin/python"
    echo "[setup] Using active virtual environment: $VIRTUAL_ENV"
else
    if [[ ! -x .venv/bin/python ]]; then
        echo "[setup] Creating .venv (no environment currently active)"
        python3 -m venv .venv
    fi
    PYTHON="$PROJECT_DIR/.venv/bin/python"
    echo "[setup] Using $PYTHON"
fi

"$PYTHON" -m pip --version >/dev/null || {
    echo "ERROR: pip is missing in this environment. Install pip or create a venv with ensurepip." >&2
    exit 1
}

"$PYTHON" - <<'PY'
import sys
print('[setup] Python:', sys.version.split()[0])
if sys.version_info < (3, 10):
    raise SystemExit('Python 3.10 or newer is required by this setup script.')
PY

# Build tools and scientific dependencies required by lie_learn and the model.
"$PYTHON" -m pip install -U 'setuptools' 'wheel' 'Cython' 'numpy' 'scipy' 'requests'

# train.py requires PyTorch; gendata.py also requires torchvision.
if ! "$PYTHON" -c 'import torch, torchvision' >/dev/null 2>&1; then
    echo "[setup] Installing PyTorch and torchvision (this may be a large download)."
    echo "[setup] For a particular HPC CUDA/PyTorch build, install that build first and rerun setup."
    "$PYTHON" -m pip install torch torchvision
fi

# Avoid shadowing the compiled package with an old vendored package.
if [[ -d lie_learn ]]; then
    echo "ERROR: A top-level lie_learn/ directory exists in the project." >&2
    echo "Rename it (e.g. mv lie_learn lie_learn_backup) before running setup." >&2
    exit 1
fi

# Build upstream lie_learn from source to obtain its Cython extension.
# Clone outside this Git repository; no nested Git repository is retained.
BUILD_DIR="$(mktemp -d "${TMPDIR:-/tmp}/sphericalcnn-lie-learn.XXXXXXXX")"
trap 'rm -rf "$BUILD_DIR"' EXIT

echo "[setup] Cloning lie_learn source"
git clone --depth 1 https://github.com/AMLab-Amsterdam/lie_learn.git "$BUILD_DIR/lie_learn"
echo "[setup] lie_learn revision: $(git -C "$BUILD_DIR/lie_learn" rev-parse HEAD)"

# The older setup.py build can fail with pip's build isolation enabled.
"$PYTHON" -m pip install --no-build-isolation --no-cache-dir "$BUILD_DIR/lie_learn"

# Some lie_learn source wheels omit these required precomputed Wigner matrices.
SOURCE_MATRICES="$BUILD_DIR/lie_learn/lie_learn/representations/SO3/pinchon_hoggan"
"$PYTHON" - "$SOURCE_MATRICES" <<'PY'
import pathlib
import shutil
import sys

src = pathlib.Path(sys.argv[1])
import lie_learn
pkg = pathlib.Path(lie_learn.__file__).resolve().parent
dest = pkg / 'representations' / 'SO3' / 'pinchon_hoggan'
dest.mkdir(parents=True, exist_ok=True)
for filename in ('J_dense_0-150.npy', 'J_block_0-150.npy'):
    source = src / filename
    if not source.is_file():
        raise SystemExit(f'Missing upstream data file: {source}')
    shutil.copy2(source, dest / filename)
    print(f'[setup] Installed {filename}')
PY

mkdir -p cache/setup_s2_fft cache/setup_so3_fft logs

# Confirm all critical imports; Python 3.13 compatibility in model.py and
# CPU FFT fix in s2cnn/soft/so3_fft.py are source-code changes committed separately.
"$PYTHON" - <<'PY'
import torch
from lie_learn.representations.SO3.irrep_bases import change_of_basis_matrix
from lie_learn.representations.SO3.wigner_d import wigner_d_matrix
from model import S2ConvNet_original
print('[setup] Imports successful; PyTorch:', torch.__version__)
print('[setup] Spherical CNN model imports successfully')
PY

if [[ ! -f s2_mnist.gz ]]; then
    echo '[setup] WARNING: s2_mnist.gz is missing. Generate or supply it before running train.py.'
fi

echo '[setup] Complete.'
echo "[setup] Activate: source ${VIRTUAL_ENV:-$PROJECT_DIR/.venv}/bin/activate"
echo '[setup] Then run: python -u train.py (or submit run.sh from this directory)'
