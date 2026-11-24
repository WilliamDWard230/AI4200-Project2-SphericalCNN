"""
Packaging for the Spherical CNN project.

Metadata and dependencies are in pyproject.toml. This file exists because
lie_learn can't be installed as an ordinary pip dependency. During the wheel
build it reproduces what setup.sh did, then bundles the result into this
project's wheel (so `import lie_learn` works after `pip install .`):

  1. git clone upstream lie_learn
  2. compile its Cython extensions (python setup.py build_ext --inplace), using
     the build requirements from pyproject.toml (numpy, Cython, ...)
  3. copy the compiled lie_learn/ package, including the precomputed Wigner J
     matrices, into this wheel
  4. verify the imports that model.py needs

(Earlier versions called `pip install` from inside the build. That can't work:
pip hides the real environment, including pip itself, from isolated builds.)

Requires git, a C compiler, and network access at install time.
Install with:  pip install .        (editable installs are not supported)

Everything the lie_learn step does is also written to a log file
(~/sphericalcnn-setup.log, or $SPHERICALCNN_LOG). Set SKIP_LIE_LEARN_BUILD=1
to skip the lie_learn step entirely.
"""
import datetime
import os
import shutil
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py
from setuptools.dist import Distribution

HERE = Path(__file__).resolve().parent

# Override with LIE_LEARN_REPO=<url or file:// path>
LIE_LEARN_REPO = os.environ.get(
    "LIE_LEARN_REPO", "https://github.com/AMLab-Amsterdam/lie_learn.git"
)
J_FILES = ("J_dense_0-150.npy", "J_block_0-150.npy")
PH_SUBDIR = Path("representations") / "SO3" / "pinchon_hoggan"

# Run in a subprocess: same imports as the "confirm critical imports" step of
# setup.sh, plus a check that the J matrices are really there.
_CHECK = f"""
import pathlib, sys
import lie_learn
from lie_learn.representations.SO3.irrep_bases import change_of_basis_matrix
from lie_learn.representations.SO3.wigner_d import wigner_d_matrix
d = pathlib.Path(lie_learn.__file__).resolve().parent / "representations" / "SO3" / "pinchon_hoggan"
missing = [n for n in {J_FILES!r} if not (d / n).is_file()]
sys.exit("missing data files: " + ", ".join(missing) if missing else 0)
"""


def _log_path():
    override = os.environ.get("SPHERICALCNN_LOG")
    if override:
        return Path(override)
    try:
        return Path.home() / "sphericalcnn-setup.log"
    except Exception:
        return Path(tempfile.gettempdir()) / "sphericalcnn-setup.log"


LOG_PATH = _log_path()


def _log(line):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except OSError:
        pass
    print(line, flush=True)


def _run(cmd, cwd=None, env=None, check=True):
    """Run cmd, streaming its output to stdout and the log file."""
    _log("[setup] $ " + " ".join(str(c) for c in cmd))
    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env, text=True, errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
    except OSError as e:
        raise RuntimeError(f"Could not run {cmd[0]!r}: {e}  (log: {LOG_PATH})") from e
    lines = []
    for line in proc.stdout:
        lines.append(line)
        _log(line.rstrip("\n"))
    rc = proc.wait()
    if check and rc != 0:
        raise RuntimeError(
            f"Command failed (exit {rc}): {' '.join(str(c) for c in cmd)}\n"
            f"Full log: {LOG_PATH}\n--- last output ---\n{''.join(lines[-40:])}"
        )
    return rc


def _vendor_lie_learn(build_lib):
    build_lib = Path(build_lib).resolve()
    with tempfile.TemporaryDirectory(prefix="sphericalcnn-lie-learn.") as tmp:
        tmp = Path(tmp)
        src = tmp / "repo"
        empty_cwd = tmp / "cwd"  # nothing named lie_learn may shadow the bundle
        empty_cwd.mkdir()

        _run(["git", "clone", "--depth", "1", LIE_LEARN_REPO, str(src)])
        # sys.executable + inherited PYTHONPATH = the build environment, which has
        # the build requirements from pyproject.toml (numpy, Cython, ...).
        _run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=src)

        pkg = src / "lie_learn"
        for name in J_FILES:
            if not (pkg / PH_SUBDIR / name).is_file():
                raise RuntimeError(f"Missing upstream data file: {pkg / PH_SUBDIR / name}")

        dest = build_lib / "lie_learn"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(pkg, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))

        n_ext = len(list(dest.rglob("*.so"))) + len(list(dest.rglob("*.pyd")))
        _log(f"[setup] bundled lie_learn into {dest} ({n_ext} compiled extension files)")
        if n_ext == 0:
            _log("[setup] WARNING: no compiled extensions found in lie_learn")

        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            p for p in (str(build_lib), env.get("PYTHONPATH")) if p
        )
        if _run([sys.executable, "-c", _CHECK], cwd=empty_cwd, env=env, check=False) != 0:
            raise RuntimeError(
                f"Bundled lie_learn failed its import check (see above). Log: {LOG_PATH}"
            )
        _log("[setup] lie_learn OK")


def install_lie_learn(build_lib, editable=False):
    if os.environ.get("SKIP_LIE_LEARN_BUILD"):
        print("[setup] SKIP_LIE_LEARN_BUILD set; not building lie_learn")
        return
    if editable:
        print("[setup] WARNING: editable install; lie_learn is NOT bundled")
        return
    if (HERE / "lie_learn").is_dir():
        raise SystemExit(
            "ERROR: A top-level lie_learn/ directory exists in the project. "
            "Rename it (e.g. mv lie_learn lie_learn_backup) before installing."
        )

    _log(f"\n===== lie_learn step {datetime.datetime.now().isoformat()} =====")
    _log(f"[setup] log file: {LOG_PATH}")
    _log(f"[setup] python: {sys.version.split()[0]}  exe: {sys.executable}")
    _log(f"[setup] project dir: {HERE}")
    _log(f"[setup] lie_learn repo: {LIE_LEARN_REPO}")
    _log(f"[setup] PYTHONPATH: {os.environ.get('PYTHONPATH')}")
    try:
        _vendor_lie_learn(build_lib)
    except BaseException:
        _log("[setup] FAILED:\n" + traceback.format_exc())
        raise


class BuildPy(build_py):
    """Build this package as usual, then bundle a compiled lie_learn into it."""

    def run(self):
        super().run()

        data = HERE / "s2_mnist.gz"
        if data.is_file():
            shutil.copy2(data, Path(self.build_lib) / "s2_mnist.gz")

        install_lie_learn(self.build_lib, editable=getattr(self, "editable_mode", False))


class BinaryDistribution(Distribution):
    """The wheel contains compiled code, so make it platform-specific."""

    def has_ext_modules(self):
        return True


setup(
    distclass=BinaryDistribution,
    cmdclass={"build_py": BuildPy},
    # Every top-level .py file (model.py, train.py, gendata.py, ...) plus the
    # s2cnn package(s). lie_learn is added by BuildPy above.
    py_modules=sorted(p.stem for p in HERE.glob("*.py") if p.name != "setup.py"),
    packages=find_packages(
        exclude=[
            "tests", "tests.*",
            "examples", "examples.*",
            "experiments", "experiments.*",
            "lie_learn", "lie_learn.*",
        ]
    ),
)