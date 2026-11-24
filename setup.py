"""
Packaging for the Spherical CNN project.

Metadata and dependencies are in pyproject.toml. This file exists because
lie_learn can't be installed as an ordinary pip dependency, so it reproduces
what setup.sh did, as part of building this package:

  1. git clone upstream lie_learn
  2. pip install it from source with --no-build-isolation
  3. copy the precomputed Wigner J matrices (which the build omits) into the
     installed package
  4. verify the imports that model.py needs

Requires git, a C compiler, and network access at install time (same as
setup.sh). Install with:  pip install .
"""
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

HERE = Path(__file__).resolve().parent

# Override with LIE_LEARN_REPO=<url or file:// path>; skip with SKIP_LIE_LEARN_BUILD=1
LIE_LEARN_REPO = os.environ.get(
    "LIE_LEARN_REPO", "https://github.com/AMLab-Amsterdam/lie_learn.git"
)
J_FILES = ("J_dense_0-150.npy", "J_block_0-150.npy")
PH_SUBDIR = Path("representations") / "SO3" / "pinchon_hoggan"

# Run in a subprocess: same imports as the "confirm critical imports" step of
# setup.sh, plus a check that the J matrices are really installed.
_CHECK = f"""
import pathlib, sys
import lie_learn
from lie_learn.representations.SO3.irrep_bases import change_of_basis_matrix
from lie_learn.representations.SO3.wigner_d import wigner_d_matrix
d = pathlib.Path(lie_learn.__file__).resolve().parent / "representations" / "SO3" / "pinchon_hoggan"
missing = [n for n in {J_FILES!r} if not (d / n).is_file()]
sys.exit("missing data files: " + ", ".join(missing) if missing else 0)
"""


def _env():
    env = dict(os.environ)
    env.pop("PIP_BUILD_TRACKER", None)  # don't share the outer pip's build tracker
    return env


def _run(cmd, **kwargs):
    print("[setup] $ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.check_call(cmd, env=_env(), **kwargs)


def _lie_learn_ok(cwd, quiet):
    # cwd must be an empty dir so nothing named lie_learn can shadow the install.
    out = subprocess.DEVNULL if quiet else None
    proc = subprocess.run(
        [sys.executable, "-c", _CHECK], cwd=cwd, env=_env(), stdout=out, stderr=out
    )
    return proc.returncode == 0


def install_lie_learn():
    if os.environ.get("SKIP_LIE_LEARN_BUILD"):
        print("[setup] SKIP_LIE_LEARN_BUILD set; not building lie_learn")
        return

    if (HERE / "lie_learn").is_dir():
        raise SystemExit(
            "ERROR: A top-level lie_learn/ directory exists in the project. "
            "Rename it (e.g. mv lie_learn lie_learn_backup) before installing."
        )

    with tempfile.TemporaryDirectory(prefix="sphericalcnn-lie-learn.") as tmp:
        tmp = Path(tmp)
        cwd = tmp / "cwd"
        cwd.mkdir()

        if _lie_learn_ok(cwd, quiet=True):
            print("[setup] lie_learn already installed and working; skipping build")
            return

        src = tmp / "repo"
        _run(["git", "clone", "--depth", "1", LIE_LEARN_REPO, str(src)])
        _run(
            [
                sys.executable, "-m", "pip", "install",
                "--no-build-isolation", "--no-cache-dir", "--no-deps",
                "--force-reinstall", str(src),
            ],
            cwd=cwd,
        )

        # Locate the installed package and copy in the J matrices.
        out = subprocess.check_output(
            [
                sys.executable, "-c",
                "import pathlib, lie_learn; "
                "print(pathlib.Path(lie_learn.__file__).resolve().parent)",
            ],
            cwd=cwd, env=_env(), text=True,
        )
        pkg = Path(out.strip().splitlines()[-1])
        dest = pkg / PH_SUBDIR
        dest.mkdir(parents=True, exist_ok=True)
        for name in J_FILES:
            source = src / "lie_learn" / PH_SUBDIR / name
            if not source.is_file():
                raise RuntimeError(f"Missing upstream data file: {source}")
            shutil.copy2(source, dest / name)
            print(f"[setup] Installed {name}")

        if not _lie_learn_ok(cwd, quiet=False):
            raise RuntimeError("lie_learn was installed but its imports failed (see above).")
        print("[setup] lie_learn OK")


class BuildPy(build_py):
    """Build lie_learn first, then build this package as usual."""

    def run(self):
        install_lie_learn()
        super().run()


setup(
    cmdclass={"build_py": BuildPy},
    # Every top-level .py file (model.py, train.py, gendata.py, ...) plus the
    # s2cnn package(s).
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
