"""Import hygiene -- the engine is napari-free, Qt-free, importable standalone.

The rule: *"The engine lives inside EpiCycle.
``packages/epicycle_annotator/reshape.py`` (or a ``reshape/`` sub-package),
napari-free and Qt-free.  Nothing in the plugin ever imports from
``notebooks/``."*

The check runs in a **subprocess**: in-process it would be meaningless, because
the pytest session has already imported napari through other test modules.

These tests also pin the lazy ``packages.epicycle_annotator.__init__`` that R0
introduced.  The package's eager ``__init__`` used to import ``sync`` /
``subviewers`` / ``crosshairs``, which import napari and vispy at module level,
so *every* module in the package -- pure ones included -- was transitively
napari-dependent.  The re-exports are now PEP 562 lazy; the names and the
objects they resolve to are unchanged, and the tests below assert exactly that.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_isolated(code: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_reshape_imports_without_napari_or_qt():
    """The mandated check, verbatim."""
    proc = _run_isolated(
        "import sys; import packages.epicycle_annotator.reshape; "
        "assert 'napari' not in sys.modules and 'qtpy' not in sys.modules"
    )
    assert proc.returncode == 0, (
        "reshape pulled in napari/Qt:\n" + proc.stdout + proc.stderr
    )


def test_reshape_imports_without_vispy_or_qt_bindings():
    proc = _run_isolated(
        "import sys; import packages.epicycle_annotator.reshape as r;\n"
        "banned = [m for m in ('napari', 'qtpy', 'vispy', 'PyQt5', 'PySide2',\n"
        "                      'PyQt6', 'PySide6', 'matplotlib')\n"
        "          if m in sys.modules]\n"
        "assert not banned, banned\n"
        "assert r.ReshapeSpec is not None\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_the_region_target_imports_without_napari_qt_or_the_plugin():
    """``reshape/region.py`` (the Add region target) is imported on its own,
    in a fresh process: no napari, no Qt binding, no vispy, and nothing
    from ``napari_epicycle`` (``packages/`` never imports the plugin)."""
    proc = _run_isolated(
        "import sys; import packages.epicycle_annotator.reshape.region as r;\n"
        "banned = [m for m in sys.modules if m.split('.')[0] in\n"
        "          ('napari', 'qtpy', 'vispy', 'PyQt5', 'PySide2', 'PyQt6',\n"
        "           'PySide6', 'napari_epicycle')]\n"
        "assert not banned, banned\n"
        "assert r.reshape_empty_region is not None\n"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_engine_never_imports_from_notebooks():
    """Nothing in the plugin ever imports from ``notebooks/``."""
    package_dir = REPO_ROOT / "packages" / "epicycle_annotator" / "reshape"
    offenders = []
    for path in sorted(package_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith(("import ", "from ")) and "notebook" in stripped:
                offenders.append("{}: {}".format(path.name, stripped))
            if "_membrane_lib" in stripped and stripped.startswith(("import ", "from ")):
                offenders.append("{}: {}".format(path.name, stripped))
    assert offenders == []


def test_reshape_public_surface_is_importable_by_name():
    import packages.epicycle_annotator.reshape as reshape

    for name in reshape.__all__:
        assert hasattr(reshape, name), name


# --------------------------------------------------------------------------- #
# the lazy parent package keeps its exact public API
# --------------------------------------------------------------------------- #
def test_parent_package_reexports_resolve_to_the_same_objects():
    import packages.epicycle_annotator as pkg
    from packages.epicycle_annotator import split as split_mod
    from packages.epicycle_annotator import orientation as orientation_mod

    assert pkg.apply_split_to_volume is split_mod.apply_split_to_volume
    assert pkg.seeded_watershed_split is split_mod.seeded_watershed_split
    assert pkg.transpose_data is orientation_mod.transpose_data


def test_parent_package_all_names_still_resolve():
    import packages.epicycle_annotator as pkg

    for name in pkg.__all__:
        assert getattr(pkg, name) is not None, name


def test_parent_package_unknown_attribute_raises_attribute_error():
    import packages.epicycle_annotator as pkg

    with pytest.raises(AttributeError):
        pkg.definitely_not_a_real_name


def test_parent_package_dir_lists_the_public_names():
    import packages.epicycle_annotator as pkg

    listed = set(dir(pkg))
    assert set(pkg.__all__).issubset(listed)


def test_parent_package_import_alone_pulls_no_napari():
    """The lazy __init__ is what makes the pure submodules genuinely pure."""
    proc = _run_isolated(
        "import sys, packages.epicycle_annotator as p; "
        "assert 'napari' not in sys.modules; "
        "assert p.transpose_data is not None"
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
