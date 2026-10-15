"""The voxel-size resolver, lifted from Step 4 into Step 1's package (R2b).

"The resolver": a value the user set
wins and is never overwritten; else Step 1's registry
(``_state.voxel_size_for_patch_path``; a processed patch its store's own
scale); else the TIFF's own tags; else none.  ``read_file=False`` (the main
thread) opens no file.  Qt-free: driven with fakes, no TIFF written.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

from napari_epicycle.data_preparation import _state as S
from napari_epicycle.data_preparation import _voxel_spacing as VS

_REL = "patches/p1.tif"


def _state(kind="raw", voxel=(2.0, 0.5, 0.5), store=None):
    src = {"source_id": "src_0", "path": "D:/raw.tif", "voxel_size": voxel}
    if store:
        src["processed_store"] = {"path": store, "format": "ome-zarr"}
    return {
        "schema_version": 2,
        "sources": [src],
        "patches": [{"patch_name": "p1", "relative_path": _REL,
                     "sidecar_path": "patches/p1.json",
                     "source_id": "src_0", "source_kind": kind}],
    }


@pytest.fixture
def reads(monkeypatch):
    """The file read as a fake: a dict path -> spacing; every call kept."""
    table = {}
    calls = []

    def fake(path):
        calls.append(str(path))
        return table.get(str(path))

    monkeypatch.setattr(VS, "_read_voxel_spacing", fake)
    return table, calls


def _p(ws):
    return str(Path(ws) / "patches" / "p1.tif")


# ---------------------------------------------------------- the order rule


def test_a_user_value_wins_over_step1_and_the_file(tmp_path, reads):
    table, calls = reads
    table[_p(tmp_path)] = (5.0, 0.25, 0.25)
    rec = {"spacing": {"zyx": [3.0, 0.7, 0.7], "source": "user"}}
    assert VS.resolve_stack_spacing(rec, _state(), _p(tmp_path),
                                    str(tmp_path)) == \
        {"zyx": [3.0, 0.7, 0.7], "source": "user"}
    assert calls == []                 # decided before any file


def test_a_user_unit_value_is_kept(tmp_path, reads):
    rec = {"spacing": {"zyx": [1, 1, 1], "source": "user"}}
    assert VS.resolve_stack_spacing(rec, _state(), _p(tmp_path),
                                    str(tmp_path)) == \
        {"zyx": [1.0, 1.0, 1.0], "source": "user"}


def test_a_non_user_record_value_is_not_rule_1(tmp_path, reads):
    rec = {"spacing": {"zyx": [9.0, 9.0, 9.0], "source": "file"}}
    assert VS.resolve_stack_spacing(rec, _state(), _p(tmp_path),
                                    str(tmp_path)) == \
        {"zyx": [2.0, 0.5, 0.5], "source": "step1"}


def test_step1_wins_over_the_file(tmp_path, reads):
    table, calls = reads
    table[_p(tmp_path)] = (5.0, 0.25, 0.25)
    assert VS.resolve_stack_spacing({}, _state(), _p(tmp_path),
                                    str(tmp_path)) == \
        {"zyx": [2.0, 0.5, 0.5], "source": "step1"}
    assert calls == []


def test_a_unit_step1_voxel_is_unknown_and_falls_to_the_file(tmp_path, reads):
    table, _ = reads
    table[_p(tmp_path)] = (5.0, 0.25, 0.25)
    assert VS.resolve_stack_spacing({}, _state(voxel=(1, 1, 1)),
                                    _p(tmp_path), str(tmp_path)) == \
        {"zyx": [5.0, 0.25, 0.25], "source": "file"}


def test_the_file_when_step1_does_not_know_the_patch(tmp_path, reads):
    table, _ = reads
    other = str(tmp_path / "elsewhere" / "q.tif")
    table[other] = (4.0, 0.3, 0.3)
    assert VS.resolve_stack_spacing({}, _state(), other, str(tmp_path)) == \
        {"zyx": [4.0, 0.3, 0.3], "source": "file"}


def test_a_path_outside_the_workspace_skips_step1(tmp_path, reads,
                                                   monkeypatch):
    monkeypatch.setattr(S, "voxel_size_for_patch_path",
                        lambda *a: pytest.fail("asked Step 1"))
    ws = tmp_path / "ws"
    out = str(tmp_path / "patches" / "p1.tif")
    assert VS.resolve_stack_spacing({}, _state(), out, str(ws)) is None


def test_none_when_nothing_knows(tmp_path, reads):
    assert VS.resolve_stack_spacing(None, None, _p(tmp_path),
                                    str(tmp_path)) is None
    table, _ = reads
    table[_p(tmp_path)] = (1.0, 1.0, 1.0)          # unit tags = unknown
    assert VS.resolve_stack_spacing({}, None, _p(tmp_path),
                                    str(tmp_path)) is None


# ---------------------------------------------------------- read_file=False


def test_read_file_false_opens_no_file(tmp_path, reads):
    table, calls = reads
    table[_p(tmp_path)] = (5.0, 0.25, 0.25)
    assert VS.resolve_stack_spacing({}, None, _p(tmp_path), str(tmp_path),
                                    read_file=False) is None
    state = _state(kind="processed", store="artifacts/step1/s.zarr")
    table[str(tmp_path / "artifacts/step1/s.zarr")] = (2.0, 0.4, 0.4)
    assert VS.resolve_stack_spacing({}, state, _p(tmp_path), str(tmp_path),
                                    read_file=False) is None
    # Step 1's real voxel still answers without a file
    assert VS.resolve_stack_spacing({}, _state(), _p(tmp_path),
                                    str(tmp_path), read_file=False) == \
        {"zyx": [2.0, 0.5, 0.5], "source": "step1"}
    assert calls == []


# ---------------------------------------------------------- a processed patch


def test_the_registry_gives_a_processed_patch_no_voxel():
    info = S.voxel_size_for_patch_path(
        _state(kind="processed", store="artifacts/step1/s.zarr"), _REL)
    assert info["voxel_size"] is None
    assert info["source_kind"] == "processed"


def test_a_processed_patch_reads_its_store_never_the_raw_voxel(tmp_path,
                                                               reads):
    table, calls = reads
    store = str(Path(tmp_path) / "artifacts/step1/s.zarr")
    table[store] = (2.0, 0.4, 0.4)
    table[_p(tmp_path)] = (5.0, 0.25, 0.25)
    state = _state(kind="processed", store="artifacts/step1/s.zarr")
    assert VS.resolve_stack_spacing({}, state, _p(tmp_path),
                                    str(tmp_path)) == \
        {"zyx": [2.0, 0.4, 0.4], "source": "step1"}
    assert calls == [store]


def test_a_processed_patch_with_no_store_scale_falls_to_its_tags(tmp_path,
                                                                 reads):
    table, calls = reads
    table[_p(tmp_path)] = (5.0, 0.25, 0.25)
    state = _state(kind="processed", store="artifacts/step1/s.zarr")
    assert VS.resolve_stack_spacing({}, state, _p(tmp_path),
                                    str(tmp_path)) == \
        {"zyx": [5.0, 0.25, 0.25], "source": "file"}
    assert len(calls) == 2


def test_the_resolver_never_raises(tmp_path, monkeypatch):
    def boom(*_a):
        raise RuntimeError("x")
    monkeypatch.setattr(S, "voxel_size_for_patch_path", boom)
    monkeypatch.setattr(VS, "_read_voxel_spacing", lambda _p: None)
    assert VS.resolve_stack_spacing({}, _state(), _p(tmp_path),
                                    str(tmp_path)) is None


@pytest.mark.parametrize("record", [
    "x",                      # a str has no .get
    5,                        # an int
    3.5,                      # a float
    ["spacing"],              # a non-empty list
    ("user",),                # a tuple
])
def test_a_record_that_is_not_a_dict_never_raises(tmp_path, reads, record):
    """'Never raises' holds for a corrupt record too: a non-dict is read as
    no record, and the later rules still answer."""
    assert VS.resolve_stack_spacing(record, None, _p(tmp_path),
                                    str(tmp_path)) is None
    reads[0][_p(tmp_path)] = (2.0, 0.3, 0.3)
    assert VS.resolve_stack_spacing(record, None, _p(tmp_path),
                                    str(tmp_path)) == \
        {"zyx": [2.0, 0.3, 0.3], "source": "file"}


def test_a_folder_whose_name_starts_with_two_dots_is_inside(tmp_path):
    """Only a path that climbs OUT (``..`` or ``../...``) is outside; a
    folder named ``..odd`` inside the workspace is not."""
    ws = tmp_path / "ws"
    assert VS._rel_to(str(ws / "..odd" / "p.tiff"), str(ws)) == \
        "..odd/p.tiff"
    assert VS._rel_to(str(ws / "..odd"), str(ws)) == "..odd"
    assert VS._rel_to(str(tmp_path / "p.tiff"), str(ws)) is None
    assert VS._rel_to(str(tmp_path), str(ws)) is None


def test_the_real_read_never_raises_on_a_missing_file(tmp_path):
    assert VS._read_voxel_spacing(str(tmp_path / "nope.tif")) is None


# ---------------------------------------------------------- the imports


_MODULE = Path(VS.__file__)


_PKG = "napari_epicycle.data_preparation"     # the module's own package


def _resolve(level: int, module: str, package: str = _PKG) -> str:
    """An ``ImportFrom``'s absolute module: ``level`` dots climb from
    *package* (``.`` = the package itself, ``..`` = its parent)."""
    if not level:
        return module
    parts = package.split(".")
    base = ".".join(parts[:len(parts) - (level - 1)])
    return base + "." + module if module else base


def _imported(tree) -> set:
    """Every module a file imports.  An ``ImportFrom`` counts its NAMES
    too (``from . import controller`` imports ``...controller``), each
    resolved against the module's package; the raw relative spelling is
    kept as well."""
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(("." * node.level) + (node.module or ""))
            mod = _resolve(node.level, node.module or "")
            mods.add(mod)
            mods |= {mod + "." + a.name for a in node.names}
    return mods


_OTHERS = ("segmentation", "fusion_reconstruction", "annotator",
           "hitl_training_annotation", "tiled_segmentation",
           "model_selection", "ui_shell")


#: the Step 1 modules it may read: ``_state`` (the registry) and, since
#: R2b part 2, ``_paths`` (``read_step1_state``'s ``step1_state_path``) --
#: both Qt-free; the rest of ``data_preparation`` is Qt
_ALLOWED_STEP1 = ("_state", "_paths")


def _violations(mods) -> list:
    """Qt / napari, every OTHER step package, and any Step 1 module but
    ``_state`` / ``_paths`` (the rest of ``data_preparation`` is Qt: the
    controller, the panel, the workers)."""
    bad = [m for m in mods
           if m.startswith(("qtpy", "PyQt", "napari.", "superqt"))
           or m == "napari"]
    bad += [m for m in mods if any(o in m.split(".") for o in _OTHERS)]
    bad += [m for m in mods if m.startswith(_PKG + ".")
            and not any(m == f"{_PKG}.{a}" or m.startswith(f"{_PKG}.{a}.")
                        for a in _ALLOWED_STEP1)]
    return bad


def test_the_module_imports_no_qt_and_no_other_step():
    """Step 1 owns the registry it reads, so ``data_preparation._state``
    is allowed (the ``ui_shell`` no-step-import rule does not apply here);
    Qt, napari and every OTHER step package are not."""
    mods = _imported(ast.parse(_MODULE.read_text(encoding="utf-8")))
    assert not _violations(mods), mods
    assert "napari_epicycle.data_preparation._state" in mods
    assert "packages.epicycle_largedata.axes" in mods


def test_the_import_check_has_teeth():
    bad = _imported(ast.parse("from ..segmentation import _step4_models\n"
                              "from qtpy.QtCore import Qt\n"
                              "import napari_epicycle.ui_shell.theme\n"))
    assert "..segmentation" in bad and "qtpy.QtCore" in bad
    assert "napari_epicycle.ui_shell.theme" in bad


@pytest.mark.parametrize("src", [
    "from . import controller\n",                  # Step 1's Qt controller
    "from .. import segmentation\n",
    "from napari_epicycle import segmentation\n",
    "from napari_epicycle import fusion_reconstruction as F\n",
    "from .controller import DataPreparationController\n",
])
def test_an_import_by_name_is_caught(src):
    """``ImportFrom.names`` count: ``from . import controller`` imports a
    module as surely as ``import ...controller`` does."""
    assert _violations(_imported(ast.parse(src))), src


def test_the_paths_module_it_reads_is_qt_free():
    """``_paths`` is allowed only because it imports nothing but the
    standard library."""
    from napari_epicycle.data_preparation import _paths as DP
    mods = _imported(ast.parse(Path(DP.__file__).read_text(
        encoding="utf-8")))
    assert all(m.split(".")[0] in ("__future__", "pathlib", "typing", "os")
               for m in mods), mods


# ---------------------------------------------------------- read_step1_state


def test_read_step1_state_reads_the_file(tmp_path):
    import json
    from napari_epicycle.data_preparation._paths import step1_state_path
    path = step1_state_path(str(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_state()), encoding="utf-8")
    assert VS.read_step1_state(str(tmp_path)) ==         json.loads(json.dumps(_state()))


@pytest.mark.parametrize("content", ["{not json", "[1, 2]"])
def test_read_step1_state_never_raises_and_never_renames(tmp_path, content):
    from napari_epicycle.data_preparation._paths import step1_state_path
    path = step1_state_path(str(tmp_path))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    assert VS.read_step1_state(str(tmp_path)) is None
    assert path.read_text(encoding="utf-8") == content     # left as it was
    assert VS.read_step1_state("") is None
    assert VS.read_step1_state(str(tmp_path / "missing")) is None


def test_read_step1_state_is_the_same_object_in_step4():
    from napari_epicycle.segmentation import _step4_models as M
    assert M.read_step1_state is VS.read_step1_state
    assert VS.read_step1_state.__module__ ==         "napari_epicycle.data_preparation._voxel_spacing"


def test_the_module_own_imports_are_not_flagged_by_the_name_rule():
    ok = ("from . import _state\n"
          "from ._state import voxel_size_for_patch_path\n")
    assert not _violations(_imported(ast.parse(ok)))
