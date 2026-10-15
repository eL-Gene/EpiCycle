"""Per-patch voxel size, the pure half (R7a, rows 10 and 29).

* ``patch_spacing`` is a WORKSPACE-LEVEL map in ``step3_state.json`` --
  one value per patch for every run: kept across a save, a reload and a
  run switch; the v3 wrap keeps it at the top level; an old v3 file without
  it loads;
* the order (user > Step 1 > the file > none) and "a ``user`` value is
  never overwritten by a ``step1`` / ``file`` re-resolve";
* ``measured_spacing`` on the run's patch record;
* the QC extraction-context sidecar per patch, the shared file read only
  when a patch has none, the staleness compared with the patch's OWN
  context;
* Step 3's spacing words and QC Settings' fallback sentence.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from napari_epicycle.data_preparation import _voxel_spacing as VS
from napari_epicycle.fusion_reconstruction import _migrate_step3_runs as MR
from napari_epicycle.fusion_reconstruction import _step3_pr_text as ST  # R7w: folded
from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
    EXTRACTION_CONTEXT_FILENAME,
    read_extraction_context,
    stale_reason,
    write_extraction_context,
)
from napari_epicycle.fusion_reconstruction.state_manager import (
    ReconstructionStateManager,
)

DEFAULT = MR.DEFAULT_RUN_ID


def _raw(ws):
    return json.loads((Path(ws) / ReconstructionStateManager.STATE_FILENAME)
                      .read_text(encoding="utf-8"))


def _mgr(ws):
    m = ReconstructionStateManager()
    m.load(str(ws))
    return m


# ---------------------------------------------------------- the map


def test_patch_spacing_is_a_workspace_level_key():
    assert "patch_spacing" in MR.WORKSPACE_LEVEL_KEYS_V1


def test_the_v3_wrap_keeps_it_at_the_top_level():
    flat = {"patches": {"a.tif": {"status": "predicted"}},
            "patch_spacing": {"a.tif": {"zyx": [2.0, 0.5, 0.5],
                                        "source": "user"}}}
    env = MR.wrap_step3_state_to_v3(flat, active_run_id="cells")
    assert env["patch_spacing"] == flat["patch_spacing"]
    assert "patch_spacing" not in env["runs"]["cells"]
    back = MR.unwrap_step3_state_v3(env, "cells")
    assert back["patch_spacing"] == flat["patch_spacing"]


def test_set_and_get(tmp_path):
    m = _mgr(tmp_path)
    assert m.get_patch_spacing("a.tif") is None
    assert m.set_patch_spacing("a.tif", (2, 0.5, 0.25), "step1") is True
    got = m.get_patch_spacing("a.tif")
    assert got == {"zyx": [2.0, 0.5, 0.25], "source": "step1"}
    got["zyx"][0] = 99.0                       # a copy
    assert m.get_patch_spacing("a.tif")["zyx"][0] == 2.0
    # the same value again changes nothing
    assert m.set_patch_spacing("a.tif", [2.0, 0.5, 0.25], "step1") is False
    assert m.get_all_patch_spacing() == {
        "a.tif": {"zyx": [2.0, 0.5, 0.25], "source": "step1"}}


@pytest.mark.parametrize("zyx", [(0, 1, 1), (-1, 1, 1), (1, 1), ("a", 1, 1),
                                 None])
def test_a_bad_value_is_refused(tmp_path, zyx):
    m = _mgr(tmp_path)
    with pytest.raises(ValueError):
        m.set_patch_spacing("a.tif", zyx, "user")
    assert m.get_patch_spacing("a.tif") is None


def test_a_bad_source_is_refused(tmp_path):
    m = _mgr(tmp_path)
    with pytest.raises(ValueError):
        m.set_patch_spacing("a.tif", (1, 1, 1), "qc")
    assert set(VS.SPACING_SOURCES) == {"user", "step1", "file"}


@pytest.mark.parametrize("source", ["step1", "file"])
def test_a_user_value_is_never_overwritten(tmp_path, source):
    m = _mgr(tmp_path)
    m.set_patch_spacing("a.tif", (3.0, 0.4, 0.4), "user")
    assert m.set_patch_spacing("a.tif", (2.0, 0.5, 0.5), source) is False
    assert m.get_patch_spacing("a.tif") == {"zyx": [3.0, 0.4, 0.4],
                                            "source": "user"}
    # a re-resolve that finds nothing leaves it too
    assert m.apply_resolved_spacing("a.tif", None, read_file=True) is False
    assert m.clear_patch_spacing("a.tif") is False
    assert m.get_patch_spacing("a.tif")["source"] == "user"
    # the user may change their own value, and clear it on purpose
    assert m.set_patch_spacing("a.tif", (4.0, 0.4, 0.4), "user") is True
    assert m.clear_patch_spacing("a.tif", user_too=True) is True
    assert m.get_patch_spacing("a.tif") is None


def test_a_user_unit_value_is_kept(tmp_path):
    """1 / 1 / 1 chosen by the user is a value (Step 4's audit D10)."""
    m = _mgr(tmp_path)
    m.set_patch_spacing("a.tif", (1, 1, 1), "user")
    assert m.get_patch_spacing("a.tif") == {"zyx": [1.0, 1.0, 1.0],
                                            "source": "user"}


# ---------------------------------------------------------- the order


_REL = "patches/p1.tif"


def _step1(voxel=(2.0, 0.5, 0.5)):
    return {"schema_version": 2,
            "sources": [{"source_id": "s0", "path": "D:/raw.tif",
                         "voxel_size": voxel}],
            "patches": [{"patch_name": "p1", "relative_path": _REL,
                         "sidecar_path": "patches/p1.json",
                         "source_id": "s0", "source_kind": "raw"}]}


@pytest.fixture
def file_tags(monkeypatch):
    table = {}
    monkeypatch.setattr(VS, "_read_voxel_spacing",
                        lambda p: table.get(str(p)))
    return table


def _resolve(m, ws, step1, *, read_file):
    path = str(Path(ws) / _REL)
    got = VS.resolve_stack_spacing({"spacing": m.get_patch_spacing("p1.tif")},
                                   step1, path, str(ws), read_file=read_file)
    return m.apply_resolved_spacing("p1.tif", got, read_file=read_file)


def test_the_order_user_then_step1_then_file(tmp_path, file_tags):
    file_tags[str(tmp_path / _REL)] = (5.0, 0.25, 0.25)
    m = _mgr(tmp_path)
    # the file only (Step 1's patch TIFFs carry no tags -- here they do)
    _resolve(m, tmp_path, None, read_file=True)
    assert m.get_patch_spacing("p1.tif")["source"] == "file"
    # Step 1 outranks the file
    _resolve(m, tmp_path, _step1(), read_file=False)
    assert m.get_patch_spacing("p1.tif") == {"zyx": [2.0, 0.5, 0.5],
                                             "source": "step1"}
    # the user outranks both, and a later re-resolve keeps it
    m.set_patch_spacing("p1.tif", (3.0, 0.7, 0.7), "user")
    assert _resolve(m, tmp_path, _step1(), read_file=True) is False
    assert m.get_patch_spacing("p1.tif") == {"zyx": [3.0, 0.7, 0.7],
                                             "source": "user"}


def test_a_main_thread_re_resolve_keeps_a_file_value(tmp_path, file_tags):
    """``read_file=False`` cannot see the file's tags, so it never drops a
    ``file`` value; a ``read_file=True`` re-resolve that finds none does."""
    file_tags[str(tmp_path / _REL)] = (5.0, 0.25, 0.25)
    m = _mgr(tmp_path)
    _resolve(m, tmp_path, None, read_file=True)
    assert _resolve(m, tmp_path, None, read_file=False) is False
    assert m.get_patch_spacing("p1.tif")["source"] == "file"
    file_tags.clear()
    assert _resolve(m, tmp_path, None, read_file=True) is True
    assert m.get_patch_spacing("p1.tif") is None


def test_a_step1_value_gone_from_step1_is_dropped(tmp_path, file_tags):
    m = _mgr(tmp_path)
    _resolve(m, tmp_path, _step1(), read_file=False)
    assert m.get_patch_spacing("p1.tif")["source"] == "step1"
    assert _resolve(m, tmp_path, _step1(voxel=None), read_file=False) is True
    assert m.get_patch_spacing("p1.tif") is None


# ---------------------------------------------------------- persistence


def test_kept_across_a_save_and_a_reload(tmp_path):
    m = _mgr(tmp_path)
    m.set_patch_spacing("a.tif", (2.0, 0.5, 0.5), "user")
    m.save()
    raw = _raw(tmp_path)
    assert raw["patch_spacing"] == {"a.tif": {"zyx": [2.0, 0.5, 0.5],
                                              "source": "user"}}
    for run in raw["runs"].values():
        assert "patch_spacing" not in run
    assert _mgr(tmp_path).get_patch_spacing("a.tif") == {
        "zyx": [2.0, 0.5, 0.5], "source": "user"}


def test_kept_across_a_run_switch_both_ways(tmp_path):
    m = _mgr(tmp_path)
    m.set_patch_status("a.tif", "predicted")
    m.set_patch_spacing("a.tif", (2.0, 0.5, 0.5), "user")
    m.create_run("cells")
    # a debounced save is pending; the switch flushes it first
    m.switch_active_run("cells")
    assert m.get_active_run_id() == "cells"
    assert m.get_patch_status("a.tif") == "unprocessed"       # per run
    assert m.get_patch_spacing("a.tif") == {"zyx": [2.0, 0.5, 0.5],
                                            "source": "user"}  # workspace
    # set on the other run, seen back on the first
    m.set_patch_spacing("b.tif", (1.5, 0.3, 0.3), "user")
    m.switch_active_run(DEFAULT)
    assert m.get_patch_spacing("b.tif")["zyx"] == [1.5, 0.3, 0.3]
    assert m.get_patch_status("a.tif") == "predicted"
    assert _raw(tmp_path)["patch_spacing"]["b.tif"]["source"] == "user"


def test_kept_across_a_switch_to_a_run_not_on_disk(tmp_path):
    """A run with no entry starts from an empty state -- but the workspace's
    map stays (it is one value per patch for every run)."""
    m = _mgr(tmp_path)
    m.set_patch_spacing("a.tif", (2.0, 0.5, 0.5), "user")
    m.switch_active_run("nuclei")
    assert m.get_all_patch_statuses() == {}
    assert m.get_patch_spacing("a.tif")["source"] == "user"
    m.save()
    assert _raw(tmp_path)["patch_spacing"]["a.tif"]["source"] == "user"


def test_an_old_v3_file_without_the_key_loads(tmp_path):
    (tmp_path / ReconstructionStateManager.STATE_FILENAME).write_text(
        json.dumps({"schema_version": 3, "active_run_id": DEFAULT,
                    "runs": {DEFAULT: {"patches": {
                        "a.tif": {"status": "reconstructed"}}}}}),
        encoding="utf-8")
    m = _mgr(tmp_path)
    assert m.get_patch_status("a.tif") == "reconstructed"
    assert m.get_patch_spacing("a.tif") is None
    assert m.get_all_patch_spacing() == {}


def test_a_corrupt_map_reads_as_none(tmp_path):
    (tmp_path / ReconstructionStateManager.STATE_FILENAME).write_text(
        json.dumps({"schema_version": 3, "active_run_id": DEFAULT,
                    "patch_spacing": {"a.tif": {"zyx": "x", "source": "user"},
                                      "b.tif": "nonsense",
                                      "c.tif": {"zyx": [2, 1, 1],
                                                "source": "qc"}},
                    "runs": {DEFAULT: {"patches": {}}}}), encoding="utf-8")
    m = _mgr(tmp_path)
    assert m.get_patch_spacing("a.tif") is None
    assert m.get_patch_spacing("b.tif") is None
    assert m.get_patch_spacing("c.tif") is None
    assert m.get_all_patch_spacing() == {}
    # and a step1 value may land over a corrupt one
    assert m.set_patch_spacing("a.tif", (2, 1, 1), "step1") is True


def test_a_write_schedules_the_debounced_save(tmp_path, monkeypatch):
    m = _mgr(tmp_path)
    calls = []
    monkeypatch.setattr(m, "schedule_save", lambda *a, **k: calls.append(1))
    m.set_patch_spacing("a.tif", (2, 1, 1), "user")
    m.set_patch_spacing("a.tif", (2, 1, 1), "user")       # unchanged
    m.set_measured_spacing("a.tif", {"zyx": [2, 1, 1], "source": "user"})
    assert len(calls) == 2


# ---------------------------------------------------------- measured_spacing


def test_measured_spacing_rides_the_runs_patch_record(tmp_path):
    m = _mgr(tmp_path)
    assert m.get_measured_spacing("a.tif") is None
    m.set_measured_spacing("a.tif", {"zyx": [2, 0.5, 0.5], "source": "step1"})
    assert m.get_measured_spacing("a.tif") == {"zyx": [2.0, 0.5, 0.5],
                                               "source": "step1"}
    m.save()
    rec = _raw(tmp_path)["runs"][DEFAULT]["patches"]["a.tif"]
    assert rec["measured_spacing"] == {"zyx": [2.0, 0.5, 0.5],
                                       "source": "step1"}
    # extracted in voxels: an explicit None
    m.set_measured_spacing("a.tif", None)
    assert m.get_measured_spacing("a.tif") is None
    assert "measured_spacing" in m._state["patches"]["a.tif"]
    # per run: another run does not see it
    m.create_run("cells")
    m.switch_active_run("cells")
    assert m.get_measured_spacing("a.tif") is None


def test_a_bad_measured_spacing_is_refused(tmp_path):
    m = _mgr(tmp_path)
    with pytest.raises(ValueError):
        m.set_measured_spacing("a.tif", {"zyx": [0, 1, 1]})
    with pytest.raises(ValueError):
        m.set_measured_spacing("a.tif", [2, 1, 1])


# ---------------------------------------------------------- the QC sidecar


def _csv(ws, stem):
    p = Path(ws) / "qc" / f"{stem}_features.csv"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("label\n", encoding="utf-8")
    return p


def test_the_sidecar_is_per_patch(tmp_path):
    a, b = _csv(tmp_path, "A"), _csv(tmp_path, "B")
    write_extraction_context(a, spacing=(2.0, 0.5, 0.5), downsample_factor=1)
    write_extraction_context(b, spacing=(4.0, 0.25, 0.25), downsample_factor=2)
    qc = tmp_path / "qc"
    assert (qc / "A_extraction_context.json").is_file()
    assert (qc / "B_extraction_context.json").is_file()
    assert not (qc / EXTRACTION_CONTEXT_FILENAME).exists()
    assert read_extraction_context(a)["spacing"] == [2.0, 0.5, 0.5]
    assert read_extraction_context(b)["spacing"] == [4.0, 0.25, 0.25]


def test_the_staleness_compares_a_patch_with_its_own_context(tmp_path):
    """Extract A, then B with another spacing: A is NOT stale at A's
    spacing (the shared file used to hold B's), and IS at B's."""
    a, b = _csv(tmp_path, "A"), _csv(tmp_path, "B")
    write_extraction_context(a, spacing=(2.0, 0.5, 0.5), downsample_factor=1)
    write_extraction_context(b, spacing=(4.0, 0.25, 0.25), downsample_factor=1)
    assert stale_reason(read_extraction_context(a),
                        current_spacing=(2.0, 0.5, 0.5),
                        current_downsample=1) is None
    assert stale_reason(read_extraction_context(a),
                        current_spacing=(4.0, 0.25, 0.25),
                        current_downsample=1)
    assert stale_reason(read_extraction_context(b),
                        current_spacing=(4.0, 0.25, 0.25),
                        current_downsample=1) is None


def test_the_shared_file_is_read_only_when_a_patch_has_none(tmp_path):
    a, b = _csv(tmp_path, "A"), _csv(tmp_path, "B")
    legacy = tmp_path / "qc" / EXTRACTION_CONTEXT_FILENAME
    legacy.write_text(json.dumps({"spacing": [9.0, 9.0, 9.0],
                                  "downsample_factor": 1}), encoding="utf-8")
    # no per-patch file: the shared one (today's behaviour, no worse)
    assert read_extraction_context(a)["spacing"] == [9.0, 9.0, 9.0]
    # a per-patch file wins, for that patch only
    write_extraction_context(a, spacing=(2.0, 0.5, 0.5), downsample_factor=1)
    assert read_extraction_context(a)["spacing"] == [2.0, 0.5, 0.5]
    assert read_extraction_context(b)["spacing"] == [9.0, 9.0, 9.0]
    # the write never touched the shared file
    assert json.loads(legacy.read_text(encoding="utf-8"))["spacing"] == \
        [9.0, 9.0, 9.0]


def test_a_malformed_per_patch_file_does_not_fall_back(tmp_path):
    a = _csv(tmp_path, "A")
    (tmp_path / "qc" / EXTRACTION_CONTEXT_FILENAME).write_text(
        json.dumps({"spacing": [9.0, 9.0, 9.0], "downsample_factor": 1}),
        encoding="utf-8")
    (tmp_path / "qc" / "A_extraction_context.json").write_text(
        "{not json", encoding="utf-8")
    assert read_extraction_context(a) is None


def test_a_csv_with_no_patch_in_its_name_keeps_the_shared_file(tmp_path):
    """``features.csv`` names no patch: the shared file, as before."""
    csv = tmp_path / "features.csv"
    write_extraction_context(csv, spacing=(1.0, 1.0, 1.0), downsample_factor=1)
    assert (tmp_path / EXTRACTION_CONTEXT_FILENAME).is_file()


# ---------------------------------------------------------- the words


def test_the_row_spacing_words():
    assert ST.spacing_row_text({"zyx": [2, 0.5, 0.25], "source": "step1"}) == \
        "2.00 × 0.50 × 0.25 µm (from Step 1)"
    assert ST.spacing_row_text({"zyx": [2, 0.5, 0.25], "source": "file"}) == \
        "2.00 × 0.50 × 0.25 µm (from the file)"
    assert ST.spacing_row_text({"zyx": [2, 0.5, 0.25], "source": "user"}) == \
        "2.00 × 0.50 × 0.25 µm (set by you)"
    for none in (None, {}, {"zyx": "x"}, {"zyx": [1, 2]}):
        assert ST.spacing_row_text(none) == ST.S3_ROW_SPACING_NONE == "voxels"
    assert ST.S3_SET_SPACING_BTN == "Set spacing…"
    assert ST.S3_MENU_SET_SPACING == "Set spacing…"


def test_the_dialog_words_fit_the_shared_dialog():
    from napari_epicycle.ui_shell._set_spacing_dialog import SPACING_TEXT_KEYS
    texts = ST.step3_spacing_texts()
    assert set(texts) <= set(SPACING_TEXT_KEYS)
    for key in ("zorient_label", "zorient_items", "axes_label", "axes_auto"):
        assert key not in texts                    # Step 3 shows no such row
    assert texts["desc_one"].format(name="p.tif") .endswith("p.tif.")
    assert "{n}" in texts["desc"]


def test_the_confirm_names_the_patches_that_go_stale():
    one = ST.spacing_confirm_text(["A.tif"])
    assert "A.tif" in one
    two = ST.spacing_confirm_text(["A.tif", "B.tif"])
    assert "A.tif" in two and "B.tif" in two and "2" in two
    many = ST.spacing_confirm_text([f"P{i}.tif" for i in range(9)])
    assert "P0.tif" in many and "P8.tif" not in many and "9" in many
    assert ST.spacing_confirm_text([]) == ""


def test_the_step1_no_tags_hint_and_the_qc_fallback_sentence():
    assert "Step 1" in ST.S3_SPACING_NO_TAGS_HINT
    assert ST.QC_SPACING_FALLBACK == (
        "Used for patches with no voxel size of their own; set one per "
        "patch in Session.")


def test_the_words_module_is_qt_free():
    import ast
    src = Path(ST.__file__).read_text(encoding="utf-8")
    mods = set()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods |= {a.name for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            mods.add(node.module or "")
    assert not any(m.startswith(("qtpy", "PyQt", "napari", "superqt",
                                 "segmentation")) for m in mods), mods


def test_qc_settings_says_its_spacing_is_the_fallback():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])  # noqa: F841
    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )
    dlg = QCExtractionDialog(None)
    try:
        lbl = dlg.spacing_fallback_label
        assert lbl.text() == ST.QC_SPACING_FALLBACK
        assert lbl.wordWrap()
        assert not lbl.isHidden()
    finally:
        dlg.deleteLater()
