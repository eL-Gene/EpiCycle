"""Where stacks come from and what each row says (D-g).  Qt-free."""
from __future__ import annotations

import json
from pathlib import Path

from napari_epicycle.segmentation import _step4_models as M
from napari_epicycle.segmentation import _step4_rows as R
from napari_epicycle.segmentation import _step4_text as T

from ._step4_panel import make_workspace, write_stack


def _step1(ws):
    return json.loads((ws / "step1_state.json").read_text(encoding="utf-8"))


def test_step1_patches_come_from_the_registry_not_a_scan(tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif"))
    # a stray file in the patch folder is NOT a registered patch
    write_stack(ws / "artifacts/step1/patches/stray.tif")
    names = [e["name"] for e in R.list_stack_sources(str(ws), _step1(ws), [])]
    assert names == ["P1.tif", "P2.tif"]


def test_a_folder_that_is_step1s_folder_lists_each_patch_once(tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif", "P2.tif"))
    folder = str(ws / "artifacts/step1/patches")
    out = R.list_stack_sources(str(ws), _step1(ws), [folder])
    assert [e["name"] for e in out] == ["P1.tif", "P2.tif"]
    assert [e["origin"] for e in out] == ["step1", "step1"]


def test_added_folders_list_tif_and_tiff_only(tmp_path):
    ws = make_workspace(tmp_path, patches=())
    extra = tmp_path / "extra"
    write_stack(extra / "a.tif")
    write_stack(extra / "b.tiff")
    np_file = extra / "c.npy"
    np_file.write_bytes(b"x")
    out = R.list_stack_sources(str(ws), _step1(ws), [str(extra)])
    assert [e["name"] for e in out] == ["a.tif", "b.tiff"]
    assert all(e["origin"] == "folder" for e in out)


def test_same_name_or_stem_is_listed_but_cannot_run(tmp_path):
    ws = make_workspace(tmp_path, patches=("P1.tif",))
    extra = tmp_path / "extra"
    write_stack(extra / "P1.tif")
    write_stack(extra / "P1.tiff")
    out = R.list_stack_sources(str(ws), _step1(ws), [str(extra)])
    assert [e["conflict"] for e in out] == [False, True, True]
    row = R.build_row(out[1], {})
    assert not row["can_run"] and not row["tickable"]
    assert T.ROW_WARN_SAME_NAME in row["warns_narrow"]


def _rec(**kw):
    base = {"status": "unprocessed", "axes": "CZYX",
            "stack_shape": [2, 6, 24, 24], "n_channels": 2,
            "spacing": {"zyx": [2.0, 0.5, 0.5], "source": "step1"}}
    base.update(kw)
    return base


ENTRY = {"name": "s.tif", "path": "", "origin": "step1", "conflict": False}


def test_row_words_colours_and_shape():
    row = R.build_row(ENTRY, _rec())
    assert row["word"] == "unprocessed" and row["colour"] == "#777777"
    assert row["icon"] == "mdi6.circle-outline"
    assert row["shape_text"] == T.ROW_SHAPE.format(Z=6, Y=24, X=24, c=2)
    assert row["spacing_text"] == T.ROW_SPACING_STEP1.format(
        z="2.00", y="0.50", x="0.50")
    rec = _rec(status="qc_extracted", n_cells=4653, labels_path="x")
    row = R.build_row(ENTRY, rec)
    assert row["word"] == "measured" and row["colour"] == "#a78bca"
    assert row["shape_text"].endswith(T.ROW_SHAPE_CELLS.format(n="4,653"))


def test_legacy_statuses_show_failed_and_measured():
    assert R.build_row(ENTRY, _rec(status="segmented"))["word"] == "failed"
    assert R.build_row(ENTRY, _rec(status="smoothed"))["word"] == "measured"


def test_no_spacing_rows_warn_in_voxels():
    row = R.build_row(ENTRY, _rec(spacing=None))
    assert row["in_voxels"] and row["spacing_text"] == T.ROW_SPACING_NONE
    # A3 fold: ①'s wide row follows Session's pattern -- no spacing on
    # line 2, no warning line; the voxel line says "voxels" in amber
    assert T.ROW_WARN_NO_SPACING not in row["warns_wide"]
    assert row["line3"] == T.ROW_SPACING_NONE and row["line3_warn"]
    assert T.ROW_WARN_NO_SPACING_SHORT not in row["warns_narrow"]  # L7
    assert row["can_run"]          # nothing refuses to run


def test_channels_that_do_not_fit_grey_the_tick():
    row = R.build_row(ENTRY, _rec(stack_shape=[6, 24, 24], axes="ZYX",
                                  n_channels=1), needs=2)
    assert not row["can_run"]
    assert T.ROW_WARN_CHANNELS.format(n=2, k=1) in row["warns_narrow"]
    # unknown needs: no warning (blank stays blank)
    assert R.build_row(ENTRY, _rec(stack_shape=[6, 24, 24], axes="ZYX",
                                   n_channels=1))["can_run"]


SNAP_A = M.model_snapshot("C:/ws/m/a/lateral", "C:/ws/m/a/axial",
                          kind="refined", workspace="C:/ws", run_id="r1",
                          bundle="a")
SNAP_B = M.model_snapshot("cyto3", "cyto3", kind="pretrained")


def test_made_by_chip_amber_when_not_the_active_model():
    rec = _rec(status="reconstructed", made_with=SNAP_A, labels_path="x")
    same = R.build_row(ENTRY, rec, active_model=SNAP_A)
    other = R.build_row(ENTRY, rec, active_model=SNAP_B)
    label = M.model_short_label(SNAP_A)
    assert same["chips"] == ((T.ROW_MADE_BY.format(model=label), False),)
    assert other["chips"] == ((T.ROW_NOT_ACTIVE_MODEL.format(model=label),
                               True),)
    legacy = R.build_row(ENTRY, _rec(status="reconstructed"))
    assert legacy["chips"] == ((T.ROW_MADE_BY_UNKNOWN, False),)


def test_2d_chip_and_stale_axes():
    rec = _rec(status="predicted", predicted_with=SNAP_B,
               predicted_axes="ZCYX")
    row = R.build_row(ENTRY, rec, active_model=SNAP_A, cache_complete=True)
    texts = [t for t, _w in row["chips"]]
    assert T.ROW_2D_NOT_ACTIVE.format(model=M.model_short_label(SNAP_B)) \
        in texts
    assert T.ROW_2D_STALE in texts and row["cache_stale"]
    assert not R.recon_ready(dict(row, ticked=True))


def test_last_run_failed_keeps_the_status():
    row = R.build_row(ENTRY, _rec(status="reconstructed",
                                  failed_stage="reconstruct"))
    assert row["word"] == "reconstructed"
    assert T.ROW_LAST_RUN_FAILED.format(stage="Reconstruct") in \
        row["warns_narrow"]


def test_running_row():
    row = R.build_row(ENTRY, _rec(), running=("predict", " · XZ 4 / 24"))
    from napari_epicycle.ui_shell.theme import ROW_SEGMENTING
    assert row["word"] == ROW_SEGMENTING and row["colour"] == "#777777"
    assert row["icon"] == "mdi6.timer-sand"
    assert row["shape_text"].endswith(" · XZ 4 / 24")


def test_the_status_table_copy_matches_the_theme():
    from napari_epicycle.ui_shell import theme as th
    for s in ("unprocessed", "predicted", "reconstructed", "qc_extracted",
              "finalized", "failed"):
        assert R._STATUS_COLORS[s] == th.STACK_STATUS_COLORS[s]
        assert R._STATUS_ICONS[s] == th.STACK_STATUS_ICONS[s]
        assert R._STATUS_DOTS[s] == th.STACK_STATUS_DOTS[s]
        assert R._STATUS_WORDS[s] == th.STACK_STATUS_WORDS[s]


def test_summaries():
    rows = [R.build_row(dict(ENTRY, name=f"s{i}.tif"),
                        _rec(spacing=None if i == 2 else _rec()["spacing"]))
            for i in range(3)]
    main, amber = R.session_summary(rows)
    assert main == T.STACKS_COUNT.format(n="3", k="2")
    assert amber == T.STACKS_IN_VOXELS.format(m="1")
    assert R.session_summary([]) == (T.STACKS_NONE, "")
    dim, amber = R.ticked_line(rows, ["s0.tif", "s2.tif"])
    assert dim == T.SEG_TICKED_LINE.format(n=2)
    assert amber == T.SEG_TICKED_VOXELS.format(stack="s2.tif")


def test_axes_resolved_from_the_header(tmp_path):
    p = write_stack(tmp_path / "a.tif", channels=2)
    info = M.resolve_stack_axes(p)
    assert info == {"axes": "CZYX", "shape": [2, 6, 24, 24], "n_channels": 2,
                    "axes_how": "metadata", "dtype": "uint8"}
    p1 = write_stack(tmp_path / "b.tif", channels=1)
    assert M.resolve_stack_axes(p1)["axes"] == "ZYX"
    assert M.resolve_stack_axes(tmp_path / "missing.tif") is None
