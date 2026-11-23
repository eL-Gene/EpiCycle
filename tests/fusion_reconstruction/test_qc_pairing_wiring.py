"""The pairing check wired through Step 3's controller and settings.

The Extract Features settings offer "Compare with:" only when the
patch has another run with labels.  Every patch has
its own choice: the one made this session, else the run (or "(none)") its
last extraction recorded.  At dispatch the main thread lists the runs once,
resolves the choice to a bundle (or a log line it writes once the
extraction is confirmed), records the files' sizes and mtimes, and the
floors come back through each runner (the subprocess sidecar, the
in-process table attrs, the batch signal) into the extraction-context
sidecar.  Recalculate and the edits blank the match values of the active
rows they touch and say the table's match values predate the edit; a
fresh Extract clears that, and so does reading the table again.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tifffile

from napari_epicycle.fusion_reconstruction import _qc_pairing_text as T
from napari_epicycle.fusion_reconstruction import qc_pairing_group as G
from tests.fusion_reconstruction.test_qc_pairing_group import (
    PATCH,
    _bundle,
    _cells,
    _core_df,
    _match_file,
    _workspace,
)


@pytest.fixture(scope="module")
def app():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def w(app):
    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    widget = FusionReconstructionCuration(viewer=None)
    yield widget
    timer = getattr(widget.controller.state_manager, "_save_timer", None)
    if timer is not None:
        timer.stop()
    widget.deleteLater()
    app.processEvents()


def _ctl(w, ws):
    c = w.controller
    c.working_dir = str(ws)
    c.state_manager.active_run_id = "default"
    c.state_manager.load(str(ws))
    return c


def _csv(ws):
    from napari_epicycle.fusion_reconstruction import _paths
    return _paths.qc_features_csv(str(ws), Path(PATCH).stem, run_id=None)


def _record_match(ws, slug="nuclei"):
    """The sidecar of a past extraction of PATCH with ``match_run`` set
    (``None``: "(none)" was chosen)."""
    from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
        write_extraction_context,
    )
    write_extraction_context(_csv(ws), spacing=(1.0, 1.0, 1.0),
                             downsample_factor=1,
                             pairing={"match_run": slug})


# --------------------------------------------------------------------------- #
# the picker
# --------------------------------------------------------------------------- #
def _dialog(app, runs, default=None, *, none=False, settings=None,
            patch=PATCH):
    from napari_epicycle.fusion_reconstruction.qc_extraction_dialog import (
        QCExtractionDialog,
    )
    return QCExtractionDialog(None, settings=settings, match_runs=runs,
                              match_default=default, match_none=none,
                              match_patch=patch)


def _shown(dlg) -> bool:
    return not dlg.match_combo.isHidden()


def test_the_picker_is_hidden_with_no_other_run(app):
    from qtpy.QtCore import Qt

    dlg = _dialog(app, [])
    assert not _shown(dlg) and dlg.match_label.isHidden()
    assert dlg.match_hint.isHidden()
    # hidden, it is still an input of the scrolled body, wheel-guarded
    assert dlg._body.isAncestorOf(dlg.match_combo)
    assert dlg.match_combo.focusPolicy() == Qt.StrongFocus
    assert "match_run" not in dlg.get_settings()
    assert dlg.get_match_run() is None
    dlg.deleteLater()


def test_the_picker_shows_the_only_other_run_by_default(app):
    dlg = _dialog(app, [("nuclei", "Nuclei DAPI")])
    assert _shown(dlg)
    assert dlg.match_label.text() == "Compare with:"
    assert dlg.match_combo.toolTip() == T.PICKER_TOOLTIP
    assert dlg.match_label.toolTip() == T.PICKER_TOOLTIP
    assert T.PICKER_TOOLTIP.startswith(
        "Compare with another run of this patch. ")
    assert T.PICKER_TOOLTIP.endswith(
        "and adds match columns you can plot, and a Colour by entry named "
        "after that run. Choose (none) to skip.")
    # the dim line under the row says what the pick does, no hover needed
    assert not dlg.match_hint.isHidden()
    assert dlg.match_hint.text() == T.PICKER_HINT
    assert dlg.match_hint.wordWrap()
    assert dlg._body.isAncestorOf(dlg.match_hint)
    assert T.PICKER_HINT.startswith("Finds, for each object of this run, ")
    items = [dlg.match_combo.itemText(i)
             for i in range(dlg.match_combo.count())]
    assert items == ["(none)", "Nuclei DAPI"]
    assert dlg.get_settings()["match_run"] == {PATCH: "nuclei"}
    dlg.match_combo.setCurrentIndex(0)
    assert dlg.get_settings()["match_run"] == {PATCH: None}
    dlg.deleteLater()


@pytest.mark.parametrize("delta", [120, -120])
def test_a_wheel_over_the_picker_leaves_the_compared_run(app, delta):
    from qtpy.QtCore import QPoint, QPointF, Qt
    from qtpy.QtGui import QWheelEvent
    from qtpy.QtWidgets import QApplication

    dlg = _dialog(app, [("nuclei", "Nuclei DAPI"), ("cells", "Cells")],
                  default="nuclei")
    dlg.show()
    app.processEvents()
    combo = dlg.match_combo
    assert dlg._body.isAncestorOf(combo)
    assert combo.focusPolicy() == Qt.StrongFocus
    combo.clearFocus()
    before = combo.currentIndex()
    assert before == 1
    ev = QWheelEvent(QPointF(5, 5), QPointF(5, 5), QPoint(0, 0),
                     QPoint(0, delta), Qt.NoButton, Qt.NoModifier,
                     Qt.NoScrollPhase, False)
    QApplication.sendEvent(combo, ev)
    app.processEvents()
    assert combo.currentIndex() == before
    assert dlg.get_match_run() == "nuclei"
    dlg.close()
    dlg.deleteLater()


def test_the_picker_row_is_a_form_row_capped_like_its_siblings(app):
    from qtpy.QtWidgets import QFormLayout
    dlg = _dialog(app, [("nuclei", "Nuclei DAPI")])
    form = dlg.match_combo.parentWidget().layout()
    forms = [lay for lay in dlg.findChildren(QFormLayout)
             if lay.indexOf(dlg.match_combo) >= 0]
    assert forms, form
    lay = forms[0]
    row, role = lay.getWidgetPosition(dlg.match_combo)
    assert role == QFormLayout.FieldRole
    assert lay.itemAt(row, QFormLayout.LabelRole).widget() is (
        dlg.match_label)
    assert dlg.match_combo.maximumWidth() == 220
    dlg.deleteLater()


def test_the_picker_row_leaves_the_field_column_where_it_was(app):
    """One line, never wrapped, and no wider than the other labels: the
    fields start where they start without the row."""
    def field_x(dlg):
        dlg.resize(693, 646)
        dlg.show()
        app.processEvents()
        x = dlg.erosion_iters_sb.mapTo(dlg, dlg.erosion_iters_sb.rect()
                                       .topLeft()).x()
        dlg.hide()
        return x

    with_row = _dialog(app, [("nuclei", "Nuclei DAPI")])
    without = _dialog(app, [])
    assert not with_row.match_label.wordWrap()
    assert field_x(with_row) == field_x(without)
    combo_x = with_row.match_combo.mapTo(
        with_row, with_row.match_combo.rect().topLeft()).x()
    assert combo_x == with_row.erosion_iters_sb.mapTo(
        with_row, with_row.erosion_iters_sb.rect().topLeft()).x()
    with_row.deleteLater()
    without.deleteLater()


def test_the_picker_defaults_to_the_recorded_run_if_it_lists(app):
    runs = [("nuclei", "Nuclei"), ("cells_b", "Cells B")]
    dlg = _dialog(app, runs, default="cells_b")
    assert dlg.get_match_run() == "cells_b"
    gone = _dialog(app, runs, default="archived_one")
    assert gone.get_match_run() is None   # two runs: none
    dlg.deleteLater()
    gone.deleteLater()


def test_a_vanished_recorded_run_reads_none_even_with_one_other_run(app):
    dlg = _dialog(app, [("nuclei", "Nuclei")], default="cells_old")
    assert dlg.get_match_run() is None
    dlg.deleteLater()


def test_none_chosen_shows_none_back(app):
    dlg = _dialog(app, [("nuclei", "Nuclei")], none=True)
    assert dlg.match_combo.currentIndex() == 0
    assert dlg.get_match_run() is None
    dlg.deleteLater()


def test_the_other_patches_choices_ride_through(app):
    settings = {"match_run": {"patch_009.tif": "nuclei",
                                PATCH: "cells_b"}}
    dlg = _dialog(app, [("nuclei", "Nuclei")], settings=settings)
    dlg.match_combo.setCurrentIndex(0)
    assert dlg.get_settings()["match_run"] == {
        "patch_009.tif": "nuclei", PATCH: None}
    hidden = _dialog(app, [], settings=settings)
    assert hidden.get_settings()["match_run"] == settings["match_run"]
    dlg.deleteLater()
    hidden.deleteLater()


def test_the_controller_fills_the_picker_from_the_workspace(w, tmp_path):
    c = _ctl(w, _workspace(tmp_path))
    runs, default, none = c._match_picker_state(PATCH)
    assert runs == [("nuclei", "Nuclei DAPI")]
    assert default is None and none is False
    _record_match(tmp_path / "ws")
    assert c._match_picker_state(PATCH)[1:] == ("nuclei", False)
    _record_match(tmp_path / "ws", None)
    assert c._match_picker_state(PATCH)[1:] == (None, True)
    # this session's choice for the patch wins over its record
    c._qc_settings["match_run"] = {PATCH: "nuclei"}
    assert c._match_picker_state(PATCH)[1:] == ("nuclei", False)
    c._qc_settings["match_run"] = {PATCH: None}
    assert c._match_picker_state(PATCH)[1:] == (None, True)
    c._qc_settings["match_run"] = {"patch_009.tif": None}
    assert c._match_picker_state(PATCH)[1:] == (None, True)   # record


def test_one_run_only_hides_the_picker_through_the_controller(w, tmp_path):
    ws = _workspace(tmp_path, nuclei_labels=False)
    c = _ctl(w, ws)
    assert c._match_picker_state(PATCH) == ([], None, False)


def test_the_run_switch_forgets_the_picks(w, tmp_path):
    c = _ctl(w, _workspace(tmp_path))
    c._qc_settings["match_run"] = {PATCH: None}
    c._match_values_predate_edits = True
    c._reset_workspace_scoped_caches()
    assert "match_run" not in c._qc_settings
    assert c._match_values_predate_edits is False


# --------------------------------------------------------------------------- #
# dispatch
# --------------------------------------------------------------------------- #
def _dispatch(w, ws, monkeypatch):
    c = _ctl(w, ws)
    lab = ws / "artifacts" / "step3" / "runs" / PATCH / "labels_3d.tif"
    c._qc_view_patch_name = PATCH
    c._curation_dirty = False
    monkeypatch.setattr(c, "_get_active_labels_path", lambda: str(lab))
    monkeypatch.setattr(c, "_get_active_source_stack_path", lambda: "")
    monkeypatch.setattr(c, "_get_active_prob_path", lambda: "")
    monkeypatch.setattr(c, "_get_active_seg_channel", lambda: 0)
    seen = []
    monkeypatch.setattr(c, "_launch_qc_extract_subprocess",
                        lambda kw: seen.append(kw) or True)
    c._extract_features_current()
    return c, seen[-1], lab


def test_without_a_pick_or_a_record_no_match_is_sent(w, tmp_path,
                                                       monkeypatch):
    c, kw, lab = _dispatch(w, _workspace(tmp_path), monkeypatch)
    assert kw["match_labels"] is None
    prov = c._qc_launch_pairing[PATCH]
    assert prov["patch_file_size"] == lab.stat().st_size
    assert "match_run" not in prov
    json.dumps(kw["match_labels"])


def test_the_picked_run_is_sent_as_a_json_safe_bundle(w, tmp_path,
                                                      monkeypatch):
    ws = _workspace(tmp_path)
    w.controller._qc_settings["match_run"] = {PATCH: "nuclei"}
    c, kw, _ = _dispatch(w, ws, monkeypatch)
    b = kw["match_labels"]
    assert b["run"] == "nuclei" and b["link_floor"] == 0.20
    assert b["path"].replace("\\", "/").endswith(
        f"nuclei/reconstructions/{PATCH}/labels_3d.tif")
    json.dumps(kw)
    prov = c._qc_launch_pairing[PATCH]
    assert prov["match_run"] == "nuclei"
    assert prov["match_file"].startswith("artifacts/step3/runs/nuclei/")
    assert prov["match_file_size"] == Path(b["path"]).stat().st_size


def test_re_extract_without_the_picker_keeps_the_recorded_run(w, tmp_path,
                                                              monkeypatch):
    ws = _workspace(tmp_path)
    _record_match(ws)
    w.controller._qc_settings.pop("match_run", None)
    _, kw, _ = _dispatch(w, ws, monkeypatch)
    assert kw["match_labels"]["run"] == "nuclei"


def test_an_explicit_none_wins_over_the_record(w, tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    _record_match(ws)
    w.controller._qc_settings["match_run"] = {PATCH: None}
    c, kw, _ = _dispatch(w, ws, monkeypatch)
    assert kw["match_labels"] is None
    # "(none)" is recorded as such, so it sticks after a restart
    assert c._qc_launch_pairing[PATCH]["match_run"] is None


def test_another_patch_s_choice_does_not_apply(w, tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    _record_match(ws)
    w.controller._qc_settings["match_run"] = {"patch_009.tif": None}
    _, kw, _ = _dispatch(w, ws, monkeypatch)
    assert kw["match_labels"]["run"] == "nuclei"


def test_a_recorded_none_sticks_without_the_picker(w, tmp_path,
                                                   monkeypatch):
    ws = _workspace(tmp_path)
    _record_match(ws, None)
    w.controller._qc_settings.pop("match_run", None)
    c, kw, _ = _dispatch(w, ws, monkeypatch)
    assert kw["match_labels"] is None
    assert c._qc_launch_pairing[PATCH]["match_run"] is None


def test_a_run_without_labels_logs_and_sends_nothing(w, tmp_path,
                                                     monkeypatch):
    ws = _workspace(tmp_path, nuclei_labels=False)
    w.controller._qc_settings["match_run"] = {PATCH: "nuclei"}
    logs = []
    monkeypatch.setattr(w.controller, "_append_log", logs.append)
    _, kw, _ = _dispatch(w, ws, monkeypatch)
    assert kw["match_labels"] is None
    assert T.NO_LABELS.format(other="Nuclei DAPI") in logs
    assert sum(T.NO_LABELS.format(other="Nuclei DAPI") in m
               for m in logs) == 1


def test_the_dispatch_lists_the_runs_once(w, tmp_path, monkeypatch):
    ws = _workspace(tmp_path)
    w.controller._qc_settings["match_run"] = {PATCH: "nuclei"}
    calls = []
    real = G.read_run_listing
    monkeypatch.setattr(G, "read_run_listing",
                        lambda ws_: calls.append(ws_) or real(ws_))
    _, kw, _ = _dispatch(w, ws, monkeypatch)
    assert kw["match_labels"]["run"] == "nuclei"
    assert len(calls) == 1


def test_the_in_process_fallback_passes_the_bundle(w, tmp_path, monkeypatch):
    import napari_epicycle.fusion_reconstruction.controller as C
    made = []

    class _W:
        def __init__(self, *a, **kw):
            made.append(kw)
            raise RuntimeError("stop here")

    monkeypatch.setattr(C, "_QCFeatureWorker", _W)
    c = w.controller
    monkeypatch.setattr(c, "_busy_hide", lambda *a, **k: None)
    kwargs = dict(labels_path="l.tif", patch_name=PATCH, csv_path="c.csv",
                  spacing=None, source_stack_path="", erosion_iters=2,
                  max_radius=5, composite_method="zscore", seg_channel=0,
                  pca_features=None, prob_path="", n_pcs=5,
                  compute_umap=False, umap_min_dist=0.1,
                  umap_n_neighbors=15, compute_tsne=False,
                  tsne_perplexity=30.0, channel_configs=[],
                  exclude_border=False, invert_z=False, metric="euclidean",
                  patch_status="reconstructed",
                  match_labels={"run": "nuclei"})
    c._launch_qc_extract_inprocess(kwargs)
    assert made[-1]["match_labels"] == {"run": "nuclei"}


# --------------------------------------------------------------------------- #
# the three runners bring the floors back
# --------------------------------------------------------------------------- #
def _core_cfg(tmp_path, bundle):
    from tests.fusion_reconstruction.test_qc_pairing_group import _cfg
    return _cfg(tmp_path, "a", match_labels=bundle)


def test_the_in_process_worker_puts_the_result_on_the_table(app, tmp_path):
    import napari_epicycle.fusion_reconstruction.controller as C
    cfg = _core_cfg(tmp_path, _bundle(_match_file(tmp_path)))
    worker = C._QCFeatureWorker(
        cfg["labels_path"], "patch001", cfg["csv_path"],
        match_labels=cfg["match_labels"])
    got = []
    worker.finished.connect(lambda df, pn: got.append(df))
    worker.error.connect(lambda m: got.append(m))
    worker.run()
    df = got[-1]
    assert not isinstance(df, str), df
    assert "match_count" in df.columns
    assert df.attrs["_pairing"]["floor_active_vx"] == 20


def test_the_batch_worker_sends_the_result_with_patch_done(app, tmp_path):
    import napari_epicycle.fusion_reconstruction.controller as C
    cfg = _core_cfg(tmp_path, _bundle(_match_file(tmp_path)))
    entry = {"labels_path": cfg["labels_path"], "patch_name": "patch001",
             "csv_path": cfg["csv_path"],
             "match_labels": cfg["match_labels"]}
    plain = {"labels_path": cfg["labels_path"], "patch_name": "patch002",
             "csv_path": str(Path(cfg["csv_path"]).with_name(
                 "patch002_features.csv"))}
    worker = C._QCFeatureBatchWorker([entry, plain])
    got = []
    worker.patch_done.connect(lambda pn, nl, info: got.append((pn, nl, info)))
    worker.error.connect(lambda m: got.append(("error", m)))
    worker.run()
    assert [g[0] for g in got] == ["patch001", "patch002"]
    assert got[0][2]["pairing"]["floor_other_vx"] == 20
    assert got[1][2] == {"pairing": None}
    from napari_epicycle.fusion_reconstruction.qc_features import (
        load_patch_csv,
    )
    assert "match_count" in load_patch_csv(entry["csv_path"]).columns
    assert "match_count" not in load_patch_csv(plain["csv_path"]).columns
    assert entry["match_labels"].get(G.OUTCOME_KEY) is None   # not mutated


class _NoThread:
    """Stands in for the batch's QThread: nothing runs."""

    class _Sig:
        def connect(self, *a, **k):
            pass

    def __init__(self, *a, **k):
        self.started = self._Sig()
        self.finished = self._Sig()

    def start(self):
        pass


def _second_patch(ws, name="patch_002.tif", *, nuclei=True):
    """A second patch in both runs (the nuclei run's only when asked)."""
    from tests.fusion_reconstruction.test_qc_pairing_group import _nuclei
    step3 = ws / "artifacts" / "step3"
    d = step3 / "runs" / name
    d.mkdir(parents=True)
    tifffile.imwrite(str(d / "labels_3d.tif"), _cells())
    if nuclei:
        nd = step3 / "runs" / "nuclei" / "reconstructions" / name
        nd.mkdir(parents=True)
        tifffile.imwrite(str(nd / "labels_3d.tif"), _nuclei())
    from napari_epicycle.fusion_reconstruction import _paths
    sp = Path(_paths.state_json_path(ws))
    state = json.loads(sp.read_text(encoding="utf-8"))
    for run in ("default", "nuclei"):
        state["runs"][run]["patches"][name] = {"status": "reconstructed"}
    sp.write_text(json.dumps(state), encoding="utf-8")
    return name


def _extract_all(w, ws, monkeypatch, *, yes=True):
    import napari_epicycle.fusion_reconstruction.controller as C
    c = _ctl(w, ws)
    made = {}

    class _Worker:
        def __init__(self, configs, **kw):
            made["configs"] = configs

        def __getattr__(self, name):
            return _NoThread._Sig()

        def moveToThread(self, t):
            pass

    monkeypatch.setattr(C, "_QCFeatureBatchWorker", _Worker)
    monkeypatch.setattr(C, "QThread", _NoThread)
    root = ws / "artifacts" / "step3" / "runs"

    def payload(pn):
        return {"labels_path": str(root / pn / "labels_3d.tif"),
                "source_stack_path": "", "seg_config": {}}

    monkeypatch.setattr(c, "_resolve_patch_payload", payload)
    order = []
    monkeypatch.setattr(c, "_append_log", lambda m: order.append(m))
    from qtpy.QtWidgets import QMessageBox

    def ask(*a, **k):
        order.append("<question>")
        return QMessageBox.Yes if yes else QMessageBox.No

    monkeypatch.setattr(C.QMessageBox, "question", ask)
    c._extract_features_all()
    return c, made.get("configs"), order


def test_extract_all_sends_each_patch_its_own_match(w, tmp_path,
                                                      monkeypatch):
    ws = _workspace(tmp_path)
    other = _second_patch(ws)
    w.controller._qc_settings["match_run"] = {PATCH: "nuclei",
                                                other: None}
    c, configs, _ = _extract_all(w, ws, monkeypatch)
    by = {cfg["patch_name"]: cfg for cfg in configs}
    assert by[PATCH]["match_labels"]["run"] == "nuclei"
    assert by[other]["match_labels"] is None
    json.dumps(configs)
    # the stash the per-patch finish writes into each sidecar
    assert c._qc_launch_pairing[PATCH]["match_run"] == "nuclei"
    assert c._qc_launch_pairing[other]["match_run"] is None
    assert "match_file" in c._qc_launch_pairing[PATCH]


def test_extract_all_logs_the_notes_after_the_question(w, tmp_path,
                                                       monkeypatch):
    ws = _workspace(tmp_path)
    other = _second_patch(ws, nuclei=False)
    # an existing CSV, so the re-extract question is asked
    csv = _csv(ws)
    csv.parent.mkdir(parents=True, exist_ok=True)
    csv.write_text("label\n1\n", encoding="utf-8")
    w.controller._qc_settings["match_run"] = {other: "nuclei"}
    _, _, order = _extract_all(w, ws, monkeypatch)
    note = f"[{other}] " + T.NO_LABELS.format(other="Nuclei DAPI")
    assert note in order
    assert order.index("<question>") < order.index(note)
    # "No": nothing about the comparison is logged
    _, _, order = _extract_all(w, ws, monkeypatch, yes=False)
    assert note not in order and "<question>" in order


def test_the_batch_patch_done_takes_the_pairing_argument(w, tmp_path):
    from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
        read_extraction_context,
    )
    ws = _workspace(tmp_path)
    c = _ctl(w, ws)
    c._qc_launch_pairing = {PATCH: {"match_run": None,
                                    "patch_file_size": 1,
                                    "patch_file_mtime": 1.0}}
    c._on_qc_batch_patch_done(PATCH, 5, {"pairing": None})
    ctx = read_extraction_context(_csv(ws))
    assert "match_run" in ctx and ctx["match_run"] is None


def test_the_subprocess_sidecar_reaches_the_finish_handler(tmp_path,
                                                           monkeypatch):
    from tests.fusion_reconstruction.test_qc_extract_subprocess import (
        _FakeProc, _make_controller_skeleton,
    )
    from tests.fusion_reconstruction.test_qc_pairing_group import _cfg
    from napari_epicycle.fusion_reconstruction._qc_extract_core import (
        run_full_qc_extraction,
    )
    ctrl = _make_controller_skeleton()
    cfg = _cfg(tmp_path, "a")
    run_full_qc_extraction(**cfg)
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    side = Path(str(config_path) + ".pairing.json")
    side.write_text(json.dumps({"run": "nuclei", "floor_active_vx": 7,
                                "refused": None}), encoding="utf-8")
    ctrl._qc_proc = _FakeProc(returncode=0)
    ctrl._qc_proc_config_path = str(config_path)
    ctrl._qc_proc_patch_name = "patch001"
    ctrl._qc_proc_csv_path = cfg["csv_path"]
    calls = {}
    monkeypatch.setattr(ctrl, "_on_qc_extract_finished",
                        lambda df, pn: calls.setdefault("df", df),
                        raising=False)
    monkeypatch.setattr(ctrl, "_on_qc_csv_saved", lambda pn, cp: None,
                        raising=False)
    ctrl._on_qc_proc_done(0, "")
    assert calls["df"].attrs["_pairing"]["floor_active_vx"] == 7
    assert not side.is_file()


def test_the_batch_patch_done_writes_the_pairing_fields(w, tmp_path):
    from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
        read_extraction_context,
    )
    ws = _workspace(tmp_path)
    c = _ctl(w, ws)
    c._qc_launch_pairing = {PATCH: {
        "patch_file_size": 11, "patch_file_mtime": 1.0,
        "match_run": "nuclei", "match_run_display_name": "Nuclei DAPI",
        "match_file": "artifacts/x.tif", "match_file_mtime": 2.0,
        "match_file_size": 22, "link_floor": 0.2}}
    c._on_qc_batch_patch_done(PATCH, 5, {"pairing": {
        "link_floor": 0.2, "floor_active_vx": 33, "floor_other_vx": 4,
        "refused": None}})
    ctx = read_extraction_context(_csv(ws))
    for key, value in (("match_run", "nuclei"),
                       ("match_run_display_name", "Nuclei DAPI"),
                       ("match_file", "artifacts/x.tif"),
                       ("match_file_mtime", 2.0),
                       ("match_file_size", 22), ("link_floor", 0.2),
                       ("floor_active_vx", 33), ("floor_other_vx", 4),
                       ("patch_file_size", 11), ("patch_file_mtime", 1.0)):
        assert ctx[key] == value, key
    assert PATCH not in c._qc_launch_pairing
    # an older two-argument call still writes the sidecar it always did
    c._on_qc_batch_patch_done(PATCH, 5)
    ctx = read_extraction_context(_csv(ws))
    assert "match_run" not in ctx and "spacing" in ctx


# --------------------------------------------------------------------------- #
# Recalculate and the edit write-backs
# --------------------------------------------------------------------------- #
def _skeleton_with_table(tmp_path):
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    df = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                             match_labels=_bundle(_match_file(tmp_path)))
    df["qc_tier"] = "good"
    df["feature_state"] = "fresh"
    df["label_status"] = "active"
    c = FusionReconstructionController.__new__(FusionReconstructionController)
    c._qc_features_df = df
    c._match_values_predate_edits = False
    c.viewer = SimpleNamespace(layers=[])
    c.ui = SimpleNamespace()
    labels = _cells()
    c._find_active_labels_layer = lambda: SimpleNamespace(data=labels)
    c._current_downsample_factor = lambda *a, **k: 1
    c._update_flagged_labels_list = lambda: None
    c._update_qc_summary = lambda df: None
    c._append_log = lambda m: None
    return c


def test_recalculate_blanks_the_match_values_and_sets_the_flag(tmp_path):
    c = _skeleton_with_table(tmp_path)
    c._on_navigator_recalculate([1])
    df = c._qc_features_df
    row = df.set_index("label").loc[1]
    assert row["feature_state"] == "fresh"
    for col in ("match_count", "match_shared", "match_overlap",
                "match_minor_ratio"):
        assert np.isnan(row[col]), col
    assert row["match_id"] == 0
    assert df["match_id"].dtype == np.int64
    # the others keep their measured values
    other = df.set_index("label").loc[5]
    assert other["match_count"] == 2 and other["match_id"] == 15
    assert c._match_values_predate_edits is True
    from napari_epicycle.fusion_reconstruction.qc_pairing import (
        status_from_columns,
    )
    words = status_from_columns(df["match_count"], df["match_shared"],
                                df["match_overlap"])
    assert words[list(df["label"]).index(1)] == "not measured"


def test_recalculate_never_leaves_an_old_match_value_on_a_fresh_row(
        tmp_path):
    c = _skeleton_with_table(tmp_path)
    c._on_navigator_recalculate([1, 2, 3, 4, 5])
    df = c._qc_features_df
    fresh = df["feature_state"] == "fresh"
    assert fresh.all()
    assert df.loc[fresh, "match_count"].isna().all()
    assert (df.loc[fresh, "match_id"] == 0).all()


def test_recalculate_of_a_table_without_match_columns_adds_none(tmp_path):
    c = _skeleton_with_table(tmp_path)
    c._qc_features_df = c._qc_features_df.drop(columns=list(
        G.MATCH_COLUMNS))
    c._on_navigator_recalculate([1])
    assert not any(col.startswith("match_")
                   for col in c._qc_features_df.columns)


def test_blank_match_row_for_a_new_row():
    cols = ["label", "area", *G.MATCH_COLUMNS]
    row = {"label": 9, "match_count": 3, "match_id": 4,
           "match_overlap": 0.5}
    G.blank_match_row(row, cols)
    assert row["match_id"] == 0
    assert np.isnan(row["match_count"]) and np.isnan(row["match_overlap"])
    assert np.isnan(row["match_shared"])
    plain = {"label": 9}
    G.blank_match_row(plain, ["label", "area"])
    assert plain == {"label": 9}


def _table_ctl(tmp_path):
    """A bare controller holding a measured match table (no Qt)."""
    c = _skeleton_with_table(tmp_path)
    from packages.epicycle_annotator.curation_undo import CurationUndoStack
    c._curation_undo_stack = CurationUndoStack()
    c._curation_dirty = False
    c._qc_settings = {}
    for name in ("_refresh_curation_ui", "_refresh_save_button_state",
                 "_scan_qc_df_for_duplicates", "_note_commit_without_qc",
                 "_refresh_flagged_after_curation", "_mark_curation_dirty"):
        setattr(c, name, lambda *a, **k: None)
    return c


def _row(df, label, status="active"):
    sel = (df["label"] == label) & (df["label_status"] == status)
    return df.loc[sel].iloc[0]


@pytest.mark.parametrize("action", ["merge", "delete", "paint_stroke",
                                    "smooth", "interpolation"])
def test_a_mask_edit_blanks_the_label_s_active_row(tmp_path, action):
    c = _table_ctl(tmp_path)
    df = c._qc_features_df
    ghost = df[df["label"] == 5].copy()
    ghost["label_status"] = "superseded"
    c._qc_features_df = df = __import__("pandas").concat(
        [df, ghost], ignore_index=True)
    try:
        c._commit_curation_action(action, 5, f"{action} 5")
    except Exception:
        # what runs after the table update needs the full widget; the
        # table and the flag are what this checks
        pass
    df = c._qc_features_df
    live = _row(df, 5)
    assert np.isnan(live["match_count"]) and live["match_id"] == 0
    old = _row(df, 5, "superseded")
    assert old["match_count"] == 2 and old["match_id"] == 15
    assert _row(df, 1)["match_count"] == 1      # untouched label
    assert c._match_values_predate_edits is True


def test_a_newly_painted_label_reads_not_measured(tmp_path):
    from types import SimpleNamespace as NS
    c = _table_ctl(tmp_path)
    vol = _cells()
    vol[12:16, 2:8, 2:8] = 9
    c._find_active_labels_layer = lambda: NS(data=vol)
    try:
        c._on_annotator_new_label_painted(9)
    except Exception:
        pass
    df = c._qc_features_df
    new = _row(df, 9)
    assert np.isnan(new["match_count"]) and new["match_id"] == 0
    assert c._match_values_predate_edits is True


def test_a_tier_only_action_keeps_the_match_values(tmp_path):
    c = _table_ctl(tmp_path)
    try:
        c._commit_curation_action("mark_good", 5, "mark good")
    except Exception:
        pass
    assert _row(c._qc_features_df, 5)["match_count"] == 2
    assert c._match_values_predate_edits is False


@pytest.mark.parametrize("curated", [True, False])
def test_batch_refine_blanks_and_sets_the_flag(tmp_path, curated):
    c = _table_ctl(tmp_path)
    try:
        c._on_annotator_batch_refined([1, 5], mark_curated=curated)
    except Exception:
        pass
    df = c._qc_features_df
    for lab in (1, 5):
        assert np.isnan(_row(df, lab)["match_count"]), lab
    assert _row(df, 2)["match_count"] == 1
    if curated:
        assert _row(df, 5, "superseded")["match_count"] == 2
    assert c._match_values_predate_edits is True


def test_recalculate_leaves_ghost_rows_alone(tmp_path):
    import pandas as pd
    c = _skeleton_with_table(tmp_path)
    df = c._qc_features_df
    ghost = df[df["label"] == 5].copy()
    ghost["label_status"] = "deleted"
    c._qc_features_df = pd.concat([df, ghost], ignore_index=True)
    c._on_navigator_recalculate([5])
    df = c._qc_features_df
    assert np.isnan(_row(df, 5)["match_count"])
    assert _row(df, 5, "deleted")["match_count"] == 2


def test_a_fresh_extract_clears_the_flag(w, tmp_path):
    ws = _workspace(tmp_path)
    c = _ctl(w, ws)
    c._qc_table_patch = PATCH
    c._match_values_predate_edits = True
    c._write_qc_extraction_context(PATCH)
    assert c._match_values_predate_edits is False


def test_reading_the_table_again_resets_the_flag(w, tmp_path, monkeypatch):
    from napari_epicycle.fusion_reconstruction.qc_features import (
        save_patch_csv,
    )
    ws = _workspace(tmp_path)
    c = _ctl(w, ws)
    c._qc_view_patch_name = PATCH
    df = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                             match_labels=_bundle(_match_file(tmp_path)))
    df["label_status"] = "active"
    save_patch_csv(df, str(_csv(ws)))
    _record_match(ws)
    c._match_values_predate_edits = True
    c._reload_current_patch_features()
    assert c._match_values_predate_edits is False   # Discard's reload
    # a row blanked on disk says so after a restart
    G.blank_match_columns(df, df["label"] == 1)
    save_patch_csv(df, str(_csv(ws)))
    c._reload_current_patch_features()
    assert c._match_values_predate_edits is True
    # so does the sidecar of a save after an edit (a delete leaves no blank)
    df = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                             match_labels=_bundle(_match_file(tmp_path)))
    df["label_status"] = "active"
    save_patch_csv(df, str(_csv(ws)))
    c._reload_current_patch_features()
    assert c._match_values_predate_edits is False
    c._match_values_predate_edits = True
    c._record_match_predate(PATCH)
    from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
        read_extraction_context,
    )
    assert read_extraction_context(_csv(ws))[
        "match_values_predate_edits"] is True
    c._match_values_predate_edits = False
    c._reload_current_patch_features()
    assert c._match_values_predate_edits is True


def test_the_single_extract_records_the_floors_through_the_finish(
        w, tmp_path, monkeypatch):
    """The worker's finished signal, the real finish handler, the sidecar:
    the floors the comparison returned reach the extraction context."""
    from napari_epicycle.fusion_reconstruction._qc_extraction_context import (
        read_extraction_context,
    )
    from napari_epicycle.fusion_reconstruction.qc_features import (
        save_patch_csv,
    )
    from qtpy.QtCore import QObject, Signal

    ws = _workspace(tmp_path)
    w.controller._qc_settings["match_run"] = {PATCH: "nuclei"}
    c, kw, lab = _dispatch(w, ws, monkeypatch)
    assert kw["match_labels"]["run"] == "nuclei"
    # what the extraction wrote and handed back
    df = G.cross_run_pairing(labels=_cells(), df=_core_df(),
                             match_labels=dict(kw["match_labels"]))
    df["label_status"] = "active"
    save_patch_csv(df, str(_csv(ws)))
    df.attrs["_pairing"] = {"run": "nuclei", "display_name": "Nuclei DAPI",
                            "link_floor": 0.2, "floor_active_vx": 21,
                            "floor_other_vx": 3, "links": 6,
                            "refused": None}

    class _Fake(QObject):
        finished = Signal(object, str)

    fake = _Fake()
    fake.finished.connect(c._on_qc_extract_finished)
    fake.finished.emit(df, PATCH)
    from qtpy.QtWidgets import QApplication
    QApplication.processEvents()
    ctx = read_extraction_context(_csv(ws))
    assert ctx is not None
    assert ctx["match_run"] == "nuclei"
    assert ctx["floor_active_vx"] == 21 and ctx["floor_other_vx"] == 3
    assert ctx["link_floor"] == 0.2
    assert ctx["match_file"].startswith("artifacts/step3/runs/nuclei/")
    assert ctx["patch_file_size"] == lab.stat().st_size
    assert "match_values_predate_edits" not in ctx
    assert c._match_values_predate_edits is False


def test_the_triage_commit_never_writes_the_display_column(tmp_path):
    import pandas as pd
    from napari_epicycle.fusion_reconstruction.controller import (
        _commit_triage_persist,
    )
    df = pd.DataFrame({"label": [1, 2], "qc_tier": ["good", "bad"],
                       "label_status": ["active", "active"],
                       "match_count": [1, 0],
                       "match_status": ["one match", "no match"]})
    out = _commit_triage_persist(
        df=df, garbage_ids=set(), lifted_mask=_cells(),
        labels_path=str(tmp_path / "labels_3d.tif"),
        csv_path=tmp_path / "p_features.csv", csv_dir=tmp_path)
    assert out["ok"] is True, out
    written = pd.read_csv(tmp_path / "p_features.csv")
    assert "match_status" not in written.columns
    assert "match_count" in written.columns
    assert "match_status" in df.columns          # the table keeps it
