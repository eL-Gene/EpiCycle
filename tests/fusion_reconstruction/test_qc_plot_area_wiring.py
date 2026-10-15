"""The plot area's lifecycle and its selection half (brick P2): the
controller's side of the QC plot area.

A fake-UI controller (``object.__new__`` + recording fakes, the pattern of
``test_tier_drag_deferral.py`` / ``test_plot_selection_live.py``): the
source pushed per door, the reset per lifecycle door, the enables after a
lock pass, the tier-drag snapshot on release only, the load's
``set_colour_by``, the Assign / Undo assign receipts, the status kinds.
"""

from __future__ import annotations

import inspect
import types

import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction import _qc_plot_text as T
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)


class _W:
    """A widget fake: enabled state + calls."""

    def __init__(self, text=""):
        self._enabled = True
        self._text = text
        self.calls = []

    def setEnabled(self, on):  # noqa: N802
        self._enabled = bool(on)

    def isEnabled(self):  # noqa: N802
        return self._enabled

    def setText(self, t):  # noqa: N802
        self._text = t

    def text(self):
        return self._text

    def currentText(self):  # noqa: N802
        return self._text

    def __getattr__(self, name):
        def f(*a, **k):
            self.calls.append((name, a, k))
        return f


class _Area:
    def __init__(self):
        self.pushes = []
        self.resets = []
        self.locked = []
        self.enabled = []
        self.frames = []
        self.messages = []

    def selection_changed(self, source):
        self.pushes.append(source)

    def reset(self, sentence_plot, sentence_see):
        self.resets.append((sentence_plot, sentence_see))
        self.frames.append(None)

    def set_locked(self, locked):
        self.locked.append(bool(locked))

    def set_enabled_state(self, has_df, has_sel, can_undo):
        self.enabled.append((has_df, has_sel, can_undo))

    def set_dataframe(self, df):
        self.frames.append(df)

    def set_card_message(self, text, kind="info"):
        self.messages.append((text, kind))

    def dataframe(self):
        return self.frames[-1] if self.frames else None


class _Plot:
    _layout_mode = "workbench"

    def __init__(self, selected=()):
        self.selected = list(selected)
        self.status_calls = []
        self.colour_calls = []
        self.marks = []
        self.redraws = []
        self._df = None
        self._stats_label = _W()

    def selected_label_ids(self):
        return list(self.selected)

    def get_selected_df_indices(self):
        return list(range(len(self.selected)))

    def set_status(self, text, kind="info"):
        self.status_calls.append((text, kind))

    def set_colour_by(self, name, *, colours=None, scale=None):
        self.colour_calls.append((name, colours, scale))
        return True

    def refresh_marks(self, df):
        self.marks.append(df)

    def _redraw(self, **k):
        self.redraws.append(k)

    def highlight_labels(self, ids):
        self.selected = list(ids)

    def clear(self):
        pass

    def reset_tier_lines(self):
        pass

    def set_features(self, df, **k):
        self._df = df


def _df(n=6):
    return pd.DataFrame({
        "label": list(range(1, n + 1)),
        "qc_score": [i / n for i in range(n)],
        "qc_tier": ["good"] * n,
        "label_status": ["active"] * n,
    })


def _ctrl(*, df=None, selected=()):
    c = object.__new__(C)
    area = _Area()
    pw = _Plot(selected)
    ui = types.SimpleNamespace(
        qc_plot_area=area, qc_plot_widget=pw,
        tier_assign_combo=_W("review"), assign_tier_btn=_W(),
        undo_tier_btn=_W(), commit_triage_btn=_W(),
        qc_summary_label=_W(),
    )
    c.ui = ui
    c._qc_features_df = df
    c._tier_undo_stack = []
    c._committing_triage = False
    c._patch_tier_thresholds = {}
    c._qc_view_patch_name = "p1.tif"
    c._isolated_label_ids = set()
    c._selection_sync = types.SimpleNamespace(dispatch=lambda sel: None)
    c._notify_plot_selection = lambda: None
    c._selected_curation_patch = lambda: "p1.tif"
    c._selected_patch_name = lambda: "p1.tif"
    c._append_log = lambda *a, **k: None
    c._update_qc_summary = lambda *a, **k: None
    c._update_flagged_labels_list = lambda *a, **k: None
    c._push_to_triage_histogram = lambda *a, **k: None
    c._refresh_save_button_state = lambda: None
    c._set_curation_status = lambda t: ui.curation_status_label.setText(t)
    ui.curation_status_label = _W("Viewing: p1")
    ui._triage_histogram_panel = _W()
    if df is not None:
        area.set_dataframe(df)          # what a load leaves behind
    return c, area, pw, ui


# ---------------------------------------------------------------------------
# The source per door
# ---------------------------------------------------------------------------

def test_a_plot_selection_pushes_plot():
    c, area, pw, _ = _ctrl(df=_df(), selected=[1, 2])
    c._on_qc_plot_label_selected([1, 2])
    assert area.pushes == ["plot"]


def test_an_empty_plot_selection_pushes_plot_too():
    c, area, *_ = _ctrl(df=_df())
    c._on_qc_plot_label_selected([])
    assert area.pushes == ["plot"]          # the area resolves it as a clear


def test_a_gate_release_pushes_gate():
    c, area, *_ = _ctrl(df=_df(), selected=[3])
    c._on_gated_drag_finished({3})
    assert area.pushes == ["gate"]


def test_a_navigator_echo_pushes_echo_and_an_empty_one_clear():
    c, area, pw, _ = _ctrl(df=_df())
    c._apply_plot_highlight_adapter(types.SimpleNamespace(label_ids=[4, 5]))
    assert pw.selected == [4, 5]
    c._apply_plot_highlight_adapter(types.SimpleNamespace(label_ids=[]))
    assert area.pushes == ["echo", "clear"]


def test_a_push_re_derives_the_assign_gate():
    c, area, pw, ui = _ctrl(df=_df(), selected=[1, 2])
    ui.assign_tier_btn.setEnabled(False)
    c._on_qc_plot_label_selected([1, 2])
    assert ui.assign_tier_btn.isEnabled()
    pw.selected = []
    c._on_qc_plot_label_selected([])
    assert not ui.assign_tier_btn.isEnabled()


def test_a_failing_area_never_breaks_the_selection_pipeline():
    c, area, *_ = _ctrl(df=_df(), selected=[1])

    def boom(source):
        raise RuntimeError("x")

    area.selection_changed = boom
    c._on_qc_plot_label_selected([1])        # no exception


# ---------------------------------------------------------------------------
# The reset per lifecycle door
# ---------------------------------------------------------------------------

def test_the_patch_switch_start_says_loading():
    c, area, pw, ui = _ctrl(df=_df())
    order = []
    for flag in ("_patch_load_in_flight", "_curation_busy",
                 "_prediction_running", "_reconstruction_running",
                 "_refinement_running", "_extraction_running",
                 "_commit_triage_in_flight", "_save_in_flight",
                 "_batch_refine_running", "_extract_all_running",
                 "_add_mode_running", "_reshape_running", "_refine_running"):
        setattr(c, flag, False)
    c._qc_view_patch_name = "old.tif"
    c._guard_unsaved_changes = lambda: True
    c._hard_clear_curation_surfaces = lambda: order.append("hard_clear")
    c._select_curation_bucket_item = lambda p: order.append("bucket")
    c._get_open_annotator_widget = lambda: None
    c.layer_lifecycle_manager = None
    c._clear_curation_layers = lambda: None
    c._clear_qc_view_layers = lambda: None
    c._load_patch_for_qc_view = lambda p: True
    real = c._reset_qc_plot_area

    def spy(a, b):
        order.append("reset")
        real(a, b)

    c._reset_qc_plot_area = spy
    c._switch_to_curation_patch("Wound_patch_1.tiff")
    assert order[:3] == ["hard_clear", "reset", "bucket"]
    assert area.resets == [(T.PLOT_NO_PATCH, "Loading Wound_patch_1…")]


def test_a_patch_without_features_resets_and_clears_the_undo(tmp_path):
    c, area, pw, ui = _ctrl(df=_df())
    c.working_dir = str(tmp_path)
    c.state_manager = None
    c._tier_undo_stack = [pd.Series(["good"])]
    c._isolated_label_ids = {1, 2}
    c._reload_current_patch_features()
    assert c._qc_features_df is None
    assert area.resets == [(T.PLOT_NO_FEATURES, T.SEE_NO_FEATURES)]
    assert c._tier_undo_stack == []
    assert c._isolated_label_ids == set()
    assert not ui.assign_tier_btn.isEnabled()
    assert not ui.tier_assign_combo.isEnabled()
    assert not ui.commit_triage_btn.isEnabled()


def test_the_stale_qc_delete_resets_the_area(tmp_path):
    c, area, *_ = _ctrl(df=_df())
    c.working_dir = str(tmp_path)
    c.state_manager = types.SimpleNamespace(_state={"patches": {}},
                                            schedule_save=lambda: None)
    c._cleanup_stale_qc("p1.tif")
    assert c._qc_features_df is None
    # the patch in view lost its features (P2 audit LOW-11)
    assert area.resets == [(T.PLOT_NO_FEATURES, T.SEE_NO_FEATURES)]


@pytest.mark.parametrize("method", ["_switch_active_run",
                                    "_adopt_workspace_quiet"])
def test_the_run_switch_and_browse_empty_the_area(method):
    # Both run long, Qt- and disk-bound paths (Browse and the shared
    # workspace both adopt through _adopt_workspace_quiet); the reset sits right after
    # the workspace-cache reset (the driver checks the result live: "RUN
    # switch empties the plot", "BROWSE empties the plot").
    src = inspect.getsource(getattr(C, method))
    i = src.index("self._reset_workspace_scoped_caches()")
    j = src.index('_area_call(self, "_reset_qc_plot_area"', i)
    assert "_plot_text.SEE_NO_PATCH" in src[j:j + 200]
    assert "blank_patch=True" in src[j:j + 200]
    assert src.count('"_reset_qc_plot_area"') == 1
    if "self._qc_features_df = None" in src:
        # HIGH-3: the reset comes AFTER the table is gone
        assert src.index("self._qc_features_df = None", i) < j


def test_reset_without_an_area_still_clears_the_plot():
    c, area, pw, ui = _ctrl(df=_df())
    del ui.qc_plot_area
    calls = []
    pw.clear = lambda: calls.append("clear")
    c._reset_qc_plot_area(T.PLOT_NO_PATCH, T.SEE_NO_PATCH)
    assert calls == ["clear"]


# ---------------------------------------------------------------------------
# The lock and the honest gates
# ---------------------------------------------------------------------------

def _lock_ctrl(**kw):
    c, area, pw, ui = _ctrl(**kw)
    c._get_open_annotator_widget = lambda: None
    c._workflow_shell = None
    for name in C._CURATION_LOCK_WIDGETS:
        if not hasattr(ui, name):
            setattr(ui, name, _W())
    return c, area, pw, ui


def test_a_lock_pass_greys_the_row_and_locks_the_area():
    c, area, pw, ui = _lock_ctrl(df=_df(), selected=[1])
    c._curation_busy = True
    c._refresh_step3_lock()
    assert area.locked[-1] is True
    for name in ("tier_assign_combo", "assign_tier_btn", "undo_tier_btn",
                 "commit_triage_btn"):
        assert not getattr(ui, name).isEnabled(), name


def test_the_unlock_pass_does_not_force_enable_the_row():
    # proven live first: with a df, a selection and an undo entry all four
    # come back on an unlock pass
    c, area, pw, ui = _lock_ctrl(df=_df(), selected=[1])
    c._tier_undo_stack = [pd.Series(["good"])]
    c._curation_busy = False
    c._refresh_step3_lock()
    assert area.locked[-1] is False
    assert all(getattr(ui, n).isEnabled() for n in (
        "tier_assign_combo", "assign_tier_btn", "undo_tier_btn",
        "commit_triage_btn"))
    # no df: the bulk loop's force-enable is taken back (the gates read the
    # table the AREA holds, P2 audit HIGH-3)
    c._qc_features_df = None
    area.set_dataframe(None)
    c._refresh_step3_lock()
    for name in ("tier_assign_combo", "assign_tier_btn", "undo_tier_btn",
                 "commit_triage_btn"):
        assert not getattr(ui, name).isEnabled(), name
    # a df, nothing selected, nothing to undo
    c._qc_features_df = _df()
    area.set_dataframe(c._qc_features_df)
    c._tier_undo_stack = []
    pw.selected = []
    c._refresh_step3_lock()
    assert ui.tier_assign_combo.isEnabled()
    assert ui.commit_triage_btn.isEnabled()
    assert not ui.assign_tier_btn.isEnabled()
    assert not ui.undo_tier_btn.isEnabled()
    assert area.enabled[-1] == (True, False, False)


def test_the_gates_between_passes_follow_the_last_lock():
    c, area, pw, ui = _lock_ctrl(df=_df(), selected=[1])
    c._curation_busy = True
    c._refresh_step3_lock()
    c._derive_tier_action_enables()          # e.g. a selection mid-op
    assert not ui.assign_tier_btn.isEnabled()


# ---------------------------------------------------------------------------
# The tier-line drag: one undo entry per RELEASE
# ---------------------------------------------------------------------------

def test_a_drag_pushes_one_snapshot_on_release_only():
    df = _df()
    c, area, pw, ui = _ctrl(df=df)
    before = df["qc_tier"].copy()
    for t in (0.1, 0.2, 0.3):
        c._on_tier_thresholds_changed((t, t + 0.2, t + 0.4))
    assert c._tier_undo_stack == []            # nothing per tick
    assert not df["qc_tier"].equals(before)
    c._on_tier_drag_finished((0.3, 0.5, 0.7))
    assert len(c._tier_undo_stack) == 1
    assert c._tier_undo_stack[0].equals(before)
    assert area.frames, "the card is refreshed on the release"
    c._on_tier_drag_finished((0.3, 0.5, 0.7))  # a release with no drag
    assert len(c._tier_undo_stack) == 1


def test_a_drag_that_changes_nothing_pushes_nothing():
    df = _df()
    df["qc_tier"] = "curated"                   # curated rows never move
    c, *_ = _ctrl(df=df)
    c._on_tier_thresholds_changed((0.1, 0.2, 0.3))
    c._on_tier_drag_finished((0.1, 0.2, 0.3))
    assert c._tier_undo_stack == []


def test_undo_assign_takes_the_drag_back():
    df = _df()
    c, area, pw, ui = _ctrl(df=df)
    before = df["qc_tier"].copy()
    assert "p1.tif" not in c._patch_tier_thresholds
    c._on_tier_thresholds_changed((0.1, 0.3, 0.5))
    c._on_tier_drag_finished((0.1, 0.3, 0.5))
    assert c._patch_tier_thresholds["p1.tif"] == (0.1, 0.3, 0.5)
    c._undo_tier_assignment()
    assert c._qc_features_df["qc_tier"].equals(before)
    # the LINES come back too: the dragged thresholds are forgotten (a
    # switch back would otherwise re-apply them) and the panel re-derives
    # its lines from qc_tier (P2 audit MED-7)
    assert "p1.tif" not in c._patch_tier_thresholds
    assert any(name == "reset_tier_lines"
               for name, _, _ in ui._triage_histogram_panel.calls)


def test_undo_assign_restores_earlier_thresholds():
    df = _df()
    c, area, pw, ui = _ctrl(df=df)
    c._patch_tier_thresholds["p1.tif"] = (0.2, 0.4, 0.6)
    c._on_tier_thresholds_changed((0.1, 0.3, 0.5))
    c._on_tier_drag_finished((0.1, 0.3, 0.5))
    c._undo_tier_assignment()
    assert c._patch_tier_thresholds["p1.tif"] == (0.2, 0.4, 0.6)


def test_undo_of_an_assignment_leaves_the_thresholds_alone():
    df = _df()
    c, area, pw, ui = _ctrl(df=df, selected=[1])
    c._tiers_dirty = False
    c._patch_tier_thresholds["p1.tif"] = (0.2, 0.4, 0.6)
    c._assign_selected_to_tier()
    c._undo_tier_assignment()
    assert c._patch_tier_thresholds["p1.tif"] == (0.2, 0.4, 0.6)
    assert not any(name == "reset_tier_lines"
                   for name, _, _ in ui._triage_histogram_panel.calls)


def test_no_drag_tick_while_commit_triage_persists():
    df = _df()
    c, *_ = _ctrl(df=df)
    c._commit_triage_in_flight = True
    before = df["qc_tier"].copy()
    c._on_tier_thresholds_changed((0.1, 0.2, 0.3))
    assert df["qc_tier"].equals(before)


# ---------------------------------------------------------------------------
# Assign / Undo assign: receipts, refresh_marks, the card
# ---------------------------------------------------------------------------

def test_assign_writes_its_receipt_and_keeps_the_view():
    df = _df()
    c, area, pw, ui = _ctrl(df=df, selected=[1, 2])
    c._tiers_dirty = False
    c._assign_selected_to_tier()
    assert list(df["qc_tier"][:2]) == ["review", "review"]
    assert pw.marks and pw.marks[-1] is df     # refresh_marks, no _redraw
    assert pw.redraws == []
    assert pw.status_calls[-1] == (
        T.STATUS_ASSIGNED.format(n="2", tier="review"), "info")
    assert len(c._tier_undo_stack) == 1
    assert area.frames[-1] is df
    assert ui.undo_tier_btn.isEnabled()


def test_assign_of_one_label_uses_the_singular():
    c, area, pw, ui = _ctrl(df=_df(), selected=[1])
    c._tiers_dirty = False
    c._assign_selected_to_tier()
    assert pw.status_calls[-1][0] == T.STATUS_ASSIGNED_ONE.format(tier="review")


def test_a_refused_assign_warns_on_the_card_and_leaves_nothing_to_undo():
    df = _df()
    df["feature_state"] = "recomputing"
    c, area, pw, ui = _ctrl(df=df, selected=[1, 2])
    c._assign_selected_to_tier()
    assert c._tier_undo_stack == []
    assert area.messages[-1] == (
        T.CARD_WARN_RECOMPUTING.format(n="2"), "warn")


def test_undo_assign_writes_its_receipt():
    df = _df()
    c, area, pw, ui = _ctrl(df=df, selected=[1, 2])
    c._tiers_dirty = False
    c._assign_selected_to_tier()
    c._undo_tier_assignment()
    assert pw.status_calls[-1] == (T.STATUS_UNDONE, "info")
    assert list(df["qc_tier"]) == ["good"] * len(df)
    assert not ui.undo_tier_btn.isEnabled()


# ---------------------------------------------------------------------------
# The load: through set_colour_by
# ---------------------------------------------------------------------------

def test_the_load_colours_by_qc_score_coolwarm_rank(tmp_path):
    from napari_epicycle.fusion_reconstruction import _paths

    c, area, pw, ui = _ctrl(df=None)
    c.working_dir = str(tmp_path)
    c.state_manager = None
    qc = _paths.qc_dir(str(tmp_path), run_id=None)
    qc.mkdir(parents=True, exist_ok=True)
    _df().to_csv(qc / "p1_features.csv", index=False)
    pw.plot_type_combo = _W()
    for name in ("_scan_qc_df_for_duplicates", "_check_csv_volume_consistency",
                 "_auto_assign_qc_tiers", "_populate_threshold_feature_combo",
                 "_check_qc_extraction_staleness",
                 "_maybe_show_qc_staleness_notification"):
        setattr(c, name, lambda *a, **k: None)
    c._reload_current_patch_features()
    assert pw.colour_calls == [("qc_score", "Coolwarm", "Rank")]
    assert area.frames and area.frames[-1] is c._qc_features_df
    assert ui.tier_assign_combo.isEnabled()
    assert ui.commit_triage_btn.isEnabled()


# ---------------------------------------------------------------------------
# The status slot
# ---------------------------------------------------------------------------

def test_set_plot_status_passes_the_kind():
    c, area, pw, ui = _ctrl()
    c._set_plot_status("x", "warn")
    c._set_plot_status("y")
    assert pw.status_calls == [("x", "warn"), ("y", "info")]


def test_set_plot_status_on_a_classic_plot_writes_its_label():
    c, area, pw, ui = _ctrl()
    pw._layout_mode = "classic"
    c._set_plot_status("z")
    assert pw.status_calls == []
    assert pw._stats_label.text() == "z"


# ---------------------------------------------------------------------------
# The P2 audits (2026-10-06)
# ---------------------------------------------------------------------------

def test_a_reset_drops_the_tier_undo_so_it_cannot_hit_the_next_patch():
    # HIGH-1: drag on A, a door, then Undo assign must not write A's tiers
    # over the next table
    c, area, pw, ui = _ctrl(df=_df())
    c._on_tier_thresholds_changed((0.1, 0.3, 0.5))
    c._on_tier_drag_finished((0.1, 0.3, 0.5))
    assert len(c._tier_undo_stack) == 1
    c._reset_qc_plot_area(T.PLOT_NO_PATCH, T.SEE_LOADING.format(patch="b"))
    assert c._tier_undo_stack == []
    assert not ui.undo_tier_btn.isEnabled()
    b = _df(9)
    c._qc_features_df = b
    area.set_dataframe(b)
    c._derive_tier_action_enables()
    assert not ui.undo_tier_btn.isEnabled()
    c._undo_tier_assignment()                   # nothing to undo
    assert list(b["qc_tier"]) == ["good"] * 9


def test_the_patch_switch_drops_the_tier_undo():
    c, area, pw, ui = _ctrl(df=_df())
    c._on_tier_thresholds_changed((0.1, 0.3, 0.5))
    c._on_tier_drag_finished((0.1, 0.3, 0.5))
    for flag in ("_patch_load_in_flight", "_curation_busy",
                 "_prediction_running", "_reconstruction_running",
                 "_refinement_running", "_extraction_running",
                 "_commit_triage_in_flight", "_save_in_flight",
                 "_batch_refine_running", "_extract_all_running",
                 "_add_mode_running", "_reshape_running", "_refine_running"):
        setattr(c, flag, False)
    c._qc_view_patch_name = "old.tif"
    c._guard_unsaved_changes = lambda: True
    c._hard_clear_curation_surfaces = lambda: None
    c._select_curation_bucket_item = lambda p: None
    c._get_open_annotator_widget = lambda: None
    c.layer_lifecycle_manager = None
    c._clear_curation_layers = lambda: None
    c._clear_qc_view_layers = lambda: None
    c._load_patch_for_qc_view = lambda p: True
    c._switch_to_curation_patch("B.tiff")
    assert c._tier_undo_stack == []
    assert not ui.undo_tier_btn.isEnabled()


def test_undo_refuses_a_snapshot_of_another_table():
    c, area, pw, ui = _ctrl(df=_df(6))
    c._tier_undo_stack = [pd.Series(["bad"] * 9)]       # another table
    c._undo_tier_assignment()
    assert list(c._qc_features_df["qc_tier"]) == ["good"] * 6
    assert area.messages[-1] == (T.CARD_WARN_UNDO_STALE, "warn")
    assert c._tier_undo_stack == []
    assert not ui.undo_tier_btn.isEnabled()


def test_the_gates_read_the_areas_table_not_the_controllers():
    # HIGH-3: Browse leaves the controller's df; the area says no patch
    c, area, pw, ui = _ctrl(df=_df(), selected=[1])
    c._derive_tier_action_enables()
    assert ui.commit_triage_btn.isEnabled()      # proven live first
    area.set_dataframe(None)
    c._derive_tier_action_enables()
    assert not ui.commit_triage_btn.isEnabled()
    assert not ui.tier_assign_combo.isEnabled()
    assert not ui.assign_tier_btn.isEnabled()


def test_a_blank_patch_reset_blanks_the_summary_and_the_status():
    c, area, pw, ui = _ctrl(df=_df())
    ui.qc_summary_label.setText("842 total · 842 curated")
    c._reset_qc_plot_area(T.PLOT_NO_PATCH, T.SEE_NO_PATCH, blank_patch=True)
    assert ui.qc_summary_label.text() == T.QC_SUMMARY_NONE
    assert ui.curation_status_label.text() == T.CURATION_STATUS_NONE
    assert not ui.commit_triage_btn.isEnabled()


def test_a_plain_reset_leaves_the_summary():
    c, area, pw, ui = _ctrl(df=_df())
    ui.qc_summary_label.setText("842 total")
    c._reset_qc_plot_area(T.PLOT_NO_PATCH, T.SEE_LOADING.format(patch="x"))
    assert ui.qc_summary_label.text() == "842 total"


def test_an_empty_gate_push_is_a_clear():
    c, area, *_ = _ctrl(df=_df())
    c._on_gated_drag_finished(set())
    assert area.pushes == ["clear"]


def test_assign_undo_and_a_lock_drop_a_pending_drag_snapshot():
    c, area, pw, ui = _ctrl(df=_df(), selected=[1])
    c._tiers_dirty = False
    c._on_tier_thresholds_changed((0.1, 0.3, 0.5))      # a release never came
    assert c._tier_drag_snapshot is not None
    c._assign_selected_to_tier()
    assert c._tier_drag_snapshot is None
    c._on_tier_thresholds_changed((0.1, 0.3, 0.5))
    c._undo_tier_assignment()
    assert c._tier_drag_snapshot is None
    c._on_tier_thresholds_changed((0.2, 0.3, 0.5))
    c._get_open_annotator_widget = lambda: None
    c._workflow_shell = None
    c._curation_busy = True
    c._refresh_step3_lock()
    assert c._tier_drag_snapshot is None


def test_the_stale_qc_delete_of_another_patch_leaves_the_area(tmp_path):
    c, area, *_ = _ctrl(df=_df())
    c.working_dir = str(tmp_path)
    c.state_manager = types.SimpleNamespace(_state={"patches": {}},
                                            schedule_save=lambda: None)
    c._cleanup_stale_qc("other.tif")
    assert area.resets == []


def test_a_receipt_goes_through_the_area_when_it_has_one():
    c, area, pw, ui = _ctrl(df=_df(), selected=[1, 2])
    got = []
    area.set_receipt = got.append
    c._tiers_dirty = False
    c._assign_selected_to_tier()
    assert got == [T.STATUS_ASSIGNED.format(n="2", tier="review")]
    c._undo_tier_assignment()
    assert got[-1] == T.STATUS_UNDONE


def test_area_call_raises_on_a_name_the_class_lacks():
    from napari_epicycle.fusion_reconstruction.controller import _area_call

    c, *_ = _ctrl(df=_df())
    with pytest.raises(AttributeError):
        _area_call(c, "_refresh_plot_kard")
    fake = types.SimpleNamespace()
    assert _area_call(fake, "_refresh_plot_card") is None   # a harness gap


def test_area_call_prints_and_continues_on_a_failure_inside(capsys):
    from napari_epicycle.fusion_reconstruction.controller import _area_call

    c, *_ = _ctrl(df=_df())

    def boom():
        raise RuntimeError("x")

    c._refresh_plot_card = boom
    assert _area_call(c, "_refresh_plot_card") is None
    assert "[QC-AREA] _refresh_plot_card failed" in capsys.readouterr().out


def test_every_area_call_names_a_method_of_the_controller():
    import ast

    from napari_epicycle.fusion_reconstruction import controller as M

    tree = ast.parse(inspect.getsource(M))
    names = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_area_call" and len(node.args) >= 2
                and isinstance(node.args[1], ast.Constant)):
            names.add(node.args[1].value)
    assert len(names) >= 10, names
    missing = [n for n in sorted(names) if not callable(getattr(C, n, None))]
    assert missing == [], missing
