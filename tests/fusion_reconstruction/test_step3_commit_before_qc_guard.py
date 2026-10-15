"""R1 — a curation edit must COMMIT even when the patch has no QC features.

The bug (ranked R1): every annotator mask-edit callback on
``FusionReconstructionController`` opened with ::

    if self._qc_features_df is None:
        return

*before* ``_commit_curation_action(...)``.  ``_commit_curation_action`` is the
only place that sets ``_curation_dirty = True`` and pushes the
``CurationUndoEntry``.  So on a patch where QC features were never extracted,
the annotator had ALREADY mutated the shared volume, the callback returned
early, no dirty flag was set, ``_guard_unsaved_changes`` never prompted, and
the next patch switch dropped the layer through ``_clear_curation_layers``.
The edit was gone with no warning — silent data loss.

The rule these tests lock in: **the dirty flag, the undo entry and the log
line are set BEFORE any optional-data early return.**  The DataFrame row
update stays optional and is skipped with a ``[CURATION-COMMIT]``
diagnostic when there is no df.  ``_on_annotator_reshape_committed`` already
had the correct shape (its df check lives inside its ``_update`` closure) and
is covered here as the reference case.

Qt-free: the controller is built with ``object.__new__`` and only the
attributes the methods under test actually touch are bound — the convention
used throughout ``tests/fusion_reconstruction/`` (see
``test_reshape_controller.py``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _FakeStack:
    """Stand-in for ``_curation_undo_stack`` — records pushed entries."""

    def __init__(self):
        self.entries = []

    def push(self, entry):
        self.entries.append(entry)

    def peek(self):
        return self.entries[-1] if self.entries else None

    def __len__(self):
        return len(self.entries)


class _FakeLabelsLayer:
    def __init__(self, data):
        self.data = data


def _base_df():
    """A minimal QC features df with two active labels."""
    return pd.DataFrame(
        {
            "label": [7, 8],
            "qc_tier": ["review", "review"],
            "feature_state": ["fresh", "fresh"],
            "label_status": ["active", "active"],
            "area": [1000, 900],
            "centroid-0": [10.0, 11.0],
            "centroid-1": [20.0, 21.0],
            "centroid-2": [30.0, 31.0],
        }
    )


def _make_controller(df=None):
    """A bare controller carrying only what the commit handlers read.

    ``df=None`` is the bug case: a patch with no QC features extracted.
    """
    c = object.__new__(FusionReconstructionController)
    c._qc_features_df = df
    c._curation_undo_stack = _FakeStack()
    c._curation_dirty = False
    c._tiers_dirty = False
    c._curation_dirty_epoch = 0
    c.logs = []
    c._append_log = c.logs.append
    c._refresh_curation_ui = lambda: None
    c._current_downsample_factor = lambda *_a, **_k: 1
    # A small volume carrying labels 7, 8 and the new ids the paint / split /
    # add handlers look for.
    vol = np.zeros((4, 4, 4), dtype=np.int32)
    vol[0, 0, 0] = 7
    vol[0, 0, 1] = 8
    vol[1, 1, 1] = 21
    vol[1, 1, 2] = 22
    vol[2, 2, 2] = 31
    c._find_active_labels_layer = lambda: _FakeLabelsLayer(vol)
    # Real method, but it does a full-volume scan; the guard-ordering
    # behaviour under test does not depend on it.
    c._basic_stats_calls = []
    c._update_label_basic_stats = (
        lambda lid, **kw: c._basic_stats_calls.append(lid)
    )
    c._on_annotator_reshape_context = lambda: {}
    return c


# The nine mask-edit callbacks that carried the guard, as
# (id, callable-name, args, kwargs, expected action_type).
ROI = (slice(0, 2), slice(0, 2), slice(0, 2))
OLD_ROI = np.zeros((2, 2, 2), dtype=np.int32)

HANDLERS = [
    ("delete", "_on_annotator_label_deleted", (7,),
     {"roi_slices": ROI, "old_roi_data": OLD_ROI}, "delete"),
    ("discard_garbage", "_on_annotator_discard_garbage", ([7, 8],),
     {"roi_slices": ROI, "old_roi_data": OLD_ROI}, "discard_garbage"),
    ("interpolation", "_on_annotator_interpolation_accepted", (7,),
     {"roi_slices": ROI, "old_roi_data": OLD_ROI}, "interpolation"),
    ("new_label_painted", "_on_annotator_new_label_painted", (31,),
     {}, "new_label_painted"),
    ("split", "_on_annotator_label_split", (7, [21, 22]),
     {"roi_slices": ROI, "old_roi_data": OLD_ROI}, "split"),
    ("smooth", "_on_annotator_label_smoothed", (7,),
     {"roi_slices": ROI, "old_roi_data": OLD_ROI}, "smooth"),
    ("add_denovo", "_on_annotator_add_committed", ([21, 22],),
     {"roi_slices": ROI, "old_roi_data": OLD_ROI}, "add_denovo"),
    ("merge", "_on_annotator_label_merged", (7, [8]),
     {"roi_slices": ROI, "old_roi_data": OLD_ROI}, "merge"),
    ("batch_refine", "_on_annotator_batch_refined", ([7, 8],),
     {"roi_slices": ROI, "old_roi_data": OLD_ROI}, "batch_refine"),
]

HANDLER_IDS = [h[0] for h in HANDLERS]


# ---------------------------------------------------------------------------
# 1. The bug: no QC features must NOT swallow the edit
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,method,args,kwargs,action_type", HANDLERS, ids=HANDLER_IDS
)
def test_commit_happens_without_qc_features(
    name, method, args, kwargs, action_type
):
    """With ``_qc_features_df is None`` the volume has already been mutated.
    The handler MUST still route through ``_commit_curation_action`` so the
    dirty flag is set and the undo entry exists."""
    c = _make_controller(df=None)

    getattr(c, method)(*args, **kwargs)

    assert c._curation_dirty is True, (
        f"{method} left _curation_dirty False on a patch with no QC "
        f"features — _guard_unsaved_changes will not prompt and the next "
        f"patch switch drops the edit (silent data loss)"
    )
    assert len(c._curation_undo_stack) == 1, (
        f"{method} pushed no curation-undo entry — the edit cannot be undone"
    )
    entry = c._curation_undo_stack.peek()
    assert entry.action_type == action_type
    assert entry.old_df_snapshot is None, (
        "there is no df to snapshot; the undo entry must carry None, not raise"
    )


@pytest.mark.parametrize(
    "name,method,args,kwargs,action_type", HANDLERS, ids=HANDLER_IDS
)
def test_no_qc_features_prints_the_skip_diagnostic(
    name, method, args, kwargs, action_type, capsys
):
    """The skipped df work must be VISIBLE in the console, not silent."""
    c = _make_controller(df=None)

    getattr(c, method)(*args, **kwargs)

    out = capsys.readouterr().out
    assert "[CURATION-COMMIT]" in out, (
        f"{method} skipped the df row update with no diagnostic line"
    )
    assert action_type in out, (
        f"the [CURATION-COMMIT] line must name the action ({action_type}) so "
        f"a console trace attributes it"
    )


@pytest.mark.parametrize(
    "name,method,args,kwargs,action_type", HANDLERS, ids=HANDLER_IDS
)
def test_no_qc_features_logs_to_the_ui(
    name, method, args, kwargs, action_type
):
    """The user-facing log must also record the edit — a commit that leaves
    no trace in the log panel reads as "nothing happened"."""
    c = _make_controller(df=None)

    getattr(c, method)(*args, **kwargs)

    assert c.logs, f"{method} appended nothing to the UI log"


@pytest.mark.parametrize(
    "name,method,args,kwargs,action_type", HANDLERS, ids=HANDLER_IDS
)
def test_no_qc_features_logs_exactly_once(
    name, method, args, kwargs, action_type
):
    """LOW-2 — ONE edit, ONE log line.

    The df-less branch of ``split`` (and of ``new_label_painted``) writes its
    own "(no QC DataFrame)" note, and the handler's normal trailing line then
    fires too because the commit now goes through.  Two lines for one edit
    reads like two edits in the log panel."""
    c = _make_controller(df=None)

    getattr(c, method)(*args, **kwargs)

    assert len(c.logs) == 1, (
        f"{method} wrote {len(c.logs)} log lines for one edit: {c.logs}"
    )


@pytest.mark.parametrize(
    "name,method,args,kwargs,action_type", HANDLERS, ids=HANDLER_IDS
)
def test_with_qc_features_never_double_logs(
    name, method, args, kwargs, action_type
):
    """The df path's logging is unchanged — never two lines for one edit.

    ``<= 1`` rather than ``== 1`` because ``_on_annotator_add_committed``
    logs NOTHING on the df path: it has no trailing ``_append_log`` at all,
    only the "(no QC DataFrame)" note.  That is pre-existing on ``main`` and
    out of scope for this fix (it loses no data — the rows and the undo
    entry are both there); noted so a future pass can give de-novo Add its
    "Added N new label(s)" line."""
    c = _make_controller(df=_base_df())

    getattr(c, method)(*args, **kwargs)

    assert len(c.logs) <= 1, (
        f"{method} wrote {len(c.logs)} log lines for one edit: {c.logs}"
    )


def test_new_label_painted_does_not_claim_a_df_registration():
    """MED-1 — the trailing log line says the label was "registered in
    DataFrame".  On a patch with no DataFrame that is simply false, and it
    is the line the user would read to decide whether the paint landed."""
    c = _make_controller(df=None)

    c._on_annotator_new_label_painted(31)

    joined = " | ".join(c.logs)
    assert "registered in DataFrame" not in joined, (
        f"claimed a DataFrame registration with no DataFrame: {c.logs}"
    )
    assert "no QC DataFrame" in joined, (
        "the honest note must survive — the user still needs to know why "
        "the label is missing from the navigator"
    )


def test_new_label_painted_still_claims_registration_with_a_df():
    c = _make_controller(df=_base_df())
    c._on_annotator_new_label_painted(31)
    assert "registered in DataFrame" in " | ".join(c.logs)


def test_split_log_says_the_df_was_skipped():
    """The single split line must still carry the "no QC DataFrame" fact —
    merging the two lines must not drop the information."""
    c = _make_controller(df=None)
    c._on_annotator_label_split(7, [21, 22], roi_slices=ROI, old_roi_data=OLD_ROI)
    assert "no QC DataFrame" in c.logs[0]
    assert "21" in c.logs[0] and "22" in c.logs[0]


# ---------------------------------------------------------------------------
# 2. The df path is unchanged — the row update still runs when there IS a df
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "name,method,args,kwargs,action_type", HANDLERS, ids=HANDLER_IDS
)
def test_commit_still_happens_with_qc_features(
    name, method, args, kwargs, action_type
):
    c = _make_controller(df=_base_df())

    getattr(c, method)(*args, **kwargs)

    assert c._curation_dirty is True
    assert len(c._curation_undo_stack) == 1
    entry = c._curation_undo_stack.peek()
    assert entry.action_type == action_type
    assert entry.old_df_snapshot is not None, (
        "with a df present the undo entry must carry the pre-edit snapshot"
    )


def test_delete_still_ghosts_the_row():
    c = _make_controller(df=_base_df())
    c._on_annotator_label_deleted(7, roi_slices=ROI, old_roi_data=OLD_ROI)
    row = c._qc_features_df[c._qc_features_df["label"] == 7].iloc[0]
    assert row["label_status"] == "deleted"


def test_discard_garbage_still_ghosts_the_rows():
    c = _make_controller(df=_base_df())
    c._on_annotator_discard_garbage([7, 8], roi_slices=ROI, old_roi_data=OLD_ROI)
    assert set(c._qc_features_df["label_status"]) == {"deleted"}


def test_smooth_still_curates_the_row():
    c = _make_controller(df=_base_df())
    c._on_annotator_label_smoothed(7, roi_slices=ROI, old_roi_data=OLD_ROI)
    df = c._qc_features_df
    active = df[(df["label"] == 7) & (df["label_status"] == "active")]
    assert active.iloc[0]["qc_tier"] == "curated"
    ghosts = df[(df["label"] == 7) & (df["label_status"] == "superseded")]
    assert len(ghosts) == 1, "the pre-edit row must survive as one ghost"


def test_interpolation_still_curates_the_row():
    c = _make_controller(df=_base_df())
    c._on_annotator_interpolation_accepted(
        7, roi_slices=ROI, old_roi_data=OLD_ROI
    )
    df = c._qc_features_df
    active = df[(df["label"] == 7) & (df["label_status"] == "active")]
    assert active.iloc[0]["qc_tier"] == "curated"


def test_batch_refine_still_curates_and_stales_the_rows():
    c = _make_controller(df=_base_df())
    c._on_annotator_batch_refined(
        [7, 8], roi_slices=ROI, old_roi_data=OLD_ROI
    )
    df = c._qc_features_df
    live = df[df["label_status"] == "active"]
    assert set(live["qc_tier"]) == {"curated"}
    assert set(live["feature_state"]) == {"stale"}


def test_split_still_registers_the_fragments():
    c = _make_controller(df=_base_df())
    c._on_annotator_label_split(
        7, [21, 22], roi_slices=ROI, old_roi_data=OLD_ROI
    )
    df = c._qc_features_df
    assert df[df["label"] == 7].iloc[0]["label_status"] == "superseded"
    assert set(df[df["label_status"] == "active"]["label"]) >= {21, 22}


def test_merge_still_supersedes_the_absorbed_row():
    c = _make_controller(df=_base_df())
    c._on_annotator_label_merged(7, [8], roi_slices=ROI, old_roi_data=OLD_ROI)
    df = c._qc_features_df
    assert df[df["label"] == 8].iloc[0]["label_status"] == "superseded"


def test_add_still_registers_the_new_rows():
    c = _make_controller(df=_base_df())
    c._on_annotator_add_committed(
        [21, 22], roi_slices=ROI, old_roi_data=OLD_ROI
    )
    assert set(c._qc_features_df["label"]) >= {21, 22}


def test_new_label_painted_still_registers_the_row():
    c = _make_controller(df=_base_df())
    c._on_annotator_new_label_painted(31)
    assert 31 in set(c._qc_features_df["label"])


# ---------------------------------------------------------------------------
# 3. Genuine no-op guards must SURVIVE the fix
# ---------------------------------------------------------------------------

def test_discard_garbage_with_no_ids_is_still_a_no_op():
    """Empty id list = nothing was mutated; no commit, no dirty flag."""
    c = _make_controller(df=None)
    c._on_annotator_discard_garbage([])
    assert c._curation_dirty is False
    assert len(c._curation_undo_stack) == 0


def test_add_with_no_ids_is_still_a_no_op():
    c = _make_controller(df=None)
    c._on_annotator_add_committed([])
    assert c._curation_dirty is False
    assert len(c._curation_undo_stack) == 0


def test_merge_with_no_absorbed_ids_is_still_a_no_op():
    c = _make_controller(df=None)
    c._on_annotator_label_merged(7, [])
    assert c._curation_dirty is False
    assert len(c._curation_undo_stack) == 0


def test_paint_over_an_existing_label_is_still_a_no_op():
    """The idempotence check only exists when there IS a df; a paint-over of
    an already-registered label must stay a no-op."""
    c = _make_controller(df=_base_df())
    c._on_annotator_new_label_painted(7)
    assert c._curation_dirty is False
    assert len(c._curation_undo_stack) == 0


# ---------------------------------------------------------------------------
# 3b. The REAL UI refresh must survive a df-less commit
# ---------------------------------------------------------------------------
#
# LOW-6.  Every test above stubs ``_refresh_curation_ui`` — which is exactly
# the step ``_commit_curation_action`` now reaches on a code path it never
# reached before.  These run the real one.

class _FakeButton:
    def __init__(self):
        self.enabled = None

    def setEnabled(self, value):
        self.enabled = bool(value)


class _FakeListWidget:
    def __init__(self):
        self.cleared = 0

    def clear(self):
        self.cleared += 1


class _FakeLabel:
    def __init__(self):
        self.text = None

    def setText(self, value):
        self.text = value


class _FakeUI:
    """Only what ``_refresh_curation_ui`` reaches with no DataFrame.

    Deliberately WITHOUT ``qc_plot_widget`` / ``_triage_histogram_panel`` —
    those are the Qt-heavy ones, and the df-None path must not touch them.
    """

    def __init__(self):
        self._flagged_labels_list = _FakeListWidget()
        self.save_curation_btn = _FakeButton()
        self.undo_curation_btn = _FakeButton()
        self.qc_summary_label = _FakeLabel()


class _FakeWindow:
    _dock_widgets = {}


class _FakeViewer:
    window = _FakeWindow()


def _make_live_ui_controller(df=None):
    """Same controller, but the REAL ``_refresh_curation_ui`` runs."""
    c = _make_controller(df=df)
    del c._refresh_curation_ui  # drop the stub — use the bound class method
    c.ui = _FakeUI()
    c.viewer = _FakeViewer()
    c._flagged_centroids = {}
    c._sync_debug_log = lambda *_a, **_k: None
    return c


@pytest.mark.parametrize(
    "name,method,args,kwargs,action_type", HANDLERS, ids=HANDLER_IDS
)
def test_real_refresh_curation_ui_survives_a_dfless_commit(
    name, method, args, kwargs, action_type
):
    """``_commit_curation_action`` step 5 calls ``_refresh_curation_ui``, which
    before this fix was never reached with ``_qc_features_df is None`` from a
    mask edit.  It must not raise, and it must light the Save button."""
    c = _make_live_ui_controller(df=None)

    getattr(c, method)(*args, **kwargs)

    assert c.ui.save_curation_btn.enabled is True, (
        "the Save button must light up — it is the only visible sign the "
        "edit is pending"
    )
    assert c.ui.undo_curation_btn.enabled is True
    assert c.ui._flagged_labels_list.cleared >= 1, (
        "the real _update_flagged_labels_list did not run"
    )


def test_real_refresh_curation_ui_still_works_with_a_df():
    """The contrast: with a df the real refresh takes the OTHER branch and
    writes the tier summary — the branch the df-None path must skip."""
    df = _base_df()
    # All curated, so the flagged list stays empty and no Qt item is built.
    df["qc_tier"] = "curated"
    c = _make_live_ui_controller(df=df)

    c._on_annotator_label_smoothed(7, roi_slices=ROI, old_roi_data=OLD_ROI)

    assert c.ui.save_curation_btn.enabled is True
    assert c.ui.qc_summary_label.text is not None, (
        "_update_qc_summary did not run on the df path"
    )


def test_real_refresh_leaves_the_summary_alone_without_a_df():
    c = _make_live_ui_controller(df=None)
    c._on_annotator_label_smoothed(7, roi_slices=ROI, old_roi_data=OLD_ROI)
    assert c.ui.qc_summary_label.text is None, (
        "there is nothing to summarise — the df branch must stay skipped"
    )


# ---------------------------------------------------------------------------
# 4. Reference case — reshape already had the correct shape
# ---------------------------------------------------------------------------

def test_reshape_commits_without_qc_features():
    """``_on_annotator_reshape_committed`` keeps its df check INSIDE its
    ``_update`` closure, so it already commits on a df-less patch.  This is
    the shape the other nine now share; it must not regress."""
    c = _make_controller(df=None)

    c._on_annotator_reshape_committed(
        7, [21, 22], roi_slices=ROI, old_roi_data=OLD_ROI
    )

    assert c._curation_dirty is True
    assert len(c._curation_undo_stack) == 1
    assert c._curation_undo_stack.peek().action_type == "reshape"


# ---------------------------------------------------------------------------
# 5. Tier-only marks are NOT in scope — they have nothing to commit
# ---------------------------------------------------------------------------

def test_mark_tier_still_returns_early_without_a_df():
    """A tier mark lives ONLY in the DataFrame — with no df there is nothing
    to record, and it must not raise the mask-edit dirty flag."""
    c = _make_controller(df=None)
    c._on_annotator_mark_tier(7, "good")
    assert c._curation_dirty is False
    assert c._tiers_dirty is False
    assert len(c._curation_undo_stack) == 0
