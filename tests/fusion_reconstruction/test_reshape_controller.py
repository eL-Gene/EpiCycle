"""Controller half of the guided Reshape family - Brick R1.

Covers the six controller-side additions:

1. the ``_reshape_running`` lock flag (unified lock OR + patch-switch guard),
2. ``_on_annotator_reshape_lock``,
3. ``_on_annotator_reshape_context``,
4. ``_on_annotator_reshape_committed`` (df rows + provenance + sidecar),
5. ``"reshape"`` in ``_MASK_MUTATING_ACTIONS``,
6. the ``guide_`` sweep branch in ``_clear_curation_layers``.

Qt-free: the controller is built with ``object.__new__`` and only the
attributes the methods under test actually touch are bound - the convention
used throughout ``tests/fusion_reconstruction/`` (see
``test_step3_unified_lock.py`` / ``test_extraction_provenance.py``).
"""

from __future__ import annotations

import contextlib

import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction import _reshape_flags
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class _W:
    def __init__(self):
        self.enabled = True

    def setEnabled(self, v):
        self.enabled = v


class _UI:
    pass


class _FakeShell:
    def __init__(self):
        self.nav_calls = []

    def lock_step_nav(self, locked):
        self.nav_calls.append(locked)


class _FakeStack:
    """Stand-in for ``_curation_undo_stack`` - records pushed entries."""

    def __init__(self):
        self.entries = []

    def push(self, entry):
        self.entries.append(entry)

    def peek(self):
        return self.entries[-1] if self.entries else None

    def __len__(self):
        return len(self.entries)


class _FakeVisibleEvent:
    @contextlib.contextmanager
    def blocker(self):
        yield


class _FakeEvents:
    def __init__(self):
        self.visible = _FakeVisibleEvent()


class _FakeLayer:
    def __init__(self, name):
        self.name = name
        self.visible = True
        self.events = _FakeEvents()


class _FakeLayerList(list):
    def remove(self, layer):
        list.remove(self, layer)


class _FakeViewer:
    def __init__(self, layers):
        self.layers = _FakeLayerList(layers)

    def __bool__(self):
        return True


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------

def _make_lock_controller():
    c = object.__new__(FusionReconstructionController)
    ui = _UI()
    ui._curation_bucket = _W()
    ui.predict_selected_btn = _W()
    ui.reconstruct_selected_btn = _W()
    c.ui = ui
    c._get_open_annotator_widget = lambda: None
    shell = _FakeShell()
    c._workflow_shell = shell
    c._curation_busy = False
    c._patch_load_in_flight = False
    c._prediction_running = False
    c._reconstruction_running = False
    c._reshape_running = False
    return c, shell


def _base_df():
    """A minimal QC features df with one active label (7) carrying an
    inherited column the reshape must preserve on its child rows."""
    return pd.DataFrame(
        {
            "label": [7],
            "qc_tier": ["review"],
            "feature_state": ["fresh"],
            "label_status": ["active"],
            "area": [1000],
            "centroid-0": [10.0],
            "centroid-1": [20.0],
            "centroid-2": [30.0],
            "sphericity": [0.42],  # inherited column
        }
    )


def _make_commit_controller(tmp_path, df=None, ds_factor=1, patch="p01.tif"):
    c = object.__new__(FusionReconstructionController)
    c.working_dir = str(tmp_path)
    c._qc_view_patch_name = patch
    c._qc_features_df = _base_df() if df is None else df
    c._curation_undo_stack = _FakeStack()
    c._curation_dirty = False
    c._tiers_dirty = False
    c._curation_dirty_epoch = 0
    c._run_epoch = 3
    c._qc_settings = {"spacing": None, "spacing_values": (2.0, 0.5, 0.5)}
    c._current_downsample_factor = lambda *_a, **_k: ds_factor
    c.logs = []
    c._append_log = c.logs.append
    c._refresh_curation_ui = lambda: None
    return c


def _active(df, label):
    rows = df[(df["label"] == label) & (df["label_status"] == "active")]
    assert len(rows) == 1, f"expected one active row for {label}, got {len(rows)}"
    return rows.iloc[0]


# ---------------------------------------------------------------------------
# 1. The lock flag
# ---------------------------------------------------------------------------

def test_reshape_flag_locks_the_step3_surface():
    """Setting _reshape_running must lock the curate widgets + step-nav
    through the unified state-driven lock, and release on clear."""
    c, shell = _make_lock_controller()
    c._reshape_running = True
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is True
    assert c.ui._curation_bucket.enabled is False
    assert c.ui.predict_selected_btn.enabled is False

    c._reshape_running = False
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False
    assert c.ui._curation_bucket.enabled is True


def test_reshape_flag_appears_in_the_lock_diagnostic(capsys):
    c, _ = _make_lock_controller()
    c._reshape_running = True
    c._refresh_step3_lock()
    out = capsys.readouterr().out
    assert "[LOAD-LOCK] step3 lock=True" in out
    assert "reshape=True" in out, (
        "the [LOAD-LOCK] diagnostic must report the reshape flag so a stuck "
        "lock is attributable"
    )


def test_reshape_flag_blocks_a_patch_switch():
    """A staged Reshape holds the target punched out of the volume - the
    patch-switch guard must refuse the switch (both entry points)."""
    c = object.__new__(FusionReconstructionController)
    c._patch_load_in_flight = False
    c._curation_busy = False
    c._prediction_running = False
    c._reconstruction_running = False
    c._reshape_running = True
    c._qc_view_patch_name = "other_patch"
    appended = []
    c._append_log = appended.append

    # Nothing else is bound on the bare instance: if the guard fails to
    # early-return, the switch raises AttributeError and the test fails.
    c._switch_to_curation_patch("some_patch")

    assert any("[LOAD-LOCK]" in m for m in appended)


def test_reshape_flag_absent_does_not_lock():
    c, shell = _make_lock_controller()
    c._refresh_step3_lock()
    assert shell.nav_calls[-1] is False


# ---------------------------------------------------------------------------
# 2. _on_annotator_reshape_lock
# ---------------------------------------------------------------------------

def test_reshape_lock_callback_sets_flag_and_refreshes():
    c = object.__new__(FusionReconstructionController)
    calls = []
    c._refresh_step3_lock = lambda: calls.append(True)

    c._on_annotator_reshape_lock(True)
    assert c._reshape_running is True
    c._on_annotator_reshape_lock(False)
    assert c._reshape_running is False
    assert len(calls) == 2


def test_reshape_lock_callback_never_raises_when_refresh_raises(capsys):
    c = object.__new__(FusionReconstructionController)

    def _boom():
        raise RuntimeError("no ui")

    c._refresh_step3_lock = _boom
    c._on_annotator_reshape_lock(True)  # must not raise
    assert c._reshape_running is True
    assert "[LOAD-LOCK] reshape lock refresh failed" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 3. _on_annotator_reshape_context
# ---------------------------------------------------------------------------

_CONTEXT_KEYS = {
    "workspace", "patch_name", "run_id", "run_epoch",
    "downsample_factor", "spacing",
}


def test_reshape_context_returns_all_six_keys(tmp_path):
    c = _make_commit_controller(tmp_path, ds_factor=2)
    ctx = c._on_annotator_reshape_context()
    assert set(ctx) == _CONTEXT_KEYS
    assert ctx["workspace"] == str(tmp_path)
    assert ctx["patch_name"] == "p01.tif"
    assert ctx["run_id"] is None  # default (flat) run
    assert ctx["run_epoch"] == 3
    assert ctx["downsample_factor"] == 2
    # spacing is None in _qc_settings -> falls back to spacing_values.
    assert ctx["spacing"] == (2.0, 0.5, 0.5)


def test_reshape_context_prefers_explicit_spacing(tmp_path):
    c = _make_commit_controller(tmp_path)
    c._qc_settings["spacing"] = (1.5, 0.25, 0.25)
    assert c._on_annotator_reshape_context()["spacing"] == (1.5, 0.25, 0.25)


def test_reshape_context_never_raises_on_half_initialised_controller():
    """A bare instance has no working_dir / patch / settings / state_manager -
    the provider must degrade to safe defaults, not raise."""
    c = object.__new__(FusionReconstructionController)
    ctx = c._on_annotator_reshape_context()
    assert set(ctx) == _CONTEXT_KEYS
    assert ctx["workspace"] == ""
    assert ctx["patch_name"] == ""
    assert ctx["run_id"] is None
    assert ctx["run_epoch"] == 0
    assert ctx["downsample_factor"] == 1
    assert ctx["spacing"] == (1.0, 1.0, 1.0)


def test_reshape_context_survives_garbage_spacing(tmp_path):
    c = _make_commit_controller(tmp_path)
    c._qc_settings["spacing"] = "not a tuple"
    c._qc_settings["spacing_values"] = None
    assert c._on_annotator_reshape_context()["spacing"] == (1.0, 1.0, 1.0)


# ---------------------------------------------------------------------------
# 5. _MASK_MUTATING_ACTIONS
# ---------------------------------------------------------------------------

def test_reshape_is_a_mask_mutating_action():
    assert "reshape" in FusionReconstructionController._MASK_MUTATING_ACTIONS


def test_reshape_commit_sets_curation_dirty_not_tiers_dirty(tmp_path):
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(
        7, [11, 12],
        piece_stats=[
            {"id": 11, "voxels": 400, "centroid_zyx": (1.0, 2.0, 3.0)},
            {"id": 12, "voxels": 600, "centroid_zyx": (4.0, 5.0, 6.0)},
        ],
    )
    assert c._curation_dirty is True, (
        "a reshape is a MASK edit - it must set _curation_dirty"
    )
    assert c._tiers_dirty is False, (
        "a reshape must NOT be misfiled as a tier-only edit (the 2026-06-09 "
        "discard/discard_garbage bug class)"
    )


def test_reshape_commit_marks_original_features_stale(tmp_path):
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(7, [11], piece_stats=[
        {"id": 11, "voxels": 400, "centroid_zyx": (1.0, 2.0, 3.0)},
    ])
    df = c._qc_features_df
    orig = df[df["label"] == 7].iloc[0]
    assert orig["feature_state"] == "stale"


# ---------------------------------------------------------------------------
# 4. The df update
# ---------------------------------------------------------------------------

def test_reshape_commit_supersedes_original_when_superseded_true(tmp_path):
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(
        7, [11, 12], superseded=True,
        piece_stats=[
            {"id": 11, "voxels": 400, "centroid_zyx": (1.0, 2.0, 3.0)},
            {"id": 12, "voxels": 600, "centroid_zyx": (4.0, 5.0, 6.0)},
        ],
    )
    df = c._qc_features_df
    assert df[df["label"] == 7].iloc[0]["label_status"] == "superseded"
    for new_id in (11, 12):
        row = _active(df, new_id)
        assert row["qc_tier"] == "curated"
        assert row["feature_state"] == "stale"
        assert row["label_status"] == "active"


def test_reshape_commit_keeps_original_active_when_superseded_false(tmp_path):
    """The user unticked a piece (its voxels went back to the original cell),
    or the arbiter kept the cell whole - the original row must stay ACTIVE."""
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(
        7, [11], superseded=False,
        piece_stats=[{"id": 11, "voxels": 400, "centroid_zyx": (1.0, 2.0, 3.0)}],
    )
    df = c._qc_features_df
    assert df[df["label"] == 7].iloc[0]["label_status"] == "active"


def test_reshape_commit_inherits_columns_from_the_original_row(tmp_path):
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(7, [11], piece_stats=[
        {"id": 11, "voxels": 400, "centroid_zyx": (1.0, 2.0, 3.0)},
    ])
    row = _active(c._qc_features_df, 11)
    assert row["sphericity"] == 0.42, (
        "new rows must be seeded from the original row so inherited columns "
        "survive the reshape"
    )
    assert row["area"] == 400


def test_reshape_commit_centroids_at_factor_1(tmp_path):
    c = _make_commit_controller(tmp_path, ds_factor=1)
    c._on_annotator_reshape_committed(7, [11], piece_stats=[
        {"id": 11, "voxels": 400, "centroid_zyx": (5.0, 6.0, 7.0)},
    ])
    row = _active(c._qc_features_df, 11)
    assert (row["centroid-0"], row["centroid-1"], row["centroid-2"]) == (
        5.0, 6.0, 7.0
    )


def test_reshape_commit_scales_centroids_by_downsample_factor(tmp_path):
    """The engine's centroids are in LOADED-volume voxels; the df stores
    FULL-RES coords.  Missing this multiply puts every new label at half its
    true position for the navigator auto-pan and the mesh export - silently."""
    c = _make_commit_controller(tmp_path, ds_factor=2)
    c._on_annotator_reshape_committed(7, [11], piece_stats=[
        {"id": 11, "voxels": 400, "centroid_zyx": (5.0, 6.0, 7.0)},
    ])
    row = _active(c._qc_features_df, 11)
    assert (row["centroid-0"], row["centroid-1"], row["centroid-2"]) == (
        10.0, 12.0, 14.0
    )


def test_reshape_commit_dup_prevent_drops_colliding_row(tmp_path, capsys):
    """A piece id that collides with an existing ACTIVE row must be
    de-duplicated with keep='last' (the fresh row wins) and logged."""
    df = _base_df()
    df = pd.concat([df, pd.DataFrame([{
        "label": 11, "qc_tier": "bad", "feature_state": "fresh",
        "label_status": "active", "area": 1, "centroid-0": 0.0,
        "centroid-1": 0.0, "centroid-2": 0.0, "sphericity": 0.1,
    }])], ignore_index=True)
    c = _make_commit_controller(tmp_path, df=df)

    c._on_annotator_reshape_committed(7, [11], piece_stats=[
        {"id": 11, "voxels": 400, "centroid_zyx": (1.0, 2.0, 3.0)},
    ])

    out = c._qc_features_df
    rows = out[(out["label"] == 11) & (out["label_status"] == "active")]
    assert len(rows) == 1
    assert rows.iloc[0]["area"] == 400  # the fresh row won
    assert "[DUP-PREVENT] _on_annotator_reshape_committed" in capsys.readouterr().out


def test_reshape_commit_pushes_exactly_one_undo_entry_with_provenance(tmp_path):
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(
        7, [11, 12],
        roi_slices=(slice(0, 2), slice(0, 2), slice(0, 2)),
        old_roi_data="OLD",
        flags={11: "low-evidence"},
        guide_run="nuclei_run",
        preset="guided_split",
        piece_stats=[
            {"id": 11, "voxels": 400, "centroid_zyx": (1.0, 2.0, 3.0)},
            {"id": 12, "voxels": 600, "centroid_zyx": (4.0, 5.0, 6.0)},
        ],
    )
    assert len(c._curation_undo_stack.entries) == 1
    entry = c._curation_undo_stack.entries[0]
    assert entry.action_type == "reshape"
    assert entry.label_id == 7  # representative label = the target
    assert entry.old_mask_roi == "OLD"
    assert entry.extra["new_ids"] == [11, 12]
    assert entry.extra["flags"] == {11: "low-evidence"}
    assert entry.extra["guide_run"] == "nuclei_run"
    assert entry.extra["preset"] == "guided_split"
    assert entry.extra["superseded"] is True
    # The pre-mutation snapshot is captured by _commit_curation_action.
    assert entry.old_df_snapshot is not None
    assert 11 not in set(entry.old_df_snapshot["label"])


# ---------------------------------------------------------------------------
# The "dividing" no-op commit
# ---------------------------------------------------------------------------

def test_dividing_verdict_still_commits(tmp_path):
    """new_ids == [] is the arbiter's most important answer and a NO-OP write.
    It must STILL route through _commit_curation_action: an undo entry is
    pushed, the flag lands on the original row, and the sidecar is written."""
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(
        7, [], superseded=False, flags={7: "dividing"},
        roi_slices=(slice(0, 2),), old_roi_data="OLD",
    )

    assert len(c._curation_undo_stack.entries) == 1, (
        "a dividing verdict must still push an undo entry"
    )
    entry = c._curation_undo_stack.entries[0]
    assert entry.action_type == "reshape"
    assert entry.extra["new_ids"] == []
    assert entry.extra["superseded"] is False

    df = c._qc_features_df
    orig = df[df["label"] == 7].iloc[0]
    assert orig["label_status"] == "active", (
        "a dividing verdict changes nothing in the volume - the cell survives"
    )
    assert orig[_reshape_flags.DEFAULT_FLAG_COLUMN] == "dividing"

    stored = _reshape_flags.read_flags(str(tmp_path), "p01")
    assert stored == {7: "dividing"}


def test_dividing_verdict_sets_curation_dirty(tmp_path):
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(
        7, [], superseded=False, flags={7: "dividing"},
    )
    assert c._curation_dirty is True
    assert c._tiers_dirty is False


# ---------------------------------------------------------------------------
# reshape_flag: df column + sidecar + survives re-extraction
# ---------------------------------------------------------------------------

def test_reshape_flag_written_to_new_rows_and_sidecar(tmp_path):
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(
        7, [11, 12], flags={11: "split-override"},
        piece_stats=[
            {"id": 11, "voxels": 400, "centroid_zyx": (1.0, 2.0, 3.0)},
            {"id": 12, "voxels": 600, "centroid_zyx": (4.0, 5.0, 6.0)},
        ],
    )
    col = _reshape_flags.DEFAULT_FLAG_COLUMN
    df = c._qc_features_df
    assert _active(df, 11)[col] == "split-override"
    assert _active(df, 12)[col] == ""  # never NaN
    assert _reshape_flags.read_flags(str(tmp_path), "p01") == {
        11: "split-override"
    }


def test_reshape_flag_survives_a_simulated_re_extraction(tmp_path):
    """A QC re-extraction preserves only qc_tier; every other
    curation-authored column is dropped.  The sidecar is what puts the flag
    back ."""
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(
        7, [11], flags={11: "low-evidence"},
        piece_stats=[{"id": 11, "voxels": 400, "centroid_zyx": (1.0, 2.0, 3.0)}],
    )

    # A freshly extracted df: no reshape_flag column at all.
    fresh = pd.DataFrame({"label": [11, 12], "area": [400, 600]})
    assert _reshape_flags.DEFAULT_FLAG_COLUMN not in fresh.columns

    stored = _reshape_flags.read_flags(str(tmp_path), "p01")
    restored = _reshape_flags.apply_flags_to_df(fresh, stored)
    col = _reshape_flags.DEFAULT_FLAG_COLUMN
    assert restored.loc[restored["label"] == 11, col].iloc[0] == "low-evidence"
    assert restored.loc[restored["label"] == 12, col].iloc[0] == ""


def test_reshape_flags_merge_across_two_accepts(tmp_path):
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(7, [], superseded=False,
                                      flags={7: "dividing"})
    c._on_annotator_reshape_committed(7, [11], flags={11: "split-override"},
                                      piece_stats=[{"id": 11, "voxels": 5,
                                                    "centroid_zyx": (0, 0, 0)}])
    assert _reshape_flags.read_flags(str(tmp_path), "p01") == {
        7: "dividing", 11: "split-override",
    }


def test_sidecar_write_failure_does_not_abort_the_commit(
    tmp_path, monkeypatch, capsys
):
    def _boom(*_a, **_k):
        raise OSError("disk on fire")

    monkeypatch.setattr(_reshape_flags, "merge_flags", _boom)

    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(
        7, [11], flags={11: "dividing"},
        piece_stats=[{"id": 11, "voxels": 400, "centroid_zyx": (1.0, 2.0, 3.0)}],
    )

    assert "[RESHAPE] flag sidecar write FAILED" in capsys.readouterr().out
    assert len(c._curation_undo_stack.entries) == 1, (
        "a sidecar failure must not cost the user their edit"
    )
    assert c._curation_dirty is True
    assert _active(c._qc_features_df, 11)[
        _reshape_flags.DEFAULT_FLAG_COLUMN
    ] == "dividing"


def test_commit_with_no_dataframe_still_pushes_undo(tmp_path):
    """A patch with no QC features loaded must still get its undo entry."""
    c = _make_commit_controller(tmp_path)
    c._qc_features_df = None
    c._scan_qc_df_for_duplicates = lambda _ctx: None
    c._on_annotator_reshape_committed(7, [], superseded=False,
                                      flags={7: "dividing"})
    assert len(c._curation_undo_stack.entries) == 1


# ---------------------------------------------------------------------------
# 6. The guide_ sweep in _clear_curation_layers
# ---------------------------------------------------------------------------

def _make_sweep_controller(layers, qc_tracked=()):
    c = object.__new__(FusionReconstructionController)
    c.viewer = _FakeViewer(layers)
    c._qc_view_layers = list(qc_tracked)
    return c


def test_clear_curation_layers_removes_guide_layers():
    guide = _FakeLayer("guide_nuclei")
    keep = _FakeLayer("some_image")
    c = _make_sweep_controller([guide, keep])
    c._clear_curation_layers()
    names = [l.name for l in c.viewer.layers]
    assert "guide_nuclei" not in names, (
        "a surviving guide_* layer becomes a foreign layer in the next "
        "patch's annotator session"
    )
    assert "some_image" in names


def test_clear_curation_layers_hides_guide_layer_before_removal():
    guide = _FakeLayer("guide_nuclei")
    c = _make_sweep_controller([guide])
    c._clear_curation_layers()
    assert guide.visible is False, (
        "hide-before-remove (freed-VBO rule)"
    )


def test_clear_curation_layers_still_removes_curation_3d_layers():
    cur = _FakeLayer("curation_3d_p01")
    guide = _FakeLayer("guide_nuclei")
    c = _make_sweep_controller([cur, guide])
    c._clear_curation_layers()
    assert list(c.viewer.layers) == []


def test_clear_curation_layers_still_skips_qc_tracked_stack_layers():
    tracked = _FakeLayer("stack_p01")
    untracked = _FakeLayer("stack_other")
    guide = _FakeLayer("guide_nuclei")
    c = _make_sweep_controller([tracked, untracked, guide],
                               qc_tracked=[tracked])
    c._clear_curation_layers()
    names = [l.name for l in c.viewer.layers]
    assert names == ["stack_p01"], (
        "QC-tracked stack_ layers are cleaned by _clear_qc_view_layers, not here"
    )


# ---------------------------------------------------------------------------
# 7. Launch wiring
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "method_name",
    [
        "_on_annotator_reshape_lock",
        "_on_annotator_reshape_committed",
        "_on_annotator_reshape_context",
    ],
)
def test_reshape_callbacks_exist(method_name):
    assert hasattr(FusionReconstructionController, method_name)


def test_reshape_callbacks_are_wired_at_both_launch_sites():
    """The three callbacks must be passed at launch AND re-wired on every
    patch switch (the set_triage_callbacks site)."""
    import inspect

    from napari_epicycle.fusion_reconstruction import controller as _mod

    module_src = inspect.getsource(_mod)
    for kw in (
        "on_reshape_lock=self._on_annotator_reshape_lock",
        "on_reshape_committed=self._on_annotator_reshape_committed",
        "on_reshape_context=self._on_annotator_reshape_context",
    ):
        assert module_src.count(kw) == 2, (
            f"{kw} must be wired at BOTH the launch site and the "
            f"set_triage_callbacks re-wire site (found "
            f"{module_src.count(kw)})"
        )




# ---------------------------------------------------------------------------
# Defensive input handling
# ---------------------------------------------------------------------------

def test_commit_ignores_a_bad_original_id(tmp_path, capsys):
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed("not-an-int", [11])
    assert len(c._curation_undo_stack.entries) == 0
    assert "[RESHAPE] commit ignored" in capsys.readouterr().out


def test_commit_tolerates_missing_piece_stats(tmp_path):
    """No stats supplied - the row is still registered (area/centroid keep the
    inherited values) rather than the commit failing."""
    c = _make_commit_controller(tmp_path)
    c._on_annotator_reshape_committed(7, [11])
    row = _active(c._qc_features_df, 11)
    assert row["qc_tier"] == "curated"
    assert row["feature_state"] == "stale"


# --------------------------------------------------------------------------- #
# The self-lock deadlock, found on the dev PC 2026-09-08.
# --------------------------------------------------------------------------- #
class _FakeAnnotatorLock:
    def __init__(self):
        self.locked = None

    def set_load_locked(self, locked):
        self.locked = bool(locked)


class TestReshapeDoesNotLockTheAnnotatorAgainstItself:
    """A guided split raises `_reshape_running` BEFORE it punches the target
    out of the volume.  `_accept_split` refuses outright while the annotator is
    load-locked, so if that flag reached `set_load_locked` the tool could never
    commit its own preview -- the cell would sit punched out with no way
    forward.  Measured live before the exclusion existed.
    """

    def _ctrl(self, **flags):
        c, _shell = _make_lock_controller()
        c.ui.save_curation_btn = _W()
        ann = _FakeAnnotatorLock()
        c._get_open_annotator_widget = lambda: ann
        for k, v in flags.items():
            setattr(c, k, v)
        return c, ann

    def test_reshape_alone_does_NOT_lock_the_annotator(self):
        c, ann = self._ctrl(_reshape_running=True)
        c._refresh_step3_lock()
        assert ann.locked is False, (
            "the annotator must stay usable for the op it raised itself"
        )

    def test_reshape_alone_still_locks_the_REST_of_step_3(self):
        c, ann = self._ctrl(_reshape_running=True)
        c._refresh_step3_lock()
        btn = getattr(c.ui, "save_curation_btn", None)
        if btn is not None:
            assert btn.enabled is False, (
                "Save must NOT be available while the cell is punched out"
            )

    def test_a_COMPETING_op_still_locks_the_annotator(self):
        for other in ("_save_in_flight", "_commit_triage_in_flight",
                      "_batch_refine_running", "_patch_load_in_flight",
                      "_add_mode_running", "_extraction_running"):
            c, ann = self._ctrl(_reshape_running=True, **{other: True})
            c._refresh_step3_lock()
            assert ann.locked is True, (
                "{} must still lock the annotator during a reshape".format(other)
            )

    def test_nothing_running_leaves_the_annotator_unlocked(self):
        c, ann = self._ctrl()
        c._refresh_step3_lock()
        assert ann.locked is False
