"""Tests for `init_spec_to_model_state_overrides` + `apply_overrides_to_run_state`
(2026-05-29, feature/step2-new-run-model-prepopulate).

When a user creates a new run via the New Run dialog and picks an init source
(browse / step3_refined / pretrained / inherited / cross_bucket / fork), the
Inference Model section's "Use:" dropdown should auto-set to reflect that
choice — currently it stays on "Auto" and the user has to manually re-pick
Imported + re-browse the same file before their first Predict.

These helpers map the init_spec to per-run state field overrides, with a
"don't clobber" rule so values the user has since adjusted aren't reset.
"""
from __future__ import annotations

from napari_epicycle.hitl_training_annotation._run_management import (
    _empty_per_run_state,
    init_spec_to_model_state_overrides,
    apply_overrides_to_run_state,
)
from packages.epicycle_io.step2_run_manifest import (
    init_spec_scratch,
    init_spec_inherited,
    init_spec_cross_bucket,
    init_spec_step3_refined,
    init_spec_pretrained,
    init_spec_browse,
    init_spec_fork,
)


# ---------------------------------------------------------------------------
# Pure mapping helper
# ---------------------------------------------------------------------------


def test_overrides_scratch_returns_empty():
    """Scratch init has no model file — no overrides."""
    assert init_spec_to_model_state_overrides(init_spec_scratch()) == {}


def test_overrides_none_returns_empty():
    assert init_spec_to_model_state_overrides(None) == {}


def test_overrides_non_dict_returns_empty():
    assert init_spec_to_model_state_overrides("not a dict") == {}
    assert init_spec_to_model_state_overrides([]) == {}


def test_overrides_unknown_mode_returns_empty():
    assert init_spec_to_model_state_overrides({"mode": "telepathy"}) == {}


def test_overrides_browse_sets_imported_path_and_checkpoint():
    spec = init_spec_browse(external_path="/tmp/foo.model")
    result = init_spec_to_model_state_overrides(spec)
    assert result["curation_model_source_mode"] == "imported"
    assert result["imported_curation_model_path"] == "/tmp/foo.model"
    assert result["starting_checkpoint"] == "/tmp/foo.model"


def test_overrides_browse_empty_path_returns_empty():
    """Defensive — browse with an empty path shouldn't produce nonsense overrides."""
    assert init_spec_to_model_state_overrides({"mode": "browse", "external_path": ""}) == {}


def test_overrides_pretrained_sets_pretrained_combo_no_checkpoint():
    spec = init_spec_pretrained(pretrained_name="nuclei")
    result = init_spec_to_model_state_overrides(spec)
    assert result["curation_model_source_mode"] == "pretrained"
    assert result["curation_pretrained_model_type"] == "nuclei"
    # pretrained is a Cellpose builtin name, not a file — no Checkpoint pre-pop.
    assert "starting_checkpoint" not in result
    assert "imported_curation_model_path" not in result


def test_overrides_inherited_uses_resolved_checkpoint():
    spec = init_spec_inherited(
        source_run_id="run_a", source_bucket="lateral", source_round_id="round_03"
    )
    result = init_spec_to_model_state_overrides(
        spec, resolved_checkpoint="/ws/run_a/lateral/round_03/weights"
    )
    assert result["curation_model_source_mode"] == "imported"
    assert result["imported_curation_model_path"] == "/ws/run_a/lateral/round_03/weights"
    assert result["starting_checkpoint"] == "/ws/run_a/lateral/round_03/weights"


def test_overrides_inherited_unresolved_returns_empty():
    """If the resolver can't find the source checkpoint, leave defaults — don't
    invent a path."""
    spec = init_spec_inherited(
        source_run_id="run_a", source_bucket="lateral", source_round_id="round_03"
    )
    assert init_spec_to_model_state_overrides(spec, resolved_checkpoint=None) == {}
    assert init_spec_to_model_state_overrides(spec, resolved_checkpoint="") == {}


def test_overrides_step3_refined_uses_resolved_checkpoint():
    spec = init_spec_step3_refined(
        source_step3_bundle="bundle_x", source_step3_bucket="axial"
    )
    result = init_spec_to_model_state_overrides(
        spec, resolved_checkpoint="/ws/artifacts/step3/refined.model"
    )
    assert result["curation_model_source_mode"] == "imported"
    assert result["imported_curation_model_path"] == "/ws/artifacts/step3/refined.model"


def test_overrides_fork_uses_resolved_checkpoint():
    spec = init_spec_fork(source_run_id="run_a", copied_pair_count=12)
    result = init_spec_to_model_state_overrides(
        spec, resolved_checkpoint="/ws/run_a/lateral/round_05/weights"
    )
    assert result["curation_model_source_mode"] == "imported"
    assert result["imported_curation_model_path"] == "/ws/run_a/lateral/round_05/weights"


def test_overrides_cross_bucket_uses_resolved_checkpoint():
    spec = init_spec_cross_bucket(source_bucket="lateral", source_round_id="round_02")
    result = init_spec_to_model_state_overrides(
        spec, resolved_checkpoint="/ws/this_run/lateral/round_02/weights"
    )
    assert result["curation_model_source_mode"] == "imported"
    assert result["imported_curation_model_path"] == "/ws/this_run/lateral/round_02/weights"


# ---------------------------------------------------------------------------
# Don't-clobber application
# ---------------------------------------------------------------------------


def test_apply_overrides_to_empty_state_writes_all_fields():
    state = _empty_per_run_state()
    apply_overrides_to_run_state(
        state,
        {
            "curation_model_source_mode": "imported",
            "imported_curation_model_path": "/foo.model",
            "starting_checkpoint": "/foo.model",
        },
    )
    assert state["curation_model_source_mode"] == "imported"
    assert state["imported_curation_model_path"] == "/foo.model"
    assert state["current_config"]["starting_checkpoint"] == "/foo.model"


def test_apply_overrides_preserves_user_set_imported_path():
    """If the user has already picked an imported path, don't clobber it."""
    state = _empty_per_run_state()
    state["imported_curation_model_path"] = "/user_picked.model"  # NOT default ""
    apply_overrides_to_run_state(
        state,
        {
            "curation_model_source_mode": "imported",
            "imported_curation_model_path": "/spec.model",
        },
    )
    # mode was at default ("auto") -> still applied
    assert state["curation_model_source_mode"] == "imported"
    # path was user-set -> preserved
    assert state["imported_curation_model_path"] == "/user_picked.model"


def test_apply_overrides_preserves_user_changed_pretrained():
    """If the user has changed the pretrained from the cyto3 default, leave it."""
    state = _empty_per_run_state()
    state["curation_pretrained_model_type"] = "cyto2"  # user changed
    apply_overrides_to_run_state(
        state,
        {
            "curation_model_source_mode": "pretrained",
            "curation_pretrained_model_type": "nuclei",
        },
    )
    assert state["curation_pretrained_model_type"] == "cyto2"


def test_apply_overrides_preserves_user_set_source_mode():
    """If the user has already chosen a source mode (anything != 'auto'),
    don't override it from the init_spec."""
    state = _empty_per_run_state()
    state["curation_model_source_mode"] = "none"  # user picked Manual/SAM only
    apply_overrides_to_run_state(
        state,
        {
            "curation_model_source_mode": "imported",
            "imported_curation_model_path": "/foo.model",
        },
    )
    # mode preserved
    assert state["curation_model_source_mode"] == "none"
    # path was at default ("") -> still applied (user can flip mode later)
    assert state["imported_curation_model_path"] == "/foo.model"


def test_apply_overrides_preserves_user_set_starting_checkpoint():
    state = _empty_per_run_state()
    state["current_config"] = {"starting_checkpoint": "/user_ckpt.model"}
    apply_overrides_to_run_state(
        state,
        {"starting_checkpoint": "/spec_ckpt.model"},
    )
    assert state["current_config"]["starting_checkpoint"] == "/user_ckpt.model"


def test_apply_overrides_with_empty_overrides_is_noop():
    state = _empty_per_run_state()
    snapshot = dict(state)
    apply_overrides_to_run_state(state, {})
    assert state == snapshot


def test_apply_overrides_ignores_unknown_keys():
    """Audit hardening (2026-05-29): unknown override keys must be silently
    IGNORED, not written through.  A future mode adding a field without
    updating the `defaults` allowlist would silently clobber user state
    if this branch wrote unknown keys unconditionally."""
    state = _empty_per_run_state()
    state["future_field"] = "user_value"  # pretend a future field user-set
    apply_overrides_to_run_state(
        state,
        {"future_field": "override_value"},
    )
    assert state["future_field"] == "user_value"


def test_apply_overrides_creates_current_config_if_missing():
    """Defensive: per-run state without a current_config key should still
    accept a starting_checkpoint override."""
    state = {"curation_model_source_mode": "auto"}  # minimal state, no current_config
    apply_overrides_to_run_state(
        state,
        {"starting_checkpoint": "/foo.model"},
    )
    assert state["current_config"]["starting_checkpoint"] == "/foo.model"


# ---------------------------------------------------------------------------
# Integration: add_run_to_state_json wires init_spec through to per-run state
# ---------------------------------------------------------------------------


def test_add_run_to_state_json_applies_browse_init_spec(tmp_path):
    """End-to-end: creating a run with a browse init_spec pre-populates the
    per-run state's Model section fields on disk."""
    from napari_epicycle.hitl_training_annotation._run_management import (
        add_run_to_state_json,
    )
    from napari_epicycle.hitl_training_annotation import _paths as _step2_paths

    ws = tmp_path / "workspace"
    ws.mkdir()
    spec = init_spec_browse(external_path=str(tmp_path / "my_model.model"))

    v3 = add_run_to_state_json(
        ws, "test_run", activate=True, init_spec=spec, target_bucket="lateral"
    )

    run_state = v3["runs"]["test_run"]
    assert run_state["curation_model_source_mode"] == "imported"
    assert run_state["imported_curation_model_path"] == str(tmp_path / "my_model.model")
    assert run_state["current_config"]["starting_checkpoint"] == str(tmp_path / "my_model.model")
    assert v3["active_run_id"] == "test_run"

    # And it's actually on disk — re-read the file proves the atomic write
    # carried the overrides through.
    import json
    state_path = _step2_paths.state_json_path(ws)
    on_disk = json.loads(state_path.read_text())
    assert on_disk["runs"]["test_run"]["curation_model_source_mode"] == "imported"


def test_add_run_to_state_json_pretrained_init_spec(tmp_path):
    from napari_epicycle.hitl_training_annotation._run_management import (
        add_run_to_state_json,
    )

    ws = tmp_path / "workspace"
    ws.mkdir()
    spec = init_spec_pretrained(pretrained_name="nuclei")

    v3 = add_run_to_state_json(ws, "test_run", init_spec=spec)

    run_state = v3["runs"]["test_run"]
    assert run_state["curation_model_source_mode"] == "pretrained"
    assert run_state["curation_pretrained_model_type"] == "nuclei"
    # pretrained doesn't pre-pop the Checkpoint widget.
    assert "starting_checkpoint" not in run_state.get("current_config", {})


def test_add_run_to_state_json_scratch_init_spec_no_overrides(tmp_path):
    """Scratch is the existing baseline behavior — nothing should change."""
    from napari_epicycle.hitl_training_annotation._run_management import (
        add_run_to_state_json,
        _empty_per_run_state,
    )

    ws = tmp_path / "workspace"
    ws.mkdir()
    spec = init_spec_scratch()

    v3 = add_run_to_state_json(ws, "test_run", init_spec=spec)

    run_state = v3["runs"]["test_run"]
    empty = _empty_per_run_state()
    # All Model-section fields stayed at their empty defaults.
    assert run_state["curation_model_source_mode"] == empty["curation_model_source_mode"]
    assert run_state["imported_curation_model_path"] == empty["imported_curation_model_path"]
    assert run_state["curation_pretrained_model_type"] == empty["curation_pretrained_model_type"]


def test_add_run_to_state_json_no_init_spec_unchanged(tmp_path):
    """Legacy callers (no init_spec arg) get the original empty-state behavior."""
    from napari_epicycle.hitl_training_annotation._run_management import (
        add_run_to_state_json,
        _empty_per_run_state,
    )

    ws = tmp_path / "workspace"
    ws.mkdir()
    v3 = add_run_to_state_json(ws, "test_run", activate=True)

    assert v3["runs"]["test_run"] == _empty_per_run_state()


def test_add_run_to_state_json_existing_slug_not_clobbered(tmp_path):
    """Re-adding an existing slug must not apply init_spec overrides — the
    user has potentially already adjusted that run's state."""
    from napari_epicycle.hitl_training_annotation._run_management import (
        add_run_to_state_json,
    )

    ws = tmp_path / "workspace"
    ws.mkdir()
    # First call — fresh run, init_spec applies.
    spec1 = init_spec_browse(external_path="/first.model")
    add_run_to_state_json(ws, "test_run", init_spec=spec1)
    # User then changes the imported path manually (simulated by direct edit
    # — in real life this is the controller writing state on user edits).
    # Now call add_run_to_state_json again with a DIFFERENT spec.
    spec2 = init_spec_browse(external_path="/second.model")
    v3 = add_run_to_state_json(ws, "test_run", init_spec=spec2)
    # Existing slug — second spec must NOT clobber.
    assert v3["runs"]["test_run"]["imported_curation_model_path"] == "/first.model"


# ---------------------------------------------------------------------------
# Dependency on the persistence branch's clobber guards
# (Agent 3's recommendation, 2026-05-29)
# ---------------------------------------------------------------------------


def test_prepopulate_depends_on_restore_blocksignals_guard():
    """The prepopulate fix puts imported_curation_model_path into the per-run
    state on disk.  On run load, _load_internal_state restores the controller
    field from state — and on baseline before fix/step2-model-state merged,
    `combo.setCurrentText("Imported model")` fired `_on_model_source_changed`
    which re-read the not-yet-populated imported_model_path widget and clobbered
    the just-restored controller field back to "".

    This branch (prepopulate) requires the merged persistence branch's
    `blockSignals` guard to actually take effect end-to-end.  Without it the
    prepopulate is silently inert.  If a future refactor removes the guard,
    this test fires and points the reader at the doc that explains why it
    is load-bearing.

    """
    import inspect
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )

    src = inspect.getsource(TrainingAnnotationController._load_internal_state)
    assert "blockSignals" in src, (
        "_load_internal_state must wrap model_source_combo.setCurrentText "
        "in blockSignals — otherwise the prepopulated "
        "imported_curation_model_path is clobbered to '' on every run load. "
            )


def test_prepopulate_depends_on_inference_sync_guard():
    """Companion to the blockSignals test: `_sync_inference_controls_from_config`
    must guard with `_init_transparency_loading` so its programmatic channel-
    combo restore doesn't fire `_sync_channel_combos` and schedule a spurious
    save / overwrite a restored channel value during run load.  Without the
    guard the prepopulate's channel persistence is silently undone on every
    workspace open.
    """
    import inspect
    from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
        TrainingAnnotationController,
    )

    src = inspect.getsource(
        TrainingAnnotationController._sync_inference_controls_from_config
    )
    assert "_init_transparency_loading" in src, (
        "_sync_inference_controls_from_config must set the "
        "_init_transparency_loading guard before calling its impl."
    )
