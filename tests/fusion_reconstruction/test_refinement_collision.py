"""Tests for the Tier 1a refinement-training collision detection.

Verifies ``_check_refinement_collision(mode)`` correctly identifies
when training would overwrite an existing bucket entry in a bundle
manifest, and routes through the dialog accordingly.

Pure-Python (no Qt event loop); the dialog itself is monkeypatched
to a deterministic return value.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from napari_epicycle.fusion_reconstruction import _paths


def _make_controller(working_dir: Path, *, bundle_name: str = ""):
    """Build a minimal fake controller for the collision check.

    Imports the real method off the controller class so we test the
    actual production code.
    """
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    class _LineEdit:
        def __init__(self, text):
            self._text = text
        def text(self):
            return self._text

    ui = SimpleNamespace()
    ui.refine_model_name = _LineEdit(bundle_name)

    ctrl = SimpleNamespace(
        working_dir=str(working_dir),
        ui=ui,
        _append_log=lambda _msg: None,
    )
    ctrl._check_refinement_collision = (
        FusionReconstructionController._check_refinement_collision.__get__(ctrl)
    )
    ctrl._show_refinement_collision_dialog = lambda bundle_name, collisions: "overwrite"
    return ctrl


def _write_bundle_manifest(working_dir: Path, bundle_name: str, manifest: dict):
    """Write a fake bundle manifest at the canonical path."""
    manifest_path = _paths.refined_model_manifest_path(working_dir, bundle_name)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


# ---------------------------------------------------------------------------
# No-collision cases (should return immediately)
# ---------------------------------------------------------------------------


class TestNoCollision:

    def test_empty_bundle_name_no_collision(self, tmp_path):
        """Auto-generated names (empty field) get unique timestamps —
        no collision possible.  Skip the check."""
        ctrl = _make_controller(tmp_path, bundle_name="")
        assert ctrl._check_refinement_collision("Fine-tune Lateral") == "no_collision"

    def test_bundle_doesnt_exist_no_collision(self, tmp_path):
        ctrl = _make_controller(tmp_path, bundle_name="brand_new")
        assert ctrl._check_refinement_collision("Fine-tune Both") == "no_collision"

    def test_bundle_exists_but_no_manifest_no_collision(self, tmp_path):
        bundle_dir = _paths.refined_model_dir(tmp_path, "no_manifest_yet")
        bundle_dir.mkdir(parents=True)
        ctrl = _make_controller(tmp_path, bundle_name="no_manifest_yet")
        assert ctrl._check_refinement_collision("Fine-tune Lateral") == "no_collision"

    def test_corrupt_manifest_no_collision(self, tmp_path):
        """Defensive: a corrupt manifest is treated as no-collision —
        finalize_refinement_pass will overwrite it cleanly."""
        bundle_dir = _paths.refined_model_dir(tmp_path, "corrupt")
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "manifest.json").write_text("NOT VALID JSON")
        ctrl = _make_controller(tmp_path, bundle_name="corrupt")
        assert ctrl._check_refinement_collision("Fine-tune Lateral") == "no_collision"

    def test_manifest_exists_but_bucket_not_in_it(self, tmp_path):
        """Training axial on a bundle that only has lateral entry —
        NOT a collision (this is the Mode=Both two-pass pattern).
        """
        _write_bundle_manifest(tmp_path, "lateral_only", {
            "schema_version": "2.0",
            "name": "lateral_only",
            "lateral": {"backend": "cellpose", "weights_path": "lateral"},
        })
        ctrl = _make_controller(tmp_path, bundle_name="lateral_only")
        assert ctrl._check_refinement_collision("Fine-tune Axial") == "no_collision"


# ---------------------------------------------------------------------------
# Collision cases
# ---------------------------------------------------------------------------


class TestCollisionDetected:

    def test_same_bucket_collision_lateral(self, tmp_path):
        _write_bundle_manifest(tmp_path, "existing_v1", {
            "schema_version": "2.0",
            "name": "existing_v1",
            "lateral": {
                "backend": "cellpose",
                "weights_path": "lateral",
                "final_loss": {"train": 0.5, "val": 0.4},
                "n_epochs": 10,
            },
        })
        ctrl = _make_controller(tmp_path, bundle_name="existing_v1")
        captured = []
        ctrl._show_refinement_collision_dialog = (
            lambda name, coll: (captured.append((name, coll)), "overwrite")[1]
        )
        result = ctrl._check_refinement_collision("Fine-tune Lateral")
        assert result == "overwrite"
        # Dialog was called with the collision info.
        assert len(captured) == 1
        bundle_name, coll = captured[0]
        assert bundle_name == "existing_v1"
        assert "lateral" in coll
        assert coll["lateral"]["final_train_loss"] == 0.5
        assert coll["lateral"]["n_epochs"] == 10

    def test_same_bucket_collision_axial(self, tmp_path):
        _write_bundle_manifest(tmp_path, "existing_v1", {
            "schema_version": "2.0",
            "name": "existing_v1",
            "axial": {"backend": "cellpose", "weights_path": "axial"},
        })
        ctrl = _make_controller(tmp_path, bundle_name="existing_v1")
        ctrl._show_refinement_collision_dialog = lambda n, c: "overwrite"
        assert ctrl._check_refinement_collision("Fine-tune Axial") == "overwrite"

    def test_mode_both_lateral_collision_only(self, tmp_path):
        """Bundle has lateral entry, training Mode=Both → collision on
        lateral bucket, no collision on axial bucket."""
        _write_bundle_manifest(tmp_path, "half_done", {
            "schema_version": "2.0",
            "name": "half_done",
            "lateral": {"backend": "cellpose", "weights_path": "lateral"},
        })
        ctrl = _make_controller(tmp_path, bundle_name="half_done")
        captured = []
        ctrl._show_refinement_collision_dialog = (
            lambda n, c: (captured.append(c), "overwrite")[1]
        )
        ctrl._check_refinement_collision("Fine-tune Both")
        coll = captured[0]
        assert "lateral" in coll
        assert "axial" not in coll  # axial entry didn't exist, no collision

    def test_mode_both_both_buckets_collision(self, tmp_path):
        _write_bundle_manifest(tmp_path, "fully_trained", {
            "schema_version": "2.0",
            "name": "fully_trained",
            "lateral": {"backend": "cellpose", "weights_path": "lateral"},
            "axial": {"backend": "cellpose", "weights_path": "axial"},
        })
        ctrl = _make_controller(tmp_path, bundle_name="fully_trained")
        captured = []
        ctrl._show_refinement_collision_dialog = (
            lambda n, c: (captured.append(c), "overwrite")[1]
        )
        ctrl._check_refinement_collision("Fine-tune Both")
        coll = captured[0]
        assert "lateral" in coll
        assert "axial" in coll

    def test_cancel_decision_propagates(self, tmp_path):
        _write_bundle_manifest(tmp_path, "existing", {
            "schema_version": "2.0",
            "name": "existing",
            "lateral": {"backend": "cellpose", "weights_path": "lateral"},
        })
        ctrl = _make_controller(tmp_path, bundle_name="existing")
        ctrl._show_refinement_collision_dialog = lambda n, c: "cancel"
        assert ctrl._check_refinement_collision("Fine-tune Lateral") == "cancel"


# ---------------------------------------------------------------------------
# Cross-bucket safety (the original Phase 6 use case)
# ---------------------------------------------------------------------------


class TestDialogFormatter:
    """Audit-driven 2026-05-21: the dialog body builder had a
    false-negative on n_epochs=0 (pre-fix used ``v or "?"`` which
    treats 0 as falsy).  Plus bool-as-int handling.  Cover both."""

    def _captured_message(self, collisions: dict) -> str:
        """Invoke the dialog builder but capture the rendered HTML
        body instead of opening the Qt dialog."""
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        captured = {}
        # Stub themed_confirm to capture the message and return True
        # (the formatter's "overwrite" branch).
        import napari_epicycle.ui_shell.theme as theme

        original = theme.themed_confirm

        def _capturing_confirm(parent, title, message, **kw):
            captured["message"] = message
            return True

        theme.themed_confirm = _capturing_confirm
        try:
            ctrl = SimpleNamespace(ui=SimpleNamespace())
            ctrl._show_refinement_collision_dialog = (
                FusionReconstructionController._show_refinement_collision_dialog.__get__(ctrl)
            )
            ctrl._show_refinement_collision_dialog("my_bundle", collisions)
        finally:
            theme.themed_confirm = original
        return captured.get("message", "")

    def test_n_epochs_zero_renders_zero_not_question_mark(self):
        message = self._captured_message({
            "lateral": {
                "final_train_loss": 0.5,
                "n_epochs": 0,
                "n_train_pairs": 100,
                "init_mode": "scratch",
            },
        })
        assert "n_epochs=0" in message
        assert "n_train_pairs=100" in message

    def test_final_loss_zero_renders_zero_not_question_mark(self):
        message = self._captured_message({
            "lateral": {
                "final_train_loss": 0.0,
                "n_epochs": 10,
                "n_train_pairs": 50,
                "init_mode": "cellpose",
            },
        })
        assert "loss=0.0000" in message

    def test_none_fields_render_as_question_mark(self):
        message = self._captured_message({
            "lateral": {
                "final_train_loss": None,
                "n_epochs": None,
                "n_train_pairs": None,
                "init_mode": None,
            },
        })
        assert "loss=?" in message
        assert "n_epochs=?" in message
        assert "n_train_pairs=?" in message
        assert "init_mode=?" in message

    def test_bool_treated_as_non_numeric(self):
        """Defensive: bool is a subclass of int.  Manifest shouldn't
        carry bools but the formatter must not crash if it does."""
        message = self._captured_message({
            "lateral": {
                "final_train_loss": True,  # corrupted manifest
                "n_epochs": 5,
                "n_train_pairs": 10,
            },
        })
        # bool should render as "?" not "1.0000".
        assert "loss=?" in message

    def test_mode_both_notice_in_body(self):
        """User-facing notice that Mode=Both writes both buckets even
        when only one collides — surfaced post-audit."""
        message = self._captured_message({
            "lateral": {"final_train_loss": 0.5, "n_epochs": 10},
        })
        # Search for the explicit caveat text.
        assert "trains both buckets" in message
        assert "BOTH will be" in message


class TestCrossBucketCoexistence:
    """Mode=Both two-pass training: lateral first writes lateral entry,
    axial second writes axial entry.  When the second pass starts,
    the manifest has only lateral; axial is fresh.  This must NOT
    trigger a collision dialog for the axial bucket.
    """

    def test_second_pass_axial_after_lateral_no_collision(self, tmp_path):
        # State after lateral pass completed:
        _write_bundle_manifest(tmp_path, "two_pass", {
            "schema_version": "2.0",
            "name": "two_pass",
            "lateral": {"backend": "cellpose", "weights_path": "lateral"},
        })
        ctrl = _make_controller(tmp_path, bundle_name="two_pass")
        # The second pass would only train axial (it's not Mode=Both
        # — the second invocation of _launch_next_refinement passes
        # the "Fine-tune Axial" mode through the queue).
        result = ctrl._check_refinement_collision("Fine-tune Axial")
        assert result == "no_collision"

    def test_mode_both_pre_check_only_warns_on_real_collisions(self, tmp_path):
        """Mode=Both pre-check on a fully-fresh bundle name → no
        collision (the manifest doesn't exist yet)."""
        ctrl = _make_controller(tmp_path, bundle_name="fresh_both")
        assert ctrl._check_refinement_collision("Fine-tune Both") == "no_collision"


# ---------------------------------------------------------------------------
# Mode=Both with an EMPTY name: both passes must land in ONE bundle
# ---------------------------------------------------------------------------


class TestBothWithEmptyNameUsesOneBundle:
    """With the name field empty, "Fine-tune Both" auto-names the bundle
    ``refined_<timestamp>``.  The name must be chosen ONCE per run: the
    axial pass starts later (from the lateral pass's finish, via a
    zero-delay timer), so a per-pass timestamp gave the axial pass a new
    name and split the run into two half bundles (one with only
    ``lateral``, one with only ``axial``).
    """

    def test_both_passes_share_one_bundle_name_and_folder(
        self, tmp_path, monkeypatch,
    ):
        from tests.fusion_reconstruction._refine_launch_harness import (
            make_launch_controller,
        )

        ctrl, h = make_launch_controller(tmp_path, monkeypatch)
        workers = h.workers
        # A bucket record left by a previous run must not survive a launch.
        ctrl._refinement_trained_buckets = {"axial"}

        ctrl._launch_refinement()           # lateral pass starts
        assert ctrl._refinement_queue == ["Fine-tune Axial"]
        assert ctrl._refinement_trained_buckets == set(), (
            "_launch_refinement must reset the trained-bucket record"
        )
        ctrl._launch_next_refinement()      # axial pass (the deferred timer)

        assert [w.bucket for w in workers] == ["lateral", "axial"]
        names = {w.bundle_name for w in workers}
        assert len(names) == 1, f"one run must write ONE bundle, got {names}"
        (name,) = names
        assert name.startswith("refined_")
        assert {w.save_path for w in workers} == {
            str(_paths.refined_model_dir(tmp_path, name))
        }
        folders = sorted(
            p.name for p in _paths.refined_models_root(tmp_path).iterdir()
        )
        assert folders == [name], f"expected one bundle folder, got {folders}"
        assert name == "refined_01", "the auto-name is the first free number"

        # The next run must choose its OWN name, not inherit this one.
        ctrl._restore_launch_button()       # the run is over
        assert ctrl._refinement_bundle_name is None
        ctrl._launch_refinement()
        assert workers[-1].bundle_name != name, (
            f"a new run re-used the previous run's name {name!r}"
        )
