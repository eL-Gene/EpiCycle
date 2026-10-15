"""Tests for ``napari_epicycle/hitl_training_annotation/_run_management.py``.

Pure-Python tests for the multi-run management ops the Phase 8 UI
will call.  Covers:

* create_run_on_disk — happy path, dir + manifest written
* create_run_on_disk — slug collision raises RunAlreadyExistsError
* create_run_on_disk — invalid slug raises InvalidRunIdError
* create_run_on_disk — manifest write failure cleans up dir
* add_run_to_state_json — fresh state file created
* add_run_to_state_json — existing entry preserved (no clobber)
* add_run_to_state_json — activate flag honored
* set_active_run_in_state_json — slug must exist
* get_active_run_id — default fallback / corrupt fallback
* remove_run_from_state_json — active falls back to default
* list_run_summaries — enriched output
* list_run_summaries — archived flag respected
* find_runs_inheriting_from — provenance dependency check
* delete_run_from_disk — requires confirm
* All exceptions importable + are Exception subclasses
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from napari_epicycle.hitl_training_annotation import _paths as _step2_paths
from napari_epicycle.hitl_training_annotation._run_management import (
    InvalidRunIdError,
    RunAlreadyExistsError,
    RunNotFoundError,
    add_run_to_state_json,
    create_run_on_disk,
    delete_run_from_disk,
    find_runs_inheriting_from,
    get_active_run_id,
    list_run_summaries,
    remove_run_from_state_json,
    set_active_run_in_state_json,
)
from napari_epicycle.hitl_training_annotation._run_management import (
    list_run_model_sources,
    list_step3_refined_sources,
    resolve_init_spec_to_checkpoint,
)
from packages.epicycle_io.step2_run_manifest import (
    RUN_MANIFEST_FILENAME,
    build_run_manifest,
    init_spec_browse,
    init_spec_cross_bucket,
    init_spec_fork,
    init_spec_inherited,
    init_spec_pretrained,
    init_spec_scratch,
    init_spec_step3_refined,
    write_run_manifest,
)


# ----------------------------------------------------------------------------
# Exception sanity
# ----------------------------------------------------------------------------

class TestExceptionSanity:

    def test_all_subclass_exception(self):
        assert issubclass(RunAlreadyExistsError, Exception)
        assert issubclass(RunNotFoundError, Exception)
        assert issubclass(InvalidRunIdError, Exception)


# ----------------------------------------------------------------------------
# create_run_on_disk
# ----------------------------------------------------------------------------

class TestCreateRunOnDisk:

    def test_happy_path_creates_dir_and_manifest(self, tmp_path):
        manifest = create_run_on_disk(
            tmp_path,
            "nuclei_v1",
            display_name="Nuclei v1",
            target="nuclei",
            init_spec=init_spec_pretrained(pretrained_name="nuclei"),
        )
        run_dir = _step2_paths.run_root(tmp_path, "nuclei_v1")
        assert run_dir.is_dir()
        assert (run_dir / RUN_MANIFEST_FILENAME).is_file()
        assert manifest["run_id"] == "nuclei_v1"
        assert manifest["display_name"] == "Nuclei v1"
        assert manifest["target"] == "nuclei"
        assert manifest["created_from"]["mode"] == "pretrained"

    def test_minimal_args_scratch_default(self, tmp_path):
        manifest = create_run_on_disk(
            tmp_path, "default", display_name="Default"
        )
        assert manifest["created_from"]["mode"] == "scratch"
        assert manifest["target"] == ""

    def test_slug_collision_raises(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        with pytest.raises(RunAlreadyExistsError, match="already exists"):
            create_run_on_disk(tmp_path, "default", display_name="Other")

    def test_invalid_slug_raises_invalid_run_id_error(self, tmp_path):
        with pytest.raises(InvalidRunIdError):
            create_run_on_disk(tmp_path, "has spaces", display_name="X")

    def test_empty_slug_raises(self, tmp_path):
        with pytest.raises(InvalidRunIdError):
            create_run_on_disk(tmp_path, "", display_name="X")

    def test_runs_root_auto_created(self, tmp_path):
        """If runs/ doesn't exist, create_run_on_disk creates it."""
        assert not _step2_paths.step2_runs_root(tmp_path).exists()
        create_run_on_disk(tmp_path, "default", display_name="Default")
        assert _step2_paths.step2_runs_root(tmp_path).is_dir()

    def test_partial_failure_cleans_up_dir(self, tmp_path, monkeypatch):
        """If manifest write raises after dir is made, dir is removed."""
        from packages.epicycle_io import step2_run_manifest

        # Make write_run_manifest fail
        def boom(*args, **kwargs):
            raise OSError("simulated write failure")

        monkeypatch.setattr(step2_run_manifest, "write_run_manifest", boom)
        # Re-import to pick up monkeypatched name in _run_management's namespace
        from napari_epicycle.hitl_training_annotation import _run_management
        monkeypatch.setattr(_run_management, "write_run_manifest", boom)

        with pytest.raises(OSError):
            create_run_on_disk(tmp_path, "doomed", display_name="X")

        run_dir = _step2_paths.run_root(tmp_path, "doomed")
        assert not run_dir.exists(), "failed create should leave no orphan dir"

    def test_with_inherited_init_spec(self, tmp_path):
        manifest = create_run_on_disk(
            tmp_path, "child", display_name="Child",
            init_spec=init_spec_inherited(
                source_run_id="default",
                source_bucket="lateral",
                source_round_id="round_03",
            ),
        )
        assert manifest["created_from"]["mode"] == "inherited"
        assert manifest["created_from"]["source_run_id"] == "default"


# ----------------------------------------------------------------------------
# add_run_to_state_json
# ----------------------------------------------------------------------------

class TestAddRunToStateJson:

    def test_creates_fresh_state_file(self, tmp_path):
        state_path = _step2_paths.state_json_path(tmp_path)
        assert not state_path.exists()

        v3 = add_run_to_state_json(tmp_path, "default")
        assert state_path.is_file()
        assert v3["schema_version"] == 3
        assert "default" in v3["runs"]
        # Not activated yet by default
        assert v3["active_run_id"] == _step2_paths.DEFAULT_RUN_ID

    def test_activate_flag_sets_active(self, tmp_path):
        v3 = add_run_to_state_json(tmp_path, "nuclei", activate=True)
        assert v3["active_run_id"] == "nuclei"

    def test_existing_entry_not_clobbered(self, tmp_path):
        # Pre-populate
        add_run_to_state_json(tmp_path, "default")
        # Hand-edit to put something in the entry
        state_path = _step2_paths.state_json_path(tmp_path)
        with open(state_path, "r", encoding="utf-8") as f:
            v3 = json.load(f)
        v3["runs"]["default"]["current_round"] = 42
        with open(state_path, "w", encoding="utf-8") as f:
            json.dump(v3, f)

        # Add again
        v3_after = add_run_to_state_json(tmp_path, "default")
        # The existing entry was preserved
        assert v3_after["runs"]["default"]["current_round"] == 42

    def test_second_run_added_alongside(self, tmp_path):
        add_run_to_state_json(tmp_path, "default")
        v3 = add_run_to_state_json(tmp_path, "nuclei_v1")
        assert set(v3["runs"].keys()) == {"default", "nuclei_v1"}

    def test_invalid_slug_raises(self, tmp_path):
        with pytest.raises(InvalidRunIdError):
            add_run_to_state_json(tmp_path, "has spaces")


# ----------------------------------------------------------------------------
# set_active_run_in_state_json
# ----------------------------------------------------------------------------

class TestSetActiveRunInStateJson:

    def test_happy_path(self, tmp_path):
        add_run_to_state_json(tmp_path, "default")
        add_run_to_state_json(tmp_path, "nuclei_v1")
        v3 = set_active_run_in_state_json(tmp_path, "nuclei_v1")
        assert v3["active_run_id"] == "nuclei_v1"

    def test_no_state_file_raises(self, tmp_path):
        with pytest.raises(RunNotFoundError, match="no step2_state.json"):
            set_active_run_in_state_json(tmp_path, "default")

    def test_unknown_slug_raises(self, tmp_path):
        add_run_to_state_json(tmp_path, "default")
        with pytest.raises(RunNotFoundError, match="not in step2_state.json"):
            set_active_run_in_state_json(tmp_path, "ghost")

    def test_invalid_slug_raises(self, tmp_path):
        with pytest.raises(InvalidRunIdError):
            set_active_run_in_state_json(tmp_path, "has spaces")


# ----------------------------------------------------------------------------
# get_active_run_id
# ----------------------------------------------------------------------------

class TestGetActiveRunId:

    def test_no_state_file_returns_default(self, tmp_path):
        assert get_active_run_id(tmp_path) == "default"

    def test_corrupt_state_returns_default(self, tmp_path):
        state_path = _step2_paths.state_json_path(tmp_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text("{not json", encoding="utf-8")
        assert get_active_run_id(tmp_path) == "default"

    def test_missing_active_run_id_returns_default(self, tmp_path):
        state_path = _step2_paths.state_json_path(tmp_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text('{"schema_version": 3, "runs": {}}', encoding="utf-8")
        assert get_active_run_id(tmp_path) == "default"

    def test_returns_set_value(self, tmp_path):
        add_run_to_state_json(tmp_path, "nuclei_v1", activate=True)
        assert get_active_run_id(tmp_path) == "nuclei_v1"


# ----------------------------------------------------------------------------
# remove_run_from_state_json
# ----------------------------------------------------------------------------

class TestRemoveRunFromStateJson:

    def test_removes_entry(self, tmp_path):
        add_run_to_state_json(tmp_path, "default")
        add_run_to_state_json(tmp_path, "nuclei_v1")
        v3 = remove_run_from_state_json(tmp_path, "nuclei_v1")
        assert "nuclei_v1" not in v3["runs"]
        assert "default" in v3["runs"]

    def test_active_falls_back_to_default(self, tmp_path):
        add_run_to_state_json(tmp_path, "nuclei_v1", activate=True)
        v3 = remove_run_from_state_json(tmp_path, "nuclei_v1")
        assert v3["active_run_id"] == "default"

    def test_no_state_file_noop(self, tmp_path):
        # No state file — should not raise
        result = remove_run_from_state_json(tmp_path, "ghost")
        assert result == {}

    def test_removing_nonexistent_run_noop(self, tmp_path):
        add_run_to_state_json(tmp_path, "default")
        v3 = remove_run_from_state_json(tmp_path, "never_existed")
        assert "default" in v3["runs"]


# ----------------------------------------------------------------------------
# list_run_summaries
# ----------------------------------------------------------------------------

class TestListRunSummaries:

    def test_empty_workspace_returns_empty(self, tmp_path):
        assert list_run_summaries(tmp_path) == []

    def test_single_run_summary_shape(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default Run",
                           target="general")
        add_run_to_state_json(tmp_path, "default", activate=True)

        summaries = list_run_summaries(tmp_path)
        assert len(summaries) == 1
        s = summaries[0]
        assert s["run_id"] == "default"
        assert s["display_name"] == "Default Run"
        assert s["target"] == "general"
        assert s["backend"] == "cellpose"
        assert s["archived"] is False
        assert s["manifest_ok"] is True
        assert s["is_active"] is True
        assert s["lateral_round_count"] == 0
        assert s["axial_round_count"] == 0
        # last_activity_iso may be None or an ISO string
        assert s["last_activity_iso"] is None or isinstance(s["last_activity_iso"], str)

    def test_run_with_no_manifest_still_appears(self, tmp_path):
        """Run dir exists but no manifest — summary returned with
        manifest_ok=False and display_name=slug fallback."""
        runs_root = _step2_paths.step2_runs_root(tmp_path)
        runs_root.mkdir(parents=True)
        (runs_root / "orphan").mkdir()

        summaries = list_run_summaries(tmp_path)
        assert len(summaries) == 1
        assert summaries[0]["run_id"] == "orphan"
        assert summaries[0]["display_name"] == "orphan"  # fallback to slug
        assert summaries[0]["manifest_ok"] is False

    def test_archived_flag_respected(self, tmp_path):
        # Build manifest with archived_at set
        runs_root = _step2_paths.step2_runs_root(tmp_path)
        runs_root.mkdir(parents=True)
        run_dir = runs_root / "archived_run"
        run_dir.mkdir()
        m = build_run_manifest(
            run_id="archived_run",
            display_name="Archived Run",
            archived_at="2026-04-15T00:00:00Z",
        )
        write_run_manifest(run_dir / RUN_MANIFEST_FILENAME, m)

        summaries = list_run_summaries(tmp_path)
        assert summaries[0]["archived"] is True

    def test_multiple_runs_with_round_counts(self, tmp_path):
        # Create 2 runs
        create_run_on_disk(tmp_path, "default", display_name="Default")
        create_run_on_disk(tmp_path, "nuclei_v1", display_name="Nuclei")

        # Add 3 TRAINED lateral rounds to default.  2026-09-07 (bucket-switch
        # item #10): a round only counts once it has a model/ subdir, so the
        # picker's (lat:N ax:M) can't run ahead of the status bar.
        lateral_rounds = _step2_paths.bucket_rounds_root(tmp_path, "default", "lateral")
        lateral_rounds.mkdir(parents=True)
        for i in range(3):
            (lateral_rounds / f"round_{i:02d}" / "model").mkdir(parents=True)
        # 1 axial round to default
        axial_rounds = _step2_paths.bucket_rounds_root(tmp_path, "default", "axial")
        axial_rounds.mkdir(parents=True)
        (axial_rounds / "round_00" / "model").mkdir(parents=True)
        # An UNTRAINED lateral round (the one being annotated right now) is
        # NOT counted.
        (lateral_rounds / "round_03").mkdir()

        summaries = {s["run_id"]: s for s in list_run_summaries(tmp_path)}
        assert summaries["default"]["lateral_round_count"] == 3
        assert summaries["default"]["axial_round_count"] == 1
        assert summaries["nuclei_v1"]["lateral_round_count"] == 0
        assert summaries["nuclei_v1"]["axial_round_count"] == 0

    def test_is_active_only_one(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        create_run_on_disk(tmp_path, "nuclei", display_name="Nuclei")
        add_run_to_state_json(tmp_path, "default")
        add_run_to_state_json(tmp_path, "nuclei", activate=True)

        summaries = {s["run_id"]: s for s in list_run_summaries(tmp_path)}
        assert summaries["nuclei"]["is_active"] is True
        assert summaries["default"]["is_active"] is False

    def test_accepted_pair_count_key_present(self, tmp_path):
        """2026-05-25 Group D: list_run_summaries surfaces accepted_pair_count.

        Audit-3 finding: key was untested.  This locks the contract so
        future schema changes don't accidentally drop the field.
        """
        create_run_on_disk(tmp_path, "default", display_name="Default")
        add_run_to_state_json(tmp_path, "default", activate=True)
        summaries = list_run_summaries(tmp_path)
        assert "accepted_pair_count" in summaries[0]
        assert summaries[0]["accepted_pair_count"] == 0


# ----------------------------------------------------------------------------
# _count_accepted_pairs (Group D 2026-05-25)
# ----------------------------------------------------------------------------

class TestCountAcceptedPairs:
    """Tests for the pair-count helper that feeds the Run dropdown +
    Manage Runs 'Pairs' column.  Audit-3 found zero coverage; these
    close the gap on a pure-function disk-walk that has 5 branches."""

    def _setup(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._run_management import (
            _count_accepted_pairs,
        )
        return _count_accepted_pairs

    def test_empty_workspace_returns_zero(self, tmp_path):
        """No runs/ root anywhere — returns 0, doesn't raise."""
        _count_accepted_pairs = self._setup(tmp_path)
        assert _count_accepted_pairs(tmp_path, "default") == 0

    def test_missing_run_dir_returns_zero(self, tmp_path):
        """runs/ exists but the specific run's dir doesn't — returns 0."""
        _count_accepted_pairs = self._setup(tmp_path)
        _step2_paths.step2_runs_root(tmp_path).mkdir(parents=True)
        assert _count_accepted_pairs(tmp_path, "missing_run") == 0

    def test_missing_accepted_subdir_returns_zero(self, tmp_path):
        """round_NN/ exists but no accepted/ inside — returns 0."""
        _count_accepted_pairs = self._setup(tmp_path)
        create_run_on_disk(tmp_path, "default", display_name="Default")
        rounds_root = _step2_paths.bucket_rounds_root(tmp_path, "default", "lateral")
        rounds_root.mkdir(parents=True)
        (rounds_root / "round_00").mkdir()
        # No accepted/ subdir
        assert _count_accepted_pairs(tmp_path, "default") == 0

    def test_counts_only_img_tif_files(self, tmp_path):
        """accepted/ contains *_img.tif + *_mask.tif + stray files —
        only img files counted (a pair == 1 img + 1 mask, so img count
        IS pair count)."""
        _count_accepted_pairs = self._setup(tmp_path)
        create_run_on_disk(tmp_path, "default", display_name="Default")
        rounds_root = _step2_paths.bucket_rounds_root(tmp_path, "default", "lateral")
        accepted = rounds_root / "round_00" / "accepted"
        accepted.mkdir(parents=True)
        (accepted / "patch1_XY_z0001_img.tif").touch()
        (accepted / "patch1_XY_z0001_mask.tif").touch()  # not counted
        (accepted / "patch1_XY_z0002_img.tif").touch()
        (accepted / "patch1_XY_z0002_mask.tif").touch()  # not counted
        (accepted / "notes.txt").touch()  # not counted
        (accepted / "metadata.json").touch()  # not counted
        assert _count_accepted_pairs(tmp_path, "default") == 2

    def test_sums_across_buckets_and_rounds(self, tmp_path):
        """Lateral + axial buckets each with multiple rounds — sum is
        across all of them."""
        _count_accepted_pairs = self._setup(tmp_path)
        create_run_on_disk(tmp_path, "default", display_name="Default")
        # Lateral: 2 rounds, 3 + 2 pairs
        lat_rounds = _step2_paths.bucket_rounds_root(tmp_path, "default", "lateral")
        for round_idx, n_pairs in [(0, 3), (1, 2)]:
            acc = lat_rounds / f"round_{round_idx:02d}" / "accepted"
            acc.mkdir(parents=True)
            for i in range(n_pairs):
                (acc / f"lat_XY_z{i:04d}_img.tif").touch()
        # Axial: 1 round, 4 pairs
        ax_rounds = _step2_paths.bucket_rounds_root(tmp_path, "default", "axial")
        acc = ax_rounds / "round_00" / "accepted"
        acc.mkdir(parents=True)
        for i in range(4):
            (acc / f"ax_XZ_z{i:04d}_img.tif").touch()
        # 3 + 2 + 4 = 9
        assert _count_accepted_pairs(tmp_path, "default") == 9


# ----------------------------------------------------------------------------
# find_runs_inheriting_from
# ----------------------------------------------------------------------------

class TestFindRunsInheritingFrom:

    def test_no_dependents(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default",
                           init_spec=init_spec_scratch())
        create_run_on_disk(tmp_path, "other", display_name="Other",
                           init_spec=init_spec_scratch())
        assert find_runs_inheriting_from(tmp_path, "default") == []

    def test_inherited_dependent(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default",
                           init_spec=init_spec_scratch())
        create_run_on_disk(
            tmp_path, "child", display_name="Child",
            init_spec=init_spec_inherited(
                source_run_id="default",
                source_bucket="lateral",
                source_round_id="round_00",
            ),
        )
        assert find_runs_inheriting_from(tmp_path, "default") == ["child"]

    def test_multiple_dependents_sorted(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default",
                           init_spec=init_spec_scratch())
        for slug in ("zebra", "alpha", "mango"):
            create_run_on_disk(
                tmp_path, slug, display_name=slug,
                init_spec=init_spec_inherited(
                    source_run_id="default",
                    source_bucket="lateral",
                    source_round_id="round_00",
                ),
            )
        assert find_runs_inheriting_from(tmp_path, "default") == ["alpha", "mango", "zebra"]

    def test_self_excluded(self, tmp_path):
        """A run's own manifest never reports itself as a dependent."""
        # Defensive: build a manifest that references itself (shouldn't happen
        # in practice but the function should handle it)
        runs_root = _step2_paths.step2_runs_root(tmp_path)
        runs_root.mkdir(parents=True)
        run_dir = runs_root / "weird"
        run_dir.mkdir()
        m = build_run_manifest(
            run_id="weird", display_name="Weird",
            created_from=init_spec_inherited(
                source_run_id="weird",
                source_bucket="lateral",
                source_round_id="round_00",
            ),
        )
        write_run_manifest(run_dir / RUN_MANIFEST_FILENAME, m)

        assert find_runs_inheriting_from(tmp_path, "weird") == []

    def test_orphan_manifest_doesnt_break(self, tmp_path):
        # One run with valid manifest, one with no manifest at all
        create_run_on_disk(tmp_path, "default", display_name="Default",
                           init_spec=init_spec_scratch())
        runs_root = _step2_paths.step2_runs_root(tmp_path)
        (runs_root / "orphan").mkdir()
        # Should not raise
        result = find_runs_inheriting_from(tmp_path, "default")
        assert result == []


# ----------------------------------------------------------------------------
# delete_run_from_disk
# ----------------------------------------------------------------------------

class TestListRunModelSources:
    """Phase 8b — discovery of init-source candidates from this workspace."""

    def test_empty_workspace_returns_empty(self, tmp_path):
        assert list_run_model_sources(tmp_path) == []

    def test_no_models_returns_empty(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        assert list_run_model_sources(tmp_path) == []

    def test_discovers_model_weights(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default Run")
        # Plant a weights file using canonical Phase 6c naming
        # `{run_id}_{bucket}` (post-7a36217).  Tests pre-Phase-6c.7
        # planted as `weights` — fallback scanner now excludes that name
        # (Commit 1 audit L-2 fix: pre-Phase-6c stale file disambiguation).
        models_dir = _step2_paths.model_dir(tmp_path, "default", "lateral", 0)
        models_dir.mkdir(parents=True)
        (models_dir / "default_lateral").write_bytes(b"fake weights")

        sources = list_run_model_sources(tmp_path)
        assert len(sources) == 1
        s = sources[0]
        assert s["run_id"] == "default"
        assert s["run_display_name"] == "Default Run"
        assert s["bucket"] == "lateral"
        assert s["round_id"] == "round_00"
        assert s["exists"] is True
        assert s["mtime_iso"] is not None

    def test_excludes_specified_run(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        create_run_on_disk(tmp_path, "other", display_name="Other")
        for run_id in ("default", "other"):
            md = _step2_paths.model_dir(tmp_path, run_id, "lateral", 0)
            md.mkdir(parents=True)
            (md / f"{run_id}_lateral").write_bytes(b"x")

        sources = list_run_model_sources(tmp_path, exclude_run_id="default")
        assert all(s["run_id"] != "default" for s in sources)
        assert any(s["run_id"] == "other" for s in sources)

    def test_excludes_archived_by_default(self, tmp_path):
        from packages.epicycle_io.step2_run_manifest import write_run_manifest
        # Create archived run
        runs_root = _step2_paths.step2_runs_root(tmp_path)
        runs_root.mkdir(parents=True)
        run_dir = runs_root / "old"
        run_dir.mkdir()
        m = build_run_manifest(run_id="old", display_name="Old",
                               archived_at="2026-01-01T00:00:00Z")
        write_run_manifest(run_dir / RUN_MANIFEST_FILENAME, m)
        md = _step2_paths.model_dir(tmp_path, "old", "lateral", 0)
        md.mkdir(parents=True)
        (md / "old_lateral").write_bytes(b"x")

        # default behavior: archived excluded
        assert list_run_model_sources(tmp_path) == []
        # opt-in: include archived
        with_archived = list_run_model_sources(tmp_path, include_archived=True)
        assert any(s["run_id"] == "old" for s in with_archived)

    def test_lists_both_buckets_and_multiple_rounds(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        for bucket in ("lateral", "axial"):
            for round_idx in range(3):
                md = _step2_paths.model_dir(tmp_path, "default", bucket, round_idx)
                md.mkdir(parents=True)
                (md / f"default_{bucket}").write_bytes(b"x")
        sources = list_run_model_sources(tmp_path)
        assert len(sources) == 6
        # Confirm coverage
        keys = {(s["bucket"], s["round_id"]) for s in sources}
        assert keys == {
            ("lateral", "round_00"), ("lateral", "round_01"), ("lateral", "round_02"),
            ("axial", "round_00"), ("axial", "round_01"), ("axial", "round_02"),
        }


class TestListRunModelSourcesManifestReadPath:
    """Commit 2 audit HIGH-2 (2026-05-26): close the test gap.

    The Phase 6c.7 Commit 1 C-1 fix made `list_run_model_sources`
    read `manifest.json::weights_filename` as the source of truth
    (was hardcoded to "weights").  The original 5 tests planted
    files directly via `model_weights_path` and never exercised
    the manifest-read code path — they all fell through to the
    fallback scanner.  These tests close that gap.
    """

    def _plant_run(self, tmp_path, run_id="default", display_name="Default"):
        create_run_on_disk(tmp_path, run_id, display_name=display_name)
        md = _step2_paths.model_dir(tmp_path, run_id, "lateral", 0)
        md.mkdir(parents=True)
        return md

    def test_reads_weights_filename_from_manifest(self, tmp_path):
        """Manifest present pointing at NON-default filename → manifest
        read path is the one that resolves the file.

        Commit 6 audit Audit-3/5 MEDIUM fix: previous version of this
        test planted only the canonical `default_lateral` file, so the
        fallback scanner would have produced the same result.  Now
        plants a manifest with `weights_filename="alt_name"` AND a
        decoy `default_lateral` file in the same dir.  If the test
        passes via the manifest-read path, `alt_name` wins.  If it
        passes via the fallback scanner, `default_lateral` would win
        (alphabetical sort: `alt_name` < `default_lateral` so actually
        `alt_name` would win there too — to truly discriminate, plant
        `zzz_alt` so it sorts AFTER `default_lateral`).
        """
        from packages.epicycle_training.step2_model_manifest import (
            build_step2_model_manifest, write_step2_model_manifest,
        )
        md = self._plant_run(tmp_path)
        # Plant decoy file at canonical name — fallback scanner would
        # surface this if manifest-read failed.
        (md / "default_lateral").write_bytes(b"decoy")
        # Plant the REAL weights at a name that sorts AFTER the decoy
        # in iterdir order.  Fallback scanner picks the first
        # non-excluded file alphabetically; manifest-read picks
        # whatever `weights_filename` says.
        (md / "zzz_real_weights").write_bytes(b"real")
        # Plant manifest pointing at the zzz file
        m = build_step2_model_manifest(
            run_id="default", bucket="lateral", round_number=0,
            source_workspace=str(tmp_path),
            weights_filename="zzz_real_weights",
            init_mode="scratch",
        )
        write_step2_model_manifest(md / "manifest.json", m)

        sources = list_run_model_sources(tmp_path)
        assert len(sources) == 1
        # If the manifest-read path is the one being exercised,
        # weights_filename is "zzz_real_weights".  If fallback scanner
        # (sorted iterdir) is in use, it would be "default_lateral".
        # This DISCRIMINATES the two code paths.
        assert sources[0]["weights_filename"] == "zzz_real_weights"
        assert sources[0]["exists"] is True
        assert sources[0]["weights_path"].endswith("zzz_real_weights")

    def test_corrupt_manifest_logs_and_falls_back(self, tmp_path, capsys):
        """Corrupt manifest.json + canonical weights file → diagnostic
        printed AND fallback picks the file."""
        md = self._plant_run(tmp_path)
        (md / "default_lateral").write_bytes(b"weights")
        # Corrupt manifest (invalid JSON)
        (md / "manifest.json").write_text("{not valid json", encoding="utf-8")

        sources = list_run_model_sources(tmp_path)
        captured = capsys.readouterr()
        # Diagnostic was printed (Audit LOW-1: silent except masking)
        assert "manifest read failed" in captured.out
        # Fallback still surfaced the canonical file
        assert len(sources) == 1
        assert sources[0]["weights_filename"] == "default_lateral"
        assert sources[0]["exists"] is True

    def test_manifest_only_no_weights_file(self, tmp_path):
        """Manifest present + canonical weights file MISSING → source
        emitted with `exists=False` (not silently dropped).

        Scenario: lift failed partway through training, manifest got
        written somehow, but the canonical weights file isn't at the
        expected path.  We surface it with exists=False so the user
        can see "this round exists but has a problem" rather than the
        round silently disappearing from the picker.

        Current behavior: the fallback scanner runs because the
        manifest read returned a weights_filename but the file isn't
        a "non-companion" — actually wait, the manifest IS read first
        and gives weights_filename.  Then the code checks
        `weights_path.is_file()` → False → exists=False emitted.
        """
        from packages.epicycle_training.step2_model_manifest import (
            build_step2_model_manifest, write_step2_model_manifest,
        )
        md = self._plant_run(tmp_path)
        # Do NOT plant the weights file
        m = build_step2_model_manifest(
            run_id="default", bucket="lateral", round_number=0,
            source_workspace=str(tmp_path),
            weights_filename="default_lateral",
            init_mode="scratch",
        )
        write_step2_model_manifest(md / "manifest.json", m)

        sources = list_run_model_sources(tmp_path)
        assert len(sources) == 1
        assert sources[0]["weights_filename"] == "default_lateral"
        assert sources[0]["exists"] is False
        assert sources[0]["mtime_iso"] is None


class TestListRunModelSourcesDualPath:
    """Phase 8b audit-fix 2026-05-23: dual-path scan also walks legacy
    `artifacts/models/round_NN/` paths (where Cellpose still writes
    post-Phase-6c-deferral)."""

    def test_finds_legacy_lateral_model(self, tmp_path):
        legacy_round = tmp_path / "models" / "round_00"
        legacy_round.mkdir(parents=True)
        (legacy_round / "test_spatial_round_00").write_bytes(b"x")

        sources = list_run_model_sources(tmp_path)
        assert len(sources) >= 1
        legacy = [s for s in sources if s.get("run_display_name") == "Default (legacy path)"]
        assert len(legacy) == 1
        assert legacy[0]["bucket"] == "lateral"
        assert legacy[0]["run_id"] == "default"
        assert legacy[0]["round_id"] == "round_00"

    def test_finds_legacy_axial_model(self, tmp_path):
        legacy_round = tmp_path / "models" / "round_02"
        legacy_round.mkdir(parents=True)
        (legacy_round / "test_axial_round_02").write_bytes(b"x")

        sources = list_run_model_sources(tmp_path)
        legacy = [s for s in sources if s["run_display_name"] == "Default (legacy path)"]
        assert len(legacy) == 1
        assert legacy[0]["bucket"] == "axial"
        assert legacy[0]["round_id"] == "round_02"

    def test_legacy_inference_via_xy_xz_substrings(self, tmp_path):
        """Model names containing _xy → lateral; _xz / _yz → axial."""
        legacy = tmp_path / "models" / "round_00"
        legacy.mkdir(parents=True)
        (legacy / "model_xy").write_bytes(b"x")
        (legacy / "model_xz").write_bytes(b"x")
        (legacy / "model_yz").write_bytes(b"x")
        (legacy / "no_hint").write_bytes(b"x")  # should be skipped

        sources = [s for s in list_run_model_sources(tmp_path)
                   if s["run_display_name"] == "Default (legacy path)"]
        buckets = sorted(s["bucket"] for s in sources)
        # _xy → lateral, _xz + _yz → axial (2 axial entries)
        assert buckets == ["axial", "axial", "lateral"]

    def test_legacy_excludes_npy_loss_companions(self, tmp_path):
        legacy = tmp_path / "models" / "round_00"
        legacy.mkdir(parents=True)
        (legacy / "test_lateral_round_00").write_bytes(b"x")  # real model
        (legacy / "test_lateral_round_00_train_loss.npy").write_bytes(b"x")
        (legacy / "test_lateral_round_00_val_loss.npy").write_bytes(b"x")
        (legacy / "config.json").write_text("{}", encoding="utf-8")

        sources = [s for s in list_run_model_sources(tmp_path)
                   if s["run_display_name"] == "Default (legacy path)"]
        assert len(sources) == 1

    def test_exclude_run_id_default_skips_legacy(self, tmp_path):
        """If user is creating run named 'default' (collision case), don't
        offer the legacy 'default' models as init source."""
        legacy = tmp_path / "models" / "round_00"
        legacy.mkdir(parents=True)
        (legacy / "test_lateral_round_00").write_bytes(b"x")

        sources = list_run_model_sources(tmp_path, exclude_run_id="default")
        assert sources == []

    def test_no_legacy_dir_returns_empty_legacy(self, tmp_path):
        """If artifacts/models doesn't exist (fresh workspace), no legacy
        entries — but the function still returns canonical-path entries."""
        create_run_on_disk(tmp_path, "default", display_name="Default")
        md = _step2_paths.model_dir(tmp_path, "default", "lateral", 0)
        md.mkdir(parents=True)
        (md / "default_lateral").write_bytes(b"x")

        sources = list_run_model_sources(tmp_path)
        assert len(sources) == 1
        assert sources[0]["run_display_name"] == "Default"  # canonical


class TestListStep3RefinedSources:
    """Phase 8b — discovery of Step 3 refinement bundles as init sources."""

    def test_empty_workspace_returns_empty(self, tmp_path):
        assert list_step3_refined_sources(tmp_path) == []

    def test_discovers_lateral_and_axial(self, tmp_path):
        bundle_dir = tmp_path / "artifacts" / "step3" / "refinement" / "models" / "my_bundle"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "lateral").write_bytes(b"lat weights")
        (bundle_dir / "axial").write_bytes(b"ax weights")

        sources = list_step3_refined_sources(tmp_path)
        assert len(sources) == 2
        buckets = sorted(s["bucket"] for s in sources)
        assert buckets == ["axial", "lateral"]
        for s in sources:
            assert s["bundle_name"] == "my_bundle"
            assert s["exists"] is True

    def test_discovers_multiple_bundles(self, tmp_path):
        root = tmp_path / "artifacts" / "step3" / "refinement" / "models"
        for bundle in ("bundle_a", "bundle_b", "bundle_c"):
            d = root / bundle
            d.mkdir(parents=True)
            (d / "lateral").write_bytes(b"x")

        sources = list_step3_refined_sources(tmp_path)
        assert len(sources) == 3
        names = sorted(s["bundle_name"] for s in sources)
        assert names == ["bundle_a", "bundle_b", "bundle_c"]

    def test_skips_hidden_dirs(self, tmp_path):
        root = tmp_path / "artifacts" / "step3" / "refinement" / "models"
        for bundle in ("real_bundle", ".cache"):
            d = root / bundle
            d.mkdir(parents=True)
            (d / "lateral").write_bytes(b"x")
        sources = list_step3_refined_sources(tmp_path)
        assert all(s["bundle_name"] == "real_bundle" for s in sources)

    def test_skips_bundles_missing_weights(self, tmp_path):
        root = tmp_path / "artifacts" / "step3" / "refinement" / "models"
        empty_bundle = root / "empty"
        empty_bundle.mkdir(parents=True)
        # No lateral / axial files
        full_bundle = root / "full"
        full_bundle.mkdir(parents=True)
        (full_bundle / "lateral").write_bytes(b"x")

        sources = list_step3_refined_sources(tmp_path)
        assert len(sources) == 1
        assert sources[0]["bundle_name"] == "full"


class TestDeleteRunFromDisk:

    def test_requires_confirm(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        with pytest.raises(ValueError, match="confirm=True"):
            delete_run_from_disk(tmp_path, "default")

    def test_happy_path_with_confirm(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        run_dir = _step2_paths.run_root(tmp_path, "default")
        assert run_dir.is_dir()
        delete_run_from_disk(tmp_path, "default", confirm=True)
        assert not run_dir.exists()

    def test_missing_run_raises(self, tmp_path):
        with pytest.raises(RunNotFoundError):
            delete_run_from_disk(tmp_path, "ghost", confirm=True)

    def test_deletes_nested_content(self, tmp_path):
        create_run_on_disk(tmp_path, "default", display_name="Default")
        run_dir = _step2_paths.run_root(tmp_path, "default")
        # Drop some nested junk
        (run_dir / "lateral" / "rounds" / "round_00").mkdir(parents=True)
        (run_dir / "lateral" / "rounds" / "round_00" / "x.bin").write_bytes(b"data")
        delete_run_from_disk(tmp_path, "default", confirm=True)
        assert not run_dir.exists()


class TestResolveInitSpecToCheckpoint:
    """Commit 2 audit H1 fix (2026-05-26): resolve init_spec from
    run_manifest.json::created_from into a concrete checkpoint path
    that can be passed to Cellpose as pretrained_model.

    Pre-fix, init_spec was never read at training time — round 0 of
    an inherited/fork/step3_refined/browse run silently trained from
    scratch.  This is a TRAINING CORRECTNESS bug from Phase 8b.
    """

    def _plant_source_run(self, tmp_path, run_id, bucket, round_idx,
                          weights_filename=None):
        """Helper: create a source run with a trained model bundle.

        Returns the canonical weights file path (str).
        """
        from packages.epicycle_training.step2_model_manifest import (
            build_step2_model_manifest, write_step2_model_manifest,
        )
        create_run_on_disk(tmp_path, run_id, display_name=run_id)
        md = _step2_paths.model_dir(tmp_path, run_id, bucket, round_idx)
        md.mkdir(parents=True)
        wf = weights_filename or f"{run_id}_{bucket}"
        (md / wf).write_bytes(b"fake weights")
        m = build_step2_model_manifest(
            run_id=run_id, bucket=bucket, round_number=round_idx,
            source_workspace=str(tmp_path), weights_filename=wf,
            init_mode="scratch",
        )
        write_step2_model_manifest(md / "manifest.json", m)
        return str(md / wf)

    def test_scratch_returns_none(self, tmp_path):
        assert resolve_init_spec_to_checkpoint(
            tmp_path, init_spec_scratch()) is None

    def test_pretrained_returns_none(self, tmp_path):
        # Cellpose base models are handled via config.model_type, not
        # a file path — resolver returns None.
        assert resolve_init_spec_to_checkpoint(
            tmp_path, init_spec_pretrained(pretrained_name="cyto3")) is None

    def test_none_init_spec_returns_none(self, tmp_path):
        assert resolve_init_spec_to_checkpoint(tmp_path, None) is None
        assert resolve_init_spec_to_checkpoint(tmp_path, {}) is None

    def test_browse_returns_external_path(self, tmp_path):
        ext = tmp_path / "external_weights.bin"
        ext.write_bytes(b"external")
        spec = init_spec_browse(external_path=str(ext))
        result = resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert result == str(ext)

    def test_browse_missing_file_returns_none(self, tmp_path):
        spec = init_spec_browse(external_path=str(tmp_path / "nope.bin"))
        assert resolve_init_spec_to_checkpoint(tmp_path, spec) is None

    def test_inherited_resolves_via_manifest(self, tmp_path):
        """inherited mode: read source run manifest, get canonical
        weights_filename, return absolute path."""
        expected = self._plant_source_run(
            tmp_path, "src_run", "lateral", 2)
        spec = init_spec_inherited(
            source_run_id="src_run",
            source_bucket="lateral",
            source_round_id="round_02",
        )
        result = resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert result == expected

    def test_inherited_missing_source_returns_none(self, tmp_path):
        spec = init_spec_inherited(
            source_run_id="nonexistent",
            source_bucket="lateral",
            source_round_id="round_00",
        )
        assert resolve_init_spec_to_checkpoint(tmp_path, spec) is None

    def test_inherited_fallback_to_canonical_naming(self, tmp_path):
        """When manifest.json is missing but the canonical
        {run_id}_{bucket} file exists, fallback resolves it."""
        create_run_on_disk(tmp_path, "src_run", display_name="src")
        md = _step2_paths.model_dir(tmp_path, "src_run", "axial", 1)
        md.mkdir(parents=True)
        expected = md / "src_run_axial"
        expected.write_bytes(b"x")

        spec = init_spec_inherited(
            source_run_id="src_run",
            source_bucket="axial",
            source_round_id="round_01",
        )
        result = resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert result == str(expected)

    def test_inherited_fallback_to_legacy_weights(self, tmp_path):
        """Pre-Phase-6c.7 bundles wrote weights as bare 'weights' file
        without a manifest.  Fallback chain: manifest -> canonical ->
        legacy bare weights."""
        create_run_on_disk(tmp_path, "src_run", display_name="src")
        md = _step2_paths.model_dir(tmp_path, "src_run", "lateral", 0)
        md.mkdir(parents=True)
        expected = md / "weights"
        expected.write_bytes(b"legacy")

        spec = init_spec_inherited(
            source_run_id="src_run",
            source_bucket="lateral",
            source_round_id="round_00",
        )
        result = resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert result == str(expected)

    def test_cross_bucket_uses_active_run(self, tmp_path):
        """cross_bucket: source_run_id is implicit = active_run_id."""
        expected = self._plant_source_run(
            tmp_path, "myrun", "lateral", 3)
        spec = init_spec_cross_bucket(
            source_bucket="lateral",
            source_round_id="round_03",
        )
        result = resolve_init_spec_to_checkpoint(
            tmp_path, spec, active_run_id="myrun")
        assert result == expected

    def test_cross_bucket_without_active_run_returns_none(self, tmp_path):
        spec = init_spec_cross_bucket(
            source_bucket="lateral", source_round_id="round_00")
        assert resolve_init_spec_to_checkpoint(tmp_path, spec) is None

    def test_fork_uses_terminal_round(self, tmp_path):
        """fork: no source_round_id; resolver picks highest round_NN."""
        import time
        from packages.epicycle_training.step2_model_manifest import (
            build_step2_model_manifest, write_step2_model_manifest,
        )
        create_run_on_disk(tmp_path, "forked", display_name="forked")
        for r in range(3):
            md = _step2_paths.model_dir(tmp_path, "forked", "lateral", r)
            md.mkdir(parents=True)
            (md / "forked_lateral").write_bytes(b"x")
            time.sleep(0.01)
            m = build_step2_model_manifest(
                run_id="forked", bucket="lateral", round_number=r,
                source_workspace=str(tmp_path),
                weights_filename="forked_lateral",
                init_mode="scratch",
            )
            write_step2_model_manifest(md / "manifest.json", m)

        spec = init_spec_fork(source_run_id="forked", copied_pair_count=5)
        result = resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert result is not None
        assert "round_02" in result
        assert "forked_lateral" in result

    def test_step3_refined_resolves(self, tmp_path):
        bundle_dir = tmp_path / "artifacts" / "step3" / "refinement" / "models" / "v1_bundle"
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "lateral").write_bytes(b"step3 lateral weights")
        (bundle_dir / "axial").write_bytes(b"step3 axial weights")

        spec = init_spec_step3_refined(
            source_step3_bundle="v1_bundle",
            source_step3_bucket="lateral",
        )
        result = resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert result == str(bundle_dir / "lateral")

    def test_step3_refined_resolves_nested_default_layout(self, tmp_path):
        """HIGH-2: after a Step-3 default-run 'Convert to unified layout',
        the refined bundle lives at the NESTED
        ``runs/default/refinement/models/...`` — the resolver must still
        find it (layout-agnostic two-path probe)."""
        bundle_dir = (
            tmp_path / "artifacts" / "step3" / "runs" / "default"
            / "refinement" / "models" / "v1_bundle"
        )
        bundle_dir.mkdir(parents=True)
        (bundle_dir / "lateral").write_bytes(b"step3 lateral weights (nested)")

        spec = init_spec_step3_refined(
            source_step3_bundle="v1_bundle",
            source_step3_bucket="lateral",
        )
        result = resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert result == str(bundle_dir / "lateral")

    def test_step3_refined_missing_returns_none(self, tmp_path):
        spec = init_spec_step3_refined(
            source_step3_bundle="missing",
            source_step3_bucket="lateral",
        )
        assert resolve_init_spec_to_checkpoint(tmp_path, spec) is None

    def test_malformed_spec_returns_none(self, tmp_path):
        assert resolve_init_spec_to_checkpoint(tmp_path, "not a dict") is None
        assert resolve_init_spec_to_checkpoint(tmp_path, {"mode": "weird"}) is None
        assert resolve_init_spec_to_checkpoint(
            tmp_path, {"mode": "inherited"}) is None

    def test_fork_picks_target_bucket_over_mtime(self, tmp_path):
        """Commit 4 audit H-A: fork from Mode=Both source must prefer
        TARGET bucket over mtime-latest opposite bucket.

        Scenario: source trained Mode=Both — first lateral round 0,
        then axial round 0.  Axial has newer mtime.  New run forks
        with target_bucket='lateral'.  Pre-H-A: picked axial weights
        (newer mtime) — Cellpose lateral training from axial init =
        silent quality degradation.  Post-H-A: picks lateral.
        """
        import time
        from packages.epicycle_training.step2_model_manifest import (
            build_step2_model_manifest, write_step2_model_manifest,
        )
        create_run_on_disk(tmp_path, "src", display_name="src")
        # Plant lateral round 0 first
        ml = _step2_paths.model_dir(tmp_path, "src", "lateral", 0)
        ml.mkdir(parents=True)
        (ml / "src_lateral").write_bytes(b"lateral")
        m_lat = build_step2_model_manifest(
            run_id="src", bucket="lateral", round_number=0,
            source_workspace=str(tmp_path),
            weights_filename="src_lateral", init_mode="scratch",
        )
        write_step2_model_manifest(ml / "manifest.json", m_lat)
        time.sleep(0.05)
        # Plant axial round 0 AFTER — newer mtime
        ma = _step2_paths.model_dir(tmp_path, "src", "axial", 0)
        ma.mkdir(parents=True)
        (ma / "src_axial").write_bytes(b"axial")
        m_ax = build_step2_model_manifest(
            run_id="src", bucket="axial", round_number=0,
            source_workspace=str(tmp_path),
            weights_filename="src_axial", init_mode="scratch",
        )
        write_step2_model_manifest(ma / "manifest.json", m_ax)

        spec = init_spec_fork(source_run_id="src", copied_pair_count=3)
        # target_bucket='lateral' MUST win over newer-mtime axial
        result = resolve_init_spec_to_checkpoint(
            tmp_path, spec, target_bucket="lateral")
        assert result is not None
        assert "src_lateral" in result
        assert "src_axial" not in result

        # Reverse: target_bucket='axial' picks axial
        result_ax = resolve_init_spec_to_checkpoint(
            tmp_path, spec, target_bucket="axial")
        assert result_ax is not None
        assert "src_axial" in result_ax
        assert "src_lateral" not in result_ax

    def test_fork_without_target_falls_back_to_mtime(self, tmp_path):
        """When target_bucket is unspecified, fork preserves the legacy
        mtime-latest behavior (axial wins because it trained last)."""
        import time
        from packages.epicycle_training.step2_model_manifest import (
            build_step2_model_manifest, write_step2_model_manifest,
        )
        create_run_on_disk(tmp_path, "src", display_name="src")
        ml = _step2_paths.model_dir(tmp_path, "src", "lateral", 0)
        ml.mkdir(parents=True)
        (ml / "src_lateral").write_bytes(b"x")
        write_step2_model_manifest(ml / "manifest.json",
            build_step2_model_manifest(
                run_id="src", bucket="lateral", round_number=0,
                source_workspace=str(tmp_path),
                weights_filename="src_lateral", init_mode="scratch"))
        time.sleep(0.05)
        ma = _step2_paths.model_dir(tmp_path, "src", "axial", 0)
        ma.mkdir(parents=True)
        (ma / "src_axial").write_bytes(b"x")
        write_step2_model_manifest(ma / "manifest.json",
            build_step2_model_manifest(
                run_id="src", bucket="axial", round_number=0,
                source_workspace=str(tmp_path),
                weights_filename="src_axial", init_mode="scratch"))

        spec = init_spec_fork(source_run_id="src", copied_pair_count=3)
        result = resolve_init_spec_to_checkpoint(tmp_path, spec)
        # No target_bucket → mtime-latest = axial
        assert result is not None
        assert "src_axial" in result

    def test_inherited_with_non_string_round_returns_none(self, tmp_path):
        """Commit 4 audit H-2: malformed user-touchable manifest with
        non-string source_round_id (e.g. int 3 instead of 'round_03').
        Pre-fix raised uncaught AttributeError.  Post-fix coerces +
        catches; resolver returns None gracefully."""
        spec = {
            "mode": "inherited",
            "source_run_id": "src",
            "source_bucket": "lateral",
            "source_round_id": 3,  # int — not "round_03"
        }
        # Should NOT raise; should return None (str(3)="3" doesn't
        # match "round_NN" pattern → ValueError → caught)
        result = resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert result is None

    def test_end_to_end_inherited_persists_and_resolves(self, tmp_path):
        """Commit 6 audit Audit-5/5 HIGH #1: integration test for the
        full chain.  Unit tests cover the resolver in isolation; this
        test wires create_run -> run_manifest.json -> read-back ->
        resolve_init_spec_to_checkpoint.

        Pre-Commit-4, init_spec was written to created_from but never
        read at training time.  This test would have FAILED pre-fix
        (the chain was broken at the read step).  Post-fix it proves
        the persistence + resolution round-trip works.
        """
        from packages.epicycle_training.step2_model_manifest import (
            build_step2_model_manifest, write_step2_model_manifest,
        )
        from packages.epicycle_io.step2_run_manifest import read_run_manifest
        # 1. Plant source run with a trained model bundle
        create_run_on_disk(tmp_path, "src", display_name="Source Run")
        md = _step2_paths.model_dir(tmp_path, "src", "lateral", 2)
        md.mkdir(parents=True)
        weights_path = md / "src_lateral"
        weights_path.write_bytes(b"source weights")
        sm = build_step2_model_manifest(
            run_id="src", bucket="lateral", round_number=2,
            source_workspace=str(tmp_path),
            weights_filename="src_lateral",
            init_mode="scratch",
        )
        write_step2_model_manifest(md / "manifest.json", sm)

        # 2. Create target run inheriting from src (write created_from
        # exactly the way `_create_new_run` does in production)
        target_init_spec = init_spec_inherited(
            source_run_id="src",
            source_bucket="lateral",
            source_round_id="round_02",
        )
        create_run_on_disk(
            tmp_path, "target", display_name="Target Run",
            init_spec=target_init_spec,
        )

        # 3. Read target's run_manifest from disk (simulating worker
        # thread starting fresh — no in-memory state preserved)
        rm_path = _step2_paths.run_manifest_path(tmp_path, "target")
        rm = read_run_manifest(rm_path)
        assert rm is not None
        assert rm.get("created_from", {}).get("mode") == "inherited"
        created_from = rm["created_from"]

        # 4. Resolve the init_spec to a concrete path
        resolved = resolve_init_spec_to_checkpoint(
            tmp_path, created_from,
            active_run_id="target",
            target_bucket="lateral",
        )

        # 5. The resolved path must match the source's canonical
        # weights file — proves the entire chain (create -> persist ->
        # read -> resolve) wires correctly.
        assert resolved == str(weights_path)
        assert Path(resolved).is_file()

    def test_thumbs_db_excluded_from_fallback_scanner(self, tmp_path):
        """Commit 6 audit Audit-3/5 LOW: Windows OS metadata files
        (Thumbs.db, desktop.ini) don't start with `.` or `~` and don't
        have `.tmp`/`.bak` suffix, so they slipped through the
        previous exclusion set.  Now in _FALLBACK_EXCLUDED_NAMES."""
        create_run_on_disk(tmp_path, "default", display_name="Default")
        md = _step2_paths.model_dir(tmp_path, "default", "lateral", 0)
        md.mkdir(parents=True)
        # Plant ONLY Windows OS metadata + canonical weights
        (md / "Thumbs.db").write_bytes(b"windows thumbnail cache")
        (md / "desktop.ini").write_bytes(b"folder customization")
        (md / "default_lateral").write_bytes(b"real weights")

        sources = list_run_model_sources(tmp_path)
        # The canonical file is picked, NOT Thumbs.db / desktop.ini
        assert len(sources) == 1
        assert sources[0]["weights_filename"] == "default_lateral"

    def test_thumbs_db_only_round_dir_returns_no_source(self, tmp_path):
        """If a round dir contains ONLY OS metadata + no real weights,
        the scanner should surface NO source (not Thumbs.db)."""
        create_run_on_disk(tmp_path, "default", display_name="Default")
        md = _step2_paths.model_dir(tmp_path, "default", "lateral", 0)
        md.mkdir(parents=True)
        (md / "Thumbs.db").write_bytes(b"x")
        (md / "desktop.ini").write_bytes(b"x")
        # NO real weights

        sources = list_run_model_sources(tmp_path)
        assert len(sources) == 0


# ============================================================================
# Phase B.2 / B.2.1 — Flat-layout + dual-path reader tests (2026-05-25)
# Per audit B.2 audit 2 (HIGH: counter helpers not dual-path) + audit 3
# (HIGH: missing legacy-only + dedup test coverage).
# ============================================================================

class TestB2DualPathCounters:
    """Phase B.2.1 fix (audit 2 HIGH): `_count_rounds` +
    `_count_accepted_pairs` now dual-path.  Pre-fix they read only
    legacy split layout → flat-layout runs showed 0 rounds in the
    Run-dropdown label `(lat:N ax:M)`."""

    def test_count_rounds_finds_flat_layout(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._run_management import (
            _count_rounds,
        )
        create_run_on_disk(tmp_path, "default", display_name="Default")
        # Plant 3 rounds at FLAT layout
        for r in range(3):
            md = _step2_paths.model_dir(tmp_path, "default", "lateral", r)
            md.mkdir(parents=True)
            (md / f"default_lateral_round_{r:02d}").write_bytes(b"x")
        # Counter pre-B.2.1 would have returned 0; post-fix returns 3
        assert _count_rounds(tmp_path, "default", "lateral") == 3

    def test_count_rounds_finds_legacy_split_layout(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._run_management import (
            _count_rounds,
        )
        create_run_on_disk(tmp_path, "default", display_name="Default")
        # Plant 2 TRAINED ROUND DIRS at LEGACY SPLIT layout
        # (bucket/rounds/round_NN/model/)
        for r in range(2):
            rd = _step2_paths.legacy_split_round_dir(tmp_path, "default", "lateral", r)
            (rd / "model").mkdir(parents=True)
        assert _count_rounds(tmp_path, "default", "lateral") == 2

    def test_count_rounds_dedupes_split_vs_flat_round_dirs(self, tmp_path):
        """Same round in BOTH layouts (round_dir AND legacy_split round_dir)
        — count once."""
        from napari_epicycle.hitl_training_annotation._run_management import (
            _count_rounds,
        )
        create_run_on_disk(tmp_path, "default", display_name="Default")
        # round_00 at FLAT
        (_step2_paths.round_dir(tmp_path, "default", "lateral", 0)
         / "model").mkdir(parents=True)
        # round_00 at LEGACY split
        (_step2_paths.legacy_split_round_dir(tmp_path, "default", "lateral", 0)
         / "model").mkdir(parents=True)
        # round_01 only at FLAT
        (_step2_paths.round_dir(tmp_path, "default", "lateral", 1)
         / "model").mkdir(parents=True)
        # Count 2 unique
        assert _count_rounds(tmp_path, "default", "lateral") == 2

    # `test_count_rounds_dedupes_split_vs_flat_round_dirs` above covers
    # the dedup case using actual round-dir paths.

    def test_count_accepted_pairs_finds_flat_layout(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._run_management import (
            _count_accepted_pairs,
        )
        create_run_on_disk(tmp_path, "default", display_name="Default")
        # Plant 3 accepted pairs in FLAT layout
        accepted = _step2_paths.round_accepted_dir(tmp_path, "default", "lateral", 0)
        accepted.mkdir(parents=True)
        for i in range(3):
            (accepted / f"stk_XY_z{i:04d}_img.tif").write_bytes(b"x")
            (accepted / f"stk_XY_z{i:04d}_mask.tif").write_bytes(b"x")
        assert _count_accepted_pairs(tmp_path, "default") == 3

    def test_count_accepted_pairs_finds_legacy_split(self, tmp_path):
        from napari_epicycle.hitl_training_annotation._run_management import (
            _count_accepted_pairs,
        )
        create_run_on_disk(tmp_path, "default", display_name="Default")
        # Plant 2 accepted pairs at LEGACY SPLIT
        legacy_round = _step2_paths.legacy_split_round_dir(tmp_path, "default", "lateral", 0)
        legacy_accepted = legacy_round / "accepted"
        legacy_accepted.mkdir(parents=True)
        for i in range(2):
            (legacy_accepted / f"stk_XY_z{i:04d}_img.tif").write_bytes(b"x")
            (legacy_accepted / f"stk_XY_z{i:04d}_mask.tif").write_bytes(b"x")
        assert _count_accepted_pairs(tmp_path, "default") == 2

    def test_count_accepted_pairs_dedupes_partial_migration(self, tmp_path):
        """Same round in BOTH layouts — count NEW only (no double-count)."""
        from napari_epicycle.hitl_training_annotation._run_management import (
            _count_accepted_pairs,
        )
        create_run_on_disk(tmp_path, "default", display_name="Default")
        # round_00 NEW: 3 pairs
        new_accepted = _step2_paths.round_accepted_dir(tmp_path, "default", "lateral", 0)
        new_accepted.mkdir(parents=True)
        for i in range(3):
            (new_accepted / f"stk_XY_z{i:04d}_img.tif").write_bytes(b"x")
        # round_00 LEGACY: 5 pairs (different content)
        legacy_round = _step2_paths.legacy_split_round_dir(tmp_path, "default", "lateral", 0)
        legacy_accepted = legacy_round / "accepted"
        legacy_accepted.mkdir(parents=True)
        for i in range(5):
            (legacy_accepted / f"stk_legacy_XY_z{i:04d}_img.tif").write_bytes(b"x")
        # Count = 3 (NEW wins, legacy round_00 skipped entirely)
        assert _count_accepted_pairs(tmp_path, "default") == 3


class TestB2ListRunModelSourcesDualPath:
    """Phase B.2.1 (audit 3 HIGH gaps 2+3): coverage for legacy-only +
    both-present dedup branches of `list_run_model_sources`."""

    def test_legacy_split_only_workspace_discovered(self, tmp_path):
        """Pre-Phase-B.2 workspace with data only at split layout —
        the dual-path scanner must still find it."""
        create_run_on_disk(tmp_path, "default", display_name="Default")
        md = _step2_paths.legacy_split_model_dir(tmp_path, "default", "lateral", 0)
        md.mkdir(parents=True)
        (md / "default_lateral").write_bytes(b"split weights")
        sources = list_run_model_sources(tmp_path)
        assert len(sources) == 1
        assert sources[0]["bucket"] == "lateral"
        assert sources[0]["round_id"] == "round_00"

    def test_both_layouts_same_round_new_wins(self, tmp_path):
        """Partial migration: same round in BOTH layouts.  New must win
        — legacy duplicate suppressed."""
        from packages.epicycle_training.step2_model_manifest import (
            build_step2_model_manifest, write_step2_model_manifest,
        )
        create_run_on_disk(tmp_path, "default", display_name="Default")
        # NEW layout — round_00 with manifest pointing at flat naming
        new_md = _step2_paths.model_dir(tmp_path, "default", "lateral", 0)
        new_md.mkdir(parents=True)
        (new_md / "default_lateral_round_00").write_bytes(b"flat weights")
        m = build_step2_model_manifest(
            run_id="default", bucket="lateral", round_number=0,
            source_workspace=str(tmp_path),
            weights_filename="default_lateral_round_00",
            init_mode="scratch",
        )
        write_step2_model_manifest(new_md / "manifest.json", m)
        # LEGACY layout — same round_00 with PRE-B.2 naming
        legacy_md = _step2_paths.legacy_split_model_dir(tmp_path, "default", "lateral", 0)
        legacy_md.mkdir(parents=True)
        (legacy_md / "default_lateral").write_bytes(b"OLD weights")

        sources = list_run_model_sources(tmp_path)
        # Exactly 1 entry — new wins, legacy duplicate suppressed
        assert len(sources) == 1
        # And the surfaced weights_filename is the FLAT-NAMED one
        assert sources[0]["weights_filename"] == "default_lateral_round_00"
        # And the path is under the flat layout
        assert "/model/" in sources[0]["weights_path"].replace("\\", "/")


class TestB2ResolveSourceCheckpointDualPath:
    """Phase B.2.1 (audit 3 HIGH gap 4): resolver dual-path coverage
    for legacy-only source runs."""

    def test_inherited_resolves_legacy_split_layout(self, tmp_path):
        """Source run at LEGACY SPLIT — resolver must still find it."""
        from napari_epicycle.hitl_training_annotation._run_management import (
            resolve_init_spec_to_checkpoint,
        )
        create_run_on_disk(tmp_path, "src_legacy", display_name="src")
        # Plant source pair at LEGACY SPLIT layout
        md = _step2_paths.legacy_split_model_dir(tmp_path, "src_legacy", "lateral", 2)
        md.mkdir(parents=True)
        weights = md / "src_legacy_lateral"
        weights.write_bytes(b"legacy weights")

        spec = init_spec_inherited(
            source_run_id="src_legacy",
            source_bucket="lateral",
            source_round_id="round_02",
        )
        result = resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert result == str(weights)

    def test_resolve_b2_naming_after_manifest_delete(self, tmp_path):
        """Phase B.2.1 audit 3 MEDIUM fix: resolver fallback chain now
        includes `{run}_{bucket}_round_NN` (Phase B.2 naming).  If the
        manifest is deleted from a B.2-trained bundle, the fallback
        must still find the weights file."""
        from napari_epicycle.hitl_training_annotation._run_management import (
            resolve_init_spec_to_checkpoint,
        )
        create_run_on_disk(tmp_path, "src", display_name="src")
        # FLAT layout, Phase B.2 naming, NO manifest
        md = _step2_paths.model_dir(tmp_path, "src", "lateral", 3)
        md.mkdir(parents=True)
        weights = md / "src_lateral_round_03"
        weights.write_bytes(b"b2 weights")
        # Intentionally NO manifest.json

        spec = init_spec_inherited(
            source_run_id="src",
            source_bucket="lateral",
            source_round_id="round_03",
        )
        result = resolve_init_spec_to_checkpoint(tmp_path, spec)
        assert result == str(weights), (
            f"B.2 fallback chain miss: expected {weights}, got {result}"
        )
