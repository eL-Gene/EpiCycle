"""Tests for `ui_shell/_workspace_facts.py` — Brick B1 of the shared-workspace
design.

Everything under test here is PURE: no Qt, no napari, no controller import.
That is deliberate — the shell must be able to answer "what does this
workspace contain?" from disk, before any step has been opened in this
session (invariant I14).

Covered:
  * `trained_rounds` — the `model/`-dir rule, dual-path (flat + legacy
    split) scan, summing across runs, missing dirs, never raising, and
    cache invalidation on a `step2_state.json` mtime bump.
  * `patch_count` — Step 1's extracted-patch count for the sidebar.
  * `load_last_workspace` / `save_last_workspace` — the `~/.epicycle`
    suggestion round-trip, including corrupt / missing / deleted-dir /
    UNC-skip behaviour (invariants I5, I7, I13).
  * `banner_state` / `banner_strips` — the three-state banner machine and
    the two-strip stacking + precedence (mockup Option A).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from napari_epicycle.ui_shell import _workspace_facts as wf


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def _make_flat_round(ws: Path, run: str, bucket: str, idx: int,
                     with_model: bool = True) -> Path:
    """NEW flat layout: {ws}/artifacts/step2/runs/{run}/{bucket}/round_NN/."""
    rd = ws / "artifacts" / "step2" / "runs" / run / bucket / f"round_{idx:02d}"
    rd.mkdir(parents=True, exist_ok=True)
    if with_model:
        (rd / "model").mkdir(exist_ok=True)
    return rd


def _make_split_round(ws: Path, run: str, bucket: str, idx: int,
                      with_model: bool = True) -> Path:
    """LEGACY split layout: .../{bucket}/rounds/round_NN/."""
    rd = (ws / "artifacts" / "step2" / "runs" / run / bucket / "rounds"
          / f"round_{idx:02d}")
    rd.mkdir(parents=True, exist_ok=True)
    if with_model:
        (rd / "model").mkdir(exist_ok=True)
    return rd


@pytest.fixture(autouse=True)
def _clear_cache():
    wf.clear_cache()
    yield
    wf.clear_cache()


# ---------------------------------------------------------------------------
# trained_rounds
# ---------------------------------------------------------------------------

class TestTrainedRounds:
    def test_missing_workspace_is_zeros(self, tmp_path):
        assert wf.trained_rounds(str(tmp_path / "nope")) == {
            "lateral": 0, "axial": 0}

    def test_empty_string_is_zeros(self):
        assert wf.trained_rounds("") == {"lateral": 0, "axial": 0}

    def test_flat_layout_counts_rounds_with_model(self, tmp_path):
        _make_flat_round(tmp_path, "cells", "lateral", 0)
        _make_flat_round(tmp_path, "cells", "lateral", 1)
        _make_flat_round(tmp_path, "cells", "axial", 0)
        assert wf.trained_rounds(str(tmp_path)) == {"lateral": 2, "axial": 1}

    def test_round_without_model_dir_not_counted(self, tmp_path):
        _make_flat_round(tmp_path, "cells", "lateral", 0, with_model=True)
        _make_flat_round(tmp_path, "cells", "lateral", 1, with_model=False)
        assert wf.trained_rounds(str(tmp_path))["lateral"] == 1

    def test_legacy_split_layout_counted(self, tmp_path):
        _make_split_round(tmp_path, "cells", "lateral", 0)
        _make_split_round(tmp_path, "cells", "axial", 0)
        _make_split_round(tmp_path, "cells", "axial", 1)
        assert wf.trained_rounds(str(tmp_path)) == {"lateral": 1, "axial": 2}

    def test_same_round_in_both_layouts_counted_once(self, tmp_path):
        _make_flat_round(tmp_path, "cells", "lateral", 0)
        _make_split_round(tmp_path, "cells", "lateral", 0)
        assert wf.trained_rounds(str(tmp_path))["lateral"] == 1

    def test_multiple_runs_are_summed(self, tmp_path):
        _make_flat_round(tmp_path, "cells", "lateral", 0)
        _make_flat_round(tmp_path, "nuclei", "lateral", 0)
        _make_flat_round(tmp_path, "nuclei", "lateral", 1)
        assert wf.trained_rounds(str(tmp_path))["lateral"] == 3

    def test_non_round_dirs_ignored(self, tmp_path):
        run = tmp_path / "artifacts" / "step2" / "runs" / "cells" / "lateral"
        (run / "scratch" / "model").mkdir(parents=True)
        assert wf.trained_rounds(str(tmp_path)) == {"lateral": 0, "axial": 0}

    def test_never_raises_on_a_file_where_a_dir_belongs(self, tmp_path):
        runs = tmp_path / "artifacts" / "step2" / "runs"
        runs.mkdir(parents=True)
        (runs / "cells").write_text("not a directory")
        assert wf.trained_rounds(str(tmp_path)) == {"lateral": 0, "axial": 0}


class TestTrainedRoundsCache:
    def test_second_call_is_served_from_cache(self, tmp_path):
        _make_flat_round(tmp_path, "cells", "lateral", 0)
        (tmp_path / "step2_state.json").write_text("{}")
        first = wf.trained_rounds(str(tmp_path))
        # A new round appears but the state file is untouched — the cached
        # answer is returned (this is the documented trade: cheap, and the
        # state file is written whenever Step 2 actually trains).
        _make_flat_round(tmp_path, "cells", "lateral", 1)
        assert wf.trained_rounds(str(tmp_path)) == first

    def test_mtime_bump_invalidates_the_cache(self, tmp_path):
        state = tmp_path / "step2_state.json"
        _make_flat_round(tmp_path, "cells", "lateral", 0)
        state.write_text("{}")
        assert wf.trained_rounds(str(tmp_path))["lateral"] == 1
        _make_flat_round(tmp_path, "cells", "lateral", 1)
        time.sleep(0.01)
        os.utime(state, (time.time() + 5, time.time() + 5))
        assert wf.trained_rounds(str(tmp_path))["lateral"] == 2

    def test_case_variant_path_hits_the_same_cache_entry(self, tmp_path):
        _make_flat_round(tmp_path, "cells", "lateral", 0)
        a = wf.trained_rounds(str(tmp_path))
        b = wf.trained_rounds(str(tmp_path).upper())
        assert a == b


# ---------------------------------------------------------------------------
# patch_count
# ---------------------------------------------------------------------------

class TestPatchCount:
    def test_zero_when_missing(self, tmp_path):
        assert wf.patch_count(str(tmp_path)) == 0

    def test_counts_tif_and_tiff(self, tmp_path):
        d = tmp_path / "artifacts" / "step1" / "patches"
        d.mkdir(parents=True)
        (d / "a.tif").write_bytes(b"x")
        (d / "b.tiff").write_bytes(b"x")
        (d / "notes.txt").write_text("x")
        assert wf.patch_count(str(tmp_path)) == 2


# ---------------------------------------------------------------------------
# last-workspace suggestion
# ---------------------------------------------------------------------------

class TestLastWorkspace:
    def test_round_trip(self, tmp_path, monkeypatch):
        home = tmp_path / "home"
        home.mkdir()
        monkeypatch.setattr(wf.Path, "home", staticmethod(lambda: home))
        ws = tmp_path / "MyProject"
        ws.mkdir()
        wf.save_last_workspace(str(ws))
        saved = home / ".epicycle" / "last_workspace.json"
        assert saved.is_file()
        payload = json.loads(saved.read_text())
        assert payload["schema_version"] == 1
        assert payload["path"] == str(ws)
        assert "written_at" in payload
        assert wf.load_last_workspace() == str(ws)

    def test_missing_file_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wf.Path, "home", staticmethod(lambda: tmp_path))
        assert wf.load_last_workspace() == ""

    def test_corrupt_json_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wf.Path, "home", staticmethod(lambda: tmp_path))
        d = tmp_path / ".epicycle"
        d.mkdir()
        (d / "last_workspace.json").write_text("{not json")
        assert wf.load_last_workspace() == ""

    def test_wrong_shape_is_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wf.Path, "home", staticmethod(lambda: tmp_path))
        d = tmp_path / ".epicycle"
        d.mkdir()
        (d / "last_workspace.json").write_text('["a list"]')
        assert wf.load_last_workspace() == ""

    def test_deleted_local_dir_is_dropped(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wf.Path, "home", staticmethod(lambda: tmp_path))
        d = tmp_path / ".epicycle"
        d.mkdir()
        gone = str(tmp_path / "gone")
        (d / "last_workspace.json").write_text(
            json.dumps({"schema_version": 1, "path": gone}))
        # Only validated when it looks like a local drive letter.
        if wf._is_local_drive_path(gone):
            assert wf.load_last_workspace() == ""
        else:
            assert wf.load_last_workspace() == gone

    def test_unc_path_is_not_validated(self, tmp_path, monkeypatch):
        monkeypatch.setattr(wf.Path, "home", staticmethod(lambda: tmp_path))
        d = tmp_path / ".epicycle"
        d.mkdir()
        unc = r"\\server\share\Project"
        (d / "last_workspace.json").write_text(
            json.dumps({"schema_version": 1, "path": unc}))
        assert wf.load_last_workspace() == unc

    def test_save_never_raises(self, tmp_path, monkeypatch):
        def _boom():
            raise OSError("no home")
        monkeypatch.setattr(wf.Path, "home", staticmethod(_boom))
        wf.save_last_workspace("C:/whatever")   # must not raise

    def test_is_local_drive_path(self):
        assert wf._is_local_drive_path(r"C:\Users\x") is True
        assert wf._is_local_drive_path("D:/data") is True
        assert wf._is_local_drive_path(r"\\server\share") is False
        assert wf._is_local_drive_path("/mnt/data") is False
        assert wf._is_local_drive_path("") is False


# ---------------------------------------------------------------------------
# banner state machine
# ---------------------------------------------------------------------------

SHARED = r"C:\Users\someone\Desktop\Project"
OTHER = r"D:\Scratch\Rerun"


class TestBannerState:
    """Symmetric rule: a step is amber only when the USER browsed it there
    (`browsed=True`) AND that path differs from the shared workspace.  A step
    that merely inherited a path is never amber — it re-adopts."""

    def test_no_workspace_anywhere(self):
        assert wf.banner_state("", "") == wf.STATE_NONE

    def test_step_on_shared(self):
        assert wf.banner_state(SHARED, SHARED, browsed=True) == wf.STATE_SHARED

    def test_step_on_shared_case_and_slash_variants(self):
        assert wf.banner_state(SHARED, SHARED.upper(), browsed=True) == \
            wf.STATE_SHARED
        assert wf.banner_state(
            SHARED, SHARED.replace("\\", "/"), browsed=True) == wf.STATE_SHARED

    def test_browsed_elsewhere_is_diverged(self):
        assert wf.banner_state(SHARED, OTHER, browsed=True) == \
            wf.STATE_DIVERGED

    def test_an_inherited_stale_path_is_stale_not_diverged(self):
        """B5: the banner must not call it "(shared across steps)".

        The banner is rendered for the ACTIVE step, and the active step is
        adopted synchronously by `set_workspace` — so an adopted path that
        still differs from the shared workspace means the controller
        DECLINED the re-adopt (Step 1 does that while a job is running).
        Saying "shared" there names a folder that is not shared.  It is not
        `diverged` either: the user did not browse it, and it moves on its
        own as soon as the job finishes.
        """
        assert wf.banner_state(SHARED, OTHER, browsed=False) == \
            wf.STATE_STALE

    def test_stale_renders_amber_and_says_why(self):
        strips = wf.banner_strips("project_setup", SHARED, OTHER,
                                  advisory="", browsed=False)
        text, colour, _elide = strips[0]
        assert colour == "amber"
        assert OTHER in text
        assert "shared across steps" not in text
        assert "still running" in text

    def test_step_empty_follows_the_shared_workspace(self):
        assert wf.banner_state(SHARED, "") == wf.STATE_SHARED


class TestBannerStrips:
    def test_state_none_step1_wording(self):
        strips = wf.banner_strips("project_setup", "", "", advisory="")
        assert strips == [("Choose a workspace folder to begin.", "red", "")]

    def test_state_none_other_steps_wording(self):
        strips = wf.banner_strips("apply_3d", "", "", advisory="")
        assert len(strips) == 1
        assert strips[0][1] == "red"
        assert "Set a workspace in Project Setup" in strips[0][0]

    def test_state_none_suppresses_the_advisory(self):
        strips = wf.banner_strips(
            "apply_3d", "", "", advisory="No completed HITL training round…")
        assert len(strips) == 1
        assert "Set a workspace" in strips[0][0]

    def test_shared_is_green_with_the_qualifier(self):
        strips = wf.banner_strips("apply_3d", SHARED, SHARED, advisory="")
        assert len(strips) == 1
        text, color, path = strips[0]
        assert color == "green"
        assert path == SHARED
        assert "(shared across steps)" in text

    def test_diverged_is_amber_and_does_not_inline_the_shared_path(self):
        strips = wf.banner_strips("apply_3d", SHARED, OTHER, advisory="",
                                  browsed=True)
        text, color, path = strips[0]
        assert color == "amber"
        assert path == OTHER
        assert text == f"Workspace: {OTHER}  (different from the shared workspace)"
        # The shared path lives in the tooltip, not on the line — two full
        # paths on one strip pushed the tail off the panel.
        assert SHARED not in text

    def test_diverged_stacks_an_advisory_below(self):
        strips = wf.banner_strips("reconstruct_curate_3d", SHARED, OTHER,
                                  advisory="careful", browsed=True)
        assert [c for _t, c, _p in strips] == ["amber", "amber"]
        assert strips[1][0] == "careful"

    def test_two_strips_workspace_first_then_advisory(self):
        strips = wf.banner_strips(
            "reconstruct_curate_3d", SHARED, SHARED,
            advisory="No completed HITL training round detected yet.")
        assert len(strips) == 2
        assert strips[0][1] == "green"           # workspace strip on top
        assert strips[1][1] == "amber"           # advisory below
        assert strips[1][0].startswith("No completed HITL")

    def test_missing_folder_is_red(self):
        strips = wf.banner_strips(
            "apply_3d", SHARED, SHARED, advisory="", missing=True)
        assert strips[0][1] == "red"
        assert "no longer where it was" in strips[0][0]

    def test_missing_folder_still_shows_the_advisory_below(self):
        strips = wf.banner_strips(
            "apply_3d", SHARED, SHARED, advisory="something", missing=True)
        assert len(strips) == 2
        assert strips[0][1] == "red"


# ---------------------------------------------------------------------------
# audit follow-ups: legacy round layout (N5), network guard (N1), local state
# ---------------------------------------------------------------------------

class TestLegacyRootModels:
    """Pre-Phase-6c workspaces keep trained models at {ws}/models/round_NN/."""

    def _legacy(self, ws, idx, filename):
        d = ws / "models" / f"round_{idx:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / filename).write_bytes(b"weights")
        return d

    def test_legacy_lateral_and_axial_counted(self, tmp_path):
        self._legacy(tmp_path, 0, "cellpose_lateral_round0")
        self._legacy(tmp_path, 1, "cellpose_lateral_round1")
        self._legacy(tmp_path, 2, "cellpose_axial_round0")
        assert wf.trained_rounds(str(tmp_path)) == {"lateral": 2, "axial": 1}

    def test_legacy_nested_models_subdir(self, tmp_path):
        d = tmp_path / "models" / "round_00" / "models"
        d.mkdir(parents=True)
        (d / "model_spatial_r0").write_bytes(b"w")
        assert wf.trained_rounds(str(tmp_path))["lateral"] == 1

    def test_companions_and_backups_are_not_models(self, tmp_path):
        self._legacy(tmp_path, 0, "train_loss_lateral.npy")
        self._legacy(tmp_path, 0, ".DS_Store_lateral")
        self._legacy(tmp_path, 0, "weights_lateral.bak")
        assert wf.trained_rounds(str(tmp_path)) == {"lateral": 0, "axial": 0}

    def test_unknown_bucket_is_not_attributed(self, tmp_path):
        self._legacy(tmp_path, 0, "some_model")
        assert wf.trained_rounds(str(tmp_path)) == {"lateral": 0, "axial": 0}

    def test_legacy_and_new_layouts_sum(self, tmp_path):
        _make_flat_round(tmp_path, "cells", "lateral", 0)
        self._legacy(tmp_path, 5, "cellpose_lateral_old")
        assert wf.trained_rounds(str(tmp_path))["lateral"] == 2


class TestCacheKeyCoversBothStateFiles:
    def test_hitl_state_mtime_invalidates(self, tmp_path):
        state = tmp_path / "hitl_state.json"
        d = tmp_path / "models" / "round_00"
        d.mkdir(parents=True)
        (d / "cellpose_lateral_r0").write_bytes(b"w")
        state.write_text("{}")
        assert wf.trained_rounds(str(tmp_path))["lateral"] == 1
        d2 = tmp_path / "models" / "round_01"
        d2.mkdir()
        (d2 / "cellpose_lateral_r1").write_bytes(b"w")
        os.utime(state, (time.time() + 5, time.time() + 5))
        assert wf.trained_rounds(str(tmp_path))["lateral"] == 2


class TestPatchCountNetworkGuard:
    def test_non_local_path_is_not_walked_by_default(self, tmp_path,
                                                     monkeypatch):
        walked = []

        real_iterdir = wf.Path.iterdir

        def _spy(self):
            walked.append(str(self))
            return real_iterdir(self)

        monkeypatch.setattr(wf.Path, "iterdir", _spy)
        assert wf.patch_count(r"\server\share\Proj") == 0
        assert walked == []

    def test_allow_network_lets_it_scan(self, tmp_path):
        d = tmp_path / "artifacts" / "step1" / "patches"
        d.mkdir(parents=True)
        (d / "a.tif").write_bytes(b"x")
        # tmp_path IS a local drive letter on Windows, so both forms work.
        assert wf.patch_count(str(tmp_path), allow_network=True) == 1


class TestLocalState:
    """F1: a step with a path but NO shared workspace is not 'shared'."""

    def test_state_is_local(self):
        assert wf.banner_state("", SHARED) == wf.STATE_LOCAL

    def test_strip_is_green_without_the_shared_qualifier(self):
        strips = wf.banner_strips("project_setup", "", SHARED, advisory="")
        text, color, path = strips[0]
        assert color == "green"
        assert path == SHARED
        assert "shared across steps" not in text
        assert text == f"Workspace: {SHARED}"

    def test_local_state_still_stacks_an_advisory(self):
        strips = wf.banner_strips("apply_3d", "", SHARED, advisory="careful")
        assert len(strips) == 2
        assert strips[1][0] == "careful"


# ---------------------------------------------------------------------------
# B2 (second-line status): Step 2 layout / first-run + Step 3 pre-v3
# ---------------------------------------------------------------------------

class TestStep2LayoutHint:
    """`step2_layout_hint` — the amber "old folder layout" second line."""

    def test_new_layout_has_no_hint(self, tmp_path):
        (tmp_path / "artifacts" / "step2" / "runs" / "cells").mkdir(parents=True)
        assert wf.step2_layout_hint(str(tmp_path)) == ""

    def test_fresh_workspace_has_no_hint(self, tmp_path):
        assert wf.step2_layout_hint(str(tmp_path)) == ""

    def test_legacy_training_data_gets_the_hint(self, tmp_path):
        (tmp_path / "training_data").mkdir()
        assert wf.step2_layout_hint(str(tmp_path)) == wf.MSG_STEP2_LEGACY_LAYOUT
        assert "Browse" in wf.MSG_STEP2_LEGACY_LAYOUT

    def test_legacy_root_models_gets_the_hint(self, tmp_path):
        (tmp_path / "models" / "round_00").mkdir(parents=True)
        assert wf.step2_layout_hint(str(tmp_path)) == wf.MSG_STEP2_LEGACY_LAYOUT

    def test_mixed_layout_gets_the_hint(self, tmp_path):
        (tmp_path / "artifacts" / "step2" / "runs").mkdir(parents=True)
        (tmp_path / "training_data").mkdir()
        assert wf.step2_layout_hint(str(tmp_path)) == wf.MSG_STEP2_LEGACY_LAYOUT

    def test_missing_workspace_is_silent(self, tmp_path):
        assert wf.step2_layout_hint(str(tmp_path / "nope")) == ""
        assert wf.step2_layout_hint("") == ""


class TestStep2HasRuns:
    def test_no_runs_dir(self, tmp_path):
        assert wf.step2_has_runs(str(tmp_path)) is False

    def test_empty_runs_dir(self, tmp_path):
        (tmp_path / "artifacts" / "step2" / "runs").mkdir(parents=True)
        assert wf.step2_has_runs(str(tmp_path)) is False

    def test_one_run(self, tmp_path):
        (tmp_path / "artifacts" / "step2" / "runs" / "cells").mkdir(parents=True)
        assert wf.step2_has_runs(str(tmp_path)) is True

    def test_a_stray_file_is_not_a_run(self, tmp_path):
        runs = tmp_path / "artifacts" / "step2" / "runs"
        runs.mkdir(parents=True)
        (runs / "notes.txt").write_text("x", encoding="utf-8")
        assert wf.step2_has_runs(str(tmp_path)) is False


class TestStep3PreV3:
    def test_no_state_file(self, tmp_path):
        assert wf.step3_pre_v3(str(tmp_path)) is False

    def test_v3_state(self, tmp_path):
        (tmp_path / "step3_state.json").write_text(
            json.dumps({"schema_version": 3, "runs": {}}), encoding="utf-8")
        assert wf.step3_pre_v3(str(tmp_path)) is False

    def test_v2_state(self, tmp_path):
        (tmp_path / "step3_state.json").write_text(
            json.dumps({"schema_version": 2, "patches": {}}), encoding="utf-8")
        assert wf.step3_pre_v3(str(tmp_path)) is True

    def test_schema_absent_is_pre_v3(self, tmp_path):
        (tmp_path / "step3_state.json").write_text(
            json.dumps({"patches": {}}), encoding="utf-8")
        assert wf.step3_pre_v3(str(tmp_path)) is True

    def test_corrupt_state_is_not_claimed_pre_v3(self, tmp_path):
        (tmp_path / "step3_state.json").write_text("{not json", encoding="utf-8")
        assert wf.step3_pre_v3(str(tmp_path)) is False

    def test_message_tells_the_user_to_browse(self):
        assert "Browse" in wf.MSG_STEP3_PRE_V3


# ---------------------------------------------------------------------------
# B2 lifecycle audit, FIX-FIRST #2: the three new facts are read on every 3 s
# poll, and (since the advisory follows the step's own path) potentially over
# a network share.  They get `patch_count`'s treatment: a drive-letter gate
# (invariant I13) and an mtime-keyed cache.
# ---------------------------------------------------------------------------

class TestB2FactsAreCachedAndGated:
    def _count_reads(self, monkeypatch):
        """Count every `Path.read_text` / `Path.iterdir` / `Path.is_dir`."""
        seen = []
        for name in ("read_text", "iterdir", "is_dir"):
            original = getattr(Path, name)

            def _spy(self, *a, _o=original, _n=name, **k):
                seen.append(_n)
                return _o(self, *a, **k)
            monkeypatch.setattr(Path, name, _spy)
        return seen

    def test_step3_pre_v3_does_not_re_read_an_unchanged_file(
            self, tmp_path, monkeypatch):
        (tmp_path / "step3_state.json").write_text(
            json.dumps({"schema_version": 2}), encoding="utf-8")
        wf.clear_cache()
        assert wf.step3_pre_v3(str(tmp_path)) is True
        seen = self._count_reads(monkeypatch)
        for _ in range(20):                      # a minute of 3 s polling
            assert wf.step3_pre_v3(str(tmp_path)) is True
        assert "read_text" not in seen, (
            "a cached fact must not re-parse the state file on every poll")

    def test_step3_pre_v3_notices_an_edit(self, tmp_path):
        state = tmp_path / "step3_state.json"
        state.write_text(json.dumps({"schema_version": 2}), encoding="utf-8")
        wf.clear_cache()
        assert wf.step3_pre_v3(str(tmp_path)) is True
        time.sleep(0.01)
        state.write_text(json.dumps({"schema_version": 3, "runs": {}}),
                         encoding="utf-8")
        assert wf.step3_pre_v3(str(tmp_path)) is False

    def test_step2_layout_hint_is_cached(self, tmp_path, monkeypatch):
        (tmp_path / "training_data").mkdir()
        wf.clear_cache()
        assert wf.step2_layout_hint(str(tmp_path)) == wf.MSG_STEP2_LEGACY_LAYOUT
        seen = self._count_reads(monkeypatch)
        for _ in range(20):
            wf.step2_layout_hint(str(tmp_path))
        assert seen == [], f"uncached filesystem calls: {seen}"

    def test_step2_has_runs_is_cached(self, tmp_path, monkeypatch):
        (tmp_path / "artifacts" / "step2" / "runs" / "cells").mkdir(parents=True)
        wf.clear_cache()
        assert wf.step2_has_runs(str(tmp_path)) is True
        seen = self._count_reads(monkeypatch)
        for _ in range(20):
            wf.step2_has_runs(str(tmp_path))
        assert seen == [], f"uncached filesystem calls: {seen}"

    def test_step2_has_runs_notices_a_new_run(self, tmp_path):
        runs = tmp_path / "artifacts" / "step2" / "runs"
        runs.mkdir(parents=True)
        wf.clear_cache()
        assert wf.step2_has_runs(str(tmp_path)) is False
        time.sleep(0.01)
        (runs / "cells").mkdir()
        assert wf.step2_has_runs(str(tmp_path)) is True

    @pytest.mark.parametrize("path", [r"\server\share\Proj", "//server/share/Proj"])
    def test_a_network_path_is_never_walked(self, path, monkeypatch):
        """Invariant I13 — a `stat` on a disconnected share stalls for
        seconds on Windows, and this now runs on the UI thread every poll."""
        wf.clear_cache()
        seen = self._count_reads(monkeypatch)
        assert wf.step2_layout_hint(path) == ""
        assert wf.step2_has_runs(path) is False
        assert wf.step3_pre_v3(path) is False
        assert seen == [], f"network path touched the filesystem: {seen}"

    def test_layout_hint_notices_a_round_added_under_models(self, tmp_path):
        """`models/round_NN` bumps `models/`, not the workspace root — so the
        cache key has to watch both."""
        (tmp_path / "models").mkdir()
        wf.clear_cache()
        assert wf.step2_layout_hint(str(tmp_path)) == ""
        time.sleep(0.01)
        (tmp_path / "models" / "round_00").mkdir()
        assert wf.step2_layout_hint(str(tmp_path)) == wf.MSG_STEP2_LEGACY_LAYOUT

    def test_clear_cache_drops_them(self, tmp_path):
        (tmp_path / "training_data").mkdir()
        wf.clear_cache()
        assert wf.step2_layout_hint(str(tmp_path)) == wf.MSG_STEP2_LEGACY_LAYOUT
        wf.clear_cache()
        assert wf._LAYOUT_CACHE == {}
        assert wf._RUNS_CACHE == {}
        assert wf._PRE_V3_CACHE == {}


class TestAdvisoryColour:
    """Second B2 audit: a brand-new project is not a warning.

    "No run yet — click New Run" is an invitation; painting it amber next to
    the genuinely amber "old folder layout" and "no trained 2D model" lines
    made three quite different situations look equally alarming.
    """

    def test_the_no_run_line_is_not_amber(self):
        strips = wf.banner_strips("roi_hitl_2d", "C:/ws", "C:/ws",
                                  advisory=wf.MSG_STEP2_NO_RUN)
        assert strips[1][1] == "green", strips

    def test_the_old_layout_line_stays_amber(self):
        strips = wf.banner_strips("roi_hitl_2d", "C:/ws", "C:/ws",
                                  advisory=wf.MSG_STEP2_LEGACY_LAYOUT)
        assert strips[1][1] == "amber", strips

    def test_the_no_trained_model_line_stays_amber(self):
        strips = wf.banner_strips("reconstruct_curate_3d", "C:/ws", "C:/ws",
                                  advisory=wf.MSG_NO_TRAINED_MODEL)
        assert strips[1][1] == "amber", strips

    def test_an_unknown_advisory_defaults_to_amber(self):
        strips = wf.banner_strips("apply_3d", "C:/ws", "C:/ws",
                                  advisory="Something needs attention.")
        assert strips[1][1] == "amber", strips

    def test_the_no_run_line_is_green_on_a_local_only_workspace_too(self):
        strips = wf.banner_strips("roi_hitl_2d", "", "C:/ws",
                                  advisory=wf.MSG_STEP2_NO_RUN)
        assert strips[1][1] == "green", strips

    def test_the_dash_is_an_em_dash(self):
        assert "\u2014" in wf.MSG_STEP2_NO_RUN
