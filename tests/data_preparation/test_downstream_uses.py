"""Brick 4a Part 2 — "is this patch already used later in the pipeline?"

Trim rewrites a patch TIFF's Z depth in place and Delete removes it; Step 2
names its curated pairs after the patch stem and Step 3 keys every artifact on
the patch name, so both actions can silently orphan work.  These tests pin the
read-only scan that the two confirm dialogs ask.

The module loads Step 3's `_paths.py` by FILE so that resolving it imports no Qt
and never the Step 3 controller; a test below asserts exactly that (the DELTA in
`sys.modules`), rather than the looser "Qt-free" claim an earlier draft made —
importing anything under `napari_epicycle` executes its `__init__`, which pulls
qtpy.  Fixtures are plain `tmp_path` directory trees.
"""
from __future__ import annotations

import json
import os

import pytest

from napari_epicycle.data_preparation._downstream import (
    DownstreamUse,
    downstream_used_names,
    downstream_uses,
)


# ---------------------------------------------------------------------------
# Fixture helpers — hand-built fake Step 2 / Step 3 trees
# ---------------------------------------------------------------------------
def _touch(path, text="x"):
    os.makedirs(os.path.dirname(str(path)), exist_ok=True)
    with open(str(path), "w", encoding="utf-8") as f:
        f.write(text)
    return str(path)


def _mkdir(path):
    os.makedirs(str(path), exist_ok=True)
    return str(path)


def _ws(tmp_path):
    return _mkdir(tmp_path / "ws")


def _step2_pair(ws, run, bucket, rnd, label):
    return _touch(os.path.join(
        ws, "artifacts", "step2", "runs", run, bucket,
        f"round_{rnd:02d}", "accepted", f"{label}_img.tif"))


# ---------------------------------------------------------------------------
# The "nothing to see here" contract
# ---------------------------------------------------------------------------
class TestEmptyAndInvalid:
    def test_a_workspace_that_never_ran_steps_2_or_3(self, tmp_path):
        ws = _ws(tmp_path)
        use = downstream_uses(ws, "P1")
        assert use == DownstreamUse.empty()
        assert use.any is False
        assert use.summary() == ""

    def test_an_invalid_workspace_never_raises(self, tmp_path):
        assert downstream_uses(str(tmp_path / "nope"), "P1").any is False
        assert downstream_uses(None, "P1").any is False
        assert downstream_uses(str(tmp_path), "").any is False
        assert downstream_uses(str(tmp_path), None).any is False

    def test_a_file_where_the_workspace_should_be(self, tmp_path):
        f = _touch(tmp_path / "notadir")
        assert downstream_uses(f, "P1").any is False


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------
class TestStep2:
    def test_one_accepted_pair_is_found_with_its_run(self, tmp_path):
        ws = _ws(tmp_path)
        _step2_pair(ws, "cells", "lateral", 1, "P1_XY_z0003")
        use = downstream_uses(ws, "P1")
        assert use.step2_pairs == 1
        assert use.step2_runs == ("cells",)
        assert use.any is True
        assert "Step 2" in use.summary()
        assert "'cells'" in use.summary()

    def test_P1_does_not_match_P10(self, tmp_path):
        ws = _ws(tmp_path)
        _step2_pair(ws, "cells", "lateral", 1, "P10_XY_z0003")
        assert downstream_uses(ws, "P1").any is False
        assert downstream_uses(ws, "P10").step2_pairs == 1

    def test_an_underscored_patch_name_parses(self, tmp_path):
        ws = _ws(tmp_path)
        _step2_pair(ws, "cells", "lateral", 1, "my_patch_v2_XY_z0001")
        assert downstream_uses(ws, "my_patch_v2").step2_pairs == 1

    def test_a_junk_filename_does_not_raise_or_count(self, tmp_path):
        ws = _ws(tmp_path)
        _step2_pair(ws, "cells", "lateral", 1, "P1_XY_z0003")
        _touch(os.path.join(ws, "artifacts", "step2", "runs", "cells",
                            "lateral", "round_01", "accepted", "readme.txt"))
        _touch(os.path.join(ws, "artifacts", "step2", "runs", "cells",
                            "lateral", "round_01", "accepted", "..tif"))
        assert downstream_uses(ws, "P1").step2_pairs == 1

    def test_staged_and_predictions_count_too(self, tmp_path):
        ws = _ws(tmp_path)
        base = os.path.join(ws, "artifacts", "step2", "runs", "cells",
                            "axial", "round_02")
        _touch(os.path.join(base, "staged", "P1_XZ_z0001_img.tif"))
        _touch(os.path.join(base, "predictions", "P1_YZ_z0002_pred_mask.tif"))
        # round_01 must exist or the dense-round scan stops before round_02.
        _mkdir(os.path.join(ws, "artifacts", "step2", "runs", "cells",
                            "axial", "round_01"))
        assert downstream_uses(ws, "P1").step2_pairs == 2

    def test_the_legacy_layout_is_scanned(self, tmp_path):
        ws = _ws(tmp_path)
        _touch(os.path.join(ws, "training_data", "round_01",
                            "accepted_pairs", "spatial", "P1_XY_z0001_img.tif"))
        use = downstream_uses(ws, "P1")
        assert use.step2_pairs == 1
        assert use.step2_runs == ("default",)


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------
class TestStep3:
    def test_flat_default_run_reconstruction(self, tmp_path):
        ws = _ws(tmp_path)
        _touch(os.path.join(ws, "artifacts", "step3", "runs", "P1",
                            "labels_3d.tif"))
        use = downstream_uses(ws, "P1")
        assert use.step3_runs == ("default",)
        assert "reconstruction" in use.step3_kinds
        assert use.any is True

    def test_a_named_run_needs_its_manifest(self, tmp_path):
        ws = _ws(tmp_path)
        runs = os.path.join(ws, "artifacts", "step3", "runs")
        _touch(os.path.join(runs, "cells", "run_manifest.json"), "{}")
        _mkdir(os.path.join(runs, "cells", "reconstructions", "P1"))
        use = downstream_uses(ws, "P1")
        assert use.step3_runs == ("cells",)

    def test_a_patch_dir_under_runs_is_not_reported_as_a_run(self, tmp_path):
        """`step3_runs_root` aliases `runs_dir`, which in the flat layout also
        holds the DEFAULT run's per-patch dirs — those are not runs."""
        ws = _ws(tmp_path)
        runs = os.path.join(ws, "artifacts", "step3", "runs")
        _mkdir(os.path.join(runs, "P1"))
        _mkdir(os.path.join(runs, "P2"))
        use = downstream_uses(ws, "P1")
        assert use.step3_runs == ("default",), use.step3_runs
        assert "P2" not in use.step3_runs

    def test_qc_curation_pairs_and_archive_each_count(self, tmp_path):
        ws = _ws(tmp_path)
        s3 = os.path.join(ws, "artifacts", "step3")
        _touch(os.path.join(s3, "qc", "P1_features.csv"))
        _touch(os.path.join(s3, "curation", "P1", "tiers.csv"))
        _mkdir(os.path.join(s3, "refinement", "pairs", "P1", "spatial"))
        _mkdir(os.path.join(s3, "training_archive", "P1"))
        use = downstream_uses(ws, "P1")
        for kind in ("QC analysis", "curation", "training pairs",
                     "training archive"):
            assert kind in use.step3_kinds, (kind, use.step3_kinds)

    def test_prediction_cache_and_snapshot_are_stem_keyed(self, tmp_path):
        ws = _ws(tmp_path)
        s3 = os.path.join(ws, "artifacts", "step3")
        _touch(os.path.join(s3, "prediction_cache", "P1_flows.npy"))
        _touch(os.path.join(s3, "snapshots", "P1_pre_commit_2026.tif"))
        use = downstream_uses(ws, "P1")
        assert "prediction cache" in use.step3_kinds
        assert "pre-commit snapshot" in use.step3_kinds
        # P1 must not be claimed by P10's cache.
        assert "prediction cache" not in downstream_uses(ws, "P10").step3_kinds

    def test_legacy_roots_are_scanned(self, tmp_path):
        ws = _ws(tmp_path)
        _mkdir(os.path.join(ws, "artifacts", "reconstruction", "P1"))
        _touch(os.path.join(ws, "artifacts", "qc", "P1_features.csv"))
        use = downstream_uses(ws, "P1")
        assert use.any is True
        assert "reconstruction" in use.step3_kinds

    @pytest.mark.parametrize("envelope", [True, False])
    def test_state_json_is_read_in_both_shapes(self, tmp_path, envelope):
        ws = _ws(tmp_path)
        rows = {"P1": {"status": "reconstructed"}}
        state = ({"schema_version": 3, "runs": {"default": {"patches": rows}}}
                 if envelope else {"schema_version": 1, "patches": rows})
        with open(os.path.join(ws, "step3_state.json"), "w",
                  encoding="utf-8") as f:
            json.dump(state, f)
        assert downstream_uses(ws, "P1").any is True

    def test_an_unprocessed_status_does_not_count(self, tmp_path):
        ws = _ws(tmp_path)
        with open(os.path.join(ws, "step3_state.json"), "w",
                  encoding="utf-8") as f:
            json.dump({"patches": {"P1": {"status": "unprocessed"}}}, f)
        assert downstream_uses(ws, "P1").any is False


# ---------------------------------------------------------------------------
# Batched sweep
# ---------------------------------------------------------------------------
class TestBatched:
    def test_matches_n_individual_calls(self, tmp_path):
        from napari_epicycle.data_preparation._state import (
            empty_state, register_patch, save_state,
        )

        ws = _ws(tmp_path)
        st = empty_state()
        for name in ("P1", "P2", "P3"):
            register_patch(
                st, patch_name=name,
                relative_path=f"artifacts/step1/patches/{name}.tiff",
                sidecar_path=f"artifacts/step1/patches/{name}_metadata.json",
                source_id="src_0")
        save_state(ws, st)

        _step2_pair(ws, "cells", "lateral", 1, "P1_XY_z0003")
        _touch(os.path.join(ws, "artifacts", "step3", "runs", "P3",
                            "labels_3d.tif"))

        batched = downstream_used_names(ws)
        one_by_one = {n for n in ("P1", "P2", "P3")
                      if downstream_uses(ws, n).any}
        assert batched == one_by_one == {"P1", "P3"}

    def test_empty_on_a_never_run_workspace(self, tmp_path):
        assert downstream_used_names(_ws(tmp_path)) == set()
        assert downstream_used_names(str(tmp_path / "nope")) == set()


# ---------------------------------------------------------------------------
# Resolving the path modules pulls no Qt and no Step 3 controller
# ---------------------------------------------------------------------------
def test_resolving_the_path_modules_pulls_no_qt_and_no_step3_package():
    """The whole reason Step 3's `_paths.py` is loaded by FILE."""
    import subprocess
    import sys

    # The parent `napari_epicycle` packages pull Qt on their own; what must
    # stay true is that resolving Steps 2 + 3's path helpers adds NOTHING —
    # in particular never `napari_epicycle.fusion_reconstruction`, whose
    # `__init__` imports the 18k-line Qt controller.
    code = (
        "import sys;"
        "import napari_epicycle.data_preparation._downstream as d;"
        "before=set(sys.modules);"
        "d._load_fusion_paths();"
        "d._load_step2_paths();"
        "added=sorted(m for m in set(sys.modules)-before "
        "if m.startswith('qtpy') or m.startswith('PyQt') "
        "or m.startswith('napari_epicycle.fusion_reconstruction'));"
        "print(added)"
    )
    out = subprocess.run([sys.executable, "-c", code],
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "[]", out.stdout


# ---------------------------------------------------------------------------
# The controller wiring: Trim + Delete ask before they act
# ---------------------------------------------------------------------------
class TestControllerGuard:
    def _ctrl(self, ws):
        import types

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )

        class _T:
            def __init__(self, v):
                self._v = v

            def text(self):
                return self._v

        c = types.SimpleNamespace(
            widget=types.SimpleNamespace(output_file_path=_T(ws)),
            _log=lambda *a, **k: None,
        )
        c._downstream_for = lambda n: C._downstream_for(c, n)
        return c

    def test_work_created_deep_in_the_tree_is_seen_on_the_next_call(
            self, tmp_path):
        """Brick 4c item 3: the dropped cache was keyed on the mtime of
        `artifacts/step2` + `artifacts/step3`, which a write two or three levels
        down never touches -- so a run created mid-session stayed invisible and
        a used patch got the PLAIN confirm."""
        ws = _ws(tmp_path)
        c = self._ctrl(ws)
        assert c._downstream_for("P1").any is False

        # A named Step 3 run, two levels below `artifacts/step3`.
        runs = os.path.join(ws, "artifacts", "step3", "runs")
        _touch(os.path.join(runs, "cells", "run_manifest.json"), "{}")
        _mkdir(os.path.join(runs, "cells", "reconstructions", "P1"))
        assert c._downstream_for("P1").any is True

        # And a Step 2 slice, three levels below `artifacts/step2`.
        c2 = self._ctrl(ws)
        assert c2._downstream_for("P2").any is False
        _step2_pair(ws, "cells", "lateral", 1, "P2_XY_z0001")
        assert c2._downstream_for("P2").any is True

    def test_no_cache_attribute_survives(self):
        """The stale-prone cache is gone, not merely bypassed."""
        import inspect

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        assert not hasattr(C, "_downstream_roots_key")
        assert "_downstream_cache" not in inspect.getsource(C._downstream_for)

    def test_a_broken_scan_answers_empty_and_never_raises(self, tmp_path,
                                                          monkeypatch):
        import napari_epicycle.data_preparation._downstream as d

        ws = _ws(tmp_path)
        c = self._ctrl(ws)
        monkeypatch.setattr(
            d, "downstream_uses",
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
        assert c._downstream_for("P1").any is False


# ---------------------------------------------------------------------------
# Brick 4c item 8 — the confirm branches, DRIVEN (not read out of the source)
# ---------------------------------------------------------------------------
class _FakeBox:
    """Enough QMessageBox for `_confirm_downstream`, recording what it was told."""

    Warning = object()
    AcceptRole = object()
    RejectRole = object()
    last = None

    def __init__(self, _parent=None):
        self.text = ""
        self.informative = ""
        self.title = ""
        self.buttons = []
        self.default = None
        self._clicked = None
        _FakeBox.last = self

    def setIcon(self, _i):
        pass

    def setWindowTitle(self, t):
        self.title = t

    def setText(self, t):
        self.text = t

    def setInformativeText(self, t):
        self.informative = t

    def addButton(self, label, role):
        btn = (label, role)
        self.buttons.append(btn)
        return btn

    def setDefaultButton(self, b):
        self.default = b

    def exec_(self):
        # Always "click" whatever was made the DEFAULT -- so a test that gets a
        # go-ahead proves the default was NOT Cancel.
        self._clicked = self.default

    def clickedButton(self):
        return self._clicked


@pytest.fixture
def fake_message_box(monkeypatch):
    qtw = pytest.importorskip("qtpy.QtWidgets")
    monkeypatch.setattr(qtw, "QMessageBox", _FakeBox, raising=False)
    return _FakeBox


class TestConfirmDownstreamDriven:
    def _ctrl(self):
        import types

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        c = types.SimpleNamespace(widget=None)
        c._confirm_downstream = lambda *a: C._confirm_downstream(c, *a)
        return c

    def _use(self, **kw):
        from napari_epicycle.data_preparation._downstream import DownstreamUse

        return DownstreamUse(**kw)

    def test_trim_asks_with_cancel_as_the_default(self, fake_message_box):
        c = self._ctrl()
        use = self._use(step2_pairs=3, step2_runs=("cells",),
                        step3_runs=("default",),
                        step3_kinds=("reconstruction",))
        went_ahead = c._confirm_downstream("P1", use, "trim")

        box = fake_message_box.last
        assert box.text == (
            "This patch is already being used later in the pipeline.")
        assert "P1 is used by" in box.informative
        assert "3 slices you already curated in Step 2" in box.informative
        assert "cells" in box.informative
        assert "slice numbering" in box.informative
        assert "will not change or delete anything" in box.informative
        assert [lbl for lbl, _r in box.buttons] == ["Trim anyway", "Cancel"]
        # The fake clicks the DEFAULT, so a False result proves Cancel is it.
        assert box.default == ("Cancel", _FakeBox.RejectRole)
        assert went_ahead is False

    def test_delete_asks_with_its_own_wording(self, fake_message_box):
        c = self._ctrl()
        use = self._use(step3_runs=("cells",), step3_kinds=("curation",))
        assert c._confirm_downstream("P1", use, "delete") is False

        box = fake_message_box.last
        assert [lbl for lbl, _r in box.buttons] == ["Delete anyway", "Cancel"]
        assert "removes the image from Step 1 only" in box.informative
        assert "no longer exists" in box.informative
        assert "delete that patch" in box.informative

    def test_going_ahead_is_reported(self, fake_message_box):
        c = self._ctrl()
        use = self._use(step2_pairs=1, step2_runs=("cells",))
        real_exec = _FakeBox.exec_
        try:
            # The user picks the non-default (go-ahead) button.
            _FakeBox.exec_ = lambda self: setattr(
                self, "_clicked", self.buttons[0])
            assert c._confirm_downstream("P1", use, "trim") is True
        finally:
            _FakeBox.exec_ = real_exec

    def test_an_unused_patch_never_reaches_this_dialog(self):
        """The plain confirms are untouched: both sites branch on `use.any`."""
        import inspect

        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C,
        )
        for fn in (C._on_set_zrange_clicked, C._on_delete_patch):
            src = inspect.getsource(fn)
            assert "if use.any:" in src
            assert "QMessageBox.No)" in src      # the plain confirm is still there


class TestStemSeparatorIsRequired:
    """`P1` must never claim `P10`'s artifacts.

    Fails loudly if `_stem_belongs` is ever loosened to a bare
    ``startswith(patch)``."""

    def test_the_separator_is_load_bearing(self):
        from napari_epicycle.data_preparation._downstream import _stem_belongs

        assert _stem_belongs("P1", "P1") is True
        assert _stem_belongs("P1_flows", "P1") is True
        assert _stem_belongs("P1_pre_commit_2026", "P1") is True
        # The loosening this guards against:
        assert _stem_belongs("P10", "P1") is False
        assert _stem_belongs("P10_flows", "P1") is False
        assert _stem_belongs("P1x", "P1") is False

    def test_end_to_end_through_the_prediction_cache(self, tmp_path):
        ws = _ws(tmp_path)
        _touch(os.path.join(ws, "artifacts", "step3", "prediction_cache",
                            "P10_flows.npy"))
        assert downstream_uses(ws, "P1").any is False
        assert downstream_uses(ws, "P10").any is True


class TestSummaryGrammar:
    def _u(self, **kw):
        from napari_epicycle.data_preparation._downstream import DownstreamUse

        return DownstreamUse(**kw)

    def test_three_kinds_read_as_a_list(self):
        s = self._u(step3_runs=("default",),
                    step3_kinds=("reconstruction", "QC analysis",
                                 "curation")).summary()
        assert s == "Step 3 reconstruction, QC analysis and curation"

    def test_two_kinds(self):
        s = self._u(step3_runs=("default",),
                    step3_kinds=("reconstruction", "curation")).summary()
        assert s == "Step 3 reconstruction and curation"

    def test_one_kind(self):
        s = self._u(step3_runs=("default",),
                    step3_kinds=("reconstruction",)).summary()
        assert s == "a Step 3 reconstruction"

    def test_singular_and_plural_slices(self):
        assert "1 slice you already curated" in self._u(
            step2_pairs=1, step2_runs=("cells",)).summary()
        assert "2 slices you already curated" in self._u(
            step2_pairs=2, step2_runs=("cells",)).summary()

    def test_both_halves_join_with_and(self):
        s = self._u(step2_pairs=1, step2_runs=("cells",),
                    step3_runs=("default",),
                    step3_kinds=("reconstruction",)).summary()
        assert " and a Step 3 reconstruction" in s


class TestRoundGapsDoNotHideWork:
    def test_a_deleted_round_does_not_hide_later_rounds(self, tmp_path):
        """Brick 4c item 8: the scan used to ``break`` at the first missing
        ``round_NN``, so archiving round 1 made every later round invisible and
        a curated patch was reported as unused."""
        ws = _ws(tmp_path)
        # round_01 absent on purpose; the work lives in round_03.
        _step2_pair(ws, "cells", "lateral", 3, "P1_XY_z0007")
        use = downstream_uses(ws, "P1")
        assert use.step2_pairs == 1
        assert use.step2_runs == ("cells",)
