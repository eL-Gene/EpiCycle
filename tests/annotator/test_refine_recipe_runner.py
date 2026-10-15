"""Recording the chain.

Bare fake, the REAL ``_preview_smooth`` and the REAL panel's values (the
``test_refine_cross_method_chain.py`` pattern).  Invariant 7: the records
match the breadcrumb after every event; a step is recorded iff it wrote.
"""

from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_annotator import refine_recipe as rr
from tests.annotator import _recipe_fake as rf


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture(autouse=True)
def _labels(monkeypatch, qapp):
    import napari_epicycle.annotator.curation_3d_widget as mod

    monkeypatch.setattr(mod, "Labels", rf._Layer)


@pytest.fixture
def panel(qapp):
    w = rf.make_panel()
    yield w
    w.setParent(None)


def _fake(panel, **kw):
    return rf.Fake(rf.vol_with_notch(), panel, **kw)


def _set(panel, method, **params):
    base = dict(panel.get_refine_step_params(method)) if method not in (
        "sdf", "contour", "settle") else {}
    if method == "sdf":
        base = {"sigma_z": 1.5, "sigma_xy": 1.0, "preserve_topology": True}
    base.update(params)
    ok, why = panel.set_refine_step_params(method, base)
    assert ok, why
    return base


def _r1(f):
    assert len(f._refine_step_records) == len(f._refine_steps), (
        f._refine_step_records, f._refine_steps)


class TestRecords:
    def test_a_three_step_chain_records_what_ran(self, panel):
        f = _fake(panel)
        a = _set(panel, "fill_holes", axis="3D")
        f._preview_smooth()
        _r1(f)
        b = _set(panel, "sdf", sigma_z=2.5, sigma_xy=0.7)
        f._preview_smooth()
        _r1(f)
        c = _set(panel, "dilate_erode", radius=1, axis="XY")
        f._preview_smooth()
        _r1(f)
        assert f._refine_steps == ["Fill holes", "Smooth", "Dilate/Erode"]
        got = [(r.method, r.params) for r in f._refine_step_records]
        assert got == [("fill_holes", a), ("sdf", b), ("dilate_erode", c)]
        for r in f._refine_step_records:
            r.validate()

    def test_a_refused_step_records_nothing(self, panel):
        f = _fake(panel)                     # no probability map
        _set(panel, "sdf")
        f._preview_smooth()
        panel.refine_method_combo.setCurrentIndex(3)       # guided grow
        f._preview_smooth()
        _r1(f)
        assert [r.method for r in f._refine_step_records] == ["sdf"]

    def test_a_step_that_moves_nothing_records_nothing(self, panel):
        f = _fake(panel)
        _set(panel, "fill_holes")
        f._preview_smooth()                  # fills the cavity
        assert f._refine_steps == ["Fill holes"]
        _set(panel, "fill_holes")
        f._preview_smooth()                  # nothing left to fill
        _r1(f)
        assert len(f._refine_step_records) == 1
        assert f._refine_pending_record is None   # captured, then dropped (L5)

    def test_a_first_step_that_is_refused_records_nothing(self, panel):
        f = _fake(panel)
        panel.refine_method_combo.setCurrentIndex(3)
        f._preview_smooth()
        _r1(f)
        assert f._refine_step_records == []

    def test_flood_fill_is_recorded_like_any_other(self, panel, monkeypatch):
        f = _fake(panel)
        _set(panel, "sdf")
        f._preview_smooth()
        from napari_epicycle.annotator import curation_3d_widget as mod

        r = mod._refine_capture_record(f, "flood_fill", {}, None)
        assert r.method == "flood_fill"
        assert set(r.params) == {"axis", "max_size"}

    def test_capture_is_AFTER_the_pump(self, panel, monkeypatch):
        """A spin-box edit the pump delivers is recorded exactly when the
        step uses it."""
        from qtpy.QtWidgets import QApplication

        f = _fake(panel)
        _set(panel, "dilate_erode", radius=1, axis="3D")
        real = QApplication.processEvents

        def pump(*a):
            panel.refine_de_radius_sb.setValue(3)
            return real(*a)

        monkeypatch.setattr(QApplication, "processEvents", staticmethod(pump))
        before = f.vol().copy()
        f._preview_smooth()
        assert f._refine_step_records[0].params["radius"] == 3
        # ...and the volume is the radius-3 result
        from packages.epicycle_annotator.refine import dilate_erode_label

        roi = f._smooth_roi
        want, _n, _m = dilate_erode_label(before[roi], 7, radius=3, axis="3D")
        assert np.array_equal(f.vol()[roi], want)


class TestTheSevenClears:
    def _staged(self, panel):
        f = _fake(panel)
        _set(panel, "sdf")
        f._preview_smooth()
        assert f._refine_step_records
        f._refine_chain_recipe_names = ["tidy"]
        return f

    def _cleared(self, f):
        assert f._refine_step_records == []
        assert f._refine_pending_record is None
        assert f._refine_chain_recipe_names == []
        _r1(f)

    def test_init(self):
        from napari_epicycle.annotator import curation_3d_widget as mod
        import inspect

        src = inspect.getsource(mod.Curation3DAnnotator.__init__)
        assert "self._refine_step_records = []" in src
        assert "self._refine_pending_record = None" in src

    def test_accept(self, panel):
        f = self._staged(panel)
        f._accept_smooth()
        self._cleared(f)

    def test_reject(self, panel):
        f = self._staged(panel)
        f._reject_smooth()
        self._cleared(f)

    def test_reset(self, panel):
        f = self._staged(panel)
        f._reset_smooth_state("x")
        self._cleared(f)

    def test_stopped(self, panel):
        f = self._staged(panel)
        f._on_refine_stopped()
        self._cleared(f)

    def test_teardown(self, panel):
        f = self._staged(panel)
        f._teardown_refine_for_switch()
        self._cleared(f)

    def test_a_new_first_preview(self, panel):
        f = self._staged(panel)
        f._smooth_running = False            # a session that ended elsewhere
        _set(panel, "sdf")
        f._preview_smooth()
        assert len(f._refine_step_records) == 1
        assert f._refine_chain_recipe_names == []

    def test_exactly_seven_sites_clear_the_breadcrumb(self):
        import inspect

        from napari_epicycle.annotator import curation_3d_widget as mod

        src = inspect.getsource(mod)
        assert src.count("self._refine_steps = []") == 7
        # six of them clear the records with it; the seventh is __init__,
        # which declares them.
        assert src.count("_refine_clear_records(self)") == 6


class TestProvenanceCopy:
    def test_accept_copies_before_it_clears(self, panel):
        f = _fake(panel)
        _set(panel, "sdf")
        f._preview_smooth()
        _set(panel, "fill_holes")
        f._preview_smooth()
        from napari_epicycle.annotator import curation_3d_widget as mod

        steps, recipes = mod._refine_provenance(f)
        assert [s["method"] for s in steps] == ["sdf", "fill_holes"]
        assert recipes == []
        assert rr.recipe_to_json(f._refine_step_records, created="",
                                 spacing_known=True, spacing=())


# ===========================================================================
# The runner (synchronous steps; the real
# worker is in test_refine_recipe_worker.py).
# ===========================================================================


def drain(f, rounds=60):
    from qtpy.QtWidgets import QApplication

    for _ in range(rounds):
        QApplication.processEvents()
        if f._refine_recipe_run is None:
            QApplication.processEvents()
            return
    raise AssertionError("the recipe run did not end")


SDF = {"sigma_z": 1.5, "sigma_xy": 1.0, "preserve_topology": True}


def recipe(*steps, known=True):
    return rr.recipe_to_json([rr.RecipeStep(m, dict(p)) for m, p in steps],
                             created="t", spacing_known=known,
                             spacing=(3.0, 1.0, 1.0) if known else ())


TIDY = (("fill_holes", {"axis": "3D"}), ("sdf", SDF),
        ("dilate_erode", {"radius": 1, "axis": "3D"}))


def manual(panel, vol, steps):
    """The same steps pressed by hand on a fresh fake."""
    g = rf.Fake(vol, panel)
    for m, p in steps:
        ok, why = panel.set_refine_step_params(m, dict(p))
        assert ok, why
        g._preview_smooth()
    return g


class TestSameVoxels:
    def test_a_recipe_equals_the_same_manual_steps(self, panel):
        f = _fake(panel)
        before = f.vol().copy()
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        drain(f)
        assert f._smooth_chain_count == 3
        assert f._refine_steps == ["Fill holes", "Smooth", "Dilate/Erode"]
        assert panel.smooth_accept_btn.text() == "Accept (3)"
        assert panel.smooth_accept_btn.isEnabled()
        assert panel.smooth_reject_btn.isEnabled()
        assert panel.smooth_status_label.text() == (
            "The recipe ran all 3 steps on object 7.")
        assert panel.refine_stage_label.toolTip() == ""
        assert f._refine_chain_recipe_names == ["tidy"]
        _r1(f)
        g = manual(panel, before.copy(), TIDY)
        assert np.array_equal(f.vol(), g.vol())

    def test_dilate_plus_10_twice_grows_the_region_exactly(self, panel):
        big = np.zeros((20, 90, 90), np.int32)
        big[6:14, 35:55, 35:55] = 7
        steps = (("dilate_erode", {"radius": 10, "axis": "XY"}),
                 ("dilate_erode", {"radius": 10, "axis": "XY"}))
        f = rf.Fake(big.copy(), panel)
        f.store["grow"] = recipe(*steps)
        f._on_recipe_run("grow")
        drain(f)
        g = manual(panel, big.copy(), steps)
        assert f._smooth_chain_count == 2
        assert np.array_equal(f.vol(), g.vol())
        f._reject_smooth()
        assert np.array_equal(f.vol(), big)

    def test_what_was_saved_is_what_ran(self, panel, monkeypatch):
        """Invariant 8 -- a spy on the engine functions."""
        import packages.epicycle_annotator.refine as refine
        import packages.epicycle_annotator.smooth as smooth

        seen = {}
        real_de, real_sdf = refine.dilate_erode_label, smooth.smooth_label_sdf

        def de(roi, lid, radius, axis):
            seen["de"] = dict(radius=radius, axis=axis)
            return real_de(roi, lid, radius=radius, axis=axis)

        def sdf(roi, lid, sigma_z, sigma_xy, preserve_topology):
            seen["sdf"] = dict(sigma_z=sigma_z, sigma_xy=sigma_xy,
                               preserve_topology=preserve_topology)
            return real_sdf(roi, lid, sigma_z=sigma_z, sigma_xy=sigma_xy,
                            preserve_topology=preserve_topology)

        monkeypatch.setattr(refine, "dilate_erode_label", de)
        monkeypatch.setattr(smooth, "smooth_label_sdf", sdf)
        steps = (("sdf", {"sigma_z": 2.5, "sigma_xy": 0.7,
                          "preserve_topology": False}),
                 ("dilate_erode", {"radius": -2, "axis": "XZ"}))
        f = _fake(panel)
        f.store["spy"] = recipe(*steps)
        f._on_recipe_run("spy")
        drain(f)
        assert seen["sdf"] == steps[0][1]
        assert seen["de"] == steps[1][1]
        assert [(r.method, r.params) for r in f._refine_step_records] == [
            (m, p) for m, p in steps]


class TestAppendAndDiscard:
    def test_append_to_a_staged_chain_and_discard_exactly(self, panel):
        f = _fake(panel)
        before = f.vol().copy()
        panel.set_refine_step_params("sdf", SDF)
        f._preview_smooth()                          # staged by hand
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        assert f.confirms == [
            "Add 'tidy' (3 steps) after the 1 step already staged on object "
            "7? Discard would then undo all 4."]
        assert f.confirm_buttons == [("Add", "Cancel")]
        drain(f)
        assert f._smooth_chain_count == 4
        assert panel.smooth_accept_btn.text() == "Accept (4)"
        f._on_refine_discard_clicked()
        assert np.array_equal(f.vol(), before)

    def test_declining_the_confirm_runs_nothing(self, panel):
        f = _fake(panel)
        panel.set_refine_step_params("sdf", SDF)
        f._preview_smooth()
        staged = f.vol().copy()
        f.confirm_answer = False
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        assert f._refine_recipe_run is None
        assert np.array_equal(f.vol(), staged) and f._smooth_chain_count == 1

    def test_no_confirm_when_nothing_is_staged(self, panel):
        f = _fake(panel)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        drain(f)
        assert f.confirms == []


class TestRefusalStopsAndKeeps:
    def test_nothing_moved_at_step_2_keeps_step_1(self, panel):
        f = _fake(panel)
        f.store["r"] = recipe(("fill_holes", {"axis": "3D"}),
                              ("fill_holes", {"axis": "3D"}), ("sdf", SDF))
        f._on_recipe_run("r")
        drain(f)
        assert f._smooth_running and f._smooth_chain_count == 1
        assert f._refine_steps == ["Fill holes"]
        assert panel.refine_stage_label.toolTip() == (
            "Stopped at 2/3 · Fill Holes — 'r'")
        assert panel.smooth_status_label.text().startswith(
            "Nothing moved: this object has no enclosed cavity.")
        assert panel.smooth_accept_btn.isEnabled()
        _r1(f)

    def test_nothing_moved_at_step_1_of_a_fresh_chain(self, panel):
        vol = rf.vol_with_notch()
        vol[12:16, 18:22, 18:22] = 7                 # no cavity
        f = rf.Fake(vol, panel)
        before = vol.copy()
        f.store["r"] = recipe(("fill_holes", {"axis": "3D"}), ("sdf", SDF))
        f._on_recipe_run("r")
        drain(f)
        assert not f._smooth_running
        assert np.array_equal(f.vol(), before)
        assert panel.refine_stage_label.toolTip() == (
            "Stopped at 1/2 · Fill Holes — 'r'")
        assert not panel.smooth_accept_btn.isEnabled()
        assert panel.smooth_preview_btn.isEnabled()

    def test_the_stage_line_stays_until_the_next_preview(self, panel):
        f = _fake(panel)
        f.store["r"] = recipe(("fill_holes", {"axis": "3D"}),
                              ("fill_holes", {"axis": "3D"}))
        f._on_recipe_run("r")
        drain(f)
        assert panel.refine_stage_label.toolTip()
        f._on_refine_preview_clicked()
        assert panel.refine_stage_label.toolTip() == ""

    def test_the_emptied_ending(self, panel):
        f = _fake(panel)
        f.store["e"] = recipe(("dilate_erode", {"radius": -10, "axis": "XY"}),
                              ("sdf", SDF))
        f._on_recipe_run("e")
        drain(f)
        assert f._smooth_running and f._smooth_chain_count == 1
        assert not (f.vol() == 7).any()
        assert panel.smooth_accept_btn.text() == "Delete object 7"
        assert panel.refine_stage_label.toolTip() == (
            "Stopped at 1/2 · the object is empty — 'e'")
        assert panel.refine_result_warning.text().startswith(
            "This removes the whole object.")

    def test_a_panel_that_cannot_be_set_ends_as_refused(self, panel,
                                                        monkeypatch):
        f = _fake(panel)
        real = panel.set_refine_step_params
        calls = []

        def setter(m, p):
            calls.append(m)
            if len(calls) == 2:
                return False, rr.STAGE_PREP_DIFFERS
            return real(m, p)

        monkeypatch.setattr(panel, "set_refine_step_params", setter)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        drain(f)
        assert f._smooth_chain_count == 1
        assert panel.refine_stage_label.toolTip() == (
            "Stopped at 2/3 · preparation differs — 'tidy'")
        assert panel.smooth_status_label.text() == (
            "Recipe stopped at step 2 of 3: its Smooth settings do not fit "
            "this page. The steps before it are still staged.")


class TestPreflight:
    """Invariant 3: refused before step 1, nothing written."""

    def _refused(self, f, name, sentence):
        before = f.vol().copy()
        steps_before = list(f._refine_steps)
        f._on_recipe_run(name)
        assert f._refine_recipe_run is None
        assert np.array_equal(f.vol(), before)
        assert f._refine_steps == steps_before
        assert f.adv.smooth_status_label.text() == sentence

    def test_no_object(self, panel):
        f = _fake(panel)
        f.layer.selected_label = 0
        f.store["tidy"] = recipe(*TIDY)
        self._refused(f, "tidy", "Select an object first.")

    def test_a_different_object(self, panel):
        f = _fake(panel)
        panel.set_refine_step_params("sdf", SDF)
        f._preview_smooth()
        f.layer.selected_label = 8
        f.store["tidy"] = recipe(*TIDY)
        self._refused(f, "tidy", "Object 7 has steps staged. Accept or "
                      "Discard them before running a recipe on object 8.")

    def test_a_blank_channel(self, panel):
        f = _fake(panel)
        panel.refine_method_combo.setCurrentIndex(6)
        settle = dict(panel.get_refine_step_params("settle"))
        f.store["s"] = recipe(("sdf", SDF), ("dilate_erode",
                              {"radius": 1, "axis": "3D"}),
                              ("settle", settle))
        self._refused(f, "s", "The recipe needs a channel for step 3 "
                      "(Settle). Pick one on this page first.")

    def test_a_greyed_guided_grow(self, panel):
        f = _fake(panel)
        panel.set_refine_guided_grow_available(False)
        f.store["g"] = recipe(("sdf", SDF), ("guided_grow", {
            "max_radius": 5, "prob_threshold": 0.5, "axis": "3D"}))
        self._refused(f, "g", "The recipe's step 2 (Guided Grow) needs a "
                      "probability map, and this patch has none.")

    def test_a_missing_solver(self, panel):
        f = _fake(panel, channel_roi=np.ones((28, 40, 40)))
        panel.set_refine_solver_available(False, "no")
        panel.refine_method_combo.setCurrentIndex(6)
        settle = dict(panel.get_refine_step_params("settle"),
                      wall_kind="distance", solver="label_diffusion")
        f.store["d"] = recipe(("sdf", SDF), ("fill_holes", {"axis": "3D"}),
                              ("settle", settle))
        self._refused(f, "d", "The recipe's step 3 (Settle) needs "
                      "u-Segment3D, which is not installed.")

    def test_a_probability_wall_with_no_map(self, panel):
        f = _fake(panel)
        panel.refine_method_combo.setCurrentIndex(6)
        settle = dict(panel.get_refine_step_params("settle"),
                      wall_kind="probability")
        f.store["p"] = recipe(("sdf", SDF), ("fill_holes", {"axis": "3D"}),
                              ("settle", settle))
        self._refused(f, "p", "The recipe's step 3 (Settle) needs a "
                      "probability map, and this patch has none.")

    def test_spacing_known_differs_both_ways(self, panel):
        f = _fake(panel, spacing=(1.0, 1.0, 1.0))      # unknown here
        f.store["k"] = recipe(*TIDY, known=True)
        self._refused(f, "k", rr.S1_SPACING_KNOWN)
        g = _fake(panel)                               # known here
        g.store["u"] = recipe(*TIDY, known=False)
        self._refused(g, "u", rr.S1_SPACING_UNKNOWN)

    def test_fragment_and_interp_refine(self, panel):
        f = _fake(panel)
        f.store["tidy"] = recipe(*TIDY)
        f._split_fragment_refine = True
        self._refused(f, "tidy", rr.HEADER_FRAGMENT)
        f._split_fragment_refine = False
        f._interp_refine_mode = True
        self._refused(f, "tidy", rr.HEADER_INTERP)

    def test_busy(self, panel):
        f = _fake(panel)
        f.store["tidy"] = recipe(*TIDY)
        f._batch_running = True
        self._refused(f, "tidy", "Finish or close Batch Refine first.")
        f._batch_running = False
        f._refine_running = True
        self._refused(f, "tidy", "Still working - wait, or press Stop.")

    def test_an_unreadable_recipe(self, panel):
        f = _fake(panel)
        f.store["old"] = [1, 2]
        self._refused(f, "old", rr.S1_UNREADABLE)


class TestClicksDuringARun:
    def test_buttons_held_and_the_thin_slots_answer(self, panel):
        f = _fake(panel)
        before = f.vol().copy()
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        # step 1 wrote synchronously; the judge is queued: mid-run
        assert f._refine_recipe_run is not None
        assert not panel.smooth_preview_btn.isEnabled()
        assert not panel.smooth_accept_btn.isEnabled()
        assert not panel.smooth_reject_btn.isEnabled()
        assert panel.smooth_preview_btn.toolTip() == rr.S1_RUNNING_BLOCK
        assert panel.refine_cancel_preview_btn.isHidden() is False
        assert panel.refine_cancel_preview_btn.isEnabled()
        assert panel.refine_stage_label.text().startswith("1/3")
        staged = f.vol().copy()
        f._on_refine_accept_clicked()
        f._on_refine_discard_clicked()
        f._on_refine_preview_clicked()
        assert panel.smooth_status_label.text() == rr.S1_RUNNING_BLOCK
        assert np.array_equal(f.vol(), staged)
        assert f.smoothed == []
        drain(f)
        assert f._smooth_chain_count == 3
        assert panel.smooth_accept_btn.isEnabled()
        assert panel.refine_cancel_preview_btn.isHidden() is True
        assert not np.array_equal(f.vol(), before)

    def test_stop_between_steps_is_whole(self, panel):
        """Invariant 5, between steps."""
        f = _fake(panel)
        before = f.vol().copy()
        panel.set_refine_step_params("sdf", SDF)
        f._preview_smooth()                          # 1 staged before
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        assert f._refine_recipe_run is not None
        assert "including the 1 step that was staged" in (
            panel.refine_cancel_preview_btn.toolTip())
        f._cancel_refine_preview()
        assert f._refine_recipe_run is None
        assert np.array_equal(f.vol(), before)
        assert not f._smooth_running
        assert panel.smooth_status_label.text() == (
            "Recipe stopped at step 1 of 3. The object is back as it was "
            "before the first staged step.")
        drain(f)                         # the queued judge does nothing
        assert np.array_equal(f.vol(), before)
        assert panel.refine_cancel_preview_btn.toolTip() == (
            "Stops, and puts the object back as it was before the first "
            "step.")

    def test_stop_between_steps_on_a_fresh_chain(self, panel):
        f = _fake(panel)
        before = f.vol().copy()
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        f._cancel_refine_preview()
        assert np.array_equal(f.vol(), before)
        assert panel.smooth_status_label.text() == (
            "Recipe stopped at step 1 of 3. The object is back as it was "
            "before the recipe.")


class TestTheJudge:
    def test_exactly_one_judge_per_step(self, panel, monkeypatch):
        f = _fake(panel)
        K = rf.K()
        calls = []
        real = K._recipe_after_step

        def spy(self_, token, i):
            calls.append((token, i))
            return real(self_, token, i)

        f._recipe_after_step = spy.__get__(f, rf.Fake)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        drain(f)
        assert [i for _t, i in calls] == [0, 1, 2]

    def test_a_sync_busy_judge_reschedules_at_most_three_times(self, panel):
        f = _fake(panel)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        f._refine_sync_busy = True
        drain(f)
        assert f._refine_recipe_run is None
        assert f._smooth_chain_count == 1       # reset: nothing more ran
        f._refine_sync_busy = False

    def test_a_sync_busy_that_clears_lets_the_run_go_on(self, panel):
        from qtpy.QtWidgets import QApplication

        f = _fake(panel)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        f._refine_sync_busy = True
        QApplication.processEvents()            # one re-schedule
        f._refine_sync_busy = False
        drain(f)
        assert f._smooth_chain_count == 3

    def test_a_stale_judge_after_a_patch_change_resets(self, panel):
        f = _fake(panel)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        f.ctx["patch_name"] = "q"
        drain(f)
        assert f._smooth_chain_count == 1

    def test_a_late_judge_does_nothing(self, panel):
        f = _fake(panel)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        token = f._refine_recipe_run["token"]
        drain(f)
        after = f.vol().copy()
        f._recipe_after_step(token, 2)
        f._recipe_on_worker_done(token, 2)
        assert np.array_equal(f.vol(), after)
        assert f._refine_recipe_run is None

    def test_the_session_gone_before_step_2_resets(self, panel, capsys):
        from qtpy.QtWidgets import QApplication

        f = _fake(panel)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        QApplication.processEvents()            # judge 1 -> step 2 ran
        assert f._refine_recipe_run["i"] == 1
        f._accept_smooth()                      # an internal path
        drain(f)
        assert f._refine_recipe_run is None
        assert f._refine_steps == []
        assert "ended: reset (the staged session is gone)" in (
            capsys.readouterr().out)

    def test_teardown_between_steps_ends_the_run(self, panel):
        f = _fake(panel)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        f._teardown_refine_for_switch()
        assert f._refine_recipe_run is None
        drain(f)
        assert f._smooth_chain_count == 1


class TestLoadSaveDelete:
    def test_save_the_staged_chain(self, panel):
        f = _fake(panel)
        for m, p in TIDY:
            panel.set_refine_step_params(m, dict(p))
            f._preview_smooth()
        f.name_answer = ("  tidy ", True)
        f._on_recipe_save()
        assert panel.smooth_status_label.text() == (
            "Saved the recipe (3 steps).")
        rec, problem = rr.recipe_from_json(f.store["tidy"])
        assert problem is None
        assert [(s.method, s.params) for s in rec.steps] == [
            (m, p) for m, p in TIDY]
        assert rec.spacing_known is True

    def test_a_case_collision_replaces_under_the_existing_spelling(self,
                                                                   panel):
        f = _fake(panel)
        f.store["Tidy"] = recipe(("sdf", SDF), ("fill_holes", {"axis": "3D"}))
        panel.set_refine_step_params("sdf", SDF)
        f._preview_smooth()
        f.name_answer = ("tidy", True)
        f._on_recipe_save()
        assert f.confirms == [
            "Replace the saved recipe 'Tidy' (Smooth -> Fill Holes) with "
            "these 1 step?"]
        assert f.confirm_buttons == [("Replace", "Cancel")]
        assert set(f.store) == {"Tidy"}
        assert len(f.store["Tidy"]["steps"]) == 1

    def test_a_bad_name(self, panel):
        f = _fake(panel)
        panel.set_refine_step_params("sdf", SDF)
        f._preview_smooth()
        f.name_answer = ("x" * 41, True)
        f._on_recipe_save()
        assert panel.smooth_status_label.text() == rr.S1_BAD_NAME
        assert f.store == {}

    def test_save_refused_while_the_file_cannot_be_read(self, panel):
        f = _fake(panel)
        panel.set_refine_step_params("sdf", SDF)
        f._preview_smooth()
        f.store_problem = rr.FILE_UNREADABLE
        f._on_recipe_save()
        assert f.saved == []
        assert panel.smooth_status_label.text() == rr.FILE_UNREADABLE

    def test_a_store_failure_is_said(self, panel):
        f = _fake(panel)
        panel.set_refine_step_params("sdf", SDF)
        f._preview_smooth()
        raw = ("[WinError 5] Access is denied: 'C:\\Users\\someone\\"
               "Desktop\\EpiCycle-Local-Testing\\artifacts\\step3\\"
               "refine_chain_recipes.json'")
        f._save_refine_recipe_cb = lambda name, obj: raw
        f._on_recipe_save()
        # fixed words on screen; the raw reason is for the log (UX M3)
        assert panel.smooth_status_label.text() == (
            "Could not save the recipe: the Step 3 folder could not be "
            "written to.")

    def test_load_sets_step_1_only_and_runs_nothing(self, panel):
        f = _fake(panel)
        before = f.vol().copy()
        f.store["tidy"] = recipe(("sdf", {"sigma_z": 2.5, "sigma_xy": 0.7,
                                          "preserve_topology": False}),
                                 ("fill_holes", {"axis": "3D"}))
        f._on_recipe_load("tidy")
        assert panel.refine_method_key() == "sdf"
        assert panel.smooth_standalone_sigma_z_sb.value() == 2.5
        assert np.array_equal(f.vol(), before) and not f._smooth_running
        assert panel.smooth_status_label.text() == (
            "Loaded step 1 of the recipe (Smooth). Nothing has run.")

    def test_delete(self, panel):
        f = _fake(panel)
        f.store = {"tidy": recipe(*TIDY), "other": recipe(*TIDY)}
        f._on_recipe_delete("tidy")
        assert f.confirms == [
            "Delete the recipe 'tidy'? This cannot be undone."]
        assert set(f.store) == {"other"}
        assert panel.smooth_status_label.text() == "Deleted the recipe."
        assert f.confirm_buttons == [("Delete", "Keep it")]


class TestThroughTheRealMenu:
    """The annotator's handlers behind the real panel menu: opened with
    `aboutToShow`, the real QAction triggered, in the user's order."""

    def _wire(self, f, panel):
        panel.set_smooth_callbacks(
            on_recipes_menu=f._on_recipes_menu_about_to_show,
            on_recipe_run=f._on_recipe_run, on_recipe_load=f._on_recipe_load,
            on_recipe_save=f._on_recipe_save,
            on_recipe_delete=f._on_recipe_delete)

    def _open(self, panel):
        panel.refine_recipes_menu.aboutToShow.emit()
        acts = {}
        for a in panel.refine_recipes_menu.actions():
            if a.menu() is not None:
                for b in a.menu().actions():
                    acts[(a.text(), b.text())] = b
            else:
                acts[a.text().strip()] = a
        return acts

    def test_save_then_run_from_the_menu(self, panel):
        f = _fake(panel)
        self._wire(f, panel)
        before = f.vol().copy()
        acts = self._open(panel)
        assert not acts["Save as a recipe\u2026 (unavailable: nothing is "
                        "staged yet)"].isEnabled()
        for m, p in TIDY:
            panel.set_refine_step_params(m, dict(p))
            f._preview_smooth()
        staged = f.vol().copy()
        acts = self._open(panel)
        acts["Save the 3 staged steps as…"].trigger()
        assert panel.smooth_status_label.text() == (
            "Saved the recipe (3 steps).")
        f._on_refine_discard_clicked()
        assert np.array_equal(f.vol(), before)
        acts = self._open(panel)
        assert "Run a recipe on object 7" in acts
        run = acts["tidy — Fill Holes -> Smooth -> Dilate/Erode"]
        assert run.isEnabled()
        run.trigger()
        drain(f)
        assert np.array_equal(f.vol(), staged)
        assert panel.smooth_accept_btn.text() == "Accept (3)"

    def test_greyed_items_carry_their_reason_as_text(self, panel):
        f = _fake(panel)
        self._wire(f, panel)
        panel.refine_method_combo.setCurrentIndex(6)
        settle = dict(panel.get_refine_step_params("settle"))
        f.store["s"] = recipe(("sdf", SDF), ("fill_holes", {"axis": "3D"}),
                              ("settle", settle))
        acts = self._open(panel)
        item = acts["s \u2014 step 3 (Settle) needs a channel; pick one "
                    "on this page first"]
        assert not item.isEnabled()
        before = f.vol().copy()
        f._on_recipe_run("s")             # the backstop for a stale menu
        assert np.array_equal(f.vol(), before)

    def test_while_a_run_is_active_the_menu_says_busy(self, panel):
        f = _fake(panel)
        self._wire(f, panel)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        acts = self._open(panel)
        assert rr.S1_RUNNING_BLOCK in acts         # the same words as S1
        assert not acts["tidy — Fill Holes -> Smooth -> Dilate/Erode"] \
            .isEnabled()
        assert not acts[("Load into the panel", "tidy")].isEnabled()
        drain(f)

    def test_delete_from_the_menu(self, panel):
        f = _fake(panel)
        self._wire(f, panel)
        f.store = {"tidy": recipe(*TIDY), "keep": recipe(*TIDY)}
        acts = self._open(panel)
        acts[("Delete", "tidy")].trigger()
        assert set(f.store) == {"keep"}
        acts = self._open(panel)
        assert ("Delete", "tidy") not in acts

    def test_no_workspace(self, panel):
        f = _fake(panel)
        self._wire(f, panel)
        f._get_saved_refine_recipes_cb = None
        acts = self._open(panel)
        assert list(acts) == [rr.NO_WORKSPACE]


def test_a_40_character_name_is_not_in_S1_but_on_the_stage_line(panel):
    f = _fake(panel)
    name = "a-very-long-recipe-name-of-forty-chars!!"
    assert len(name) == 40
    f.store[name] = recipe(*TIDY)
    f._on_recipe_run(name)
    assert panel.refine_stage_label.toolTip().endswith(
        "\u2014 '{}'".format(name))
    drain(f)
    assert name not in panel.smooth_status_label.text()
    assert panel.smooth_status_label.text() == (
        "The recipe ran all 3 steps on object 7.")
    assert f._refine_chain_recipe_names == [name]


def test_accept_after_a_recipe_hands_the_controller_the_trail(panel):
    f = _fake(panel)
    panel.set_refine_step_params("sdf", SDF)
    f._preview_smooth()                              # one step by hand
    f.store["tidy"] = recipe(*TIDY)
    f._on_recipe_run("tidy")
    drain(f)
    f._on_refine_accept_clicked()
    assert len(f.smoothed) == 1
    got = f.smoothed[0]
    assert got["recipes"] == ["tidy"]
    assert [s["method"] for s in got["steps"]] == [
        "sdf", "fill_holes", "sdf", "dilate_erode"]
    assert f._refine_step_records == []


# ---------------------------------------------------------------------------
# Audit H1 -- Stop delivered INSIDE a synchronous step's processEvents pump
# ---------------------------------------------------------------------------


def _stop_inside_pump_of_step(monkeypatch, f, step_index):
    """Deliver the Stop click from inside `_preview_smooth`'s own pump, the
    first time it runs for step ``step_index`` of the recipe."""
    from qtpy.QtWidgets import QApplication

    real = QApplication.processEvents
    fired = []

    def pump(*a):
        run = f._refine_recipe_run
        if (not fired and run is not None and run["i"] == step_index
                and f._refine_sync_busy):
            fired.append(1)
            f._cancel_refine_preview()          # the queued Stop click
        return real(*a)

    monkeypatch.setattr(QApplication, "processEvents", staticmethod(pump))
    return fired


class TestStopInsideTheSyncPump:
    def test_a_sync_step_does_not_write_after_stop(self, panel, monkeypatch):
        f = _fake(panel)
        before = f.vol().copy()
        fired = _stop_inside_pump_of_step(monkeypatch, f, 1)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        drain(f)
        assert fired == [1]
        assert f._refine_recipe_run is None
        assert np.array_equal(f.vol(), before), "a write landed after Stop"
        assert not f._smooth_running and f._smooth_snapshot is None
        assert f._refine_lock_fired is False
        assert f.lock_calls and f.lock_calls[-1] is False
        assert f.lock_calls.count(False) == 1, f.lock_calls
        assert "stopped at step 2 of 3." in panel.smooth_status_label.text()

    def test_a_pre_staged_chain_comes_back_whole(self, panel, monkeypatch):
        f = _fake(panel)
        before = f.vol().copy()
        panel.set_refine_step_params("sdf", SDF)
        f._preview_smooth()
        fired = _stop_inside_pump_of_step(monkeypatch, f, 0)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        drain(f)
        assert fired == [1]
        assert np.array_equal(f.vol(), before)
        assert not f._smooth_running and f._refine_lock_fired is False


class TestAuditUX:
    def test_the_running_stage_line_carries_the_name_last(self, panel):
        f = _fake(panel)
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_run("tidy")
        assert panel.refine_stage_label.toolTip() == (
            "1/3 · Fill Holes — 'tidy'")
        drain(f)

    def test_save_suggests_the_steps_summary(self, panel):
        f = _fake(panel)
        for m, p in TIDY:
            panel.set_refine_step_params(m, dict(p))
            f._preview_smooth()
        seen = []
        f._recipe_ask_name = lambda n, sug: (seen.append(sug),
                                             ("tidy", True))[1]
        f._on_recipe_save()
        assert seen == ["Fill Holes -> Smooth -> Dilate/Erode"]

    def test_the_real_name_box_takes_40_characters(self, panel, qapp):
        from qtpy.QtCore import QTimer
        from qtpy.QtWidgets import QApplication, QInputDialog, QLineEdit

        f = _fake(panel)
        got = {}

        def answer():
            w = QApplication.activeModalWidget()
            if isinstance(w, QInputDialog):
                edit = w.findChildren(QLineEdit)[0]
                got["max"] = edit.maxLength()
                got["text"] = w.textValue()
                w.reject()
            else:
                QTimer.singleShot(20, answer)

        QTimer.singleShot(20, answer)
        K = rf.K()
        text, ok = K._recipe_ask_name(f, 3, "Smooth -> Settle")
        assert got == {"max": 40, "text": "Smooth -> Settle"} and not ok

    def test_delete_is_refused_during_a_run(self, panel):
        f = _fake(panel)
        f.store = {"tidy": recipe(*TIDY), "keep": recipe(*TIDY)}
        f._on_recipe_run("tidy")
        f._on_recipe_delete("keep")
        assert "keep" in f.store and f.confirms == []
        assert panel.smooth_status_label.text() == rr.S1_RUNNING_BLOCK
        drain(f)

    def test_loaded_on_a_staged_chain_keeps_the_review_dot(self, panel):
        f = _fake(panel)
        panel.set_refine_step_params("sdf", SDF)
        f._preview_smooth()
        kinds = []
        real = f._update_smooth_ui

        def spy(**kw):
            if "status_kind" in kw:
                kinds.append(kw["status_kind"])
            return real(**kw)

        f._update_smooth_ui = spy
        f.store["tidy"] = recipe(*TIDY)
        f._on_recipe_load("tidy")
        assert kinds[-1] == "review"
        f._reject_smooth()
        f._on_recipe_load("tidy")
        assert kinds[-1] == "idle"

    def test_no_workspace_from_the_controller_is_said(self, panel):
        f = _fake(panel)
        f._get_saved_refine_recipes_cb = lambda: (None, rr.NO_WORKSPACE)
        entries = []
        panel.populate_refine_recipes_menu = entries.extend
        f._on_recipes_menu_about_to_show()
        assert [e[1] for e in entries] == [rr.NO_WORKSPACE]


@pytest.mark.parametrize("kind", ["refused", "emptied"])
def test_refused_and_emptied_endings_turn_the_dot_amber(panel, kind):
    """Refused, emptied and stopped endings use "warn" -- also
    when the runner writes no sentence of its own."""
    from napari_epicycle.annotator.advanced_features_widget import (
        _STATUS_DOT_STYLES,
    )

    f = _fake(panel)
    steps = ((("fill_holes", {"axis": "3D"}), ("fill_holes", {"axis": "3D"}))
             if kind == "refused" else
             (("dilate_erode", {"radius": -10, "axis": "XY"}), ("sdf", SDF)))
    f.store["r"] = recipe(*steps)
    f._on_recipe_run("r")
    drain(f)
    assert panel._refine_status_dot.styleSheet() == _STATUS_DOT_STYLES["warn"]
