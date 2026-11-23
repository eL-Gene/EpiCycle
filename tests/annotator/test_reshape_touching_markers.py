"""The touching-marker rule for markers taken from labels.

When the markers of a split come from another segmentation (another run of
the patch, or a mask file), every label that touches the object becomes a
marker -- including the neighbours that only graze it, and each of those
grows its own piece.  The rule keeps a label as a marker only if it covers
at least a tenth of the object, OR more than half of the label lies inside
the object (the second half ignored for a label under 20 voxels, so a
one-voxel speck wholly inside the object does not grow a full piece).
Every other label is listed, unticked, as "touching".

The rule's numbers and the measured contacts are read from
``fixtures/reshape_touching_markers.json``: every contact of the merged
cells and merged nuclei measured on the Wound patches, each with its
verdict (real / grazing / in between).

Part one is pure numpy.  Part two drives the panel offscreen and the
orchestrator through Qt-free fakes, the split every Reshape suite uses.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from packages.epicycle_annotator.reshape import seeds as S

FIXTURE = Path(__file__).with_name("fixtures") / "reshape_touching_markers.json"


def _fixture():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# part one -- the rule
# --------------------------------------------------------------------------- #
class TestTheRuleNumbers:
    def test_the_fixture_says_where_it_came_from_and_nothing_private(self):
        fx = _fixture()
        assert fx["source"] == "measured on the Wound patches, 2026-10-29"
        text = FIXTURE.read_text(encoding="utf-8")
        for bad in ("Users", "Desktop", ".html", ".md", "SPEC", "spec"):
            assert bad not in text, bad

    def test_the_engine_constants_are_the_fixture_numbers(self):
        rule = _fixture()["rule"]
        assert S.TOUCH_OBJECT_SHARE == pytest.approx(rule["object_share"])
        assert S.TOUCH_LABEL_SHARE == pytest.approx(rule["label_share"])
        assert S.TOUCH_MIN_LABEL_VOXELS == rule["label_side_min_voxels"]

    def test_the_fixture_holds_every_measured_case(self):
        fx = _fixture()
        assert len(fx["cases"]) == 24
        assert sum(len(c["contacts"]) for c in fx["cases"]) == 89
        verdicts = [k["verdict"] for c in fx["cases"] for k in c["contacts"]]
        assert verdicts.count("in between") == 2


class TestBothDirectionsOnTheMeasuredNumbers:
    """Each half alone fails one direction; together they hold both."""

    @staticmethod
    def _contacts():
        for case in _fixture()["cases"]:
            for k in case["contacts"]:
                yield case, k

    def test_every_real_marker_is_kept(self):
        for case, k in self._contacts():
            if k["verdict"] == "real":
                assert S.touching_keeps(
                    k["inside"], k["label_voxels"], case["object_voxels"]
                ), (case["name"], k)

    def test_every_grazing_label_is_left_out(self):
        for case, k in self._contacts():
            if k["verdict"] == "grazing":
                assert not S.touching_keeps(
                    k["inside"], k["label_voxels"], case["object_voxels"]
                ), (case["name"], k)

    def test_both_in_between_labels_are_kept(self):
        fx = _fixture()
        seen = {}
        for case, k in self._contacts():
            if k["verdict"] == "in between":
                seen["{} label {}".format(case["name"], k["label"])] = (
                    S.touching_keeps(
                        k["inside"], k["label_voxels"], case["object_voxels"]))
        assert set(seen) == set(fx["in_between_object_share_percent"])
        assert all(seen.values()), seen
        assert fx["in_between_object_share_percent"] == {
            "p3_nuc_97 label 684": 21.0, "p4_nuc_91 label 289": 13.3}

    def test_the_object_side_half_alone_would_drop_real_nuclei(self):
        """A nucleus is small next to a cell: 1.7 % of cell 127."""
        dropped = [
            (case["name"], k["label"]) for case, k in self._contacts()
            if k["verdict"] == "real"
            and 10 * k["inside"] < case["object_voxels"]
        ]
        assert ("p4_cell_127", 355) in dropped

    def test_the_label_side_half_alone_would_drop_real_cells(self):
        dropped = [
            (case["name"], k["label"]) for case, k in self._contacts()
            if k["verdict"] == "real" and 2 * k["inside"] <= k["label_voxels"]
        ]
        assert len(dropped) > 10

    def test_the_whole_pipeline_reproduces_every_case(self):
        """``marker_contacts`` on arrays built to each case's counts, with
        the outside part of every label OUTSIDE the ROI, so the label size
        must come from the whole mask."""
        for case in _fixture()["cases"]:
            obj = int(case["object_voxels"])
            width = obj + 2 + sum(int(k["label_voxels"])
                                  for k in case["contacts"])
            guide = np.zeros((2, 1, width), np.int32)
            labels = np.zeros_like(guide)
            labels[0, 0, :obj] = 9
            pos = 0
            for k in case["contacts"]:
                guide[0, 0, pos:pos + k["inside"]] = k["label"]
                pos += k["inside"]
            assert pos <= obj
            out = 0
            for k in case["contacts"]:
                n_out = k["label_voxels"] - k["inside"]
                guide[1, 0, out:out + n_out] = k["label"]
                out += n_out
            roi = (slice(0, 1), slice(0, 1), slice(0, width))
            region = labels[roi] == 9
            seeds_in = np.where(region, guide[roi], 0)
            got = S.marker_contacts(seeds_in, region, guide)
            assert sorted(got) == sorted(k["label"] for k in case["contacts"])
            for k in case["contacts"]:
                c = got[k["label"]]
                assert (c.inside, c.label_voxels, c.object_voxels) == (
                    k["inside"], k["label_voxels"], obj), (case["name"], k)
                assert c.keep == (k["verdict"] != "grazing"), (
                    case["name"], k)


class TestTheArithmetic:
    def test_it_is_exact_integer_arithmetic_at_the_object_boundary(self):
        assert S.touching_keeps(10, 10_000, 100)       # exactly 10 %
        assert not S.touching_keeps(9, 10_000, 100)

    def test_the_label_side_needs_MORE_than_half(self):
        assert not S.touching_keeps(50, 100, 100_000)   # exactly half
        assert S.touching_keeps(51, 100, 100_000)

    def test_an_object_under_ten_voxels_keeps_every_contact(self):
        for obj in range(1, 10):
            assert S.touching_keeps(1, 10_000, obj), obj

    def test_a_speck_wholly_inside_a_big_object_is_left_out(self):
        """A one-voxel label passes "more than half of itself inside" and
        would grow a full piece; under 20 voxels the label side is not
        counted, so it stands or falls on its share of the object."""
        assert not S.touching_keeps(1, 1, 22_589)
        assert not S.touching_keeps(19, 19, 22_589)
        assert S.touching_keeps(20, 20, 22_589)

    def test_a_speck_that_covers_a_tenth_of_a_small_object_stays(self):
        assert S.touching_keeps(1, 1, 8)


class TestMarkerContactsOnArrays:
    def _straddle(self):
        """One label straddling the ROI edge, one inside, one negative."""
        guide = np.zeros((6, 4, 4), np.int16)
        labels = np.zeros((6, 4, 4), np.int32)
        labels[0:3] = 5                       # the object fills the ROI
        guide[0:3, 0, :] = 3                  # 12 inside ...
        guide[3:6, 0, :] = 3                  # ... 12 outside the ROI
        guide[0:3, 1:4, 0] = 4                # 9 inside, none outside
        guide[0, 3, 3] = -2                   # a negative id, 1 voxel
        roi = (slice(0, 3), slice(0, 4), slice(0, 4))
        region = labels[roi] == 5
        return guide, region, np.where(region, guide[roi], 0)

    def test_a_label_crossing_the_roi_edge_is_sized_in_the_whole_mask(self):
        guide, region, seeds_in = self._straddle()
        got = S.marker_contacts(seeds_in, region, guide)
        assert got[3].inside == 12
        assert got[3].label_voxels == 24
        assert got[3].object_voxels == 48

    def test_a_negative_id_is_counted_like_any_other(self):
        guide, region, seeds_in = self._straddle()
        got = S.marker_contacts(seeds_in, region, guide)
        assert -2 in got
        assert (got[-2].inside, got[-2].label_voxels) == (1, 1)

    def test_background_is_never_a_marker(self):
        guide, region, seeds_in = self._straddle()
        assert 0 not in S.marker_contacts(seeds_in, region, guide)

    def test_the_slab_size_does_not_change_the_answer(self):
        guide, region, seeds_in = self._straddle()
        a = S.marker_contacts(seeds_in, region, guide, slab=1)
        b = S.marker_contacts(seeds_in, region, guide, slab=64)
        assert a == b

    def test_an_empty_object_has_no_contacts(self):
        guide, region, seeds_in = self._straddle()
        region = np.zeros_like(region)
        assert S.marker_contacts(np.zeros_like(seeds_in), region, guide) == {}

    def test_it_never_mutates_its_inputs(self):
        guide, region, seeds_in = self._straddle()
        g, r, s = guide.copy(), region.copy(), seeds_in.copy()
        S.marker_contacts(seeds_in, region, guide)
        assert np.array_equal(g, guide)
        assert np.array_equal(r, region)
        assert np.array_equal(s, seeds_in)

    def test_an_id_past_the_int32_range_is_keyed_as_the_cut_keys_it(self):
        """The cut reads label markers as int32 (``seeds_from_labels``).
        A uint32 id of 2**31 or more must be measured under that same
        id on both paths, and sized in the whole mask."""
        pytest.importorskip("qtpy")
        from napari_epicycle.annotator.reshape_worker import (
            touching_verdicts,
        )

        big = 2 ** 31 + 5
        labels = np.zeros((6, 4, 4), np.int32)
        labels[0:3] = 5
        guide = np.zeros((6, 4, 4), np.uint32)
        guide[0:3, 0, :] = big                # 12 inside ...
        guide[3:6, :, :] = big                # ... 48 outside the object
        roi = (slice(0, 3), slice(0, 4), slice(0, 4))
        region = labels[roi] == 5
        cut_ids = set(np.unique(S.seeds_from_labels(guide[roi], region)))
        wrapped = int((cut_ids - {0}).pop())
        assert wrapped != big                 # the cut's own id
        found = touching_verdicts(labels, guide, 5, roi,
                                  seeds_in_region=S.seeds_from_labels(
                                      guide[roi], region))
        no_find = touching_verdicts(labels, guide, 5, roi)
        assert set(found) == set(no_find) == {wrapped}
        assert found == no_find
        assert found[wrapped].inside == 12
        assert found[wrapped].label_voxels == 60

    def test_a_stop_ends_the_sweep(self):
        import threading

        guide, region, seeds_in = self._straddle()
        stop = threading.Event()
        stop.set()
        assert S.marker_contacts(seeds_in, region, guide,
                                 cancel_event=stop) == {}
        assert S.marker_contacts(seeds_in, region, guide,
                                 cancel_event=threading.Event())

    def test_the_contact_is_frozen(self):
        guide, region, seeds_in = self._straddle()
        c = S.marker_contacts(seeds_in, region, guide)[3]
        with pytest.raises(Exception):
            c.keep = False


# --------------------------------------------------------------------------- #
# part two -- the workers carry the rule onto their own thread
# --------------------------------------------------------------------------- #
def _grazed_object():
    """Object 7 holding two real labels and one grazing neighbour.

    Label 3 lies mostly OUTSIDE the search box, so its size has to come
    from the whole mask: 240 voxels inside (3.3 % of the object), 2,040 in
    all (11.8 % of itself inside) -- left out by both halves.
    """
    labels = np.zeros((12, 44, 40), np.int32)
    labels[2:10, 5:35, 5:35] = 7                    # 7,200 voxels
    guide = np.zeros_like(labels)
    guide[2:10, 5:20, 5:35] = 1                     # 3,600, all inside
    guide[2:10, 20:34, 5:35] = 2                    # 3,360, all inside
    guide[2:10, 34:35, 5:35] = 3                    # 240 inside ...
    guide[0:12, 39:44, 5:35] = 3                    # ... 1,800 far outside
    roi = (slice(0, 12), slice(1, 39), slice(1, 39))
    return labels, guide, roi


def _label_spec():
    from packages.epicycle_annotator.reshape import ReshapeSpec

    return ReshapeSpec(target="label", seeds="labels", walls="distance",
                       params={"wall_channel": None})


class _Spy:
    def __init__(self):
        self.calls = []

    def __call__(self, *args):
        self.calls.append(args if len(args) != 1 else args[0])


class TestTheFindWorkerFillsTheVerdicts:
    def _run(self, touching_rule):
        pytest.importorskip("qtpy")
        from napari_epicycle.annotator import reshape_worker as RW

        labels, guide, roi = _grazed_object()
        w = RW.SeedFindWorker(labels, {}, _label_spec(), target_id=7,
                              roi=roi, seed_layer=guide,
                              touching_rule=touching_rule)
        spy = _Spy()
        w.finished.connect(spy)
        w.run()
        assert len(spy.calls) == 1
        return spy.calls[0]

    def test_with_the_rule_every_marker_has_a_verdict(self, capsys):
        report = self._run(True)
        assert report.ok
        assert sorted(report.contacts) == [1, 2, 3]
        assert report.contacts[1].keep and report.contacts[2].keep
        c = report.contacts[3]
        assert (c.inside, c.label_voxels, c.object_voxels, c.keep) == (
            240, 2040, 7200, False)
        out = capsys.readouterr().out
        assert "[RESHAPE-FIND] touching rule wall=" in out
        assert "kept=2 left_unticked=1" in out

    def test_the_marker_dicts_keep_their_four_keys(self):
        report = self._run(True)
        for m in report.markers:
            assert set(m) == {"id", "voxels", "centroid_zyx", "bbox"}

    def test_without_the_rule_nothing_is_measured(self):
        report = self._run(False)
        assert report.ok
        assert report.contacts == {}


class TestThePreviewWorkerWithNoFind:
    def _run(self, *, touching_rule, keep_seed_ids=None, guide=None):
        pytest.importorskip("qtpy")
        from napari_epicycle.annotator import reshape_worker as RW

        labels, g, roi = _grazed_object()
        w = RW.ReshapeWorker(labels, {}, _label_spec(), target_id=7,
                             roi=roi, seed_layer=g if guide is None else guide,
                             keep_seed_ids=keep_seed_ids,
                             touching_rule=touching_rule)
        spy = _Spy()
        w.finished.connect(spy)
        w.run()
        assert len(spy.calls) == 1
        return spy.calls[0]

    def test_the_cut_uses_the_kept_labels_only(self):
        result = self._run(touching_rule=True)
        assert result.status == "ok"
        assert result.n_seeds == 2
        ids = set(int(v) for v in np.unique(result.seeds)) - {0}
        assert ids == {1, 2}

    def test_without_the_rule_every_label_is_a_marker(self):
        result = self._run(touching_rule=False)
        assert result.n_seeds == 3

    def test_ticks_handed_over_win_over_the_rule(self):
        result = self._run(touching_rule=True, keep_seed_ids=[1, 2, 3])
        assert result.n_seeds == 3

    def test_none_kept_is_refused_without_the_engine(self, monkeypatch):
        """Every label only touches: the cut is refused in words about
        markers taken from labels -- no prominence, no wall built."""
        from napari_epicycle.annotator import _reshape_text as T
        from napari_epicycle.annotator import reshape_worker as RW

        def _never(*_a, **_k):
            raise AssertionError("the engine ran on an empty marker set")

        monkeypatch.setattr(RW, "reshape_region", _never)
        labels, guide, roi = _grazed_object()
        only_grazer = np.where(guide == 3, 3, 0).astype(guide.dtype)
        result = self._run(touching_rule=True, guide=only_grazer)
        assert result.status == "refused"
        assert result.refusal_reason == "no-seeds"
        assert result.roi == roi
        assert result.messages == [T.MARKERS_ALL_TOUCHING]
        assert not any("prominence" in m for m in result.messages)

    def test_the_cut_says_which_markers_it_left_out(self):
        """No list on screen to show them, so the cut names them first."""
        from napari_epicycle.annotator import _reshape_text as T

        result = self._run(touching_rule=True)
        assert result.status == "ok"
        assert result.messages[0] == T.MARKERS_LEFT_OUT.format(n=1, s="")
        assert sum(1 for m in result.messages
                   if "touching marker" in m) == 1

    def test_none_kept_says_it_once_not_twice(self):
        """The refusal is the reason; the left-out remark would only say
        the same thing again in front of it."""
        labels, guide, roi = _grazed_object()
        only_grazer = np.where(guide == 3, 3, 0).astype(guide.dtype)
        result = self._run(touching_rule=True, guide=only_grazer)
        assert not any("touching marker" in m for m in result.messages)
        assert not any(m.startswith("No markers found")
                       for m in result.messages)

    def test_both_workers_hand_their_stop_to_the_sweep(self, monkeypatch):
        pytest.importorskip("qtpy")
        from napari_epicycle.annotator import reshape_worker as RW

        seen = []

        def _record(*_a, cancel_event=None, **_k):
            seen.append(cancel_event)
            return {}

        monkeypatch.setattr(RW, "touching_verdicts", _record)
        labels, g, roi = _grazed_object()
        cut = RW.ReshapeWorker(labels, {}, _label_spec(), target_id=7,
                               roi=roi, seed_layer=g, touching_rule=True)
        cut.run()
        find = RW.SeedFindWorker(labels, {}, _label_spec(), target_id=7,
                                 roi=roi, seed_layer=g, touching_rule=True)
        find.run()
        assert len(seen) == 2
        assert seen[0] is cut.cancel_event
        assert seen[1] is find.cancel_event

    def test_the_sweep_hands_the_stop_to_the_count(self, monkeypatch):
        pytest.importorskip("qtpy")
        from napari_epicycle.annotator.reshape_worker import (
            touching_verdicts,
        )

        seen = []

        def _record(*_a, cancel_event=None, **_k):
            seen.append(cancel_event)
            return {}

        monkeypatch.setattr(S, "marker_contacts", _record)
        import threading

        stop = threading.Event()
        labels, g, roi = _grazed_object()
        touching_verdicts(labels, g, 7, roi, cancel_event=stop)
        assert seen == [stop]

    def test_nothing_left_out_says_nothing(self):
        for kw in ({"touching_rule": False},
                   {"touching_rule": True, "keep_seed_ids": [1, 2, 3]}):
            result = self._run(**kw)
            assert not any("touching marker" in m for m in result.messages)

    def test_a_stop_during_the_sweep_is_still_a_stop(self):
        """Stop pressed while the labels are counted: the cut never
        runs on a half-measured set."""
        pytest.importorskip("qtpy")
        from napari_epicycle.annotator import reshape_worker as RW

        labels, g, roi = _grazed_object()
        w = RW.ReshapeWorker(labels, {}, _label_spec(), target_id=7,
                             roi=roi, seed_layer=g, touching_rule=True)
        w.cancel()
        spy = _Spy()
        w.finished.connect(spy)
        w.run()
        assert len(spy.calls) == 1
        result = spy.calls[0]
        assert result.status == "cancelled"
        assert not any("touching marker" in m for m in result.messages)


# --------------------------------------------------------------------------- #
# part three -- the Find lists the touching markers unticked, and parked
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover - CI without Qt
            pytest.skip("cannot create a QApplication: {}".format(exc))
    return app


@pytest.fixture
def panel(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    yield w
    try:
        w.setParent(None)
    except Exception:
        pass


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


class _Layer:
    def __init__(self, data):
        self.data = data
        self.selected_label = 0

    def refresh(self):
        pass


class _Sync:
    def __init__(self, shared):
        self.shared_labels = shared


class Session:
    """A started split of object 7 with the real panel beside it."""

    BOUND = (
        "_on_seed_find_finished", "_seed_find_thread_cleanup",
        "_marker_source_materialises", "_materialise_found_markers",
        "_drop_materialised_markers", "_painted_marker_ids_on_layer",
        "_marker_source_index", "_sync_marker_parking", "_park_marker",
        "_unpark_marker", "_parked_marker_ids", "_unpark_all_markers",
        "_unticked_marker_ids", "_checked_marker_ids",
        "_on_reshape_marker_check_changed", "_render_find_status",
        "_touching_start_off", "_touching_rule_wanted",
        "_remove_temp_layer", "_drop_walls_picture",
    )

    def __init__(self, adv, source):
        labels, guide, roi = _grazed_object()
        self.labels, self.guide = labels, guide
        self._split_roi = roi
        self._refine_layer = _Layer(labels.copy())
        self._refine_baseline = labels[roi].copy()
        self._refine_pre_preview = None
        self._split_label_id = 7
        self._split_running = True
        self._split_previewed = False
        self._split_painted_marker_ids = []
        self._split_seed_ids = []
        self._materialised_ids = []
        self._parked_marker_voxels = {}
        self._reshape_result = None
        self._reshape_found_seeds = None
        self._seed_find_running = True
        self._split_busy_kind = "find"
        self._seed_find_thread = None
        self._seed_find_worker = None
        self._seed_find_t0 = 0.0
        self._temp_guide_layers = {}
        self._active_labels_name = "x"
        self.sync_manager = _Sync({"x": labels})
        self.viewer = None
        self.adv = adv
        self.ui_calls = []
        adv.split_seeds_from_combo.setCurrentIndex(source)
        adv.update_split_ui_state(split_active=True)
        adv.set_split_callbacks(
            on_marker_check_changed=self._tick_changed)
        K = _K()
        for name in self.BOUND:
            if hasattr(K, name):
                setattr(self, name, getattr(K, name).__get__(self, Session))

    def _tick_changed(self):
        self._on_reshape_marker_check_changed()

    # -- seams -------------------------------------------------------
    def _seed_find_is_stale(self, result_token=None):
        return False

    def _result_token(self):
        return None

    def _get_advanced_features_widget(self):
        return self.adv

    def _update_split_ui(self, **kw):
        self.ui_calls.append(kw)

    def _refresh_refine_in_subviewers(self):
        pass

    def _leave_reshape_review(self, reason):
        pass

    def _review_is_on_screen(self):
        return False

    def status(self):
        for kw in reversed(self.ui_calls):
            if "status_text" in kw:
                return kw["status_text"]
        return None

    # -- the Find --------------------------------------------------------
    def find(self, *, touching_rule=True):
        from napari_epicycle.annotator import reshape_worker as RW

        w = RW.SeedFindWorker(self.labels, {}, _label_spec(), target_id=7,
                              roi=self._split_roi, seed_layer=self.guide,
                              touching_rule=touching_rule)
        spy = _Spy()
        w.finished.connect(spy)
        w.run()
        self._seed_find_running = True
        self._on_seed_find_finished(spy.calls[0])


def _rows(panel):
    from qtpy.QtCore import Qt

    lst = panel.split_marker_list
    out = []
    for i in range(lst.count()):
        it = lst.item(i)
        out.append((it.data(Qt.UserRole), it.checkState() == Qt.Checked,
                    it.text(), it.toolTip()))
    return out


class TestAFindOnMarkersFromLabels:
    def test_the_touching_row_starts_unticked_and_says_why(self, panel):
        from napari_epicycle.annotator import _reshape_text as T

        s = Session(panel, 2)
        s.find()
        rows = _rows(panel)
        assert len(rows) == 3
        ticked = [r for r in rows if r[1]]
        off = [r for r in rows if not r[1]]
        assert len(ticked) == 2 and len(off) == 1
        assert off[0][2].endswith(T.MARKER_ROW_TOUCHING)
        assert T.MARKER_TOUCHING_TIP.format(
            n=240, s="s", pct="3.3") in off[0][3]
        for r in ticked:
            assert T.MARKER_ROW_TOUCHING not in r[2]

    def test_the_touching_marker_is_parked_off_the_object(self, panel):
        s = Session(panel, 2)
        s.find()
        off_id = [r[0] for r in _rows(panel) if not r[1]][0]
        assert off_id in s._parked_marker_voxels
        assert not (s._refine_layer.data == off_id).any(), (
            "an unticked marker is lifted off the object")
        for mid in [r[0] for r in _rows(panel) if r[1]]:
            assert (s._refine_layer.data == mid).any()

    def test_the_count_line_and_the_status_say_so(self, panel):
        from napari_epicycle.annotator import _reshape_text as T

        s = Session(panel, 2)
        s.find()
        assert panel.split_marker_count_label.text() == (
            T.MARKERS_KEPT_LINE.format(kept=2, s="s", n=1, s2=""))
        assert s.status() == T.STATUS_FOUND.format(
            id=7, n=3, s="s", ticked=2)
        assert panel.split_seed_label.text() == T.MARKERS_COUNT.format(n=2)

    def test_preview_would_cut_with_the_kept_markers_only(self, panel):
        s = Session(panel, 2)
        s.find()
        kept = panel.get_checked_marker_ids()
        assert len(kept) == 2
        assert set(kept) == {r[0] for r in _rows(panel) if r[1]}

    def test_re_ticking_a_touching_row_unparks_it_and_recounts(self, panel):
        from qtpy.QtCore import Qt

        from napari_epicycle.annotator import _reshape_text as T

        s = Session(panel, 2)
        s.find()
        lst = panel.split_marker_list
        row = [i for i in range(lst.count())
               if lst.item(i).checkState() != Qt.Checked][0]
        off_id = lst.item(row).data(Qt.UserRole)
        lst.item(row).setCheckState(Qt.Checked)
        assert off_id not in s._parked_marker_voxels
        assert (s._refine_layer.data == off_id).any()
        assert panel.split_marker_count_label.text() == (
            T.MARKERS_FOUND_LINE.format(n=3, s="s"))
        assert s.status() == T.STATUS_FOUND.format(
            id=7, n=3, s="s", ticked=3)
        assert panel.split_seed_label.text() == T.MARKERS_COUNT.format(n=3)

    def test_find_again_starts_the_ticks_over_by_the_rule(self, panel):
        from qtpy.QtCore import Qt

        s = Session(panel, 2)
        s.find()
        lst = panel.split_marker_list
        for i in range(lst.count()):
            lst.item(i).setCheckState(Qt.Checked)
        assert panel.get_unticked_marker_ids() == []
        s.find()
        assert len(panel.get_unticked_marker_ids()) == 1

    def test_none_kept_says_so(self, panel):
        from napari_epicycle.annotator import _reshape_text as T

        s = Session(panel, 2)
        s.guide = np.where(s.guide == 3, 3, 0).astype(s.guide.dtype)
        s.find()
        assert panel.get_checked_marker_ids() == []
        # one marker reads in the singular
        assert panel.split_marker_count_label.text() == (
            T.MARKER_NONE_KEPT_ONE)
        assert T.MARKER_NONE_KEPT_ONE.startswith("The only marker ")
        assert "1 markers" not in panel.split_marker_count_label.text()

    def test_none_kept_of_many_says_so_in_the_plural(self, panel):
        from napari_epicycle.annotator import _reshape_text as T

        s = Session(panel, 2)
        # two grazers: label 3 (240 inside) and a second thin sliver
        g = np.where(s.guide == 3, 3, 0).astype(s.guide.dtype)
        g[2:10, 5:6, 5:35] = 4                       # 240 inside ...
        g[0:12, 0:1, 5:35] = 4                       # ... 360 outside
        s.guide = g
        s.find()
        assert panel.get_checked_marker_ids() == []
        assert panel.split_marker_count_label.text() == (
            T.MARKERS_NONE_KEPT.format(n=2))

    def test_a_find_without_verdicts_ticks_everything(self, panel):
        """An image-channel Find carries no verdicts: nothing starts off."""
        s = Session(panel, 4)
        s.find(touching_rule=False)
        assert panel.get_unticked_marker_ids() == []
        assert all(r[1] for r in _rows(panel))


class TestWhoAsksForTheRule:
    """Markers from another run (2) or a mask file (3) only."""

    @pytest.mark.parametrize("source, seeds, want", [
        (0, "paint", False), (1, "distance", False), (2, "labels", True),
        (3, "labels", True), (4, "hmax", False),
        # a found set already painted on the object is read back as paint
        (2, "paint", False), (3, "paint", False),
    ])
    def test_the_rule_is_wanted_for_label_markers_only(
            self, panel, source, seeds, want):
        s = Session(panel, source)

        class _Spec:
            pass

        spec = _Spec()
        spec.seeds = seeds
        assert s._touching_rule_wanted(spec) is want

    def test_the_find_launch_sends_it(self, panel, monkeypatch):
        got = self._launch(panel, monkeypatch, 2, "labels", find=True)
        assert got["touching_rule"] is True

    @pytest.mark.parametrize("source, seeds", [
        (0, "paint"), (1, "distance"), (4, "hmax")])
    def test_other_sources_never_send_it(self, panel, monkeypatch, source,
                                         seeds):
        got = self._launch(panel, monkeypatch, source, seeds, find=True)
        assert got["touching_rule"] is False

    def test_a_preview_with_no_find_sends_it(self, panel, monkeypatch):
        got = self._launch(panel, monkeypatch, 3, "labels", find=False)
        assert got["touching_rule"] is True
        assert got["keep_seed_ids"] is None

    @pytest.mark.parametrize("source, seeds", [
        (0, "paint"), (1, "distance"), (4, "hmax"), (2, "paint")])
    def test_a_preview_on_other_sources_never_sends_it(
            self, panel, monkeypatch, source, seeds):
        got = self._launch(panel, monkeypatch, source, seeds, find=False)
        assert got["touching_rule"] is False

    @staticmethod
    def _launch(panel, monkeypatch, source, seeds, *, find):
        s = Session(panel, source)
        got = _wire_launch(monkeypatch, s, seeds)
        K = _K()
        if find:
            K._find_reshape_markers(s)
        else:
            K._preview_reshape(s)
        assert got, "the worker was never built"
        return got

    def test_a_re_ticked_touching_marker_reaches_the_preview(
            self, panel, monkeypatch):
        """Ticked by hand after the Find: the cut gets it, and the rule is
        not applied again over the user's tick."""
        from qtpy.QtCore import Qt

        s = Session(panel, 2)
        s.find()
        lst = panel.split_marker_list
        row = [i for i in range(lst.count())
               if lst.item(i).checkState() != Qt.Checked][0]
        off_id = int(lst.item(row).data(Qt.UserRole))
        lst.item(row).setCheckState(Qt.Checked)
        got = _wire_launch(monkeypatch, s, "labels")
        _K()._preview_reshape(s)
        assert got, "the worker was never built"
        assert off_id in got["keep_seed_ids"]
        assert sorted(got["keep_seed_ids"]) == sorted(
            int(r[0]) for r in _rows(panel))
        assert got["touching_rule"] is False


class TestPreviewWhenTheRuleKeptNone:
    """Every marker came up touching and the user pressed Preview."""

    def _session(self, panel):
        s = Session(panel, 2)
        s.guide = np.where(s.guide == 3, 3, 0).astype(s.guide.dtype)
        s.find()
        assert panel.get_checked_marker_ids() == []
        return s

    def test_the_count_line_keeps_the_rule_and_the_status_says_what_to_do(
            self, panel, monkeypatch):
        from napari_epicycle.annotator import _reshape_text as T

        s = self._session(panel)
        got = _wire_launch(monkeypatch, s, "labels")
        _K()._preview_reshape(s)
        assert not got, "a refusal never starts the worker"
        text = panel.split_marker_count_label.text()
        assert text == T.MARKER_NONE_KEPT_ONE
        assert "You unticked every marker" not in text
        assert s.status() == T.STATUS_TICK_A_TOUCHING

    def test_unticking_by_hand_still_says_you_unticked_them(
            self, panel, monkeypatch):
        from qtpy.QtCore import Qt

        from napari_epicycle.annotator import _reshape_text as T

        s = Session(panel, 2)
        s.find()
        lst = panel.split_marker_list
        for i in range(lst.count()):
            lst.item(i).setCheckState(Qt.Unchecked)
        got = _wire_launch(monkeypatch, s, "labels")
        _K()._preview_reshape(s)
        assert not got
        assert panel.split_marker_count_label.text().startswith(
            "You unticked every marker")
        assert s.status() != T.STATUS_TICK_A_TOUCHING

    def test_the_cut_from_the_painted_markers_says_the_same(self):
        """After a Find the found markers are painted on the object, so
        the recipe reads them as painted ones and refuses there."""
        from napari_epicycle.annotator import _reshape_text as T

        from .test_reshape_markers_on_refine import _recipe_fake

        f = _recipe_fake(2, painted=[(101, (0, 0, 0))], materialised=[101])
        f._adv.set_marker_rows([101], off=[101])
        f._adv._touching_ids_in_list = lambda: {101}
        spec, reason, _seeds = f._reshape_recipe()
        assert spec is None
        assert reason == T.STATUS_TICK_A_TOUCHING

    def test_the_painted_refusal_is_unchanged_when_unticked_by_hand(self):
        from .test_reshape_markers_on_refine import _recipe_fake

        f = _recipe_fake(2, painted=[(101, (0, 0, 0))], materialised=[101])
        f._adv.set_marker_rows([101], off=[101])
        f._adv._touching_ids_in_list = lambda: set()
        spec, reason, _seeds = f._reshape_recipe()
        assert spec is None
        assert reason.startswith("You unticked every marker")

    def test_a_ticked_marker_with_no_voxels_is_not_called_touching(self):
        """Marker 101 is ticked but not on the object; only the touching
        102 is unticked.  The refusal is not the touching one."""
        from napari_epicycle.annotator import _reshape_text as T

        from .test_reshape_markers_on_refine import _recipe_fake

        f = _recipe_fake(2, painted=[(102, (0, 0, 0))],
                         materialised=[101, 102])
        f._split_painted_marker_ids = [101, 102]
        f._adv.set_marker_rows([101, 102], off=[102])
        f._adv._touching_ids_in_list = lambda: {102}
        spec, reason, _seeds = f._reshape_recipe()
        assert spec is None
        assert reason != T.STATUS_TICK_A_TOUCHING


def _wire_launch(monkeypatch, s, seeds):
    """Fake the worker and its thread on session ``s``; record the kwargs
    the worker would have been built with."""
    import qtpy.QtCore as QC

    from napari_epicycle.annotator import reshape_worker as RW

    got = {}

    class _Sig:
        def connect(self, *_a):
            pass

    class FakeWorker:
        def __init__(self, *a, **kw):
            got.update(kw)
            for sig in ("finished", "error", "progress", "message"):
                setattr(self, sig, _Sig())

        def moveToThread(self, _t):
            pass

        def run(self):
            pass

    class FakeThread:
        def __init__(self, *a):
            self.started = _Sig()

        def start(self):
            pass

    monkeypatch.setattr(RW, "SeedFindWorker", FakeWorker)
    monkeypatch.setattr(RW, "ReshapeWorker", FakeWorker)
    monkeypatch.setattr(QC, "QThread", FakeThread)
    s._seed_find_running = False
    s._reshape_running = False
    s._reshape_promoted = False

    class _Spec:
        walls = "distance"
        solver = "watershed"
        arbiter = False

    spec = _Spec()
    spec.seeds = seeds
    s._reshape_recipe = lambda for_find=False: (spec, {}, s.guide)
    s._reshape_labels_for_engine = lambda: s.labels
    s._reshape_context = lambda: {"patch_name": "P", "run_epoch": 0,
                                  "spacing": (1.0, 1.0, 1.0)}
    s._marker_rows_from_refine = lambda: []
    s._lock_subviewers_for_preview = lambda: None
    s._restore_ready_to_draw_state = lambda: None
    return got
