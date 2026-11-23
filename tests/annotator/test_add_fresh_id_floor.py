"""Every new id stays above BOTH floors, and an emptied main Undo closes the
Add review.

The widget's Clear floor is a plain number dropped at every patch switch and
close; the controller's floor (the ``get_label_id_floor`` callback) knows the
table's ghost rows and the session's Clear book.  A cleared or deleted id
held only by the controller's floor must not be handed out again by
Reshape / Split / the brush's next label / interpolation
(``_next_label_id_after_accept``), by "Find from shape" (``_detect_seeds``)
or by the found markers drawn onto the Refine layer
(``_materialise_found_markers``).
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


# --------------------------------------------------------------------------- #
# the one floor
# --------------------------------------------------------------------------- #
def test_the_fresh_floor_is_the_larger_of_the_two():
    K = _K()
    ann = types.SimpleNamespace(_add_clear_floor=30,
                                _get_label_id_floor_cb=lambda: 12)
    assert K._fresh_label_floor(ann) == 30
    ann._get_label_id_floor_cb = lambda: 80
    assert K._fresh_label_floor(ann) == 80


def test_the_fresh_floor_without_a_callback_or_with_a_raising_one():
    K = _K()
    ann = types.SimpleNamespace(_add_clear_floor=7)
    assert K._fresh_label_floor(ann) == 7, "no callback: the Clear floor"

    def boom():
        raise RuntimeError("no table")

    ann._get_label_id_floor_cb = boom
    assert K._fresh_label_floor(ann) == 7, "a raising callback counts as 0"
    assert K._fresh_label_floor(types.SimpleNamespace()) == 0


# --------------------------------------------------------------------------- #
# after a patch round-trip: the widget's Clear floor is 0 again, the
# controller still knows the cleared id 50
# --------------------------------------------------------------------------- #
def _after_switch(vol, nav_ids=None):
    nav = None
    if nav_ids is not None:
        nav = types.SimpleNamespace(_items=[{"label_id": i} for i in nav_ids])
    return types.SimpleNamespace(
        _navigator_widget=nav, _active_labels_name="L",
        sync_manager=types.SimpleNamespace(shared_labels={"L": vol}),
        _add_clear_floor=0, _get_label_id_floor_cb=lambda: 50)


def test_the_next_label_after_an_accept_skips_an_id_only_the_controller_knows():
    vol = np.zeros((4, 4, 4), np.int32)
    vol[0, 0, 0] = 10
    nxt = _K()._next_label_id_after_accept
    ann = _after_switch(vol, nav_ids=[3, 10])
    assert nxt(ann) == 51, "the navigator max is below the controller floor"
    ann._navigator_widget = None
    assert nxt(ann) == 51, "the volume max is below the controller floor"
    vol[1, 1, 1] = 60
    assert nxt(ann) == 61, "above the floor: max + 1 as before"


def test_find_from_shape_numbers_its_seeds_above_the_controller_floor(
        monkeypatch):
    from packages.epicycle_annotator import seed_detection as SD

    monkeypatch.setattr(
        SD, "detect_seed_peaks",
        lambda *a, **k: ([{"edt_value": 2.0}, {"edt_value": 3.0}], None))
    vol = np.zeros((4, 4, 4), np.int32)
    vol[1:3, 1:3, 1:3] = 9
    ann = _after_switch(vol)
    ann._split_running = True
    ann._split_previewed = False
    ann._split_label_id = 9
    ann._refine_layer = object()
    ann._refine_baseline = vol.copy()
    ann._leave_reshape_review = lambda *a, **k: None
    ann._get_advanced_features_widget = lambda: None
    ann._update_split_ui = lambda **k: None
    ann._apply_seed_threshold = lambda t: None
    _K()._detect_seeds(ann)
    assert ann._seed_id_base == 51


def test_found_markers_are_drawn_with_ids_above_the_controller_floor():
    from .test_reshape_recipe_promotion import Fake, Report, _bind

    roi = np.zeros((4, 4, 4), np.int32)
    roi[1:, :, :] = 7
    fake = Fake(roi, painted=[], seeds_from=1)
    fake._add_clear_floor = 0
    fake._get_label_id_floor_cb = lambda: 50
    _bind(fake, "_materialise_found_markers", "_drop_materialised_markers",
          "_marker_source_materialises", "_marker_source_index",
          "_painted_marker_ids_on_layer")
    seeds = np.zeros((4, 4, 4), np.int32)
    seeds[2, 2, 2] = 1
    seeds[3, 3, 3] = 2
    markers = [{"id": 1, "voxels": 1, "centroid_zyx": (2, 2, 2)},
               {"id": 2, "voxels": 1, "centroid_zyx": (3, 3, 3)}]
    ids, _stats = fake._materialise_found_markers(Report(seeds), markers)
    assert ids and min(ids) > 50, ids
    data = fake._refine_layer.data
    assert int(data[2, 2, 2]) > 50 and int(data[3, 3, 3]) > 50


# --------------------------------------------------------------------------- #
# Commit Triage (or a re-extraction) empties the main Undo during a review
# --------------------------------------------------------------------------- #
@pytest.fixture
def fakes(monkeypatch):
    from .add_harness import install_fakes

    install_fakes(monkeypatch)


def test_an_emptied_main_undo_closes_the_review(fakes):
    from .test_add_c2_fixpass2 import OBJ, ROI0, _accept, _page_and_controller

    hh, ctrl, vol = _page_and_controller()
    _accept(hh, ROI0, {50: OBJ})
    assert hh._add_review and hh._add_session
    ctrl._clear_all_undo_stacks()            # what Commit Triage runs
    assert not hh._add_review, "the review closed with nothing to undo"
    assert hh._add_session == []
    assert hh.adv.last("review_active") is False
    assert (vol == 50).any(), "the Accept stands"
    # "Undo all" can no longer be pressed into "use the main Undo button";
    # the page says why the review closed
    assert hh.adv.last("status_text") == AP.ADD_S1_UNDO_EMPTIED


def test_an_emptied_main_undo_with_no_review_changes_nothing(fakes):
    from .add_harness import Harness

    hh = Harness(np.zeros((4, 4, 4), np.int32))
    hh._add_review = False
    hh._add_after_undo_emptied()
    assert hh._add_review is False
    assert hh.adv.last("review_active") is None

def test_a_box_from_add_another_forgets_the_session_when_undo_empties(fakes):
    """"+ Add another" keeps the review's session for its resume; with
    the main Undo emptied meanwhile, the resume must not bring back a
    review whose Undo all can only refuse.  The box itself stays."""
    from .add_harness import Harness

    hh = Harness(np.zeros((4, 4, 4), np.int32))
    hh._add_review = False
    hh._add_running = True                   # the new box is up
    hh._add_session = [{"ids": [50], "accept_ids": [50]}]
    hh._add_session_clear_seqs = [3]
    hh._add_after_undo_emptied()
    assert hh._add_session == []
    assert hh._add_session_clear_seqs == []
    assert hh._add_running is True, "the box is left alone"
    assert hh.adv.last("review_active") is None


# --------------------------------------------------------------------------- #
# the other new-label paths: the brush at a patch switch / open, New Label
# --------------------------------------------------------------------------- #
class _Txt:
    def __init__(self, text=""):
        self.text = text

    def currentText(self):
        return self.text

    def setText(self, text):
        self.text = text


def test_the_brush_after_a_patch_switch_is_above_every_cleared_id():
    """Clear the top ids, Save, switch away and back: the page's Clear
    floor is 0 again, the controller still holds 50; the brush label the
    switch sets is above it."""
    vol = np.zeros((4, 4, 4), np.int32)
    vol[0, 0, 0] = 10
    layer = types.SimpleNamespace(data=vol)
    picked = []
    ann = types.SimpleNamespace(
        labels_combo=_Txt("L"), _active_labels_name="L",
        viewer=types.SimpleNamespace(layers=[layer]),
        _layer_by_name=lambda layers, name: layer,
        _set_selected_label_all=picked.append,
        _interp_status_label=_Txt(),
        _add_clear_floor=0, _get_label_id_floor_cb=lambda: 50)
    _K()._set_label_to_next_unused(ann)
    assert picked == [51]


def test_new_label_stays_above_the_page_clear_floor_too():
    from napari.components import ViewerModel

    vm = ViewerModel()
    vol = np.zeros((3, 6, 6), np.int32)
    vol[1, 1:3, 1:3] = 7
    vm.add_labels(vol, name="curation_3d_p01")
    picked = []
    ann = types.SimpleNamespace(
        viewer=vm, _set_selected_label_all=picked.append,
        _add_clear_floor=60, _get_label_id_floor_cb=lambda: 12)
    ann._add_label_id_floor = lambda: 12
    _K()._on_new_label_from_navigator(ann)
    assert picked == [61]
