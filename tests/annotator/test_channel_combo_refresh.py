"""Channel names: when they are refreshed, and what a miss means.

All four channel combos are filled with LAYER NAMES and read back by name.
They were refreshed on dock creation and at the moment a tool starts, and
nowhere else -- no layers.events hook, no tab-change hook -- so a layer
added or removed mid-session left a stale list.

Worse, all four did ``clear()`` -> ``addItems()`` -> ``findText(prev)`` and,
on a MISS, did nothing -- while Qt had already auto-selected index 0 inside
the blockSignals window.  A renamed or removed layer therefore silently
reselected *the first image layer*, and the tool went on to segment against
the wrong channel with no message at all.  Adding more refresh triggers
would have fired that bug more often, not less.

Pattern 3 (offscreen QApplication) for the widget, Pattern 2 (bind the
real unbound method to a fake) for the annotator's wiring.
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover - CI without Qt
            pytest.skip("cannot create a QApplication: {}".format(exc))
    return app


@pytest.fixture
def widget(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    yield w
    try:
        w.setParent(None)
    except Exception:
        pass


def _ann():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


#: (refresher, combo attribute) for all four.
REFRESHERS = (
    # `refresh_split_image_combo` / `split_image_combo` were RETIRED in
    # Brick 2b together with the manual pair they served.
    ("refresh_add_image_combo", "add_channel_combo"),
    # Brick B1: the three per-method Refine combos were RETIRED and
    # replaced by ONE shared channel picker on the same policy (REFINE
    # ).  `TestNoSilentFallback` can assert the Refine case for
    # the first time, because there is no longer a fallback to assert
    # around.
    ("set_refine_channels", "refine_channel_combo"),
    # Brick 2a: the MARKER channel is the fifth combo, on the same
    # policy verbatim.
    ("set_seed_channels", "split_seed_channel_combo"),
)


def _refresh(widget, name, names):
    getattr(widget, name)(list(names))


# --------------------------------------------------------------------------- #
# policy 4 -- a restore miss is an EMPTY selection, not silently index 0
# --------------------------------------------------------------------------- #
class TestARestoreMissClearsTheSelection:
    @pytest.mark.parametrize("refresher,combo_attr", REFRESHERS)
    def test_a_removed_layer_does_not_silently_select_another(
            self, widget, refresher, combo_attr):
        _refresh(widget, refresher, ["membrane", "dapi"])
        combo = getattr(widget, combo_attr)
        combo.setCurrentIndex(1)
        assert combo.currentText() == "dapi"
        _refresh(widget, refresher, ["membrane", "brightfield"])
        assert combo.currentIndex() == -1, (
            "the chosen channel is gone; selecting the first one instead "
            "would segment against the wrong image with no message"
        )
        assert combo.currentText() == ""

    @pytest.mark.parametrize("refresher,combo_attr", REFRESHERS)
    def test_a_surviving_layer_keeps_its_selection(
            self, widget, refresher, combo_attr):
        _refresh(widget, refresher, ["membrane", "dapi"])
        combo = getattr(widget, combo_attr)
        combo.setCurrentIndex(1)
        _refresh(widget, refresher, ["membrane", "dapi", "extra"])
        assert combo.currentText() == "dapi"

    @pytest.mark.parametrize("refresher,combo_attr", REFRESHERS)
    def test_the_first_fill_still_selects_something(
            self, widget, refresher, combo_attr):
        """There is nothing to restore on the first fill, so index 0 is the
        right answer there -- the miss rule must not break that."""
        _refresh(widget, refresher, ["membrane", "dapi"])
        assert getattr(widget, combo_attr).currentIndex() == 0

    def test_the_wall_channel_combo_follows_the_same_rule(self, widget):
        widget.set_wall_channels(["membrane", "dapi"])
        widget.split_wall_channel_combo.setCurrentIndex(1)
        widget.set_wall_channels(["membrane", "brightfield"])
        assert widget.split_wall_channel_combo.currentIndex() == -1

    def test_an_empty_current_still_means_keep_the_live_one(self, widget):
        """set_wall_channels(names, current="") reads the live text, so an
        empty `current` preserves the selection rather than defeating it."""
        widget.set_wall_channels(["membrane", "dapi"])
        widget.split_wall_channel_combo.setCurrentIndex(1)
        widget.set_wall_channels(["membrane", "dapi"], current="")
        assert widget.split_wall_channel_combo.currentText() == "dapi"


class TestTheBlankStateDoesNotHealITSELF:
    """The live driver's d6/d7/d8, reproduced with no napari at all.

    The FIRST refresh after a rename blanked the combo correctly.  The
    SECOND one read `prev = combo.currentText()`, got "" (because the
    first one had just blanked it), took that for "nothing has ever been
    chosen", and let Qt's `addItems` auto-selection at index 0 stand --
    silently selecting the renamed layer.  The driver switches tool away
    and back, which is two refreshes; Start calls set_wall_channels
    again, which is the same thing.

    So the empty state healed into a WRONG selection, and every
    downstream "that layer is gone" error became unreachable: the combo
    named a layer that existed.
    """

    OLD, NEW = "stack_patch_6", "stack_patch_6__renamed"

    @pytest.mark.parametrize("refresher,combo_attr", REFRESHERS)
    def test_a_second_refresh_does_not_re_select_the_first_layer(
            self, widget, refresher, combo_attr):
        combo = getattr(widget, combo_attr)
        _refresh(widget, refresher, [self.OLD])
        combo.setCurrentIndex(0)
        _refresh(widget, refresher, [self.NEW])
        assert combo.currentIndex() == -1, "the first refresh must blank it"
        _refresh(widget, refresher, [self.NEW])
        assert combo.currentIndex() == -1, (
            "the second refresh silently selected {!r} -- the blank state "
            "healed into a wrong channel".format(combo.currentText())
        )

    @pytest.mark.parametrize("refresher,combo_attr", REFRESHERS)
    def test_renaming_the_layer_BACK_restores_the_choice(
            self, widget, refresher, combo_attr):
        """Which is why the choice is remembered rather than forgotten."""
        combo = getattr(widget, combo_attr)
        _refresh(widget, refresher, [self.OLD, "dapi"])
        combo.setCurrentIndex(0)
        _refresh(widget, refresher, [self.NEW, "dapi"])
        assert combo.currentIndex() == -1
        _refresh(widget, refresher, [self.OLD, "dapi"])
        assert combo.currentText() == self.OLD

    @pytest.mark.parametrize("refresher,combo_attr", REFRESHERS)
    def test_a_later_pick_replaces_the_remembered_one(
            self, widget, refresher, combo_attr):
        combo = getattr(widget, combo_attr)
        _refresh(widget, refresher, ["membrane", "dapi"])
        combo.setCurrentIndex(1)
        _refresh(widget, refresher, ["membrane", "dapi"])
        assert combo.currentText() == "dapi"
        combo.setCurrentIndex(0)
        _refresh(widget, refresher, ["membrane", "dapi"])
        assert combo.currentText() == "membrane"

    def test_the_wall_combo_survives_a_second_fill_too(self, widget):
        """d8: Start re-fills it through set_wall_channels(current="")."""
        combo = widget.split_wall_channel_combo
        widget.set_wall_channels([self.OLD])
        combo.setCurrentIndex(0)
        widget.set_wall_channels([self.NEW])
        assert combo.currentIndex() == -1
        widget.set_wall_channels([self.NEW])          # <- Start
        assert combo.currentIndex() == -1, (
            "Start re-selected {!r}".format(combo.currentText())
        )

    def test_an_explicit_current_still_wins(self, widget):
        """set_wall_channels(names, current=X) is a caller SAYING which one;
        that must override whatever was remembered."""
        combo = widget.split_wall_channel_combo
        widget.set_wall_channels(["membrane", "dapi"])
        combo.setCurrentIndex(0)
        widget.set_wall_channels(["membrane", "dapi"], current="dapi")
        assert combo.currentText() == "dapi"


# --------------------------------------------------------------------------- #
# policy 1 -- when the refresh happens
# --------------------------------------------------------------------------- #
class _FakeViewer:
    class _Layers:
        pass

    def __init__(self):
        self.layers = self._Layers()


class _FakeAnnotator:
    """Only what the refresh path touches."""

    def __init__(self, visible=True, boom=()):
        self._visible = visible
        self.calls = []
        self._boom = set(boom)
        self._active_labels_name = "labels"

    def isVisible(self):
        return self._visible

    def _raise_if(self, tag):
        if tag in self._boom:
            raise RuntimeError(
                "wrapped C/C++ object of type QComboBox has been deleted")

    def _refresh_refine_inputs(self):
        # Brick B1: the three per-method combos became ONE, and its
        # refresher also hands the page its spacing.
        self._raise_if("refine")
        self.calls.append("refine")

    def _refresh_add_image_combo(self):
        self._raise_if("add")
        self.calls.append("add")

    def _refresh_wall_channels(self):
        self._raise_if("wall")
        self.calls.append("wall")

    def _refresh_seed_channels(self):
        self._raise_if("seed")
        self.calls.append("seed")

    def _refresh_batch_chain_channels(self, first=False):
        # Batch recipe brick: the Batch dialog's chain channel combo is
        # the sixth, refreshed on the same layer insert / remove.
        self._raise_if("batch")
        self.calls.append("batch")


class TestWhenTheRefreshHappens:
    def test_refresh_all_image_combos_touches_every_channel_combo(self):
        f = _FakeAnnotator()
        _ann()._refresh_all_image_combos(f)
        # "split" left this list in Brick 2b: `_refresh_split_image_combos`
        # served the retired manual pair and nothing else.
        assert sorted(f.calls) == ["add", "batch", "refine", "seed", "wall"]

    def test_the_membrane_channel_combo_is_refreshed_too(self):
        """It is filled by set_wall_channels from _refresh_reshape_inputs,
        which runs at Start and at a picker change -- so a layer added or
        removed while the panel sits open never reached it."""
        f = _FakeAnnotator()
        _ann()._refresh_all_image_combos(f)
        assert "wall" in f.calls

    def test_the_wall_refresh_does_not_rescan_the_guide_sources(self):
        """`_refresh_reshape_inputs` also scans other runs ON DISK for guide
        masks.  A layer insert/remove must not pay for that."""
        src = inspect.getsource(_ann()._refresh_wall_channels)
        assert "list_guide_sources" not in src
        assert "set_guide_sources" not in src

    def test_the_marker_channel_combo_is_refreshed_too(self):
        """The fifth combo joins BOTH refresh sets: this one (layer
        insert / remove / tab change) and `_refresh_reshape_inputs`
        (tool start / picker change)."""
        f = _FakeAnnotator()
        _ann()._refresh_all_image_combos(f)
        assert "seed" in f.calls

    def test_the_tool_start_path_refreshes_it_as_well(self):
        import inspect

        src = inspect.getsource(_ann()._refresh_reshape_inputs)
        assert "_refresh_seed_channels" in src

    def test_the_marker_refresh_uses_the_UNFILTERED_image_list(self):
        """The ndim filter the split/refine/add combos apply would turn a
        4-D stack into a hard error.  Driven, not read: a 4-D
        Image layer has to come out the other side."""
        import numpy as np
        from napari.layers import Image, Labels

        four_d = Image(np.zeros((2, 4, 4, 4), np.float32), name="stack4d")
        flat = Image(np.zeros((4, 4, 4), np.float32), name="membrane")
        temp = Image(np.zeros((4, 4, 4), np.float32), name="guide_walls")
        temp._epicycle_temporary = True
        labels = Labels(np.zeros((4, 4, 4), np.int32), name="curation_3d")

        seen = {}

        class _Adv:
            def set_seed_channels(self, names, current=""):
                seen["names"] = list(names)

        class _F:
            viewer = type("V", (), {
                "layers": [four_d, flat, temp, labels]})()

            def _get_advanced_features_widget(self):
                return _Adv()

        _ann()._refresh_seed_channels(_F())

        assert "stack4d" in seen["names"], (
            "a 4-D stack was filtered out; the marker search would then "
            "refuse a channel that is right there")
        assert "membrane" in seen["names"]
        assert "guide_walls" not in seen["names"], (
            "a temporary layer must never be offered as a channel")
        assert "curation_3d" not in seen["names"]

    def test_a_destroyed_widget_cannot_break_the_close(self):
        """`_close_cleanup` destroys the Advanced dock at step 4b and only
        disconnects `layers.events.removed` at step 6, so any layer removed
        in between reaches a half-destroyed widget.  A RuntimeError here
        propagates into napari's event emitter during window close.
        """
        for boom in ("refine", "add", "wall", "seed"):
            f = _FakeAnnotator(boom=[boom])
            _ann()._refresh_all_image_combos(f)   # must not raise
            assert boom not in f.calls

    def test_it_early_returns_while_the_dock_is_hidden(self):
        """I-11: the only new per-event work is a combo repopulate, and it
        must not run when nobody can see it."""
        f = _FakeAnnotator(visible=False)
        _ann()._refresh_all_image_combos(f)
        assert f.calls == []

    def test_a_layer_insert_refreshes_BEFORE_the_auto_load_early_return(self):
        src = inspect.getsource(_ann()._on_layer_inserted)
        i_refresh = src.index("_refresh_all_image_combos")
        i_return = src.index("if self._active_labels_name")
        assert i_refresh < i_return, (
            "_on_layer_inserted returns early once the subviewers are "
            "populated, which is every session after the first load"
        )

    def test_a_layer_removal_is_wired_too(self):
        wire = inspect.getsource(_ann()._wire_signals)
        assert "layers.events.removed.connect" in wire
        assert "layers.events.inserted.connect" in wire

    def test_the_removed_handler_is_disconnected_with_the_inserted_one(self):
        """An unpaired connect accumulates one handler per relaunch --
        exactly the class the event-handler pairing rule exists for."""
        src = inspect.getsource(_ann()._close_cleanup)
        assert "layers.events.inserted.disconnect" in src
        assert "layers.events.removed.disconnect" in src

    def test_the_tab_change_handler_is_disconnected_too(self):
        """Same rule: _open_advanced_features connects it on every click."""
        src = inspect.getsource(_ann()._close_cleanup)
        assert "currentChanged.disconnect" in src
        assert "_on_tool_tab_changed" in src

    def test_switching_tab_refreshes_and_the_connect_is_guarded(self):
        """_open_advanced_features is user-clickable and the launcher REUSES
        an open dock, so its wiring block re-runs on every click."""
        src = inspect.getsource(_ann()._open_advanced_features)
        assert "currentChanged" in src
        i_disc = src.index("currentChanged.disconnect")
        i_conn = src.index("currentChanged.connect")
        assert i_disc < i_conn, "disconnect-before-connect"


# --------------------------------------------------------------------------- #
# policy 3 -- an unresolvable channel is an ERROR, not a quiet substitution
# --------------------------------------------------------------------------- #
class TestNoSilentFallback:
    def test_the_add_path_reports_instead_of_returning_in_silence(self):
        """_get_add_channel_roi returns None at three points and both
        callers just returned, so the histogram and the threshold simply
        never updated."""
        for name in ("_compute_add_histogram", "_on_add_otsu_reset"):
            src = inspect.getsource(getattr(_ann(), name))
            assert "_push_add_ui" in src
            assert "status_text" in src, (
                "{} still swallows a missing channel".format(name)
            )
