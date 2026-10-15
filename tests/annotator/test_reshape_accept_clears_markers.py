"""Brick U1 -- Bug A: Accept never cleared the marker ticks.

`clear_split_guided_ui()` empties the marker list, so
`get_checked_marker_ids()` answers ``None`` again.  It was called from
`_teardown_reshape_for_switch` and `_reject_split` -- and from nowhere
else.  So after accepting a cut on object 139, the next object's Preview
read the PREVIOUS object's ticked ids, handed them over as
``keep_seed_ids``, and `np.isin(seeds, keep)` filtered away every marker
object 140 actually had: "No markers found in this object".

The clear is GATED on a commit actually happening, because `_accept_split`
has a `_load_locked` early return whose whole contract is "op state is
preserved, re-click after unlock works".
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("qtpy")


def _K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


# --------------------------------------------------------------------------- #
# the gate, read off the source
# --------------------------------------------------------------------------- #
class TestTheClearIsGatedOnACommit:
    def test_accept_clears_the_panel(self):
        src = inspect.getsource(_K()._accept_split)
        assert "clear_split_guided_ui" in src, (
            "Bug A: Accept never cleared the marker ticks")

    def test_the_clear_is_in_the_finally(self):
        src = inspect.getsource(_K()._accept_split)
        i_finally = src.index("finally:")
        # the CALL, not the comment that explains the bug
        i_clear = src.index('"clear_split_guided_ui"')
        assert i_clear > i_finally, (
            "the clear must run after `_write_reshape_result`, which "
            "reads the piece ticks")

    def test_it_is_guarded_by_a_committed_flag(self):
        src = inspect.getsource(_K()._accept_split)
        assert "committed = False" in src
        assert "if committed:" in src

    def test_the_load_locked_refusal_returns_before_committing(self):
        """A-H3: that path changes nothing, so it must change nothing."""
        src = inspect.getsource(_K()._accept_split)
        i_flag = src.index("committed = False")
        i_locked = src.index('getattr(self, "_load_locked", False)')
        i_true = src.index("committed = True")
        assert i_flag < i_locked < i_true

    def test_the_no_result_refusal_does_not_commit(self):
        src = inspect.getsource(_K()._accept_split)
        i_none = src.index("if _reshape_commit is None:")
        i_true = src.index("new_ids = list(_reshape_commit[\"new_ids\"])")
        assert i_none < i_true, (
            "the refusal must return before `committed` is set")

    def test_the_expected_count_tick_is_reset_too(self):
        src = inspect.getsource(_K()._accept_split)
        assert "clear_expected_count" in src

    def test_the_painted_markers_are_dropped(self):
        src = inspect.getsource(_K()._accept_split)
        assert "self._split_painted_marker_ids = []" in src


# --------------------------------------------------------------------------- #
# the survive / reset table, on the real panel
# --------------------------------------------------------------------------- #
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


def _armed(widget):
    """A session with markers found, a wall channel chosen and the
    expected count armed at 3."""
    from napari_epicycle.annotator import advanced_features_widget as M

    widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
    widget.set_seed_channels(["nuclei", "membrane"])
    widget.set_wall_channels(["membrane"])
    widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
    widget.split_target_count_sb.setValue(3)
    widget.split_target_count_cb.setChecked(True)
    widget.update_split_ui_state(split_active=True, marker_ids=[1, 2, 3])


class TestWhatResetsAndWhatSurvives:
    def test_the_marker_list_is_emptied(self, widget):
        _armed(widget)
        assert widget.get_checked_marker_ids() == [1, 2, 3]
        widget.clear_split_guided_ui()
        assert widget.get_checked_marker_ids() is None
        assert widget.split_marker_list.count() == 0

    def test_the_expected_count_TICK_goes_off(self, widget):
        _armed(widget)
        widget.clear_expected_count()
        assert widget.split_target_count_cb.isChecked() is False

    def test_but_the_spin_keeps_its_number(self, widget):
        """Re-arming it on the next object is one click and nothing the
        user chose is forgotten."""
        _armed(widget)
        widget.clear_expected_count()
        assert widget.split_target_count_sb.value() == 3

    def test_the_summary_stops_saying_expect_3(self, widget):
        _armed(widget)
        assert "expect 3" in widget.split_guided_summary_btn.toolTip()
        widget.clear_expected_count()
        assert "expect 3" not in widget.split_guided_summary_btn.toolTip()

    def test_the_summary_SHOWS_expect_while_it_is_armed(self, widget):
        """A-M5: a stale expected count used to be invisible once the
        card was collapsed -- and it is the one survivor that produces a
        WRONG cut rather than an inconvenient one."""
        _armed(widget)
        assert "expect 3" in widget.split_guided_summary_btn.toolTip()

    def test_the_sources_survive(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        _armed(widget)
        widget.clear_split_guided_ui()
        widget.clear_expected_count()
        assert widget.split_seeds_from_combo.currentIndex() == (
            M.SEEDS_FROM_IMAGE)
        assert widget.split_walls_from_combo.currentIndex() == (
            M.WALLS_FROM_MEMBRANE)

    def test_the_channels_survive(self, widget):
        _armed(widget)
        before = (widget.split_seed_channel_combo.currentText(),
                  widget.split_wall_channel_combo.currentText())
        widget.clear_split_guided_ui()
        widget.clear_expected_count()
        assert (widget.split_seed_channel_combo.currentText(),
                widget.split_wall_channel_combo.currentText()) == before

    def test_the_prepare_ticks_survive(self, widget):
        _armed(widget)
        widget.split_seed_prep_clahe_cb.setChecked(True)
        widget.clear_split_guided_ui()
        widget.clear_expected_count()
        assert widget.split_seed_prep_clahe_cb.isChecked() is True

    def test_clear_expected_count_is_idempotent(self, widget):
        _armed(widget)
        widget.clear_expected_count()
        widget.clear_expected_count()
        assert widget.split_target_count_cb.isChecked() is False
        assert widget.split_target_count_sb.value() == 3

    def test_a_second_preview_is_not_refused_for_no_markers(self, widget):
        """The bug, stated as its symptom: `None` means "no Find has
        happened, find them fresh"; a stale `[1, 2, 3]` filters away
        every marker the NEXT object has."""
        _armed(widget)
        widget.clear_split_guided_ui()
        assert widget.get_checked_marker_ids() is None
