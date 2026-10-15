"""Batch Refine's fourth target, "Plot selection (N)".

The target is what the QC plot shows highlighted RIGHT NOW, read live from
the plot through the controller's getter -- never a cache, because
``highlight_labels`` rewrites the plot's selection without emitting.

Covered here:
- the dialog's fourth radio, ``target_spec()`` mode ``"plot"``,
- ``set_target_counts`` rewriting every radio text (the "Selected (N)" text
  used to be frozen at construction),
- ``refresh_count`` re-querying while the plot radio is checked,
- ``BATCH_PLOT_EMPTY`` / ``BATCH_PLOT_NONE`` on the count line, Run disabled,
- the annotator's ``"plot"`` resolve through ``resolve_target_ids``,
- ``notify_plot_selection_changed`` restarting the debounced recount, and
  the recount pushing every radio's count,
- ``set_curation_callbacks(get_plot_selection=)``.

Plain ``QApplication`` (pytest-qt is not installed).
"""
import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from packages.epicycle_annotator.batch_refine import (  # noqa: E402
    BATCH_PLOT_EMPTY,
    BATCH_PLOT_NONE,
    BATCH_TARGET_ALL,
    BATCH_TARGET_PLOT,
    BATCH_TARGET_SELECTED,
    label_bboxes,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def widgets(qapp):
    made = []
    yield made
    for w in made:
        try:
            if hasattr(w, "mark_clean_close"):
                w.mark_clean_close()
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    qapp.processEvents()


def _dialog(widgets, counts):
    """A dialog whose count provider reads ``counts[mode]`` live."""
    from napari_epicycle.annotator.batch_refine_dialog import BatchRefineDialog

    dlg = BatchRefineDialog(
        count_provider=lambda spec: counts.get(spec["mode"], 0),
        all_count=counts.get("all", 0),
        selected_count=counts.get("selected", 0),
        available_tiers=["good", "review"],
        on_preview_one=lambda: None,
        on_run=lambda: None,
        on_cancel=lambda: None,
        on_accept=lambda: None,
        on_reject=lambda: None,
        on_close=lambda: None,
    )
    widgets.append(dlg)
    return dlg


# ---------------------------------------------------------------- dialog


class TestTheRadio:
    def test_a_fourth_radio_between_selected_and_tier(self, widgets):
        dlg = _dialog(widgets, {"all": 10, "selected": 2, "plot": 0})
        buttons = dlg._target_group.buttons()
        assert buttons == [
            dlg._radio_all, dlg._radio_selected, dlg._radio_plot,
            dlg._radio_tier,
        ]
        assert dlg._radio_plot.text() == BATCH_TARGET_PLOT.format(n=0)

    def test_target_spec_mode_plot(self, widgets):
        dlg = _dialog(widgets, {"all": 10, "plot": 3})
        dlg._radio_plot.setChecked(True)
        assert dlg.target_spec()["mode"] == "plot"

    def test_count_follows_the_plot_radio(self, widgets):
        counts = {"all": 10, "selected": 2, "plot": 5}
        dlg = _dialog(widgets, counts)
        dlg._radio_plot.setChecked(True)
        assert "5" in dlg._count_lbl.text()
        assert dlg._run_btn.isEnabled()


class TestSetTargetCounts:
    def test_rewrites_every_radio_text(self, widgets):
        dlg = _dialog(widgets, {"all": 10, "selected": 2})
        dlg.set_target_counts(all=12, selected=4, plot=7)
        assert dlg._radio_all.text() == BATCH_TARGET_ALL.format(n=12)
        assert dlg._radio_selected.text() == BATCH_TARGET_SELECTED.format(n=4)
        assert dlg._radio_plot.text() == BATCH_TARGET_PLOT.format(n=7)

    def test_the_selected_text_is_no_longer_frozen(self, widgets):
        dlg = _dialog(widgets, {"all": 10, "selected": 2})
        assert dlg._radio_selected.text() == BATCH_TARGET_SELECTED.format(n=2)
        dlg.set_target_counts(selected=9)
        assert dlg._radio_selected.text() == BATCH_TARGET_SELECTED.format(n=9)

    def test_a_keyword_not_passed_leaves_its_radio_alone(self, widgets):
        dlg = _dialog(widgets, {"all": 10, "selected": 2})
        dlg.set_target_counts(plot=3)
        assert dlg._radio_all.text() == BATCH_TARGET_ALL.format(n=10)
        assert dlg._radio_selected.text() == BATCH_TARGET_SELECTED.format(n=2)

    def test_keywords_are_keyword_only_and_default_none(self):
        import inspect

        from napari_epicycle.annotator.batch_refine_dialog import (
            BatchRefineDialog,
        )

        sig = inspect.signature(BatchRefineDialog.set_target_counts)
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            assert p.kind is inspect.Parameter.KEYWORD_ONLY, name
            assert p.default is None, name


class TestRefreshCount:
    def test_requeries_while_plot_is_checked(self, widgets):
        counts = {"all": 10, "plot": 2}
        dlg = _dialog(widgets, counts)
        dlg._radio_plot.setChecked(True)
        counts["plot"] = 6
        dlg.refresh_count()
        assert "6" in dlg._count_lbl.text()

    def test_still_requeries_selected(self, widgets):
        counts = {"all": 10, "selected": 1}
        dlg = _dialog(widgets, counts)
        dlg._radio_selected.setChecked(True)
        counts["selected"] = 4
        dlg.refresh_count()
        assert "4" in dlg._count_lbl.text()


class TestReasons:
    def test_empty_plot_selection_says_lasso_first(self, widgets):
        counts = {"all": 10, "plot": 0}
        dlg = _dialog(widgets, counts)
        dlg.set_target_counts(plot=0, plot_available=True)
        dlg._radio_plot.setChecked(True)
        assert dlg._count_lbl.text() == BATCH_PLOT_EMPTY
        assert not dlg._run_btn.isEnabled()

    def test_no_plot_on_this_patch(self, widgets):
        counts = {"all": 10, "plot": 0}
        dlg = _dialog(widgets, counts)
        dlg.set_target_counts(plot=0, plot_available=False)
        dlg._radio_plot.setChecked(True)
        assert dlg._count_lbl.text() == BATCH_PLOT_NONE
        assert not dlg._run_btn.isEnabled()

    def test_other_targets_at_zero_keep_the_count_line(self, widgets):
        dlg = _dialog(widgets, {"all": 10, "selected": 0})
        dlg._radio_selected.setChecked(True)
        assert dlg._count_lbl.text() not in (BATCH_PLOT_EMPTY, BATCH_PLOT_NONE)

    def test_the_sentences_are_plain_and_distinct(self):
        assert BATCH_PLOT_EMPTY != BATCH_PLOT_NONE
        assert "QC plot" in BATCH_PLOT_EMPTY and "QC plot" in BATCH_PLOT_NONE


# ------------------------------------------------------------- annotator


class _FakeAnnotator(types.SimpleNamespace):
    """Carries only the attributes the batch-target methods read; every
    METHOD is the real ``Curation3DAnnotator`` one, bound to this fake."""

    def __getattr__(self, name):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        fn = Curation3DAnnotator.__dict__.get(name)
        if callable(fn):
            return fn.__get__(self)
        raise AttributeError(name)


def _vol():
    v = np.zeros((4, 4, 4), dtype=np.int32)
    v[0, 0, 0] = 1
    v[1, 1, 1] = 2
    v[2, 2, 2] = 3
    return v


def _ann(plot_cb, *, nav_sel=()):
    vol = _vol()
    nav = types.SimpleNamespace(
        selected_label_ids=lambda: set(nav_sel), _items=[])
    return _FakeAnnotator(
        _active_labels_name="lbl",
        sync_manager=types.SimpleNamespace(shared_labels={"lbl": vol}),
        _navigator_widget=nav,
        _batch_explicit_targets=None,
        _batch_present_ids=frozenset(label_bboxes(vol)),
        _get_plot_selection_cb=plot_cb,
    )


class TestResolvePlot:
    def test_plot_ids_intersect_present(self):
        ann = _ann(lambda: [3, 1, 99])
        assert ann._resolve_batch_targets({"mode": "plot"}) == [1, 3]

    def test_read_live_at_every_call(self):
        box = {"ids": [1]}
        ann = _ann(lambda: box["ids"])
        assert ann._resolve_batch_targets({"mode": "plot"}) == [1]
        box["ids"] = [2, 3]
        assert ann._resolve_batch_targets({"mode": "plot"}) == [2, 3]

    def test_no_plot_resolves_to_nothing(self):
        assert _ann(lambda: None)._resolve_batch_targets(
            {"mode": "plot"}) == []

    def test_no_getter_resolves_to_nothing(self):
        assert _ann(None)._resolve_batch_targets({"mode": "plot"}) == []

    def test_a_getter_that_raises_resolves_to_nothing(self):
        def boom():
            raise RuntimeError("plot gone")

        assert _ann(boom)._resolve_batch_targets({"mode": "plot"}) == []

    def test_independent_of_the_navigator(self):
        ann = _ann(lambda: [2], nav_sel=(1, 3))
        assert ann._resolve_batch_targets({"mode": "plot"}) == [2]
        assert ann._resolve_batch_targets({"mode": "selected"}) == [1, 3]


class _Dlg:
    def __init__(self):
        self.counts = []
        self.refreshed = 0

    def set_target_counts(self, **kw):
        self.counts.append(kw)

    def refresh_count(self):
        self.refreshed += 1


class TestTheRecount:
    def _bind(self, ann, name):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        return getattr(Curation3DAnnotator, name).__get__(ann)

    def test_recount_pushes_every_radio_count(self):
        ann = _ann(lambda: [1, 2, 99], nav_sel=(3,))
        ann._batch_refine_dialog = _Dlg()
        self._bind(ann, "_do_refresh_batch_count")()
        (kw,) = ann._batch_refine_dialog.counts
        assert kw == {
            "all": 3, "selected": 1, "plot": 2, "plot_available": True,
            "plot_other_patch": False}
        assert ann._batch_refine_dialog.refreshed == 1

    def test_recount_says_when_there_is_no_plot(self):
        ann = _ann(lambda: None)
        ann._batch_refine_dialog = _Dlg()
        self._bind(ann, "_do_refresh_batch_count")()
        (kw,) = ann._batch_refine_dialog.counts
        assert kw["plot"] == 0 and kw["plot_available"] is False
        assert kw["plot_other_patch"] is False

    def test_notify_is_a_no_op_without_a_dialog(self):
        ann = _ann(lambda: [1])
        ann._batch_refine_dialog = None
        self._bind(ann, "notify_plot_selection_changed")()
        assert getattr(ann, "_batch_count_timer", None) is None

    def test_notify_restarts_the_debounce_timer(self, qapp):
        started = []

        class _T:
            def start(self, ms):
                started.append(ms)

        ann = _ann(lambda: [1])
        ann._batch_refine_dialog = _Dlg()
        ann._batch_count_timer = _T()
        self._bind(ann, "notify_plot_selection_changed")()
        assert started == [120]


class TestTheCallbackIsStored:
    def test_set_curation_callbacks_stores_the_getter(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        obj = types.SimpleNamespace()
        fn = Curation3DAnnotator.set_curation_callbacks.__get__(obj)

        def getter():
            return [1]

        fn(get_plot_selection=getter)
        assert obj._get_plot_selection_cb is getter
        fn()  # a partial call leaves it alone
        assert obj._get_plot_selection_cb is getter
