"""The QC plot's live selection, as Batch Refine's "Plot selection" reads it.

The plot's highlighted set changes through four
emitting sources (lasso, click, category, the triage-histogram gate) and
one NON-emitting one, ``highlight_labels`` -- which the controller's
``_apply_plot_highlight_adapter`` calls whenever the navigator, the
annotator or the flagged list selects.  So the target reads the plot's own
state, live, and the controller pokes the annotator from the two places
the highlighted set changes.

No QApplication: the real methods are bound to small harnesses carrying
exactly the attributes they read.
"""
import types

import pandas as pd
import pytest


# ------------------------------------------------------------ the plot


class _Plot:
    """The QCPlotWidget selection attributes, with the real methods bound."""

    def __init__(self, df, *, ghost_mask=None):
        from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
            QCPlotWidget,
        )

        self._df = df
        active = df[~ghost_mask] if ghost_mask is not None else df
        self._scatter_label_ids = active["label"].values.tolist()
        self._scatter_to_df_index = list(active.index)
        self._clicked_df_index = None
        self._selected_indices = set()
        self._selected_categories = set()
        self._selected_category_df_indices = []
        self._clear_sel_btn = types.SimpleNamespace(setVisible=lambda v: None)
        self._stats_label = types.SimpleNamespace(setText=lambda t: None)
        self.redraws = 0
        for name in (
            "get_selected_df_indices", "selected_label_ids",
            "highlight_labels",
        ):
            setattr(self, name, types.MethodType(
                getattr(QCPlotWidget, name), self))

    def _redraw(self, *a, **k):
        self.redraws += 1


def _df():
    return pd.DataFrame({"label": [11, 12, 13, 14], "area": [1, 2, 3, 4]})


class TestSelectedLabelIds:
    def test_lasso(self):
        pw = _Plot(_df())
        pw._selected_indices = {0, 2}
        assert pw.selected_label_ids() == [11, 13]

    def test_category(self):
        pw = _Plot(_df())
        pw._selected_category_df_indices = [1, 3]
        assert pw.selected_label_ids() == [12, 14]

    def test_click(self):
        pw = _Plot(_df())
        pw._clicked_df_index = 2
        assert pw.selected_label_ids() == [13]

    def test_lasso_wins_over_category_and_click(self):
        pw = _Plot(_df())
        pw._selected_indices = {0}
        pw._selected_category_df_indices = [1]
        pw._clicked_df_index = 3
        assert pw.selected_label_ids() == [11]

    def test_highlight_labels_is_seen_without_any_signal(self):
        """The non-emitting source: a navigator selection repaints the plot."""
        pw = _Plot(_df())
        pw._selected_indices = {0, 1, 2}
        pw.highlight_labels([14])
        assert pw.selected_label_ids() == [14]
        pw.highlight_labels([])
        assert pw.selected_label_ids() == []

    def test_ghost_rows_map_back_to_the_full_frame(self):
        df = _df()
        ghost = pd.Series([False, True, False, False])
        pw = _Plot(df, ghost_mask=ghost.values)
        # scatter position 1 is df row 2 once row 1 is a ghost
        pw._selected_indices = {1}
        assert pw.selected_label_ids() == [13]

    def test_no_dataframe(self):
        pw = _Plot(_df())
        pw._df = None
        pw._selected_indices = {0}
        assert pw.selected_label_ids() == []

    def test_nothing_selected(self):
        assert _Plot(_df()).selected_label_ids() == []


# -------------------------------------------------------- the controller


class _Ctl(types.SimpleNamespace):
    """The controller attributes the methods read; the methods are real."""

    def __getattr__(self, name):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        fn = FusionReconstructionController.__dict__.get(name)
        if callable(fn):
            return fn.__get__(self)
        raise AttributeError(name)


class _Annotator:
    def __init__(self):
        self.notified = 0

    def notify_plot_selection_changed(self):
        self.notified += 1


def _ctl(*, pw, patch="p1", plot_patch="p1", df=True, annotator=None):
    dispatched = []
    ctl = _Ctl(
        ui=types.SimpleNamespace(
            qc_plot_widget=pw,
            assign_tier_btn=types.SimpleNamespace(setEnabled=lambda b: None),
        ),
        _qc_features_df=pd.DataFrame({"label": [1]}) if df else None,
        _qc_view_patch_name=plot_patch,
        _selection_sync=types.SimpleNamespace(dispatch=dispatched.append),
        dispatched=dispatched,
    )
    ctl._selected_curation_patch = lambda: patch
    ctl._selected_patch_name = lambda: ""
    ctl._get_curation_annotator_widget = lambda: annotator
    return ctl


class TestGetPlotSelection:
    def test_the_current_patch_reads_the_plot_live(self):
        pw = _Plot(_df())
        ctl = _ctl(pw=pw)
        pw._selected_indices = {0}
        assert ctl._get_plot_selection() == [11]
        pw.highlight_labels([12, 13])
        assert ctl._get_plot_selection() == [12, 13]

    def test_another_patchs_plot_says_so(self):
        """UX audit M6: not ``[]`` (the dialog would say "lasso first") but
        the PLOT_OTHER_PATCH marker, which the annotator turns into ``[]``
        and ``BATCH_PLOT_OTHER_PATCH``."""
        from packages.epicycle_annotator.batch_refine import PLOT_OTHER_PATCH

        pw = _Plot(_df())
        pw._selected_indices = {0}
        assert _ctl(pw=pw, plot_patch="p2")._get_plot_selection() ==             PLOT_OTHER_PATCH

    def test_no_features_means_no_plot(self):
        assert _ctl(pw=_Plot(_df()), df=False)._get_plot_selection() is None

    def test_no_plot_widget_means_no_plot(self):
        assert _ctl(pw=None)._get_plot_selection() is None

    def test_a_plot_that_raises_gives_nothing(self):
        pw = _Plot(_df())

        def boom():
            raise RuntimeError("stale plot")

        pw.selected_label_ids = boom
        assert _ctl(pw=pw)._get_plot_selection() == []


class TestNotify:
    def test_lasso_click_or_category_notifies(self):
        ann = _Annotator()
        pw = _Plot(_df())
        pw._selected_indices = {0, 1}
        ctl = _ctl(pw=pw, annotator=ann)
        ctl._on_qc_plot_label_selected([11, 12])
        assert ann.notified == 1
        assert ctl.dispatched  # the selection pipeline still ran first

    def test_a_clear_notifies(self):
        ann = _Annotator()
        ctl = _ctl(pw=_Plot(_df()), annotator=ann)
        ctl._on_qc_plot_label_selected([])
        assert ann.notified == 1

    def test_the_gate_path_notifies(self):
        ann = _Annotator()
        pw = _Plot(_df())
        pw._selected_indices = {3}
        ctl = _ctl(pw=pw, annotator=ann)
        ctl._on_gated_drag_finished([14])
        assert ann.notified == 1

    def test_highlight_adapter_notifies_after_the_repaint(self):
        seen = []
        pw = _Plot(_df())

        class _A:
            def notify_plot_selection_changed(self):
                seen.append(pw.selected_label_ids())

        ctl = _ctl(pw=pw, annotator=_A())
        ctl._apply_plot_highlight_adapter(
            types.SimpleNamespace(label_ids=[13]))
        assert seen == [[13]]

    def test_no_annotator_is_fine(self):
        ctl = _ctl(pw=_Plot(_df()), annotator=None)
        ctl._on_qc_plot_label_selected([])  # must not raise

    def test_a_notify_that_raises_is_swallowed(self):
        class _Bad:
            def notify_plot_selection_changed(self):
                raise RuntimeError("annotator half torn down")

        pw = _Plot(_df())
        ctl = _ctl(pw=pw, annotator=_Bad())
        ctl._apply_plot_highlight_adapter(types.SimpleNamespace(label_ids=[11]))
        ctl._on_qc_plot_label_selected([11])

    def test_the_annotator_lookup_raising_is_swallowed(self):
        ctl = _ctl(pw=_Plot(_df()))

        def boom():
            raise RuntimeError("no window")

        ctl._get_curation_annotator_widget = boom
        ctl._notify_plot_selection()


# (The source-string check of the launch call was dropped, wiring audit L6:
# test_launcher_wiring.py and test_curation_callbacks_partial_set.py cover
# the get_plot_selection kwarg at runtime.)


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-q"])
