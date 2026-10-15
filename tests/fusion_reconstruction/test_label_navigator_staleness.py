"""Label Navigator staleness guard on patch switch.

When the user switches to a patch that has no QC features extracted,
``_qc_features_df`` is set to None.  Pre-fix, neither the side-panel
flagged list nor the floating Label Navigator dock was cleared — the
navigator kept showing the previous patch's rows + tier badges, which
is confusing and lets the user click into rows that no longer match
the loaded patch.

The fix is two-part:
  - ``_update_flagged_labels_list`` pushes empty items to the floating
    Label Navigator BEFORE early-returning when df is None.
  - ``_reload_current_patch_features`` calls
    ``_update_flagged_labels_list`` in its no-CSV early-return branch.

These tests pin the contract.  They use the ``object.__new__`` pattern
to construct a controller without going through ``__init__`` so we
don't need a Qt event loop or a real workspace on disk.
"""

from unittest.mock import MagicMock


def _make_controller_with_fake_dock():
    """Return (ctrl, annotator_widget) with annotator dock wired up."""
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = object.__new__(FusionReconstructionController)
    ctrl._qc_features_df = None
    ctrl._flagged_centroids = {}

    ctrl.ui = MagicMock()
    ctrl.ui._flagged_labels_list = MagicMock()

    annotator_widget = MagicMock()
    annotator_widget.set_flagged_labels = MagicMock()
    dock = MagicMock()
    dock.isVisible.return_value = True
    dock.widget.return_value = annotator_widget

    ctrl.viewer = MagicMock()
    ctrl.viewer.window._dock_widgets = {
        "EpiCycle 3D Curation Annotator": dock,
    }
    return ctrl, annotator_widget


class TestUpdateFlaggedLabelsListWhenDfIsNone:
    """`_update_flagged_labels_list` must clear the floating navigator
    when there are no QC features to display."""

    def test_pushes_empty_to_annotator(self):
        ctrl, annotator_widget = _make_controller_with_fake_dock()
        ctrl._qc_features_df = None

        ctrl._update_flagged_labels_list()

        annotator_widget.set_flagged_labels.assert_called_once_with([])

    def test_clears_side_panel_list(self):
        ctrl, _annotator = _make_controller_with_fake_dock()
        ctrl._qc_features_df = None

        ctrl._update_flagged_labels_list()

        ctrl.ui._flagged_labels_list.clear.assert_called()

    def test_clears_flagged_centroids(self):
        ctrl, _annotator = _make_controller_with_fake_dock()
        ctrl._qc_features_df = None
        ctrl._flagged_centroids = {(7,): (10, 20, 30), (8,): (1, 2, 3)}

        ctrl._update_flagged_labels_list()

        assert ctrl._flagged_centroids == {}

    def test_no_op_when_dock_hidden(self):
        ctrl, annotator_widget = _make_controller_with_fake_dock()
        ctrl._qc_features_df = None
        # Dock not visible — push helper short-circuits.
        ctrl.viewer.window._dock_widgets[
            "EpiCycle 3D Curation Annotator"
        ].isVisible.return_value = False

        ctrl._update_flagged_labels_list()

        # Side panel still cleared regardless.
        ctrl.ui._flagged_labels_list.clear.assert_called()
        # Annotator NOT pushed because dock is hidden.
        annotator_widget.set_flagged_labels.assert_not_called()


class TestReloadCurrentPatchFeaturesNoCSV:
    """`_reload_current_patch_features` early-return when no QC CSV
    exists must invoke `_update_flagged_labels_list` so navigator
    + side panel both clear."""

    def test_calls_update_flagged_when_no_csv(self, tmp_path):
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        ctrl = object.__new__(FusionReconstructionController)
        ctrl._qc_features_df = MagicMock()  # stale df from previous patch

        # Working dir exists but no CSV under artifacts/qc/
        ctrl.working_dir = str(tmp_path)

        ctrl.ui = MagicMock()
        ctrl.ui.qc_summary_label = MagicMock()
        ctrl.ui.qc_plot_widget = MagicMock()
        ctrl.ui._curation_bucket = MagicMock()

        # Patch lookup: simulate "patch_A.tif" selected.
        item = MagicMock()
        item.data.return_value = "patch_A.tif"
        ctrl.ui._curation_bucket.selectedItems.return_value = [item]
        ctrl._selected_patch_name = MagicMock(return_value="patch_A.tif")

        # Spy on _update_flagged_labels_list to confirm it fires.
        ctrl._update_flagged_labels_list = MagicMock()

        ctrl._reload_current_patch_features()

        # df was reset to None
        assert ctrl._qc_features_df is None
        # Navigator clear path was invoked
        ctrl._update_flagged_labels_list.assert_called_once()
