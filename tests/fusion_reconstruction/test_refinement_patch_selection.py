"""Tests for Model Refinement patch selection and status gating.

Phase 2: Only accepted/finalized patches appear in the refinement patch list.
Select All / Deselect All wiring. Extraction uses selected patches only.
"""

import pytest
from unittest.mock import MagicMock, patch

from napari_epicycle.fusion_reconstruction import _step3_mr_text as _MRT


# ---------------------------------------------------------------------------
# Fake UI — lightweight stand-in for FusionReconstructionCuration widget
# ---------------------------------------------------------------------------

class FakeListWidgetItem:
    """Minimal QListWidgetItem stand-in."""

    def __init__(self, text=""):
        self._text = text
        self._data = {}
        self._foreground = None
        self._selected = False
        self._checked = False

    def text(self):
        return self._text

    def setData(self, role, value):
        self._data[role] = value

    def data(self, role):
        return self._data.get(role)

    def setForeground(self, brush):
        self._foreground = brush

    def foreground(self):
        return self._foreground

    def isSelected(self):
        return self._selected

    def setSelected(self, val):
        self._selected = val

    def isChecked(self):
        return self._checked

    def setChecked(self, val):
        self._checked = bool(val)


class FakeListWidget:
    """Minimal QListWidget stand-in with MultiSelection semantics."""

    def __init__(self):
        self._items = []

    def clear(self):
        self._items.clear()

    def addItem(self, item):
        self._items.append(item)

    def count(self):
        return len(self._items)

    def item(self, index):
        return self._items[index]

    def selectedItems(self):
        return [it for it in self._items if it.isSelected()]

    def selectAll(self):
        for it in self._items:
            it.setSelected(True)

    def clearSelection(self):
        for it in self._items:
            it.setSelected(False)


class FakeButton:
    """Minimal QPushButton stand-in with signal connection tracking."""

    def __init__(self):
        self._enabled = True
        self._connections = []

    def setEnabled(self, val):
        self._enabled = val

    def isEnabled(self):
        return self._enabled

    @property
    def clicked(self):
        return self

    def connect(self, slot):
        self._connections.append(slot)

    def disconnect(self):
        self._connections.clear()

    def setText(self, text):
        self._text = text

    def setIcon(self, icon):
        pass

    def setIconSize(self, size):
        pass


class FakeLabel:
    """Minimal QLabel stand-in."""

    def __init__(self, text=""):
        self._text = text

    def setText(self, text):
        self._text = text

    def text(self):
        return self._text

    def setStyleSheet(self, style):
        pass

    def text(self):
        return self._text


class FakeProgressBar:
    def __init__(self):
        self._value = 0

    def setValue(self, v):
        self._value = v

    def setRange(self, lo, hi):
        pass


_STATUS_BADGE = {
    "predicted": "P ",
    "committed": "\u2192 ",
    "reconstructed": "R ",
    "curating": "C ",
    "accepted": "\u2714 ",
    "finalized": "\u25a0 ",
}


class FakeUI:
    """Minimal UI stand-in with only Model Refinement widgets."""

    def __init__(self):
        self.refine_patch_list = FakeListWidget()
        self.refine_select_all_btn = FakeButton()
        self.refine_deselect_all_btn = FakeButton()
        self.extract_training_pairs_btn = FakeButton()
        self.launch_refinement_btn = FakeButton()
        self.pairs_summary_label = FakeLabel("No training pairs extracted")
        self.refinement_status_label = FakeLabel("")
        # 2026-05-19: extraction status was moved to its own label
        # so messages render next to the extraction progress bar.
        self.refine_extraction_status_label = FakeLabel("")
        self.refinement_progress = FakeProgressBar()
        self.refine_extraction_progress = FakeProgressBar()
        self.log_output = type("FakeLog", (), {
            "append": lambda self, text: None,
        })()

    # The ui-side tick API (Brick B1 provides it, B2's controller reads it).
    def refine_ticked_names(self):
        return [it.data(256) for it in self.refine_patch_list._items
                if it.isChecked()]

    def set_refine_ticks(self, on):
        for it in self.refine_patch_list._items:
            it.setChecked(bool(on))

    def refine_row_count(self):
        return self.refine_patch_list.count()

    def refresh_refinement_patch_list(
        self, statuses, extraction_info=None,
        stale_stems=None, mixed_settings_stems=None,
        axes_by_patch=None, info_by_patch=None, shapes=None,
    ):
        """Fake version — uses FakeListWidgetItem instead of QListWidgetItem.

        2026-05-19: accepts the new stale_stems / mixed_settings_stems
        kwargs so the controller's call signature lines up.  Test fake
        doesn't render the markers, just absorbs the args.
        """
        self.refine_patch_list.clear()
        for name in sorted(statuses):
            status = statuses[name]
            badge = _STATUS_BADGE.get(status, "")
            item = FakeListWidgetItem(f"{badge}{name}")
            item.setData(256, name)  # Qt.UserRole == 256
            self.refine_patch_list.addItem(item)


# ---------------------------------------------------------------------------
# Fake state manager
# ---------------------------------------------------------------------------

class FakeStateManager:
    """Minimal state manager with configurable patch statuses."""

    def __init__(self, patch_statuses=None):
        self._statuses = patch_statuses or {}

    def get_all_patch_statuses(self):
        return dict(self._statuses)

    def set_patch_status(self, name, status, **kw):
        self._statuses[name] = status


# ---------------------------------------------------------------------------
# Helper: build the refresh + select methods under test
# ---------------------------------------------------------------------------

# These are the functions we'll implement in controller.py.
# Import them here to test. The actual import path:
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController,
)


def _make_controller(patch_statuses):
    """Create a minimal controller with fake UI and state manager."""
    ui = FakeUI()
    state_mgr = FakeStateManager(patch_statuses)

    # Build a minimal controller mock — we only need a few attributes
    ctrl = object.__new__(FusionReconstructionController)
    ctrl.ui = ui
    ctrl.state_manager = state_mgr
    ctrl.working_dir = "/fake/workspace"
    ctrl.loaded_stack_dir = "/fake/stacks"
    ctrl._log_lines = []
    return ctrl


# ---------------------------------------------------------------------------
# Tests: Patch list population
# ---------------------------------------------------------------------------

class TestRefreshRefinementPatchList:
    """Test _refresh_refinement_patch_list() filters by status."""

    def test_only_finalized_shown(self):
        ctrl = _make_controller({
            "patch_1.tiff": "unprocessed",
            "patch_2.tiff": "predicted",
            "patch_3.tiff": "reconstructed",
            "patch_4.tiff": "curating",
            "patch_5.tiff": "accepted",
            "patch_6.tiff": "finalized",
        })
        ctrl._refresh_refinement_patch_list()

        items = ctrl.ui.refine_patch_list._items
        names = [it.data(256) for it in items]  # Qt.UserRole == 256
        assert sorted(names) == ["patch_6.tiff"]

    def test_empty_when_no_qualifying_patches(self):
        ctrl = _make_controller({
            "patch_1.tiff": "unprocessed",
            "patch_2.tiff": "predicted",
            "patch_3.tiff": "reconstructed",
        })
        ctrl._refresh_refinement_patch_list()

        assert ctrl.ui.refine_patch_list.count() == 0

    def test_all_shown_when_all_finalized(self):
        ctrl = _make_controller({
            "patch_1.tiff": "finalized",
            "patch_2.tiff": "finalized",
            "patch_3.tiff": "finalized",
        })
        ctrl._refresh_refinement_patch_list()

        assert ctrl.ui.refine_patch_list.count() == 3

    def test_items_store_patch_name_in_user_role(self):
        ctrl = _make_controller({
            "patch_A.tiff": "finalized",
        })
        ctrl._refresh_refinement_patch_list()

        item = ctrl.ui.refine_patch_list.item(0)
        assert item.data(256) == "patch_A.tiff"

    def test_items_have_patch_name_in_text(self):
        ctrl = _make_controller({
            "patch_1.tiff": "finalized",
            "patch_2.tiff": "finalized",
        })
        ctrl._refresh_refinement_patch_list()

        texts = [ctrl.ui.refine_patch_list.item(i).text()
                 for i in range(ctrl.ui.refine_patch_list.count())]
        assert any("patch_1.tiff" in t for t in texts)
        assert any("patch_2.tiff" in t for t in texts)

    def test_refresh_clears_previous_items(self):
        ctrl = _make_controller({"patch_1.tiff": "finalized"})
        ctrl._refresh_refinement_patch_list()
        assert ctrl.ui.refine_patch_list.count() == 1

        # Change state — patch_1 no longer finalized
        ctrl.state_manager._statuses = {"patch_1.tiff": "curating"}
        ctrl._refresh_refinement_patch_list()
        assert ctrl.ui.refine_patch_list.count() == 0

    def test_empty_state_manager(self):
        ctrl = _make_controller({})
        ctrl._refresh_refinement_patch_list()
        assert ctrl.ui.refine_patch_list.count() == 0


# ---------------------------------------------------------------------------
# Tests: Select All / Deselect All
# ---------------------------------------------------------------------------

class TestSelectDeselectAll:
    """Test select/deselect all button behavior."""

    def test_select_all_selects_all_items(self):
        ctrl = _make_controller({
            "p1.tiff": "finalized",
            "p2.tiff": "finalized",
            "p3.tiff": "finalized",
        })
        ctrl._refresh_refinement_patch_list()
        assert len(ctrl.ui.refine_ticked_names()) == 0

        ctrl._on_refine_select_all()
        assert len(ctrl.ui.refine_ticked_names()) == 3

    def test_deselect_all_clears_selection(self):
        ctrl = _make_controller({
            "p1.tiff": "finalized",
            "p2.tiff": "finalized",
        })
        ctrl._refresh_refinement_patch_list()
        ctrl._on_refine_select_all()
        assert len(ctrl.ui.refine_ticked_names()) == 2

        ctrl._on_refine_deselect_all()
        assert len(ctrl.ui.refine_ticked_names()) == 0

    def test_select_all_on_empty_list(self):
        ctrl = _make_controller({})
        ctrl._refresh_refinement_patch_list()
        ctrl._on_refine_select_all()
        assert len(ctrl.ui.refine_ticked_names()) == 0


# ---------------------------------------------------------------------------
# Tests: Selected patches passed to extraction
# ---------------------------------------------------------------------------

class TestExtractionUsesSelectedPatches:
    """Test that _extract_training_pairs passes only selected patch names."""

    def test_only_selected_patches_passed_to_worker(self):
        ctrl = _make_controller({
            "p1.tiff": "finalized",
            "p2.tiff": "finalized",
            "p3.tiff": "finalized",
        })
        ctrl._refresh_refinement_patch_list()

        # Select only p1 and p3
        for i in range(ctrl.ui.refine_patch_list.count()):
            item = ctrl.ui.refine_patch_list.item(i)
            name = item.data(256)
            item.setChecked(name in ("p1.tiff", "p3.tiff"))

        # Mock the worker class to capture args without starting a thread
        with patch(
            "napari_epicycle.fusion_reconstruction.controller._ExtractionWorker"
        ) as MockWorker, patch(
            "napari_epicycle.fusion_reconstruction.controller.QThread"
        ):
            mock_instance = MagicMock()
            MockWorker.return_value = mock_instance

            ctrl._extract_training_pairs()

            MockWorker.assert_called_once()
            call_args = MockWorker.call_args
            patch_names = call_args[0][1]  # second positional arg
            assert sorted(patch_names) == ["p1", "p3"]

    def test_no_selection_extracts_nothing(self):
        ctrl = _make_controller({
            "p1.tiff": "finalized",
        })
        ctrl._refresh_refinement_patch_list()
        # Don't select anything

        ctrl._extract_training_pairs()

        # 2026-05-19: extraction status moved to refine_extraction_status_label
        # so messages render colocated with the extraction progress bar.
        status_text = (
            ctrl.ui.refine_extraction_status_label.text().lower()
        )
        assert status_text == _MRT.MR_EXTRACT_NEEDS_TICK.lower()
