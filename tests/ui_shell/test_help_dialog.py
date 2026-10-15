"""Tests for the per-subsection help-dialog infrastructure (2026-05-18).

Design decision (revised): every Step 3 *subsection* (Model Source,
Patch State, Curate, QC Features, Patch Selection, Extraction Settings,
Fine-Tuning, Training Configuration) carries its own `?` button inline
with the existing SECTION_HEADER_STYLE label.  The accordion section
headers are unchanged — earlier attempts at putting the `?` in the
accordion header band looked "off" per user feedback.

These tests verify:
  1. `help_button` returns a configured QPushButton with click wired.
  2. `themed_help_dialog` opens non-modal, renders markdown, dedups
     across same-path opens, cleans up on close.
  3. Missing markdown shows the friendly placeholder, not a crash.
  4. Model Refinement subsection guides exist on disk.

Tests run headless via pytest-qt's QApplication fixture pattern.
"""
from __future__ import annotations

import pytest
from pathlib import Path
from unittest.mock import MagicMock

qtpy = pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication, QDialog, QPushButton, QWidget


@pytest.fixture(scope="module")
def app():
    """Ensure a QApplication exists for QWidget instantiation."""
    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    new_app = QApplication([])
    yield new_app


@pytest.fixture
def parent(app):
    w = QWidget()
    yield w
    w.deleteLater()


@pytest.fixture
def guide_file(tmp_path: Path) -> Path:
    p = tmp_path / "test_guide.md"
    p.write_text(
        "# Test Guide\n\n"
        "Some body text.\n\n"
        "## Section\n\n"
        "- bullet one\n"
        "- bullet two\n",
        encoding="utf-8",
    )
    return p


# ---------------------------------------------------------------------------
# help_button
# ---------------------------------------------------------------------------


class TestHelpButton:

    def test_returns_qpushbutton(self, parent, guide_file):
        from napari_epicycle.ui_shell.theme import help_button
        btn = help_button(parent, str(guide_file), "Test Guide")
        assert isinstance(btn, QPushButton)

    def test_fixed_size(self, parent, guide_file):
        from napari_epicycle.ui_shell.theme import help_button
        btn = help_button(parent, str(guide_file), "Test Guide")
        # 22×22 per design — fits inline with section text without
        # dominating the layout.
        assert btn.width() == 22
        assert btn.height() == 22

    def test_tooltip_set(self, parent, guide_file):
        from napari_epicycle.ui_shell.theme import help_button
        btn = help_button(parent, str(guide_file), "Test Guide",
                          tooltip="Hover hint")
        assert btn.toolTip() == "Hover hint"

    def test_accent_applied_in_stylesheet(self, parent, guide_file):
        """Accent color must appear in the rendered stylesheet so
        amber-for-Step-3 / green-for-Step-4 visual is enforced."""
        from napari_epicycle.ui_shell.theme import help_button
        amber_btn = help_button(parent, str(guide_file), "T",
                                accent="#d4a843")
        green_btn = help_button(parent, str(guide_file), "T",
                                accent="#6abf69")
        assert "212,168,67" in amber_btn.styleSheet()
        assert "106,191,105" in green_btn.styleSheet()


# ---------------------------------------------------------------------------
# themed_help_dialog
# ---------------------------------------------------------------------------


class TestThemedHelpDialog:

    def test_returns_qdialog(self, parent, guide_file):
        from napari_epicycle.ui_shell.theme import themed_help_dialog
        dlg = themed_help_dialog(parent, "Test Guide", str(guide_file))
        assert isinstance(dlg, QDialog)
        dlg.close()

    def test_non_modal(self, parent, guide_file):
        """Non-modal is the entire point — verify setModal(False)
        was called."""
        from napari_epicycle.ui_shell.theme import themed_help_dialog
        dlg = themed_help_dialog(parent, "Test Guide", str(guide_file))
        assert dlg.isModal() is False
        dlg.close()

    def test_loads_markdown_content(self, parent, guide_file):
        from napari_epicycle.ui_shell.theme import themed_help_dialog
        dlg = themed_help_dialog(parent, "Test Guide", str(guide_file))
        from qtpy.QtWidgets import QTextBrowser
        browsers = dlg.findChildren(QTextBrowser)
        assert len(browsers) == 1
        text = browsers[0].toPlainText()
        assert "Test Guide" in text
        assert "bullet one" in text
        dlg.close()

    def test_missing_file_renders_placeholder(self, parent, tmp_path):
        from napari_epicycle.ui_shell.theme import themed_help_dialog
        bogus = tmp_path / "does_not_exist.md"
        dlg = themed_help_dialog(parent, "Bogus", str(bogus))
        from qtpy.QtWidgets import QTextBrowser
        browser = dlg.findChild(QTextBrowser)
        text = browser.toPlainText()
        assert "not yet written" in text.lower() or "placeholder" in text.lower()
        dlg.close()

    def test_open_twice_raises_existing(self, parent, guide_file):
        """Clicking `?` twice for the same guide should NOT stack two
        dialogs — should raise/focus the existing one."""
        from napari_epicycle.ui_shell.theme import themed_help_dialog
        dlg1 = themed_help_dialog(parent, "Test Guide", str(guide_file))
        dlg2 = themed_help_dialog(parent, "Test Guide", str(guide_file))
        assert dlg1 is dlg2
        dlg1.close()

    def test_open_two_different_guides_creates_two_dialogs(
            self, parent, tmp_path):
        from napari_epicycle.ui_shell.theme import themed_help_dialog
        g1 = tmp_path / "g1.md"
        g1.write_text("# Guide 1\n", encoding="utf-8")
        g2 = tmp_path / "g2.md"
        g2.write_text("# Guide 2\n", encoding="utf-8")
        dlg1 = themed_help_dialog(parent, "G1", str(g1))
        dlg2 = themed_help_dialog(parent, "G2", str(g2))
        assert dlg1 is not dlg2
        dlg1.close()
        dlg2.close()

    def test_closes_clear_registry(self, parent, guide_file):
        from napari_epicycle.ui_shell.theme import (
            themed_help_dialog, _OPEN_HELP_DIALOGS,
        )
        dlg1 = themed_help_dialog(parent, "T", str(guide_file))
        key = str(guide_file)
        assert key in _OPEN_HELP_DIALOGS
        dlg1.close()
        QApplication.processEvents()
        assert key not in _OPEN_HELP_DIALOGS


# ---------------------------------------------------------------------------
# Step 3 subsection guide presence
# ---------------------------------------------------------------------------


class TestStep3GuidesPresent:
    """All 10 Step 3 subsection guides must exist on disk.

    If you're seeing a failure here, either author the missing
    markdown file under ``docs/guides/`` or remove the ``?`` button
    wiring from the corresponding section in
    ``fusion_reconstruction/ui.py``.

    (Earlier in this branch's history, the non-Model-Refinement
    guides were placeholders — the dialog rendered a friendly
    'Guide not yet written' message rather than crashing.  Content
    landed 2026-05-18; this test promotes them to required.)
    """

    @pytest.mark.parametrize("guide_filename", [
        # Model Refinement (shipped first)
        "step3_model_refinement_patch_selection.md",
        "step3_model_refinement_fine_tuning.md",
        "step3_model_refinement_training_config.md",
        # Predict + Reconstruct
        "step3_predict_reconstruct_model_source.md",
        "step3_predict_reconstruct_patch_state.md",
        "step3_predict_reconstruct_settings.md",
        # Curation
        "step3_curation_curate.md",
        "step3_curation_qc_features.md",
        "step3_curation_qc_plot.md",
    ])
    def test_subsection_guide_exists(self, guide_filename):
        repo_root = Path(__file__).resolve().parents[2]
        guide_path = repo_root / "docs" / "guides" / guide_filename
        assert guide_path.exists(), (
            f"Step 3 subsection guide missing: {guide_path}. "
            f"Either write the markdown or unwire the ? button "
            f"from fusion_reconstruction/ui.py."
        )

    @pytest.mark.parametrize("guide_filename", [
        "step3_model_refinement_patch_selection.md",
        "step3_model_refinement_fine_tuning.md",
        "step3_model_refinement_training_config.md",
        "step3_predict_reconstruct_model_source.md",
        "step3_predict_reconstruct_patch_state.md",
        "step3_predict_reconstruct_settings.md",
        "step3_curation_curate.md",
        "step3_curation_qc_features.md",
        "step3_curation_qc_plot.md",
    ])
    def test_subsection_guide_nonempty(self, guide_filename):
        """Sanity check: guides have real content, not stub
        placeholders forgotten in place."""
        repo_root = Path(__file__).resolve().parents[2]
        guide_path = repo_root / "docs" / "guides" / guide_filename
        if not guide_path.exists():
            pytest.skip("existence covered by sibling test")
        text = guide_path.read_text(encoding="utf-8")
        assert len(text) > 200, (
            f"{guide_filename} is suspiciously short ({len(text)} chars) — "
            f"looks like a stub.  Either flesh it out or remove the ? wiring."
        )
        assert text.count("\n") > 5, (
            f"{guide_filename} has too few lines — looks like a stub."
        )
