"""Release hide list 2026-10-12: Step 5 nav item + utility placeholders hidden."""
from __future__ import annotations

import inspect

import pytest

pytest.importorskip("qtpy")
from qtpy.QtWidgets import QApplication

from napari_epicycle.ui_shell import workflow_shell as ws
from napari_epicycle.ui_shell.workflow_shell import EpiCycleWorkflowShell


@pytest.fixture(scope="module")
def app():
    yield QApplication.instance() or QApplication([])


def test_left_panel_hides_whole_tissue_and_placeholders(app):
    shell = EpiCycleWorkflowShell.__new__(EpiCycleWorkflowShell)
    shell._step_specs = shell._build_step_specs()
    shell._step_section_color = {}
    shell._select_step = lambda sid: None
    panel = shell._build_left_panel()
    panel.show()
    assert shell._nav_items["whole_tissue"].isHidden()
    others = [k for k in shell._nav_items if k != "whole_tissue"]
    assert len(others) == 4
    assert all(not shell._nav_items[k].isHidden() for k in others)
    for name in ("Visualization", "Animations", "Tools"):
        assert shell._nav_sections[name].isHidden()


def test_left_panel_source_has_no_spacing_gap():
    src = inspect.getsource(EpiCycleWorkflowShell._build_left_panel)
    assert "addSpacing" not in src
