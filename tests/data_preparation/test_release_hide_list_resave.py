"""Release hide list 2026-10-12: the dead "Re-save as OME-Zarr" button is hidden."""
from __future__ import annotations

import inspect

from napari_epicycle.data_preparation.ui import DataPreparationWidget


def test_resave_button_hidden_and_text_does_not_promise_it():
    src = inspect.getsource(DataPreparationWidget.confirm_non_eager_load)
    assert "convert.setVisible(False)" in src
    assert "Re-saving it as" not in src
