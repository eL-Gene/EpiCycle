"""The prediction cache writer runs on the worker thread, so it must not
touch the log widget; the main-thread patch-done handler logs the line."""
from __future__ import annotations

import ast
import inspect
import textwrap
from unittest.mock import MagicMock

from napari_epicycle.fusion_reconstruction import _step3_pr_text as _pr_text
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)


def test_the_cache_writer_never_calls_append_log():
    src = textwrap.dedent(inspect.getsource(C._persist_prediction_cache))
    tree = ast.parse(src)
    names = [n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)]
    assert "_append_log" not in names


def _handler_ctrl():
    c = C.__new__(C)
    c._predicting_patches = {"A.tif"}
    c._serial_pred_launch_epoch = None
    c._pipeline_last_view_total = 0
    c._pipeline_slices_done = 0
    c.ui = MagicMock()
    c.state_manager = MagicMock()
    for n in ("_refresh_patch_list_colors", "_refresh_summary_dashboard",
              "_refresh_patch_state_list", "_update_button_states"):
        setattr(c, n, lambda: None)
    c.logged = []
    c._append_log = lambda t: c.logged.append(t)
    return c


def test_the_patch_done_handler_logs_the_saved_cache():
    c = _handler_ctrl()
    c._on_serial_pred_patch_done(
        "A.tif", {"cache_saved": True, "cache_stem": "A"})
    assert _pr_text.S3_LOG_CACHE_SAVED.format(stem="A") in c.logged


def test_the_patch_done_handler_is_silent_when_no_cache_was_written():
    c = _handler_ctrl()
    c._on_serial_pred_patch_done("A.tif", {"cache_saved": False})
    assert not any("cache saved" in t.lower() for t in c.logged)
