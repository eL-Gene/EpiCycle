"""Tests for the Step-1 large-data logger, now a forwarder to the session log."""

from __future__ import annotations

import sys

import pytest

from napari_epicycle.data_preparation import _largedata_diag_logger as diag
from napari_epicycle.utils import crash_logger


@pytest.fixture
def clean(monkeypatch):
    prev_hook = sys.excepthook
    saved = (sys.stdout, sys.stderr)
    monkeypatch.setenv("EPICYCLE_SESSION_LOG", "0")
    crash_logger._INSTALLED = False
    crash_logger._LOG_FH = None
    crash_logger._LOG_PATH = None
    yield
    if crash_logger._LOG_FH is not None:
        crash_logger._LOG_FH.close()
    crash_logger._LOG_FH = None
    crash_logger._LOG_PATH = None
    crash_logger._INSTALLED = False
    sys.excepthook = prev_hook
    sys.stdout, sys.stderr = saved


def test_install_returns_the_session_path(tmp_path, clean):
    path = diag.install(log_dir=tmp_path)
    assert path is not None and path == crash_logger.log_path()
    assert diag.log_path() == path


def test_breadcrumb_reaches_the_session_file_and_no_second_file(tmp_path, clean):
    path = diag.install(log_dir=tmp_path)
    diag.breadcrumb("a crumb here")
    assert "a crumb here" in path.read_text(encoding="utf-8")
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


def test_install_is_idempotent(tmp_path, clean):
    assert diag.install(log_dir=tmp_path) == diag.install(log_dir=tmp_path)


def test_breadcrumb_and_log_path_are_safe_before_install(clean):
    diag.breadcrumb("no logger armed yet")
    assert diag.log_path() is None
