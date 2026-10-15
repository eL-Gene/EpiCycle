"""Tests for the persistent crash + breadcrumb logger.

Pure-Python (no Qt/napari viewer) so it runs in CI.  The module keeps
process-global state (faulthandler + sys.excepthook + an open file handle),
so each test resets that state via the ``reset_crash_logger`` fixture.
"""

from __future__ import annotations

import sys

import pytest

from napari_epicycle.utils import crash_logger


@pytest.fixture
def reset_crash_logger():
    """Reset module-global state + restore sys.excepthook around each test."""
    prev_hook = sys.excepthook
    # Reset before the test so install_crash_logging() actually runs.
    crash_logger._INSTALLED = False
    if crash_logger._LOG_FH is not None:
        try:
            crash_logger._LOG_FH.close()
        except Exception:
            pass
    crash_logger._LOG_FH = None
    crash_logger._LOG_PATH = None
    crash_logger._PREV_EXCEPTHOOK = None
    yield
    # Teardown: close the handle + restore the original excepthook so we don't
    # leak our hook into pytest's own error handling.
    if crash_logger._LOG_FH is not None:
        try:
            crash_logger._LOG_FH.close()
        except Exception:
            pass
    crash_logger._LOG_FH = None
    crash_logger._INSTALLED = False
    sys.excepthook = prev_hook


def test_install_creates_session_log_with_banner(tmp_path, reset_crash_logger):
    path = crash_logger.install_crash_logging(log_dir=tmp_path)
    assert path is not None
    assert path.exists()
    assert path.parent == tmp_path
    text = path.read_text(encoding="utf-8")
    assert "crash logging ENABLED" in text
    assert "faulthandler" in text


def test_install_enables_faulthandler(tmp_path, reset_crash_logger):
    import faulthandler
    crash_logger.install_crash_logging(log_dir=tmp_path)
    assert faulthandler.is_enabled()


def test_install_sets_excepthook(tmp_path, reset_crash_logger):
    crash_logger.install_crash_logging(log_dir=tmp_path)
    assert sys.excepthook is crash_logger._excepthook


def test_install_is_idempotent(tmp_path, reset_crash_logger):
    first = crash_logger.install_crash_logging(log_dir=tmp_path)
    # A second call (even with a different dir) must NOT re-install — it returns
    # the already-active path so we don't lose the running session's log.
    second = crash_logger.install_crash_logging(log_dir=tmp_path / "other")
    assert first == second


def test_breadcrumb_writes_flushed_line(tmp_path, reset_crash_logger):
    path = crash_logger.install_crash_logging(log_dir=tmp_path)
    crash_logger.breadcrumb("RCLICK-ENTER layer='foo'")
    # No explicit flush in the test — the breadcrumb must flush itself so the
    # line survives a hard crash.  Read WITHOUT closing the writer.
    text = path.read_text(encoding="utf-8")
    assert "RCLICK-ENTER layer='foo'" in text


def test_breadcrumb_is_noop_before_install():
    # Guard: calling breadcrumb before install (or after a failed install)
    # must never raise.
    crash_logger._LOG_FH = None
    crash_logger.breadcrumb("should not raise")  # no assertion — just must not raise


def test_liveness_handles_none_and_plain_objects():
    out = crash_logger.liveness(a=None, b=object())
    assert "a=none" in out
    # A plain (non-sip) object is reported alive-ish, never crashes.
    assert "b=" in out
    assert "a=none" in out and "b=" in out


def test_excepthook_logs_and_chains(tmp_path, reset_crash_logger):
    chained = {}

    def fake_prev(exc_type, exc_value, exc_tb):
        chained["called"] = (exc_type, exc_value)

    crash_logger.install_crash_logging(log_dir=tmp_path)
    crash_logger._PREV_EXCEPTHOOK = fake_prev
    try:
        raise ValueError("boom-xyz")
    except ValueError:
        crash_logger._excepthook(*sys.exc_info())
    text = crash_logger._LOG_PATH.read_text(encoding="utf-8")
    assert "UNCAUGHT PYTHON EXCEPTION" in text
    assert "boom-xyz" in text
    assert chained.get("called") is not None  # original hook still ran
