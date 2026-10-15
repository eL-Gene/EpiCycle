"""Tests for the Step-1 large-data session logger (observe-only diagnostics).

Verifies the piece that matters for debugging the progressive-3D GL errors: the
CAUGHT vispy draw tracebacks (printed, not crashed on) are captured by teeing
stdout+stderr to a per-session file — plus idempotence and the no-crash contract
of every entry point.  Runs headless; restores the global streams/excepthook it
touches so it never leaks into other tests.
"""

from __future__ import annotations

import sys

from napari_epicycle.data_preparation import _largedata_diag_logger as diag


def _fresh(monkeypatch):
    """Force a clean (un-installed) module state for this test."""
    monkeypatch.setattr(diag, "_INSTALLED", False)
    monkeypatch.setattr(diag, "_LOG_FH", None)
    monkeypatch.setattr(diag, "_LOG_PATH", None)


def test_install_tees_stdout_and_stderr_to_the_session_file(tmp_path, monkeypatch):
    _fresh(monkeypatch)
    saved = (sys.stdout, sys.stderr, sys.excepthook)
    try:
        path = diag.install(log_dir=tmp_path)
        assert path is not None and path.exists()
        assert diag.log_path() == path
        print("MARKER-STDOUT-abc")
        sys.stderr.write("MARKER-STDERR-def\n")
        diag.breadcrumb("a crumb here")
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout, sys.stderr, sys.excepthook = saved

    text = path.read_text(encoding="utf-8", errors="replace")
    assert "MARKER-STDOUT-abc" in text          # stdout teed
    assert "MARKER-STDERR-def" in text          # stderr teed
    assert "a crumb here" in text               # breadcrumb landed
    assert "session log" in text.lower()        # header written


def test_install_is_idempotent(tmp_path, monkeypatch):
    _fresh(monkeypatch)
    saved = (sys.stdout, sys.stderr, sys.excepthook)
    try:
        p1 = diag.install(log_dir=tmp_path)
        p2 = diag.install(log_dir=tmp_path)     # second call is a no-op
        assert p1 is not None and p1 == p2
    finally:
        sys.stdout, sys.stderr, sys.excepthook = saved


def test_breadcrumb_and_log_path_are_safe_before_install(monkeypatch):
    _fresh(monkeypatch)
    diag.breadcrumb("no logger armed yet")      # must not raise
    assert diag.log_path() is None


def test_rotation_keeps_only_recent_sessions(tmp_path, monkeypatch):
    _fresh(monkeypatch)
    # Pre-seed more than the keep-limit of stale session files.
    for i in range(diag._KEEP_SESSIONS + 4):
        (tmp_path / f"session_old{i:02d}.log").write_text("x", encoding="utf-8")
    saved = (sys.stdout, sys.stderr, sys.excepthook)
    try:
        diag.install(log_dir=tmp_path)          # rotates on install
    finally:
        sys.stdout, sys.stderr, sys.excepthook = saved
    remaining = list(tmp_path.glob("session_*.log"))
    # keep-limit stale ones + the one just created.
    assert len(remaining) <= diag._KEEP_SESSIONS + 1
