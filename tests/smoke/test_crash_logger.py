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


# ---------------------------------------------------------------------------
# Console mirror
# ---------------------------------------------------------------------------

import io
import logging
import threading
import time


class _Rec:
    """A recording stand-in for a console stream."""

    encoding = "utf-8"

    def __init__(self):
        self.calls = []
        self.text = ""

    def write(self, s):
        self.calls.append(("write", s))
        self.text += s
        return 7  # a distinctive return value

    def flush(self):
        self.calls.append(("flush",))

    def isatty(self):
        return True

    def fileno(self):
        return 5


class _Raising(_Rec):
    def fileno(self):
        raise io.UnsupportedOperation("fileno")


@pytest.fixture
def mirror(tmp_path, reset_crash_logger, monkeypatch):
    """An installed logger (no auto mirror) plus clean mirror state."""
    monkeypatch.setenv("EPICYCLE_SESSION_LOG", "0")
    path = crash_logger.install_crash_logging(log_dir=tmp_path)
    saved = (sys.stdout, sys.stderr)
    yield path
    crash_logger.restore_console()
    sys.stdout, sys.stderr = saved
    crash_logger._MIRRORED = False
    crash_logger._MIRROR_STOPPED = False


def _lines(path):
    return path.read_text(encoding="utf-8").splitlines()


def test_console_copy_is_unchanged(mirror):
    fake = _Rec()
    tee = crash_logger._Tee(fake)
    ret = [tee.write("abc"), tee.write("\n"), tee.write(""),
           tee.write("\r 10%|#  |"), tee.write("\r100%|###|\n"),
           tee.write("caf\u00e9 \u2713\n")]
    tee.flush()
    assert ret == [7] * 6
    assert fake.calls == [
        ("write", "abc"), ("write", "\n"), ("write", ""),
        ("write", "\r 10%|#  |"), ("write", "\r100%|###|\n"),
        ("write", "caf\u00e9 \u2713\n"), ("flush",),
    ]


def test_print_calls_reach_original_in_order(mirror):
    fake = _Rec()
    tee = crash_logger._Tee(fake)
    print("x", end="", file=tee)
    print("y", file=tee)
    assert fake.text == "xy\n"


def test_attribute_delegation(mirror):
    tee = crash_logger._Tee(_Rec())
    assert tee.isatty() is True
    assert tee.encoding == "utf-8"
    assert tee.fileno() == 5
    bad = crash_logger._Tee(_Raising())
    with pytest.raises(io.UnsupportedOperation):
        bad.fileno()


def test_file_side_whole_lines_with_timestamps(mirror):
    import re
    tee = crash_logger._Tee(_Rec())
    tee.write("abc")
    assert "abc" not in mirror.read_text(encoding="utf-8")  # no newline yet
    tee.write("\n")
    # on disk right after the newline, without closing the writer
    assert any(l.endswith(" abc") for l in _lines(mirror))
    tee.write("\r 10%|#  |")
    tee.write("\r100%|###|\n")
    lines = _lines(mirror)
    assert any(l.endswith(" 10%|#  |") for l in lines)
    assert any(l.endswith(" 100%|###|") for l in lines)
    stamped = [l for l in lines if l.endswith(" abc")]
    assert re.match(r"^\d\d:\d\d:\d\d\.\d{3} abc$", stamped[0])


def test_two_threads_never_mix_a_line(mirror):
    tee = crash_logger._Tee(_Rec())
    barrier = threading.Barrier(2)

    def work(tag):
        barrier.wait()
        for i in range(1000):
            tee.write(tag)
            tee.write(f"-{i}")
            tee.write("\n")

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        ts = [threading.Thread(target=work, args=(t,))
              for t in ("AAAA", "BBBB")]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        sys.setswitchinterval(old)
    body = [l.split(" ", 1)[1] for l in _lines(mirror)
            if " AAAA-" in l or " BBBB-" in l]
    assert len(body) == 2000
    for b in body:
        assert b.startswith(("AAAA-", "BBBB-")) and b.count("-") == 1


def test_file_failure_stops_once_and_console_keeps_everything(mirror):
    class _Bad:
        def write(self, s):
            raise OSError("disk gone")

        def flush(self):
            pass

        def fileno(self):
            return 0

        def close(self):
            pass

    good = crash_logger._LOG_FH
    crash_logger._LOG_FH = _Bad()
    try:
        fake = _Rec()
        tee = crash_logger._Tee(fake)
        tee.write("one\n")
        tee.write("two\n")
        tee.write("three\n")
        assert fake.text.count("[SESSION-LOG] mirror stopped") == 1
        assert "one\n" in fake.text and "two\n" in fake.text
        assert "three\n" in fake.text
    finally:
        crash_logger._LOG_FH = good


def test_fileno_without_a_console_raises_the_io_error(mirror):
    with pytest.raises(io.UnsupportedOperation):
        crash_logger._Tee(None).fileno()


def test_other_write_errors_do_not_stop_the_mirror(mirror):
    class _Odd:
        n = 0

        def write(self, s):
            _Odd.n += 1
            if _Odd.n == 1:
                raise ValueError("odd")

        def flush(self):
            pass

    good = crash_logger._LOG_FH
    crash_logger._LOG_FH = _Odd()
    try:
        tee = crash_logger._Tee(_Rec())
        tee.write("one" + '\n')
        tee.write("two" + '\n')
        assert _Odd.n == 2 and crash_logger._MIRROR_STOPPED is False
    finally:
        crash_logger._LOG_FH = good


def test_original_none_writes_file_only(mirror):
    tee = crash_logger._Tee(None)
    assert tee.write("only file\n") == len("only file\n")
    tee.flush()
    assert tee.isatty() is False
    assert any(l.endswith("only file") for l in _lines(mirror))


def test_handler_repointing_and_restore(mirror):
    orig = io.StringIO()
    handler = logging.StreamHandler(orig)
    lg = logging.getLogger("epicycle_test_repoint")
    lg.addHandler(handler)
    lg.propagate = False
    lg.setLevel(logging.INFO)
    try:
        assert crash_logger.mirror_console(stdout=orig, stderr=io.StringIO())
        lg.info("handler-line-xyz")
        assert any(l.endswith("handler-line-xyz") for l in _lines(mirror))
        assert "handler-line-xyz" in orig.getvalue()
        crash_logger.restore_console()
        assert handler.stream is orig
    finally:
        lg.removeHandler(handler)


def test_mirror_is_idempotent_and_restore_respects_foreign(mirror):
    a, b = io.StringIO(), io.StringIO()
    assert crash_logger.mirror_console(stdout=a, stderr=b)
    first = sys.stdout
    assert crash_logger.mirror_console(stdout=a, stderr=b)
    assert sys.stdout is first
    foreign = io.StringIO()
    sys.stdout = foreign
    crash_logger.restore_console()
    assert sys.stdout is foreign


def test_install_twice_rearms_faulthandler(tmp_path, reset_crash_logger,
                                           monkeypatch):
    import faulthandler
    monkeypatch.setenv("EPICYCLE_SESSION_LOG", "0")
    crash_logger.install_crash_logging(log_dir=tmp_path)
    seen = []
    monkeypatch.setattr(faulthandler, "enable",
                        lambda **kw: seen.append(kw))
    crash_logger.install_crash_logging()
    assert len(seen) == 1 and seen[0]["file"] is crash_logger._LOG_FH


def test_pytest_guard_skips_without_streams(mirror, capsys, monkeypatch):
    monkeypatch.delenv("EPICYCLE_SESSION_LOG", raising=False)
    before = sys.stdout
    assert crash_logger.mirror_console() is False
    assert sys.stdout is before
    out = capsys.readouterr().out
    assert "[SESSION-LOG] skipped" in out and "under pytest" in out


def test_install_without_dir_skips_under_pytest(reset_crash_logger, capsys):
    assert crash_logger.install_crash_logging() is None
    assert crash_logger.log_path() is None
    assert "[SESSION-LOG] skipped" in capsys.readouterr().out


def test_header_has_no_host_or_user_and_footer_written_once(mirror):
    import getpass
    import platform
    text = mirror.read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("EpiCycle session log.")
    assert "versions:" in text
    node = platform.node()
    if node:
        assert node not in text
    try:
        user = getpass.getuser()
    except Exception:
        user = ""
    if user and len(user) > 3:
        assert user not in text
    crash_logger.write_footer()
    crash_logger.write_footer()  # only once
    assert mirror.read_text(encoding="utf-8").count(
        "=== session end (clean) ===") == 1


def test_log_path_matches_install(mirror):
    assert crash_logger.log_path() == mirror


def test_mirror_speed_loose_bound(mirror):
    tee = crash_logger._Tee(_Rec())
    t0 = time.perf_counter()
    for i in range(10000):
        tee.write(f"line {i}\n")
    assert time.perf_counter() - t0 < 5.0
