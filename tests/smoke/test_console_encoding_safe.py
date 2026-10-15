r"""Guard tests for the Windows console-encoding hazard (2026-09-07).

On Windows, when ``sys.stdout`` is not UTF-8 (cp1252 — the default when
output is redirected to a file, or under some terminal configurations),
``print()`` of a string containing a character outside cp1252 (e.g. the
arrows "→" / "←") raises ``UnicodeEncodeError`` — a subclass of
``ValueError``.

That is not merely cosmetic: a ``print`` that RAISES is a control-flow
hazard.  Two real incidents:

  * ``_persist_curated_training_pair`` — the arrow print sat inside a
    ``try`` whose ``except Exception`` reassigns ``out_dir`` to the
    LEGACY ``training_data/`` folder.  Two accepted training pairs were
    written to the wrong place on the user's real workspace.
  * ``launcher._DockCloseCleanupFilter.eventFilter`` — the arrow prints
    ran BEFORE ``_mark_and_cleanup()``, so the dock-close cleanup was
    aborted by the raise.

Two independent defenses, both tested here:

  1. A GLOBAL guard in ``napari_epicycle/__init__.py`` that reconfigures
     ``sys.stdout`` / ``sys.stderr`` with ``errors="backslashreplace"``
     (never touching ``encoding``), so any print survives.
  2. This AST scan, which fails if any ``print`` / ``_diag`` / logger
     call in ``napari_epicycle/`` or ``packages/`` contains a string
     constant with a character that cannot be encoded in cp1252.

The bar this scan enforces is **cp1252-codable**, not strict ASCII.  The
codebase legitimately uses the em-dash "—" (U+2014) throughout its prose
and cp1252 encodes it fine (0x97); 116 of them survive the scan today.
On a narrower console codepage (cp850 / cp437 — e.g. a raw ``cmd.exe``)
the em-dash is NOT codable, and there the global guard in defense 1 is
the defense.  That is precisely why both layers exist: this scan removes
the characters we can cheaply remove, the guard survives the rest.
"""

from __future__ import annotations

import ast
import io
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCAN_ROOTS = ("napari_epicycle", "packages")

_CALL_NAMES = {"print", "_diag"}
_LOG_METHODS = {
    "debug", "info", "warning", "warn", "error", "exception", "critical",
}

# ``QMessageBox.warning(...)`` / ``.critical(...)`` / ``.information(...)``
# collide with the logger method names above, but their text is rendered by
# Qt and never encoded to the console — an arrow in a dialog body is
# legitimate and must not fail this test.  None of the 17 such call sites is
# flagged today; this skip keeps a future one from failing it spuriously.
_QT_DIALOG_OWNERS = {"QMessageBox", "QtWidgets"}


def _is_qt_dialog_call(func) -> bool:
    """True for ``QMessageBox.warning(...)`` and
    ``QtWidgets.QMessageBox.critical(...)`` style calls, whose text is Qt
    widget text rather than console output."""
    if not isinstance(func, ast.Attribute):
        return False
    owner = func.value
    if isinstance(owner, ast.Name):
        return owner.id in _QT_DIALOG_OWNERS
    # QtWidgets.QMessageBox.warning(...) -> owner is itself an Attribute
    if isinstance(owner, ast.Attribute):
        return owner.attr in _QT_DIALOG_OWNERS
    return False


def _uncodable_cp1252(text: str) -> str:
    """Return the characters in ``text`` that cp1252 cannot encode."""
    bad = []
    for ch in text:
        try:
            ch.encode("cp1252")
        except UnicodeEncodeError:
            bad.append(ch)
    return "".join(sorted(set(bad)))


def _scan_console_calls():
    """AST-walk the source roots; yield (path, lineno, func, bad_chars)."""
    findings = []
    for root_name in SCAN_ROOTS:
        root = REPO_ROOT / root_name
        if not root.is_dir():
            continue
        for py in sorted(root.rglob("*.py")):
            # Vendored third-party code is out of our control.
            if "_vendor_" in str(py):
                continue
            try:
                tree = ast.parse(py.read_text(encoding="utf-8"))
            except (SyntaxError, UnicodeDecodeError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                func = node.func
                if isinstance(func, ast.Name):
                    name = func.id
                elif isinstance(func, ast.Attribute):
                    if _is_qt_dialog_call(func):
                        continue
                    name = func.attr
                else:
                    continue
                if name not in _CALL_NAMES and name not in _LOG_METHODS:
                    continue
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Constant) and isinstance(sub.value, str):
                        bad = _uncodable_cp1252(sub.value)
                        if bad:
                            findings.append(
                                (
                                    py.relative_to(REPO_ROOT).as_posix(),
                                    sub.lineno,
                                    name,
                                    bad,
                                )
                            )
    return findings


def test_no_non_cp1252_chars_in_console_calls():
    """No print / _diag / logger string constant may contain a character
    that cp1252 cannot encode — such a print RAISES when stdout is cp1252
    and that raise is a control-flow hazard (see module docstring)."""
    findings = _scan_console_calls()
    if findings:
        lines = "\n".join(
            f"  {path}:{lineno}  {func}(...)  bad chars: {bad!r}"
            for path, lineno, func, bad in findings
        )
        pytest.fail(
            "Non-cp1252 characters found in console-output string "
            f"constants ({len(findings)} site(s)).  Console strings must be "
            "cp1252-codable: use -> and <- instead of the arrows.  (The "
            "em-dash IS cp1252-codable and is fine here.)  Qt widget text "
            f"is exempt.\n{lines}"
        )


def test_scanner_skips_qt_dialog_statics():
    """An arrow inside ``QMessageBox.warning(...)`` is Qt widget text and must
    NOT be flagged; the same arrow in a bare ``warning(...)`` logger call must
    still be flagged."""
    src = (
        "QMessageBox.warning(self, 'T', 'lateral \u2192 axial')\n"
        "QtWidgets.QMessageBox.critical(self, 'T', 'a \u2192 b')\n"
        "logger.warning('bare logger \u2192 arrow')\n"
    )
    flagged = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            if _is_qt_dialog_call(func):
                continue
            name = func.attr
        else:
            continue
        if name not in _CALL_NAMES and name not in _LOG_METHODS:
            continue
        for sub_node in ast.walk(node):
            if isinstance(sub_node, ast.Constant) and isinstance(sub_node.value, str):
                if _uncodable_cp1252(sub_node.value):
                    flagged.append(sub_node.value)

    assert flagged == ["bare logger \u2192 arrow"], flagged


def test_em_dash_is_cp1252_codable():
    """The bar is cp1252-codable, not ASCII: the em-dash used throughout the
    codebase's prose must pass."""
    assert _uncodable_cp1252("a \u2014 b") == ""


def test_scanner_detects_a_known_bad_char():
    """Sanity: the scanner itself is not vacuous."""
    assert _uncodable_cp1252("a → b") == "→"
    assert _uncodable_cp1252("a -> b") == ""


# ---------------------------------------------------------------------------
# The global stdout/stderr guard
# ---------------------------------------------------------------------------

def test_apply_console_encoding_guard_makes_cp1252_prints_safe(monkeypatch):
    """With a cp1252 stdout installed, an arrow print must not raise once
    the guard has run; the char is written as a backslash escape."""
    from napari_epicycle import _apply_console_encoding_guard

    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="cp1252", newline="")
    monkeypatch.setattr(sys, "stdout", stream)

    # Pre-condition: without the guard this stream raises.
    with pytest.raises(UnicodeEncodeError):
        stream.write("a → b")

    _apply_console_encoding_guard()

    print("a → b")  # must NOT raise
    sys.stdout.flush()
    written = raw.getvalue().decode("cp1252")
    assert r"\u2192" in written  # backslash-escaped, not the raw char
    assert "a " in written


def test_guard_is_noop_on_none_streams(monkeypatch):
    """pythonw / detached processes have ``sys.stdout is None`` — the guard
    must swallow that, not raise at import time."""
    from napari_epicycle import _apply_console_encoding_guard

    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    _apply_console_encoding_guard()  # must not raise


def test_guard_is_noop_on_non_reconfigurable_stream(monkeypatch):
    """IPython's OutStream and pytest's capture objects have no
    ``reconfigure``; the guard must swallow the AttributeError."""
    from napari_epicycle import _apply_console_encoding_guard

    class _NoReconfigure:
        def write(self, s):
            return len(s)

        def flush(self):
            pass

    monkeypatch.setattr(sys, "stdout", _NoReconfigure())
    monkeypatch.setattr(sys, "stderr", _NoReconfigure())
    _apply_console_encoding_guard()  # must not raise


def test_guard_does_not_change_encoding(monkeypatch):
    """The guard must only touch ``errors`` — changing ``encoding`` would
    silently alter what the user's terminal receives."""
    from napari_epicycle import _apply_console_encoding_guard

    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
    monkeypatch.setattr(sys, "stdout", stream)
    _apply_console_encoding_guard()
    assert stream.encoding.lower().replace("-", "") == "cp1252"
    assert stream.errors == "backslashreplace"
