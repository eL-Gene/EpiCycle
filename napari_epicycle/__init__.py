import sys
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from importlib.metadata import version as _dist_version

    __version__ = _dist_version("napari-epicycle")
except Exception:  # not installed as a distribution (run from a source tree)
    __version__ = "0.1.0"


def _apply_console_encoding_guard():
    """Make ``print()`` unable to raise on a non-UTF-8 console (Windows).

    On Windows ``sys.stdout`` is frequently cp1252 rather than UTF-8 —
    notably whenever napari's output is redirected to a file.  A
    ``print()`` whose text contains a character cp1252 cannot encode then
    raises ``UnicodeEncodeError`` (a ``ValueError`` subclass).

    A raising ``print`` is not cosmetic: it is a CONTROL-FLOW hazard.  A
    diagnostic print inside a ``try`` whose ``except`` changes behaviour,
    or inside a Qt slot / ``eventFilter`` / callback, silently alters
    what the app does.  Two real incidents (2026-09-07) — accepted Step 2
    training pairs written to the legacy ``training_data/`` folder, and
    an aborted annotator dock-close cleanup.

    We set ``errors="backslashreplace"`` ONLY.  ``encoding`` is
    deliberately left alone: forcing UTF-8 would change the bytes the
    user's terminal actually receives (mojibake on a real cp1252
    console), whereas ``backslashreplace`` keeps the console's own
    encoding and degrades only the offending characters.

    Best-effort and never fatal:
      * ``sys.stdout`` is ``None`` under ``pythonw`` / detached processes
        -> ``AttributeError``
      * IPython's ``OutStream`` and pytest's capture objects have no
        ``reconfigure`` -> ``AttributeError``
      * an already-detached or closed stream -> ``ValueError`` / ``OSError``

    This is the belt; the braces are the ASCII-only rule for console
    strings, locked by ``tests/smoke/test_console_encoding_safe.py``.
    """
    for name in ("stdout", "stderr"):
        stream = getattr(sys, name, None)
        try:
            stream.reconfigure(errors="backslashreplace")
        except (AttributeError, ValueError, OSError):
            pass


_apply_console_encoding_guard()

from ._widget import EpiCycleMainWidget


def napari_experimental_provide_dock_widget():
    return EpiCycleMainWidget
