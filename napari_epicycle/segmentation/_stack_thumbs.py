"""Step 4's Session thumbnails: an ALIAS of ``ui_shell/_stack_thumbs.py``.

The module moved to ``ui_shell`` so Step 3 can draw its Session rows with
it.  This path is the SAME module object (not a star re-export): the
underscore names (``_StackThumbWorker``, ``_ThumbThread``, ``_CALLER_SLOTS``,
``_LIVE_THUMB_WORKERS``, ``_write_atomic``...) keep working, and a
``monkeypatch`` on either path patches the one module the code reads.
"""
import sys

from ..ui_shell import _stack_thumbs as _m

sys.modules[__name__] = _m
