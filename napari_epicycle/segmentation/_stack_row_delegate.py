"""Step 4's stack rows: an ALIAS of ``ui_shell/_stack_row_delegate.py``.

The module moved to ``ui_shell`` so Step 3 can paint its rows with it.  This
path is the SAME module object (not a star re-export): the underscore names
Step 4 and its tests import keep working, and a ``monkeypatch`` on either
path patches the one module the code reads.
"""
import sys

from ..ui_shell import _stack_row_delegate as _m

sys.modules[__name__] = _m
