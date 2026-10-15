"""Custom exceptions for the training subsystem.

The key design choice here is that ``TrainingAbortedError`` derives from
``BaseException`` (NOT ``Exception``).  This is intentional and makes mid-
epoch cancellation actually work.

Why BaseException?
------------------
When a long-running training call runs through third-party code that
wraps things in ``try/except Exception:`` (Python's
``logging.callHandlers``, Cellpose's internal loops, etc.), an
``Exception``-derived abort gets silently swallowed.  Deriving from
``BaseException`` bypasses every ``except Exception`` block in the
abort's call path — allowing the cancel signal to propagate through
Cellpose's training loop within a single epoch instead of waiting for
the next inter-phase abort check.

Catchers
--------
Code that handles training cancellation MUST use either:

  * ``except TrainingAbortedError:`` (preferred — explicit)
  * ``except BaseException:`` (rare — only for outer-most cleanup
    handlers that want to catch absolutely everything)

A plain ``except Exception:`` will NOT catch this — that's the whole
point.

Reuse note
----------
The same pattern applies more broadly: Step 3 refinement training, Step 4 stack segmentation, Step 5
tiled segmentation, and Step 3 reconstruction all run long blocking
calls that could benefit from the same approach.  Each section should
define its own ``*AbortedError(BaseException)`` rather than reusing
this one, so the catch sites are unambiguous.
"""

from __future__ import annotations


class TrainingAbortedError(BaseException):
    """Raised when ``_check_abort`` sees the cancellation flag set.

    Derives from ``BaseException`` (not ``Exception``) so it propagates
    cleanly through third-party ``except Exception`` blocks — notably
    Python's logging chain and Cellpose's training loop.  See module
    docstring for the architectural rationale.
    """
    pass
