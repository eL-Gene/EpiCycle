"""The carved-out ``print`` sites -- a message sink instead of stdout.

The original membrane-sandbox library printed at eight sites.  Six of them are not debug noise: they carry the
**only** signal that a guard fired -- guided filter unavailable, guided refine
would erase a cell, random walker failed, diffusion dropped a cell,
u-Segment3D failed, wanted N pieces but got fewer.

A pure module must not print, and the plugin must not lose those messages.  So every one of them becomes an entry in
``ReshapeResult.messages``, and an optional ``log`` callback lets a caller
mirror them live (the UI console, a test, a notebook).
"""

from __future__ import annotations

from typing import Callable, Iterable


class MessageSink:
    """Collects plain-language messages; optionally mirrors them to ``log``.

    ``log`` is any one-argument callable (``print``, a Qt console appender, a
    test spy).  Exceptions raised by ``log`` are swallowed: a broken logger must
    never take down a segmentation.
    """

    __slots__ = ("messages", "_log")

    def __init__(self, log: Callable[[str], None] | None = None) -> None:
        self.messages: list[str] = []
        self._log = log

    def emit(self, text: str) -> None:
        text = str(text)
        self.messages.append(text)
        if self._log is not None:
            try:
                self._log(text)
            except Exception:
                pass

    def extend(self, texts: Iterable[str]) -> None:
        for text in texts:
            self.emit(text)

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self):
        return iter(self.messages)
