"""Shared skip marker for tests that pin the Windows font's pixel metrics."""
import sys

import pytest

windows_font_metrics = pytest.mark.skipif(
    sys.platform != "win32",
    reason="pins pixel metrics of the Windows font; other fonts differ by a few pixels",
)
