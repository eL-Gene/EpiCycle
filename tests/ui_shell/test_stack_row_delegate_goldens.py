"""The delegate's new keys are default-off.

Goldens (``tests/ui_shell/goldens/*.png``) were rendered at ``533d9cb7``
from the UNTOUCHED delegate: one Step 3 narrow row, one Step 3 wide row
(Session, with a tier line) and one Step 4 row.  The renderer runs in a
subprocess (offscreen, pinned font) and the pixels must match byte for byte.
A teeth test paints a record WITH the new keys and asserts it differs.
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = os.path.join(HERE, "goldens")
RENDER = os.path.join(HERE, "_delegate_golden_render.py")
ROOT = os.path.dirname(os.path.dirname(HERE))


@pytest.fixture(scope="module")
def rendered(tmp_path_factory):
    """One subprocess render of every row, the new-keys row included."""
    out = tmp_path_factory.mktemp("delegate_goldens")
    _render(out, extra=True)
    return out


def _render(tmp_path, extra=False):
    env = dict(os.environ, PYTHONUTF8="1", QT_QPA_PLATFORM="offscreen",
               PYTHONPATH=ROOT)
    cmd = [sys.executable, RENDER, str(tmp_path)]
    if extra:
        cmd.append("--extra")
    r = subprocess.run(cmd, env=env, capture_output=True, text=True,
                       timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]


def _pixels(path):
    from qtpy.QtGui import QImage
    img = QImage(str(path)).convertToFormat(QImage.Format_ARGB32)
    return (img.width(), img.height(),
            bytes(img.constBits().asstring(img.sizeInBytes())))


@pytest.mark.parametrize("name", ["s3_narrow", "s3_wide", "s4_wide"])
def test_rows_without_new_keys_match_goldens(rendered, name):
    assert _pixels(rendered / f"{name}.png") == _pixels(
        os.path.join(GOLD, f"{name}.png"))


def test_a_row_with_new_keys_differs(rendered):
    # same constructor kwargs as the golden: only the record's new keys differ
    new = _pixels(rendered / "new_keys_wide.png")
    assert new != _pixels(os.path.join(GOLD, "s3_wide.png"))


def test_the_new_keys_are_honoured_one_visual_change_shows(rendered):
    # teeth: the same record and kwargs, ONE strip colour changed -> the
    # bytes must differ, so the delegate really paints the strip it is given
    a = _pixels(rendered / "new_keys_wide.png")
    b = _pixels(rendered / "new_keys_wide_b.png")
    assert a[:2] == b[:2] and a[2] != b[2]


def test_the_narrow_keys_have_teeth(rendered):
    new = _pixels(rendered / "new_keys_narrow.png")
    assert new != _pixels(os.path.join(GOLD, "s3_narrow.png"))
