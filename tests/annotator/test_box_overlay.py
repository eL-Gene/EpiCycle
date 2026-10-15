"""Headless tests for the Add ROI-box vispy overlay (geometry + lifecycle).

No display needed: the overlay is constructed with ``None`` QtViewers (so no
vispy visuals are created) and we exercise the pure geometry + no-op safety.
"""

import numpy as np

from packages.epicycle_annotator.box_overlay import AddBoxOverlay, _rect
from packages.epicycle_annotator.roi_box import BoxRec


def test_rect_is_closed_5point_col_row():
    # rows 2..5, cols 1..4  → scene points are (x=col, y=row), closed loop.
    r = _rect(2, 5, 1, 4)
    assert r.shape == (5, 2)
    assert np.allclose(r[0], r[-1])          # closed
    assert np.allclose(r[0], [1, 2])         # (c0, r0)
    assert np.allclose(r[1], [4, 2])         # (c1, r0)
    assert np.allclose(r[2], [4, 5])         # (c1, r1)
    assert np.allclose(r[3], [1, 5])         # (c0, r1)


def test_overlay_with_no_viewers_is_safe_noop():
    # No QtViewers → no visuals created; update/repaint/destroy must not raise.
    ov = AddBoxOverlay(None, None, None, None)
    box = BoxRec.centered((20, 30, 40))
    ov.update(box, (20, 30, 40), scale=[2.0, 1.0, 1.0], translate=[0.0, 0.0, 0.0])
    ov.repaint()
    ov.destroy()
    ov.destroy()  # idempotent


def test_overlay_update_handles_edge_clamped_box():
    # A box hugging the volume edge still produces valid bounds (no raise).
    ov = AddBoxOverlay(None, None, None, None)
    box = BoxRec(cz=0, cy=0, cx=0, sz=100, sy=100, sx=100)
    ov.update(box, (10, 10, 10))
    ov.destroy()


def test_update_keys_skips_the_main_wireframe_during_a_drag():
    """'Live redraw': ``keys=("XY", "XZ", "YZ")`` redraws the three
    side rectangles and leaves the MAIN wireframe alone; no ``keys`` is a
    full update."""
    ov = AddBoxOverlay(None, None, None, None)
    seen = []
    ov._set = lambda key, pos, connect: seen.append(key)
    box = BoxRec.centered((20, 30, 40))
    ov.update(box, (20, 30, 40), keys=("XY", "XZ", "YZ"))
    assert seen == ["XY", "XZ", "YZ"]
    seen.clear()
    ov.update(box, (20, 30, 40))
    assert seen == ["MAIN", "XY", "XZ", "YZ"]
    seen.clear()
    ov.update(box, (20, 30, 40), keys=("XZ",))
    assert seen == ["XZ"]


class _FakeLine:
    def __init__(self):
        self.colors = []

    def set_data(self, pos=None, connect=None, color=None):
        if color is not None:
            self.colors.append(tuple(color))


def _overlay_with_fake_lines():
    ov = AddBoxOverlay(None, None, None, None)
    ov._lines = {k: _FakeLine() for k in ("MAIN", "XY", "XZ", "YZ")}
    return ov


def test_a_side_rectangle_off_its_slice_range_is_dimmed():
    """B3 fix pass LOW-4: drawn on every slice, grabbable only inside the
    box's range -- so off the range it is dimmed, and back to full colour
    inside it."""
    ov = _overlay_with_fake_lines()
    ov.set_in_range("XY", False)
    assert ov.is_dimmed("XY")
    assert ov._lines["XY"].colors[-1][3] < 0.5
    ov.set_in_range("XY", False)                 # no change, no redraw
    assert len(ov._lines["XY"].colors) == 1
    ov.set_in_range("XY", True)
    assert not ov.is_dimmed("XY")
    assert ov._lines["XY"].colors[-1][3] == 1.0
    assert ov._lines["XZ"].colors == [] and ov._lines["MAIN"].colors == []


def test_update_carries_the_in_range_state():
    ov = _overlay_with_fake_lines()
    box = BoxRec.centered((20, 30, 40))
    ov.update(box, (20, 30, 40), keys=("XY", "XZ", "YZ"),
              in_range={"XY": True, "XZ": False, "YZ": True, "MAIN": False})
    assert ov.is_dimmed("XZ") and not ov.is_dimmed("XY")
    assert not ov.is_dimmed("MAIN"), "the 3D wireframe is never dimmed"
    ov.update(box, (20, 30, 40))                 # no in_range: kept
    assert ov.is_dimmed("XZ")
