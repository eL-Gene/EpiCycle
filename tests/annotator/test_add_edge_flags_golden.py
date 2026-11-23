"""The edge flags of an ordinary box are FROZEN.

``add_segment.edge_flag_ids`` was rewritten to one bounding-box pass over
compacted ids (a whole patch of hundreds of objects took seconds with one
full-volume mask per object).  On an ordinary box -- no face left open --
the flagged ids must be exactly what the per-object version gave.  The
expected sets were recorded from that version before the rewrite and live
in ``fixtures/add_edge_flags_golden.json``.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from packages.epicycle_annotator.add_segment import edge_flag_ids

GOLDEN = pathlib.Path(__file__).parent / "fixtures" / "add_edge_flags_golden.json"
MARGINS = (0, 1, 3, 5)


def _blobs(seed, shape, n, dtype=np.int32, ids=None):
    rng = np.random.default_rng(seed)
    out = np.zeros(shape, dtype)
    ids = list(ids) if ids is not None else list(range(1, n + 1))
    for i in ids:
        size = rng.integers(1, 7, size=3)
        start = [int(rng.integers(0, max(1, s - int(k) + 1)))
                 for s, k in zip(shape, size)]
        sl = tuple(slice(a, a + int(k)) for a, k in zip(start, size))
        out[sl] = i
    return out


def cases():
    """Every volume the golden covers, by name (deterministic)."""
    out = {
        "blobs_int32": _blobs(0, (20, 30, 25), 30),
        "blobs_uint16_sparse": _blobs(1, (14, 22, 18), 12, np.uint16,
                                      ids=[7, 9, 300, 301, 1000, 4096, 5000,
                                           12345, 40000, 50000, 65000,
                                           65535]),
        "blobs_uint32_huge": _blobs(2, (12, 16, 16), 8, np.uint32,
                                    ids=[1, 2, 3, 2 ** 31 - 1, 2 ** 32 - 1,
                                         17, 18, 19]),
        "thin_z": _blobs(3, (2, 20, 20), 10),
    }
    lone = np.zeros((10, 10, 10), np.int32)
    lone[0:4, 0:4, 0:4] = 5
    out["lone_piece"] = lone
    far = np.zeros((12, 12, 12), np.int32)
    far[5:7, 5:7, 5:7] = 1            # the middle
    far[11, 5:7, 5:7] = 2             # the last z slice only
    far[5:7, 5:7, 0] = 3              # the first x column only
    far[3:5, 9:12, 3:5] = 4           # reaches the far y face
    out["one_face_each"] = far
    out["empty"] = np.zeros((6, 6, 6), np.int32)
    return out


def _record():
    """The golden as the CURRENT code computes it (run once, before the
    rewrite, to write the fixture)."""
    return {name: {str(m): sorted(int(i) for i in edge_flag_ids(vol, m))
                   for m in MARGINS}
            for name, vol in cases().items()}


def test_the_golden_fixture_exists():
    assert GOLDEN.is_file()


@pytest.mark.parametrize("name", sorted(cases()))
def test_an_ordinary_box_flags_exactly_the_recorded_ids(name):
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))[name]
    vol = cases()[name]
    for m in MARGINS:
        got = sorted(int(i) for i in edge_flag_ids(vol, m))
        assert got == want[str(m)], (name, m)


def test_the_golden_is_not_vacuous():
    want = json.loads(GOLDEN.read_text(encoding="utf-8"))
    assert want["blobs_int32"]["3"], "some objects touch the edge zone"
    assert len(want["blobs_int32"]["3"]) < 30, "and some do not"
    assert want["lone_piece"]["3"] == []
    assert 2 ** 31 - 1 in want["blobs_uint32_huge"]["5"] or \
        2 ** 32 - 1 in want["blobs_uint32_huge"]["5"]
