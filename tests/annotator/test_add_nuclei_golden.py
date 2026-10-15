"""The Nuclei preset's answer is FROZEN.

Recorded at ``f33078e6`` -- the Add orchestrator still at ``44dbdcfc``'s
behaviour, BEFORE any B2 change -- by composing exactly what today's
``_preview_add`` does at the widget's default settings:

    add_segment_region(image_roi, existing_roi, clahe=False, clahe_clip=0.01,
                       threshold=<Otsu of the box>, min_size=0, separate=True,
                       min_distance=3, next_label=max(volume) + 1,
                       spacing=main.scale)
    -> edge flags on the RAW result (margin 3)
    -> Curation3DAnnotator._smooth_add_result(result, 1.5)

The widget defaults of that day are frozen beside it (``DEFAULTS``).  The
rebuilt Nuclei preset (``add_preview.run_add_preview(preset="nuclei")``,
B2.1) must reproduce the voxels byte for byte; the ids may differ ONLY by the
id floor (new ids skip deleted ids), by exactly
``floor - max(volume)``.

To re-record after a DELIBERATE change (never for the Add rebuild):
``python -m tests.annotator.test_add_nuclei_golden`` and paste the printed
table over ``RECORDED``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

ISO = (1.0, 1.0, 1.0)
ANISO = (2.0, 1.0, 1.0)

# get_add_params() at the widget's defaults, 44dbdcfc (connectivity and
# marker are removed by B2.4; every other key keeps its value)
DEFAULTS = {
    "clahe": False, "clahe_clip": 0.01, "min_size": 0, "separate": True,
    "min_distance": 3, "clear_border": True, "edge_margin": 3,
    "auto_smooth": True, "smooth_sigma": 1.5,
}


def nuclei_fixture(seed):
    """A 24 x 48 x 48 nuclear stain: 14 bright blobs (two pairs touching),
    noise, and a label volume holding three kept objects (one over a blob)."""
    rng = np.random.default_rng(seed)
    shape = (24, 48, 48)
    img = rng.normal(0.1, 0.03, shape).astype(np.float32)
    z, y, x = np.indices(shape)
    centres = [(12, 8, 8), (12, 8, 20), (12, 8, 27), (6, 24, 10),
               (18, 24, 22), (12, 40, 40), (12, 30, 36), (4, 40, 12),
               (20, 12, 40), (12, 20, 30), (12, 36, 24), (8, 44, 30),
               (16, 2, 44), (12, 24, 46)]
    for i, (cz, cy, cx) in enumerate(centres):
        r = 3.5 + (i % 3) * 0.6
        blob = ((z - cz) / 1.2) ** 2 + (y - cy) ** 2 + (x - cx) ** 2 <= r * r
        img[blob] += 0.8 + 0.05 * (i % 4)
    labels = np.zeros(shape, np.int32)
    labels[2:8, 30:38, 2:8] = 5
    labels[10:14, 38:44, 36:44] = 9            # over the (12, 40, 40) blob
    labels[18:22, 2:6, 2:6] = 17
    return np.clip(img, 0, None), labels


def box_of(shape):
    return (slice(2, shape[0] - 2), slice(1, shape[1] - 1), slice(0, shape[2]))


def todays_nuclei_answer(img, labels, roi, spacing, next_label=None):
    """What ``_preview_add`` produced at 44dbdcfc (see module doc)."""
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )
    from packages.epicycle_annotator.add_segment import (
        _resolve_threshold,
        add_segment_region,
        add_threshold_space,
    )

    image_roi = img[roi].astype(np.float64)
    existing = labels[roi]
    proc = add_threshold_space(image_roi, clahe=False, clahe_clip=0.01)
    thr = _resolve_threshold(proc, "otsu")
    if next_label is None:
        next_label = int(labels.max()) + 1
    result = add_segment_region(
        image_roi, existing, clahe=False, clahe_clip=0.01, threshold=thr,
        min_size=0, separate=True, min_distance=3, next_label=next_label,
        spacing=spacing)
    edge = Curation3DAnnotator._add_edge_flags(None, result, 3)
    result = Curation3DAnnotator._smooth_add_result(None, result, 1.5)
    return result, sorted(int(v) for v in edge), thr


def digest(result):
    a = np.ascontiguousarray(result)
    return {"shape": tuple(int(v) for v in a.shape), "dtype": str(a.dtype),
            "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "ids": [int(v) for v in np.unique(a) if v]}


CASES = {"iso_seed0": (0, ISO), "aniso_seed1": (1, ANISO)}

RECORDED = {
    'aniso_seed1': {'shape': (20, 46, 48), 'dtype': 'int32', 'sha256': 'f16e990e7259206d215c13411f11f9711c65009d3bc9dfd3f642ef1d4c2d684f', 'ids': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], 'edge': [18, 19, 21, 23, 24, 25, 27, 31]},
    'iso_seed0': {'shape': (20, 46, 48), 'dtype': 'int32', 'sha256': '0658ccd9895cd2a5e14dedee5505a2f9c4b6ed06893d308537348b007166caf1', 'ids': [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], 'edge': [18, 19, 21, 23, 24, 26, 27, 31]},
}


def _compute(name):
    seed, spacing = CASES[name]
    img, labels = nuclei_fixture(seed)
    roi = box_of(labels.shape)
    result, edge, _thr = todays_nuclei_answer(img, labels, roi, spacing)
    d = digest(result)
    d["edge"] = edge
    return d


@pytest.mark.parametrize("name", sorted(CASES))
def test_todays_nuclei_answer_is_frozen(name):
    assert _compute(name) == RECORDED[name]


def test_the_fixture_is_not_trivial():
    """The golden must exercise a split, an edge flag and the kept clip."""
    for name in CASES:
        rec = RECORDED[name]
        assert len(rec["ids"]) >= 8
        assert rec["edge"]
        assert len(rec["edge"]) < len(rec["ids"])


if __name__ == "__main__":  # pragma: no cover - the recorder
    for _name in sorted(CASES):
        print("    {!r}: {!r},".format(_name, _compute(_name)))
