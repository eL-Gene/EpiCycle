"""The label target of ``reshape_region`` / ``find_seeds`` is FROZEN.

invariant 13: the region target plugs in as ONE
early dispatch in front of the label path, and "every line of the label
path stays textually unchanged".  This file is the proof.  It was recorded
at the code BEFORE any region-target change (``538f359e``, which carries
``44dbdcfc``'s engine unchanged) and holds, for three existing engine
fixtures, the partition, the markers, the region and the elevation byte for
byte (as SHA-256 of the raw bytes, with shape and dtype), plus every
message, every count and every per-piece statistic of both
``reshape_region`` and ``find_seeds``.

``EPICYCLE_STRICT_GOLDEN=1`` (the pre-merge setting) compares the float
elevation BYTE for byte as well.  Without it the elevation is compared on
shape, dtype and a few summary numbers, so a different BLAS on another
machine cannot fail a day-to-day run -- the integer arrays (partition,
markers, region) are always byte-equal.

To re-record after a DELIBERATE label-path change (never for the Add
rebuild): ``python -m tests.annotator.test_reshape_label_path_golden`` and
paste the printed tables over ``RECORDED`` / ``REFUSAL_RECORDED``.

Pure pytest -- no Qt, no napari.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import pytest

from packages.epicycle_annotator.reshape import ReshapeSpec, reshape_region
from packages.epicycle_annotator.reshape.engine import find_seeds

try:  # run as a test
    from .reshape_fixtures import dumbbell, stacked_in_z, two_columns
except ImportError:  # pragma: no cover - the recorder, run as a module
    from tests.annotator.reshape_fixtures import dumbbell, stacked_in_z, two_columns

STRICT = os.environ.get("EPICYCLE_STRICT_GOLDEN") == "1"

ISO = (1.0, 1.0, 1.0)
ANISO = (5.0, 1.0, 1.0)


def _digest(array):
    if array is None:
        return None
    a = np.ascontiguousarray(np.asarray(array))
    return {
        "shape": tuple(int(v) for v in a.shape),
        "dtype": str(a.dtype),
        "sha256": hashlib.sha256(a.tobytes()).hexdigest(),
    }


def _float_summary(array):
    a = np.asarray(array, np.float64)
    return (float(a.min()), float(a.max()), float(a.mean()), float(a.std()))


# --------------------------------------------------------------------------- #
# the three cases
# --------------------------------------------------------------------------- #
def _case_two_columns():
    labels, guide, mem, _ = two_columns(0.4)
    spec = ReshapeSpec(target="label", seeds="labels", walls="intensity",
                       params={"wall_channel": "mem", "clip": 0.0})
    return dict(labels=labels, images={"mem": mem}, spec=spec,
                seed_layer=guide, spacing=ISO)


def _case_stacked_membrane_aniso():
    labels, guide, mem, _ = stacked_in_z(0.4)
    spec = ReshapeSpec(target="label", seeds="labels", walls="membrane",
                       refine=("merge_small",),
                       params={"wall_channel": "mem", "min_size": 5})
    return dict(labels=labels, images={"mem": mem}, spec=spec,
                seed_layer=guide, spacing=ANISO)


def _case_dumbbell_distance():
    labels = dumbbell()
    spec = ReshapeSpec(target="label", seeds="distance", walls="distance")
    return dict(labels=labels, images={}, spec=spec, seed_layer=None,
                spacing=ISO)


CASES = {
    "two_columns": _case_two_columns,
    "stacked_membrane_aniso": _case_stacked_membrane_aniso,
    "dumbbell_distance": _case_dumbbell_distance,
}


def _observe(name):
    case = CASES[name]()
    out = reshape_region(case["labels"], case["images"], case["spec"],
                         target_id=7, seed_layer=case["seed_layer"],
                         spacing=case["spacing"])
    found = find_seeds(case["labels"], case["images"], case["spec"],
                       target_id=7, roi=out.roi, seed_layer=case["seed_layer"],
                       spacing=case["spacing"])
    return {
        "status": out.status,
        "refusal_reason": out.refusal_reason,
        "roi": tuple((sl.start, sl.stop) for sl in out.roi),
        "result": _digest(out.result),
        "seeds": _digest(out.seeds),
        "region": _digest(out.region),
        "elevation": _digest(out.elevation),
        "elevation_summary": _float_summary(out.elevation),
        "n_seeds": out.n_seeds,
        "n_seeds_found": out.n_seeds_found,
        "n_seeds_dropped": out.n_seeds_dropped,
        "n_pieces": out.n_pieces,
        "n_pieces_before_merge": out.n_pieces_before_merge,
        "piece_sizes": list(out.piece_sizes),
        "piece_stats": [dict(r) for r in out.piece_stats],
        "coverage": out.coverage,
        "messages": list(out.messages),
        "find_status": found.status,
        "find_seeds": _digest(found.seeds),
        "find_n_found": found.n_found,
        "find_n_in_region": found.n_in_region,
        "find_messages": list(found.messages),
        "find_markers": [dict(r) for r in found.markers],
    }


def _observe_refusal():
    """A zero-marker refusal: its reason and its exact sentence."""
    labels, _guide, mem, _ = two_columns(0.4)
    spec = ReshapeSpec(target="label", seeds="labels", walls="intensity",
                       params={"wall_channel": "mem", "clip": 0.0})
    empty = np.zeros_like(labels)
    out = reshape_region(labels, {"mem": mem}, spec, target_id=7,
                         seed_layer=empty, spacing=ISO)
    found = find_seeds(labels, {"mem": mem}, spec, target_id=7,
                       seed_layer=empty, spacing=ISO)
    return {
        "status": out.status,
        "refusal_reason": out.refusal_reason,
        "messages": list(out.messages),
        "find_status": found.status,
        "find_refusal_reason": found.refusal_reason,
        "find_messages": list(found.messages),
    }


# --------------------------------------------------------------------------- #
# the recording (made at 538f359e, before any region-target change)
# --------------------------------------------------------------------------- #
RECORDED = {'dumbbell_distance': {'coverage': 1.0,
                       'elevation': {'dtype': 'float32',
                                     'sha256': '1680a1bdd199efe96f6dd63e2e8ce2fb16b23349b7fae729562c297d61771c88',
                                     'shape': (23, 39, 23)},
                       'elevation_summary': (0.0,
                                             1.0,
                                             0.9288365338803909,
                                             0.20062454981425173),
                       'find_markers': [{'bbox': (10, 14, 10, 14, 10, 14),
                                         'centroid_zyx': (12.0,
                                                          12.377358490566039,
                                                          12.0),
                                         'id': 1,
                                         'voxels': 53},
                                        {'bbox': (10, 14, 26, 30, 10, 14),
                                         'centroid_zyx': (12.0,
                                                          27.62264150943396,
                                                          12.0),
                                         'id': 2,
                                         'voxels': 53}],
                       'find_messages': [],
                       'find_n_found': 2,
                       'find_n_in_region': 2,
                       'find_seeds': {'dtype': 'int32',
                                      'sha256': 'f07120b0e4a238d990c9e5955f523560e59d12d3f0396cf4b27b17971ff609b7',
                                      'shape': (23, 39, 23)},
                       'find_status': 'ok',
                       'messages': [],
                       'n_pieces': 2,
                       'n_pieces_before_merge': 2,
                       'n_seeds': 2,
                       'n_seeds_dropped': 0,
                       'n_seeds_found': 2,
                       'piece_sizes': [1468, 1443],
                       'piece_stats': [{'bbox': (5, 19, 5, 20, 5, 19),
                                        'centroid_zyx': (12.0,
                                                         12.250681198910081,
                                                         12.0),
                                        'id': 1,
                                        'voxels': 1468},
                                       {'bbox': (5, 19, 21, 35, 5, 19),
                                        'centroid_zyx': (12.0,
                                                         27.883575883575883,
                                                         12.0),
                                        'id': 2,
                                        'voxels': 1443}],
                       'refusal_reason': None,
                       'region': {'dtype': 'bool',
                                  'sha256': 'bb0f841417e5a176133162f3101e89cf955c8e65146153ff6c640f3e6c060544',
                                  'shape': (23, 39, 23)},
                       'result': {'dtype': 'int32',
                                  'sha256': '4af222d12138fe2bf88d5387d36e6d6516f66fd3c6983fd7fcc1a2e3d952fb3b',
                                  'shape': (23, 39, 23)},
                       'roi': ((1, 24), (1, 40), (1, 24)),
                       'seeds': {'dtype': 'int32',
                                 'sha256': 'f07120b0e4a238d990c9e5955f523560e59d12d3f0396cf4b27b17971ff609b7',
                                 'shape': (23, 39, 23)},
                       'status': 'ok'},
 'stacked_membrane_aniso': {'coverage': 1.0,
                            'elevation': {'dtype': 'float32',
                                          'sha256': '5c0b7e2330d440e89707fa27254bdf4ec7b53b56e7434a7e0273438e171e6c3d',
                                          'shape': (24, 24, 24)},
                            'elevation_summary': (0.0,
                                                  1.0,
                                                  0.3547216958614924,
                                                  0.3002536991694006),
                            'find_markers': [{'bbox': (4, 6, 10, 13, 10, 13),
                                              'centroid_zyx': (5.0, 11.5, 11.5),
                                              'id': 1,
                                              'voxels': 48},
                                             {'bbox': (17, 19, 10, 13, 10, 13),
                                              'centroid_zyx': (18.0, 11.5, 11.5),
                                              'id': 2,
                                              'voxels': 48}],
                            'find_messages': [],
                            'find_n_found': 2,
                            'find_n_in_region': 2,
                            'find_seeds': {'dtype': 'int32',
                                           'sha256': '1531fb2a03e776e7092c9799faa7ff81c8de93d6a5a919b11483ddd4b3ddfc00',
                                           'shape': (24, 24, 24)},
                            'find_status': 'ok',
                            'messages': ['Wall contrast 0.15-1.00.'],
                            'n_pieces': 2,
                            'n_pieces_before_merge': 2,
                            'n_seeds': 2,
                            'n_seeds_dropped': 0,
                            'n_seeds_found': 2,
                            'piece_sizes': [2654, 2466],
                            'piece_stats': [{'bbox': (2, 21, 4, 19, 4, 19),
                                             'centroid_zyx': (7.2388847023360965,
                                                              11.336850037678975,
                                                              11.380934438583271),
                                             'id': 1,
                                             'voxels': 2654},
                                            {'bbox': (2, 21, 4, 19, 4, 19),
                                             'centroid_zyx': (16.085969180859692,
                                                              11.67558799675588,
                                                              11.628142741281428),
                                             'id': 2,
                                             'voxels': 2466}],
                            'refusal_reason': None,
                            'region': {'dtype': 'bool',
                                       'sha256': 'bd0f30bdd38c0e0e39362cac1884fe312df64588978b0cebdab1502612b6c7d4',
                                       'shape': (24, 24, 24)},
                            'result': {'dtype': 'int32',
                                       'sha256': 'b0fa98b7e8e7f32f9b332e5d46a51f06fa4786ce18eeb6adbbf17356799ad958',
                                       'shape': (24, 24, 24)},
                            'roi': ((0, 24), (0, 24), (0, 24)),
                            'seeds': {'dtype': 'int32',
                                      'sha256': '1531fb2a03e776e7092c9799faa7ff81c8de93d6a5a919b11483ddd4b3ddfc00',
                                      'shape': (24, 24, 24)},
                            'status': 'ok'},
 'two_columns': {'coverage': 1.0,
                 'elevation': {'dtype': 'float32',
                               'sha256': '0ac0c87423e0e72f694ce4ea8b6a599c89f0f817a0a352dcb2487496cceef9f2',
                               'shape': (12, 35, 23)},
                 'elevation_summary': (0.0,
                                       1.0,
                                       0.24441303185362748,
                                       0.2831268808594774),
                 'find_markers': [{'bbox': (0, 11, 14, 14, 16, 16),
                                   'centroid_zyx': (5.5, 14.0, 16.0),
                                   'id': 1,
                                   'voxels': 12},
                                  {'bbox': (0, 11, 26, 26, 16, 16),
                                   'centroid_zyx': (5.5, 26.0, 16.0),
                                   'id': 2,
                                   'voxels': 12}],
                 'find_messages': [],
                 'find_n_found': 2,
                 'find_n_in_region': 2,
                 'find_seeds': {'dtype': 'int32',
                                'sha256': 'ffa3656c4521626fbdfbc696db011bf925ecbdbb34232ec805d6697e471181a7',
                                'shape': (12, 35, 23)},
                 'find_status': 'ok',
                 'messages': ['Wall contrast 0.33-1.00.'],
                 'n_pieces': 2,
                 'n_pieces_before_merge': 2,
                 'n_seeds': 2,
                 'n_seeds_dropped': 0,
                 'n_seeds_found': 2,
                 'piece_sizes': [2125, 2111],
                 'piece_stats': [{'bbox': (0, 11, 7, 20, 9, 23),
                                  'centroid_zyx': (5.509647058823529,
                                                   13.805176470588234,
                                                   15.999529411764707),
                                  'id': 1,
                                  'voxels': 2125},
                                 {'bbox': (0, 11, 20, 33, 9, 23),
                                  'centroid_zyx': (5.490288962576978,
                                                   26.235907153008053,
                                                   16.000473709142586),
                                  'id': 2,
                                  'voxels': 2111}],
                 'refusal_reason': None,
                 'region': {'dtype': 'bool',
                            'sha256': '69856bd0776dcc7ebfeea2162798d37183e79dbcf7fed0a4452f701e914a48a7',
                            'shape': (12, 35, 23)},
                 'result': {'dtype': 'int32',
                            'sha256': '9accd00cb06135dd2cf996681706528fae6f3a99bb8d523ba9ac06def5428859',
                            'shape': (12, 35, 23)},
                 'roi': ((0, 12), (3, 38), (5, 28)),
                 'seeds': {'dtype': 'int32',
                           'sha256': 'ffa3656c4521626fbdfbc696db011bf925ecbdbb34232ec805d6697e471181a7',
                           'shape': (12, 35, 23)},
                 'status': 'ok'}}

REFUSAL_RECORDED = {'find_messages': ['No markers found in this object -- lower the prominence, change '
                   'the marker source, or paint one. (Nothing was changed.)'],
 'find_refusal_reason': 'no-seeds',
 'find_status': 'refused',
 'messages': ['Wall contrast 0.33-1.00.',
              'No markers found in this object -- lower the prominence, change where '
              'the markers come from, or paint one. (Nothing was changed.)'],
 'refusal_reason': 'no-seeds',
 'status': 'refused'}


def _compare(name, got, want):
    assert set(got) == set(want), name
    for key, value in want.items():
        if key == "elevation" and not STRICT:
            assert got[key]["shape"] == value["shape"], (name, key)
            assert got[key]["dtype"] == value["dtype"], (name, key)
            continue
        if key == "elevation_summary":
            np.testing.assert_allclose(got[key], value, rtol=1e-6, atol=1e-9,
                                       err_msg="{} {}".format(name, key))
            continue
        assert got[key] == value, "{}: {} changed".format(name, key)


@pytest.mark.parametrize("name", sorted(CASES))
def test_label_path_is_byte_identical(name):
    assert RECORDED is not None and name in RECORDED
    _compare(name, _observe(name), RECORDED[name])


def test_a_zero_marker_refusal_is_unchanged():
    assert REFUSAL_RECORDED is not None
    assert _observe_refusal() == REFUSAL_RECORDED


def test_find_seeds_without_a_target_id_still_raises_on_the_label_path():
    """``target_id=None`` becomes the default for the region target;
    on the label path it must still raise, as a missing argument did."""
    labels, guide, mem, _ = two_columns(0.4)
    spec = ReshapeSpec(target="label", seeds="labels", walls="intensity",
                       params={"wall_channel": "mem"})
    with pytest.raises((TypeError, ValueError)):
        find_seeds(labels, {"mem": mem}, spec, seed_layer=guide)
    with pytest.raises((TypeError, ValueError)):
        find_seeds(labels, {"mem": mem}, spec, target_id=None,
                   seed_layer=guide)


if __name__ == "__main__":  # pragma: no cover - the recorder
    import pprint

    table = {name: _observe(name) for name in sorted(CASES)}
    print("RECORDED = ", end="")
    pprint.pprint(table, width=88, sort_dicts=True)
    print()
    print("REFUSAL_RECORDED = ", end="")
    pprint.pprint(_observe_refusal(), width=88, sort_dicts=True)
