"""The Cells run count on the Add page equals what the Preview grows from.

The box-info line "{n} in the box · {m} over empty space become markers"
and the outline-drops amber read the engine's own seed report
(``find_region_seeds`` with the same outline the Preview gets).  Pinned
here over a world that has every drop at once: a kept object lying wholly
over one run cell (its marker is "over kept"), a kept object over most of
another (a sliver under Spread), a run object deep in the dark, one half in
the dark, and three tissue outlines -- under Spread and Fill, on a box the
size of the patch and on a sub-box handed the whole-patch outline, and on a
whole patch (filled to the walls with a background marker, whatever the
hidden Grow by says).
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import region as R

from .test_reshape_region import _threshold, _tissue

ISO = (1.0, 1.0, 1.0)
SHAPE = (12, 40, 40)
SUB = (slice(1, 11), slice(2, 34), slice(4, 38))


def _world(seed):
    mem, nuc, cells = _tissue(shape=SHAPE, seed=seed)
    ch = {"membrane": mem, "nuclei": nuc}
    run = cells.copy()
    run[3:9, 30:36, 3:9] = 900          # deep in the dark space: dropped dark
    run[3:9, 20:28, 32:38] = 901        # half in the tissue, half dark
    ids = [int(i) for i in np.unique(cells) if i]
    labels = np.zeros_like(cells)
    labels[cells == ids[0]] = 5000      # a whole cell kept: its marker is over kept
    m = cells == ids[1]
    xx = np.nonzero(m)[2]
    xcut = int(np.percentile(xx, 30))
    labels[m & (np.arange(SHAPE[2])[None, None, :] >= xcut)] = 5001  # a sliver
    return ch, run, labels


def _outlines():
    o1 = np.zeros(SHAPE, bool)
    o1[:, :24, :] = True
    o1[:, 13:24, 21:] = False           # leaves two cells out
    o2 = np.ones(SHAPE, bool)           # everything inside
    o3 = np.zeros(SHAPE, bool)
    o3[:, :, :15] = True                # cuts cells in half
    return {"o1": o1, "o2": o2, "o3": o3}


_CASES = (list(itertools.product((0, 1), ("o1", "o2", "o3"),
                                 ("compete", "flood"), ("whole", "sub")))
          + list(itertools.product((0, 1), ("o1", "o2", "o3"),
                                   ("fill",), ("whole",))))
_SEEN = {"slivers": 0, "dark": 0, "over_kept": 0, "outside": 0}


@pytest.mark.parametrize("seed,oname,growth,where", _CASES)
def test_the_run_count_is_the_previews_markers(monkeypatch, seed, oname,
                                               growth, where):
    ch, run, labels = _world(seed)
    outline = _outlines()[oname]
    thr = _threshold(ch)
    # "fill": a whole patch, its hidden Grow by left at Spread
    whole = growth == "fill"
    roi = tuple(slice(0, s) for s in SHAPE) if where == "whole" else SUB
    crop = where == "sub"
    params = {"preset": "cells", "dark_outside": False, "outline": "cleared",
              "growth": "compete" if whole else growth, "whole_patch": whole}
    info = AP.add_box_info(
        existing_roi=(labels[roi] if crop else labels), dark_sources=None,
        dark_rois=({k: v[roi] for k, v in ch.items()} if crop else ch),
        spacing=ISO, stride=1, dark_cache=thr, source=AP.SOURCE_RUN,
        run_guide_roi=(run[roi] if crop else run), nuclei_roi=None,
        params=params, roi=roi, outline=outline)

    seen = []
    real = R.reshape_empty_region

    ran = []

    def spy(*a, **kw):
        ran.append(a[2].params["growth"])     # the spec the engine got
        res = real(*a, **kw)
        seen.append(res)
        return res

    monkeypatch.setattr(R, "reshape_empty_region", spy)
    res = AP.run_add_preview(
        preset="cells", roi=roi, existing_roi=np.array(labels[roi], copy=True),
        next_label=10000, spacing=ISO,
        params=dict(params, source=AP.SOURCE_RUN, auto_smooth=False,
                    min_size=0),
        wall_roi=ch["membrane"][roi],
        dark_rois={k: v[roi] for k, v in ch.items()},
        run_guide_roi=run[roi], dark_threshold=thr, outline_roi=outline)
    assert len(seen) == 1
    engine = seen[0]
    # what the engine ran: the fill on a whole patch (no tiles), else the
    # Grow by pick
    assert ran == [("fill_with_background" if whole else growth)]
    if whole:
        assert engine.n_tiles == 0

    assert info.run_count is not None
    n_found, n_markers = info.run_count
    assert n_found == engine.n_seeds_found
    assert n_markers == (engine.n_seeds if engine.ok else 0)
    assert info.n_outside_outline == res.n_outside_outline
    if engine.refusal_reason == "no-markers-outline":
        assert n_markers == 0
    _SEEN["slivers"] += res.n_slivers
    _SEEN["dark"] += res.n_dropped_dark
    _SEEN["over_kept"] += res.n_markers_over_kept
    _SEEN["outside"] += res.n_outside_outline


def test_the_world_has_every_drop():
    """Teeth: the cases above met slivers, dark drops, markers over kept
    objects and outline drops (run after them, in file order)."""
    if sum(_SEEN.values()) == 0:
        pytest.skip("run with the parity cases")
    assert all(v > 0 for v in _SEEN.values()), _SEEN


def test_the_outline_dropping_every_marker_counts_zero():
    """An outline that leaves every marker outside: the engine refuses
    with its outline reason, and the count reads (found, 0)."""
    ch, run, labels = _world(0)
    thr = _threshold(ch)
    nothing = np.zeros(SHAPE, bool)
    nothing[0, 0, 0] = True
    roi = tuple(slice(0, s) for s in SHAPE)
    info = AP.add_box_info(
        existing_roi=labels, dark_sources=None, dark_rois=ch, spacing=ISO,
        stride=1, dark_cache=thr, source=AP.SOURCE_RUN, run_guide_roi=run,
        nuclei_roi=None, roi=roi, outline=nothing,
        params={"preset": "cells", "dark_outside": False,
                "outline": "cleared", "growth": "compete",
                "whole_patch": True})
    assert info.run_count is not None
    assert info.run_count[1] == 0 and info.run_count[0] > 0
    assert info.n_outside_outline > 0


def test_a_no_markers_refusal_hands_the_walls_back():
    """The walls are built before the markers: a refusal for no markers
    returns them, so a whole patch's retry does not build them again."""
    ch, run, labels = _world(0)
    thr = _threshold(ch)
    nothing = np.zeros(SHAPE, bool)
    nothing[0, 0, 0] = True
    roi = tuple(slice(0, s) for s in SHAPE)
    res = AP.run_add_preview(
        preset="cells", roi=roi, existing_roi=labels.copy(), next_label=10000,
        spacing=ISO,
        params={"preset": "cells", "dark_outside": False, "outline": "cleared",
                "growth": "compete", "whole_patch": True,
                "source": AP.SOURCE_RUN, "auto_smooth": False, "min_size": 0},
        wall_roi=ch["membrane"], dark_rois=ch, run_guide_roi=run,
        dark_threshold=thr, outline_roi=nothing)
    assert res.status == "refused"
    assert res.refusal_reason == "no-markers-outline"
    assert res.refusal == AP.ADD_S1_NO_MARKERS_OUTLINE_CLEARED
    assert res.elevation is not None and res.elevation.shape == SHAPE


@pytest.mark.parametrize("where", ["box", "whole"])
@pytest.mark.parametrize("case", ["none_here", "dark", "kept"])
def test_zero_count_means_the_engine_refuses(case, where):
    """A run count of no marker greys Preview before the press: that is only
    honest when the Preview would refuse, and the line says why.  Three
    worlds: the run has no object here; its one object lies deep in the
    dark with nothing kept (the dark tick on); every object of the run is
    kept already -- on a box and on a whole patch (filled to the walls)."""
    ch, run, _labels = _world(0)
    thr = _threshold(ch)
    labels = np.zeros_like(run)
    if case == "none_here":
        run = np.zeros_like(run)
    elif case == "dark":
        run = np.where(run == 900, run, 0)          # deep in the dark
    else:
        labels = run.copy()                          # every object kept
    whole = where == "whole"
    dark_on = case == "dark"
    roi = tuple(slice(0, s) for s in SHAPE)
    params = {"preset": "cells", "dark_outside": dark_on, "outline": "none",
              "growth": "flood", "whole_patch": whole}
    info = AP.add_box_info(
        existing_roi=labels, dark_sources=None, dark_rois=ch, spacing=ISO,
        stride=1, dark_cache=thr, source=AP.SOURCE_RUN, run_guide_roi=run,
        nuclei_roi=None, params=params, roi=roi)
    assert info.run_count is not None
    n_found, n_markers = info.run_count
    assert n_markers == 0
    assert (n_found == 0) is (case == "none_here")
    state = {"has_box": True, "preset": "cells", "box_shape": SHAPE,
             "wall_channel": "membrane", "source": AP.SOURCE_RUN,
             "run_id": "r", "has_other_run": True, "whole_patch": whole,
             "n_run_found": n_found, "n_run_markers": n_markers,
             "n_outside_outline": info.n_outside_outline,
             "outline": "none", "empty_share": info.empty_share,
             "dark_outside": dark_on}
    want = {
        ("none_here", "box"): AP.ADD_S1_RUN_NONE_HERE,
        ("none_here", "whole"): AP.ADD_S1_RUN_NONE_HERE_PATCH,
        ("dark", "box"): AP.ADD_S1_RUN_NO_MARKERS_EMPTY.format(
            dark=AP.ADD_DARK_OUTSIDE_LABEL),
        ("dark", "whole"): AP.ADD_S1_RUN_NO_MARKERS_EMPTY_PATCH.format(
            dark=AP.ADD_DARK_OUTSIDE_LABEL),
        ("kept", "box"): AP.ADD_S1_RUN_NO_MARKERS,
        ("kept", "whole"): AP.ADD_S1_RUN_NO_MARKERS_PATCH,
    }[(case, where)]
    assert AP.preview_block(state) == want
    res = AP.run_add_preview(
        preset="cells", roi=roi, existing_roi=labels.copy(),
        next_label=10000, spacing=ISO,
        params=dict(params, source=AP.SOURCE_RUN, auto_smooth=False,
                    min_size=0),
        wall_roi=ch["membrane"], dark_rois=ch, run_guide_roi=run,
        dark_threshold=thr)
    assert res.status == "refused"
    assert res.refusal_reason == "no-markers"
