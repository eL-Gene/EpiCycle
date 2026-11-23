"""The ``[ADD-PERF]`` laps of the Add candidate and the click balls.

One timing line per call, in the file's ``_StepTimer`` style
(``<tag> name=ms ... key=value total=ms``):

* ``[ADD-PERF] render`` -- one line per candidate draw, ``path=build`` for
  the first draw of a box (the old layer out, the zero array, the fill, the
  Labels build, the append, the side copies, the outline, the side colours,
  the property watch, the active layers) and ``path=diff`` for every later
  one (the old result zeroed, the new one written, the colours, the show,
  the render repair);
* ``[ADD-PERF] remove`` -- one line per candidate taken down (nothing when
  there is no candidate);
* ``[ADD-PERF] hide`` -- one line per invalidation of a drawn candidate;
* ``[ADD-PERF] ball`` -- one line per ball draw (one per click).

The laps only time: the candidate's data is the result, as before.
"""

from __future__ import annotations

import contextlib
import io

import numpy as np
import pytest

from .add_harness import install_fakes, marker_zyx
from .add_layer_rig import LayerRig, blocks, result_for
from .test_add_click_markers import h  # noqa: F401  (the ball fixture)

SHAPE = (12, 40, 40)
IDS = (100, 101, 102, 103)

RENDER_LAPS = ("remove", "zeros", "fill", "build", "append", "sides",
               "outline", "colours", "watch", "active")
DIFF_LAPS = ("zeros", "fill", "colours", "show", "heal")
HIDE_LAPS = ("zeros", "hide")
REMOVE_LAPS = ("unwatch", "unsync", "sides", "unregister", "layer", "gc")
BALL_LAPS = ("build", "erase", "draw", "views", "visible")


@pytest.fixture
def rig(monkeypatch):
    install_fakes(monkeypatch)
    vol = np.zeros(SHAPE, np.int32)
    hh = LayerRig(vol, ndisplay=3)
    hh.adv.preset = "cells"
    hh.place_box(tuple(slice(0, s) for s in SHAPE))
    hh._add_click_markers = []
    hh._add_markers_base = 500
    yield hh
    with contextlib.suppress(Exception), contextlib.redirect_stdout(
            io.StringIO()):
        hh._remove_add_candidate()


def _result():
    res = blocks(SHAPE, 4, first=IDS[0], block=8)
    return result_for(tuple(slice(0, s) for s in SHAPE), res)


def _lines(text, tag):
    return [ln for ln in text.splitlines()
            if ln.startswith("[ADD-PERF] {} ".format(tag))]


def _check_line(line, tag, laps, extras=()):
    tokens = line[len("[ADD-PERF] {} ".format(tag)):].split()
    keys = [t.split("=", 1)[0] for t in tokens]
    assert keys == list(laps) + list(extras) + ["total"], line
    for t in tokens[:len(laps)]:
        assert t.endswith("ms"), line
        float(t.split("=", 1)[1][:-2])
    assert tokens[-1].startswith("total=") and tokens[-1].endswith("ms")


def test_one_render_line_with_every_lap(rig, capsys):
    out = _result()
    capsys.readouterr()
    rig._apply_add_result(out)
    text = capsys.readouterr().out
    lines = _lines(text, "render")
    assert len(lines) == 1, text
    _check_line(lines[0], "render", RENDER_LAPS, extras=("path", "n"))
    assert lines[0].split()[-3] == "path=build"
    assert lines[0].split()[-2] == "n={}".format(len(IDS))
    # nothing to take down on the first build: no remove line
    assert _lines(text, "remove") == []
    # the laps only time: the candidate holds the result
    cand = rig._add_candidate_layer
    assert cand is not None
    np.testing.assert_array_equal(np.asarray(cand.data), out.result)


def test_a_second_render_takes_the_diff_path(rig, capsys):
    """The layer is made once per box: a later render writes into the same
    array -- no remove line, no new layer."""
    rig._apply_add_result(_result())
    first = rig._add_candidate_layer
    capsys.readouterr()
    rig._invalidate_add_preview()               # a click
    text = capsys.readouterr().out
    assert _lines(text, "remove") == [], text
    hides = _lines(text, "hide")
    assert len(hides) == 1, text
    _check_line(hides[0], "hide", HIDE_LAPS)
    out = _result()
    rig._apply_add_result(out)
    text = capsys.readouterr().out
    lines = _lines(text, "render")
    assert len(lines) == 1, text
    _check_line(lines[0], "render", DIFF_LAPS, extras=("path", "n"))
    assert lines[0].split()[-3] == "path=diff"
    assert _lines(text, "remove") == [], text
    cand = rig._add_candidate_layer
    assert cand is first and first in rig.viewer.layers
    np.testing.assert_array_equal(np.asarray(cand.data), out.result)


def test_close_box_prints_one_remove_line(rig, capsys):
    """The remove line now belongs to the exits: Close box takes the four
    layers down once, every lap named."""
    rig._apply_add_result(_result())
    rig._invalidate_add_preview()
    capsys.readouterr()
    rig._reject_add()
    removes = _lines(capsys.readouterr().out, "remove")
    assert len(removes) == 1, removes
    _check_line(removes[0], "remove", REMOVE_LAPS)
    assert rig._add_candidate_layer is None


def test_removing_nothing_prints_nothing(rig, capsys):
    capsys.readouterr()
    rig._remove_add_candidate()
    assert _lines(capsys.readouterr().out, "remove") == []


def test_one_ball_line_per_click(h, capsys):  # noqa: F811
    capsys.readouterr()
    assert h._add_marker_click((12, 20, 20))
    text = capsys.readouterr().out
    lines = _lines(text, "ball")
    assert len(lines) == 1, text
    _check_line(lines[0], "ball", BALL_LAPS, extras=("n", "full"))
    assert marker_zyx(h) == [(12, 20, 20)]
    # a second click: one more line, a partial draw
    assert h._add_marker_click((14, 10, 10))
    lines = _lines(capsys.readouterr().out, "ball")
    assert len(lines) == 1
    _check_line(lines[0], "ball", BALL_LAPS, extras=("n", "full"))
    assert lines[0].split()[-2] == "full=0"
