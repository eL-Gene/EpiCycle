"""The prediction window's reads and pictures (``_prediction_preview``,
Qt-free) and its loader on a real ``QThread``.

* What the model saw is Cellpose's own rule, rebuilt from the recorded
  codes: code 0 = the mean of every channel (float32), code k = channel k-1,
  a helper h > 0 = channel h-1, a one-channel image = that channel whatever
  the codes -- checked against ``cellpose.transforms.reshape`` on the same
  slice, for ``[0,0]`` / ``[0,2]`` / ``[1,0]`` / ``[2,1]`` on ZCYX, CZYX,
  ZYX and ZCYX with one channel.
* Each map is read with ONE header read and ONE full read of its own file
  (never mapped, never the image again); the size limit comes first.
* The probability goes through the fixed -6 ... 6; the flows' colours are
  ``cellpose.plot.dx_to_circ``'s, the angle taken per drawn plane; the plane
  follows the pass, YZ turned to the annotator's YZ panel; a plane is
  thinned to about twice the pane before it is coloured.
* The loader answers on a real thread through real queued signals.

Run one file at a time with ``-o addopts=""``.
"""
from __future__ import annotations

import ast
import json
import os
from pathlib import Path

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction import _prediction_preview as PP
from napari_epicycle.fusion_reconstruction import _step3_pr_text as T
from tests.fusion_reconstruction._one_channel import ZYX_SHAPE, write_zyx_patch

_REPO = Path(__file__).resolve().parents[2]
_FR = _REPO / "napari_epicycle" / "fusion_reconstruction"


def _channels(n=3, z=4, y=8, x=8):
    """``n`` channels that cannot be told apart by brightness alone."""
    rng = np.random.default_rng(7)
    return [rng.integers(0, 1000 * (k + 1), size=(z, y, x)).astype(np.uint16)
            for k in range(n)]


def _image(chs, axes):
    if axes == "ZYX":
        return chs[0]
    stack = np.stack(chs, axis=0)                       # C, Z, Y, X
    return stack if axes == "CZYX" else stack.transpose(1, 0, 2, 3)


def _u8(vol):
    return PP._u8(np.asarray(vol))


# ------------------------------------------------------------ the rule

CASES = [(0, 0), (0, 2), (1, 0), (2, 1)]


@pytest.mark.parametrize("axes", ["ZCYX", "CZYX"])
@pytest.mark.parametrize("codes", CASES)
def test_the_model_input_follows_the_codes_on_two_channels(axes, codes):
    chs = _channels(2)
    out = PP.model_input(_image(chs, axes), axes, list(codes))
    c1, c2 = codes
    if c1 == 0:
        assert out["mode"] == "avg" and out["helper"] is None
        mean = np.mean(np.stack(chs, 1).astype(np.float32), axis=1,
                       dtype=np.float32)
        assert np.array_equal(out["seg"], _u8(mean))
    else:
        assert np.array_equal(out["seg"], _u8(chs[c1 - 1]))
        if c2:
            assert out["mode"] == "pair"
            assert np.array_equal(out["helper"], _u8(chs[c2 - 1]))
        else:
            assert out["mode"] == "seg" and out["helper"] is None
    assert out["shape"] == (4, 8, 8)


@pytest.mark.parametrize("codes", CASES)
@pytest.mark.parametrize("layout", ["ZYX", "ZCYX_C1"])
def test_a_one_channel_image_is_that_channel_in_grey_for_every_code(
        layout, codes):
    ch = _channels(1)[0]
    image = ch if layout == "ZYX" else ch[:, np.newaxis]
    out = PP.model_input(image, "ZCYX", list(codes))
    assert out["mode"] == "one" and out["helper"] is None
    assert np.array_equal(out["seg"], _u8(ch))
    assert PP.left_caption(out) == T.PRED_LEFT_ONE_CHANNEL


def test_a_patch_recorded_as_one_channel_reads_grey():
    chs = _channels(2)
    out = PP.model_input(_image(chs, "ZCYX"), "ZCYX", [2, 1],
                         one_channel=True)
    assert out["mode"] == "one"
    assert PP.left_caption(out) == T.PRED_LEFT_ONE_CHANNEL


@pytest.mark.parametrize("codes", CASES)
def test_the_rule_is_cellposes_reshape(codes):
    """The same slice through ``cellpose.transforms.reshape``: its first
    channel is the channel to segment, its second the helper (or zeros)."""
    tr = pytest.importorskip("cellpose.transforms")
    chs = _channels(3)
    zcyx = np.stack(chs, axis=1)
    out = PP.model_input(zcyx, "ZCYX", list(codes))
    z = 2
    data = tr.reshape(zcyx[z].astype(np.float32), channels=list(codes))
    seg = data[..., 0]
    if codes[0] == 0:
        mine = np.mean(zcyx[z], axis=0, dtype=np.float32)
    else:
        mine = zcyx[z, codes[0] - 1].astype(np.float32)
    assert np.allclose(seg, mine, rtol=1e-6)
    if codes[0] and codes[1]:
        assert np.array_equal(data[..., 1], zcyx[z, codes[1] - 1])
        assert out["mode"] == "pair"
    else:
        assert not np.any(data[..., 1])
        assert out["helper"] is None


def test_a_code_past_the_channels_is_an_error():
    with pytest.raises(ValueError):
        PP.model_input(_image(_channels(2), "ZCYX"), "ZCYX", [3, 0])


def test_no_record_reads_the_first_channel_and_says_it_is_a_guess():
    chs = _channels(2)
    out = PP.model_input(_image(chs, "ZCYX"), "ZCYX", None)
    assert out["mode"] == "seg" and out["guessed"]
    assert np.array_equal(out["seg"], _u8(chs[0]))
    assert PP.left_caption(out) == "no record — shown with 1: red"
    one = PP.model_input(_channels(1)[0], "ZCYX", None)
    assert PP.left_caption(one) == T.PRED_LEFT_ONE_CHANNEL
    known = PP.model_input(_image(chs, "ZCYX"), "ZCYX", [1, 0])
    assert not known["guessed"] and PP.left_caption(known) == "1: red"


def test_the_captions_speak_the_channel_boxes_words():
    base = {"seg": None, "helper": None, "shape": (1, 1, 1)}
    assert PP.left_caption(dict(base, mode="seg", codes=(1, 0))) == "1: red"
    assert PP.left_caption(dict(base, mode="pair", codes=(2, 1))) == \
        "2: green (segment) + 1: red (helper)"
    assert PP.left_caption(dict(base, mode="avg", codes=(0, 2))) == \
        T.PRED_LEFT_GRAY_AVG
    assert PP.left_caption(dict(base, mode="one", codes=(2, 1))) == \
        T.PRED_LEFT_ONE_CHANNEL


# ------------------------------------------------------------ the lines

def _manifest(codes, **extra):
    from napari_epicycle.fusion_reconstruction import _model_provenance as MP
    m = {"seg_config": {"channels": list(codes)}, "axes": "ZCYX",
         "predicted_at": "2026-10-26T21:34:00+00:00",
         MP.PROVENANCE_KEY: MP.provenance_block(
             "nuclei", "nuclei", now="2026-10-26T21:00:00+00:00")}
    m.update(extra)
    return m


def test_the_two_lines_name_the_codes_as_the_boxes_do():
    lines = PP.window_lines(_manifest([1, 0]), "nuclei-fixed")
    assert lines["line1"].startswith("nuclei-fixed · nuclei (built-in) · "
                                     "predicted 2026-10-2")
    assert lines["line2"] == "channel to segment 1: red · helper 0: none"
    lines = PP.window_lines(_manifest([0, 1]), "nuclei")
    assert lines["line2"] == ("channel to segment 0: gray · helper 1: red"
                              + T.PRED_HELPER_UNUSED)
    lines = PP.window_lines(_manifest([2, 1], one_channel=True), "r")
    assert lines["line2"] == T.PRED_LINE2_ONE_CHANNEL


def test_a_one_channel_record_says_what_ran_not_what_was_asked():
    """A one-channel patch ran as 0: gray whatever the run asked for: its
    second line says so, without the requested codes."""
    lines = PP.window_lines(_manifest([2, 1], one_channel=True), "Default")
    assert lines["line2"] == "one-channel image · ran as 0: gray"
    assert "2: green" not in lines["line2"]


def test_a_blank_box_one_channel_record_keeps_the_model_and_the_time():
    """The record a blank "Channel to segment" leaves on a one-channel
    patch (``channels: null`` + ``one_channel: true``) is a record."""
    m = _manifest([0, 0], one_channel=True)
    m["seg_config"]["channels"] = None
    lines = PP.window_lines(m, "nuclei")
    assert lines["line1"].startswith("nuclei · nuclei (built-in) · "
                                     "predicted 2026-10-2")
    assert lines["line2"] == T.PRED_LINE2_ONE_CHANNEL


def test_no_manifest_says_so():
    lines = PP.window_lines(None, "default")
    assert lines == {"line1": T.PRED_LINE_NO_RECORD.format(run="default"),
                     "line2": ""}


def test_an_absent_one_channel_key_reads_multi_channel():
    assert PP.recorded(_manifest([2, 1]))["one_channel"] is False
    assert PP.recorded(_manifest([2, 1], one_channel=True))["one_channel"]
    assert PP.recorded(None) == {"codes": None, "axes": None,
                                 "one_channel": False}


# ------------------------------------------------------------ the maps

def _cache(tmp_path, stem="p", shape=(4, 8, 8), seed=3):
    rng = np.random.default_rng(seed)
    cache = tmp_path / "cache"
    cache.mkdir(exist_ok=True)
    for v in PP.PASSES:
        np.save(cache / f"{stem}_{v}_probs.npy",
                rng.normal(0, 4, size=shape).astype(np.float16))
        np.save(cache / f"{stem}_{v}_flows.npy",
                rng.normal(0, 1, size=(2,) + shape).astype(np.float16))
    return cache


def test_the_header_gives_the_shape_and_the_limit_refuses(tmp_path):
    cache = _cache(tmp_path)
    assert PP.read_header(cache / "p_xy_probs.npy") == (4, 8, 8)
    assert PP.read_header(cache / "p_xz_flows.npy") == (2, 4, 8, 8)
    big = tmp_path / "big_xy_probs.npy"
    with open(big, "wb") as fh:
        np.lib.format.write_array_header_1_0(fh, {
            "descr": "<f2", "fortran_order": False,
            "shape": (1000, 1000, 1000)})
    with pytest.raises(PP.PreviewTooLarge) as err:
        PP.load_map(tmp_path, "big", "xy", "probs")
    assert err.value.shape == (1000, 1000, 1000)
    assert PP.too_large_sentence(err.value.shape) == (
        "This patch is too large to show its prediction here: "
        "1000 × 1000 × 1000 voxels, and the window shows up to 200 million.")


def test_the_probability_goes_through_minus_six_to_six(tmp_path):
    cache = tmp_path
    np.save(cache / "q_xy_probs.npy",
            np.array([[[-9.0, -6.0, 0.0, 6.0, 12.0]]], np.float16))
    m = PP.load_map(cache, "q", "xy", "probs")
    assert m["prob"].tolist() == [[[0, 0, 128, 255, 255]]]
    assert m["prob"].dtype == np.uint8 and m["shape"] == (1, 1, 5)
    assert PP.PRED_PROB_LIMITS == (-6.0, 6.0)


@pytest.mark.parametrize("pass_", PP.PASSES)
def test_the_flows_are_cellposes_colours(tmp_path, pass_):
    """One plane per pass: ``render_flow`` on the stored components with
    the length's range from ``load_map`` against ``dx_to_circ`` on the
    same two components (one plane, so the ranges are the same)."""
    cplot = pytest.importorskip("cellpose.plot")
    rng = np.random.default_rng(11)
    comps = rng.normal(0, 1, size=(2, 1, 24, 24)).astype(np.float16)
    np.save(tmp_path / f"f_{pass_}_flows.npy", comps)
    m = PP.load_map(tmp_path, "f", pass_, "flows")
    assert m["comps"].dtype == np.float16           # as stored, no angle
    assert "angle" not in m
    got = PP.render_flow(m["comps"][0, 0], m["comps"][1, 0], m["lo"],
                         m["hi"])
    want = cplot.dx_to_circ(comps[:, 0].astype(np.float32))
    diff = np.abs(got.astype(int) - want.astype(int))
    assert diff.max() <= 1, diff.max()
    assert got.flags["C_CONTIGUOUS"] and got.dtype == np.uint8


def test_the_yz_flows_turn_with_the_plane():
    """YZ is drawn Y down, Z across (the annotator's YZ panel), so its
    stored (dz, dy) become (dy, dz) of the turned plane."""
    comps = np.zeros((2, 4, 8, 6), np.float32)
    comps[0] = 1.0                                   # dz
    comps[1] = 2.0                                   # dy
    down, across = PP.flow_plane(comps, "yz", 3)
    assert down.shape == (8, 4) and np.all(down == 2.0)
    assert np.all(across == 1.0)
    down, across = PP.flow_plane(comps, "xz", 3)
    assert down.shape == (4, 6) and np.all(down == 1.0)


@pytest.mark.parametrize("pass_,want", [("xy", (8, 6)), ("xz", (4, 6)),
                                        ("yz", (8, 4))])
def test_the_plane_follows_the_pass(pass_, want):
    vol = np.arange(4 * 8 * 6).reshape(4, 8, 6)
    assert PP.plane(vol, pass_, 1).shape == want
    assert PP.plane_count(vol.shape, pass_) == {"xy": 4, "xz": 8,
                                                "yz": 6}[pass_]
    assert PP.SLIDER_AXIS == {"xy": "z", "xz": "y", "yz": "x"}
    if pass_ == "xz":
        assert np.array_equal(PP.plane(vol, pass_, 1), vol[:, 1, :])
    if pass_ == "yz":           # turned: Y down, Z across
        assert np.array_equal(PP.plane(vol, pass_, 1), vol[:, :, 1].T)


def test_a_plane_is_thinned_to_about_twice_the_pane():
    assert PP.thin_step((2048, 2048), 496) == 5
    assert PP.thin_step((81, 256), 496) == 1
    big = np.zeros((2048, 1024), np.uint8)
    assert PP.thin(big, PP.thin_step(big.shape, 496)).shape == (410, 205)


def test_a_choice_opens_only_its_own_file(tmp_path, monkeypatch):
    cache = _cache(tmp_path)
    opened = []
    real_open, real_load = open, np.load

    def spy_open(path, *a, **k):
        opened.append(Path(str(path)).name)
        return real_open(path, *a, **k)

    def spy_load(path, *a, **k):
        assert "mmap_mode" not in k
        opened.append(Path(str(path)).name)
        return real_load(path, *a, **k)

    import builtins
    import tifffile
    monkeypatch.setattr(builtins, "open", spy_open)
    monkeypatch.setattr(PP.np, "load", spy_load)
    monkeypatch.setattr(tifffile, "imread",
                        lambda *a, **k: pytest.fail("the image read again"))
    PP.load_map(cache, "p", "xz", "flows")
    assert set(opened) == {"p_xz_flows.npy"}
    opened.clear()
    PP.load_map(cache, "p", "yz", "probs")
    assert set(opened) == {"p_yz_probs.npy"}


def test_no_array_file_is_ever_mapped():
    for mod in ("_prediction_preview.py", "_prediction_preview_dialog.py"):
        assert "mmap_mode" not in (_FR / mod).read_text(encoding="utf-8")


def test_the_open_read_on_a_one_channel_patch(tmp_path):
    cache = _cache(tmp_path, stem="one_ch", shape=ZYX_SHAPE)
    stack = write_zyx_patch(tmp_path / "patches")
    man = cache / "one_ch_prediction_manifest.json"
    man.write_text(json.dumps(_manifest([2, 1], axes="ZYX")),
                   encoding="utf-8")
    out = PP.load_open(cache, "one_ch", stack, man, "default")
    assert out["image"]["mode"] == "one"
    assert out["caption"] == T.PRED_LEFT_ONE_CHANNEL
    assert out["map"]["pass"] == "xy" and out["map"]["kind"] == "probs"
    assert out["shape"] == ZYX_SHAPE and not out["shape_changed"]
    assert out["lines"]["line2"] == "channel to segment 2: green · helper 1: red"


def test_a_patch_that_lost_a_channel_keeps_the_map_only(tmp_path):
    cache = _cache(tmp_path, stem="one_ch", shape=ZYX_SHAPE)
    stack = write_zyx_patch(tmp_path / "patches")
    man = cache / "one_ch_prediction_manifest.json"
    # recorded on a third channel the patch on disk does not have
    import tifffile
    tifffile.imwrite(str(stack), np.zeros((4, 2, 8, 8), np.uint16),
                     metadata={"axes": "ZCYX"})
    man.write_text(json.dumps(_manifest([3, 0], axes="ZCYX")),
                   encoding="utf-8")
    out = PP.load_open(cache, "one_ch", stack, man, "default")
    assert out["image"] is None and out["shape_changed"]
    assert out["map"] is not None


def test_a_patch_trimmed_since_its_predict_keeps_the_map_only(tmp_path):
    cache = _cache(tmp_path, stem="one_ch", shape=(4, 8, 6))
    stack = write_zyx_patch(tmp_path / "patches")
    out = PP.load_open(cache, "one_ch", stack, cache / "none.json", "r")
    assert out["image"] is None and out["shape_changed"]
    assert out["map"]["shape"] == (4, 8, 6)


def test_a_4d_patch_with_no_record_is_resolved_as_the_predict_does(tmp_path):
    import tifffile
    chs = _channels(2)
    path = tmp_path / "c.tif"
    tifffile.imwrite(str(path), _image(chs, "CZYX"))
    seen = []

    def resolve(image, image_path=None, fallback=None):
        seen.append((image.shape, fallback))
        return "CZYX"

    out = PP.load_image(path, None, None, resolve=resolve)
    assert seen == [((2, 4, 8, 8), "CZYX")]
    assert np.array_equal(out["seg"], _u8(chs[0]))


def test_the_pictures_are_contiguous_rgb():
    seg = np.arange(15, dtype=np.uint8).reshape(3, 5)
    grey = PP.render_image(seg)
    assert grey.shape == (3, 5, 3) and grey.flags["C_CONTIGUOUS"]
    pair = PP.render_image(seg, seg[::-1].copy())
    assert np.array_equal(pair[..., 1], seg)          # segment green
    assert np.array_equal(pair[..., 0], seg[::-1])    # helper red
    lut = np.stack([np.arange(256)] * 3, -1).astype(np.uint8)
    assert PP.render_prob(seg, lut).flags["C_CONTIGUOUS"]


def test_the_reads_module_imports_no_qt():
    tree = ast.parse((_FR / "_prediction_preview.py").read_text("utf-8"))
    names = {a.name for n in ast.walk(tree)
             if isinstance(n, (ast.Import, ast.ImportFrom))
             for a in n.names}
    mods = {n.module or "" for n in ast.walk(tree)
            if isinstance(n, ast.ImportFrom)}
    assert not any("qt" in m.lower() for m in mods | names)


# ------------------------------------------------------------ a real thread

def test_the_loader_answers_on_a_real_thread(tmp_path):
    """A real ``QThread`` and real queued signals: the answer reaches a
    plain object's bound method on the main thread, the thread quits on
    both exits (``finished`` fires)."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    pytest.importorskip("qtpy")
    from qtpy.QtCore import QThread
    from qtpy.QtWidgets import QApplication

    from napari_epicycle.fusion_reconstruction._prediction_preview_dialog \
        import PredictionPreviewWorker
    app = QApplication.instance() or QApplication([])
    cache = _cache(tmp_path)

    class Sink:
        __slots__ = ("got", "ended", "thread_ids", "__weakref__")

        def __init__(self):
            self.got, self.ended, self.thread_ids = [], 0, []

        def loaded(self, result):
            self.got.append(("loaded", result))
            self.thread_ids.append(int(QThread.currentThreadId()))

        def failed(self, error):
            self.got.append(("failed", error))

        def finished(self):
            self.ended += 1

    main_id = int(QThread.currentThreadId())
    for args, want in (((cache, "p", "xz", "probs"), "loaded"),
                       ((cache, "zz", "xy", "probs"), "failed")):
        sink = Sink()
        worker = PredictionPreviewWorker(PP.load_map, args)
        thread = QThread()
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.loaded.connect(sink.loaded)
        worker.failed.connect(sink.failed)
        worker.loaded.connect(thread.quit)
        worker.failed.connect(thread.quit)
        thread.finished.connect(sink.finished)
        thread.start()
        import time
        t0 = time.monotonic()
        while (not sink.ended) and time.monotonic() - t0 < 10:
            app.processEvents()
            time.sleep(0.005)
        assert thread.wait(2000)
        assert sink.ended == 1, sink.got
        assert [k for k, _v in sink.got] == [want]
        if want == "loaded":
            assert sink.got[0][1]["shape"] == (4, 8, 8)
            assert sink.thread_ids == [main_id]       # delivered queued
        else:
            assert sink.got[0][1][0] == "error"
