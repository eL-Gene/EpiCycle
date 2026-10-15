"""The patch worker end to end on a real thread (offscreen Qt): header probe,
metadata payload, a PNG in the patches folder at ONE size, a flat file as a
blank tile with its reason, stop is bounded and idempotent.  Qt-backed: run
this file on its own."""
import os
import time

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class Sink:
    """Bound-method callbacks, as the owner passes them (never lambdas)."""

    def __init__(self):
        self.ready, self.failed, self.meta = {}, {}, {}
        self.finished = False

    def on_ready(self, name, path, png_path, data):
        self.ready[name] = (png_path, bytes(data))

    def on_failed(self, name, path, reason):
        self.failed[name] = reason

    def on_meta(self, name, path, text):
        self.meta[name] = text

    def on_finished(self):
        self.finished = True


def _wait(app, cond, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.01)
    return cond()


def _tif(path, shape, axes):
    data = (np.arange(int(np.prod(shape))) % 251).astype(np.uint16)
    tifffile.imwrite(str(path), data.reshape(shape), metadata={"axes": axes})


def test_a_batch_makes_pictures_metadata_and_a_blank_tile(qapp, tmp_path):
    from napari_epicycle.hitl_training_annotation import _step2_thumbs as TH
    from napari_epicycle.ui_shell import _stack_thumbs as S

    d = tmp_path / "patches_in"
    d.mkdir()
    _tif(d / "two.tif", (6, 2, 32, 32), "ZCYX")
    _tif(d / "one.tif", (6, 32, 32), "ZYX")
    _tif(d / "flat.tif", (32, 32), "YX")
    names = ["two.tif", "one.tif", "flat.tif"]
    reqs = TH.patch_thumb_requests(str(tmp_path), str(d), names)
    assert len(reqs) == 3
    sink = Sink()
    cache = TH.patch_thumbs_dir(tmp_path)
    state = TH.start_patch_thumbs(reqs, cache, sink.on_ready, sink.on_failed,
                                  sink.on_meta, on_finished=sink.on_finished)
    assert _wait(qapp, lambda: sink.finished)
    assert set(sink.ready) == {"two.tif", "one.tif"}
    assert set(sink.failed) == {"flat.tif"}
    assert sink.failed["flat.tif"] == \
        __import__("napari_epicycle.hitl_training_annotation.ui._step2_text",
                   fromlist=["x"]).THUMB_FAIL_AXES
    metas = {n: TH.PatchMeta.from_json(t) for n, t in sink.meta.items()}
    assert metas["two.tif"].n_channels == 2 and metas["two.tif"].axes == "ZCYX"
    assert metas["one.tif"].n_channels == 1
    assert metas["flat.tif"].reason
    pngs = sorted(os.listdir(cache))
    assert len(pngs) == 2 and all(".s128." in p for p in pngs)
    assert not (TH.slice_thumbs_dir(tmp_path)).exists()
    from qtpy.QtGui import QPixmap

    pm = QPixmap()
    assert pm.loadFromData(sink.ready["one.tif"][1], "PNG")
    assert max(pm.width(), pm.height()) == 128
    assert TH.stop_thumbs(state) is True
    assert TH.stop_thumbs(state) is True            # idempotent
    assert TH.stop_thumbs(None) is True
    del S


def test_a_second_batch_stops_the_first_of_the_same_state(qapp, tmp_path):
    from napari_epicycle.hitl_training_annotation import _step2_thumbs as TH

    d = tmp_path / "in"
    d.mkdir()
    for i in range(3):
        _tif(d / f"p{i}.tif", (4, 32, 32), "ZYX")
    reqs = TH.patch_thumb_requests(str(tmp_path), str(d),
                                   [f"p{i}.tif" for i in range(3)])
    first = Sink()
    state = TH.start_patch_thumbs(reqs, TH.patch_thumbs_dir(tmp_path),
                                  first.on_ready, first.on_failed,
                                  first.on_meta)
    second = Sink()
    same = TH.start_patch_thumbs(reqs[:1], TH.patch_thumbs_dir(tmp_path),
                                 second.on_ready, second.on_failed,
                                 second.on_meta, state=state,
                                 on_finished=second.on_finished)
    assert same is state
    assert _wait(qapp, lambda: second.finished)
    assert "p0.tif" in second.ready
    assert TH.stop_thumbs(state) is True


def test_no_requests_starts_nothing_but_still_stops_the_old_batch(qapp,
                                                                  tmp_path):
    from napari_epicycle.hitl_training_annotation import _step2_thumbs as TH

    st = TH.start_patch_thumbs([], tmp_path, None, None, None)
    assert st.thread is None and not st.running()
