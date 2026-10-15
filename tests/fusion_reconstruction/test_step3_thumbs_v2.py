"""Step 3 thumbnail engine v2:
inferno for one channel, label outlines on reconstructed patches, the
request builder's ``labels_for``.  Step 4's grey and key stay as they were.
Real worker on a real ``QThread`` under a plain offscreen ``QApplication``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("qtpy")
tifffile = pytest.importorskip("tifffile")

from napari_epicycle.fusion_reconstruction import _step3_thumbs as S3  # noqa: E402
from napari_epicycle.ui_shell import _stack_thumbs as T  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _ramp(h=48, w=48):
    return np.tile(np.linspace(0, 1000, w, dtype=np.float32), (h, 1))


def _png_rgb(data: bytes) -> np.ndarray:
    from qtpy.QtGui import QImage

    img = QImage.fromData(data, "PNG").convertToFormat(QImage.Format_RGB888)
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()
    arr = np.frombuffer(img.constBits().asstring(bpl * h), np.uint8)
    return arr.reshape(h, bpl)[:, : w * 3].reshape(h, w, 3).copy()


def _patch(tmp_path, name="p.tif", z=6):
    vol = np.stack([_ramp() for _ in range(z)]).astype(np.uint16)
    path = tmp_path / name
    tifffile.imwrite(str(path), vol, metadata={"axes": "ZYX"})
    return path


def _labels(tmp_path, name="labels_3d.tif", z=6):
    lab = np.zeros((z, 48, 48), np.uint16)
    lab[:, 10:30, 10:30] = 1
    path = tmp_path / name
    tifffile.imwrite(str(path), lab)
    return path


def _req(path, labels=None):
    st = os.stat(path)
    kw = {}
    if labels is not None:
        kw = dict(labels_path=str(labels),
                  labels_mtime_ns=int(os.stat(labels).st_mtime_ns))
    return T.ThumbRequest("p.tif", str(path), "ZYX", int(st.st_mtime_ns),
                          int(st.st_size), **kw)


def _overlay_diff(rgb, plain):
    """Pixels the labels overlay changed (A3: per-instance colours, no
    single outline colour any more)."""
    return int((rgb.astype(int) != plain.astype(int)).any(-1).sum())


# -- compose_rgb ------------------------------------------------------------

def test_the_lut_is_inferno_256():
    lut = np.asarray(T.THUMB_INFERNO)
    assert lut.shape == (256, 3)
    assert tuple(lut[0]) == (0, 0, 4)
    assert lut[255][0] > 240                  # inferno ends pale yellow


def test_single_none_is_grey_and_inferno_is_not():
    plane = _ramp()
    grey = T.compose_rgb(plane[None])
    assert (grey[..., 0] == grey[..., 1]).all()
    assert (grey[..., 1] == grey[..., 2]).all()
    inf = T.compose_rgb(plane[None], single="inferno")
    assert not (inf[..., 0] == inf[..., 2]).all()
    lut = np.asarray(T.THUMB_INFERNO, np.uint8)
    assert (inf == lut[T.thumb_norm8(plane)]).all()


def test_inferno_does_not_touch_two_channels():
    planes = np.stack([_ramp(), _ramp()[:, ::-1]])
    assert (T.compose_rgb(planes)
            == T.compose_rgb(planes, single="inferno")).all()


def test_an_unknown_rule_raises():
    with pytest.raises(ValueError):
        T.compose_rgb(_ramp()[None], single="viridis")


# -- the PNG key ------------------------------------------------------------

def test_step3_and_step4_pngs_of_one_file_do_not_collide(tmp_path):
    a = T.thumb_png_path(tmp_path, "p.tif", "x/p.tif", 5, 6)
    b = T.thumb_png_path(tmp_path, "p.tif", "x/p.tif", 5, 6, single="inferno")
    c = T.thumb_png_path(tmp_path, "p.tif", "x/p.tif", 5, 6, single="inferno",
                         labels_mtime_ns=99)
    assert len({a.name, b.name, c.name}) == 3
    assert a.name.endswith(".c2.s128.png")       # Step 4's name + the A3 size


def test_stale_pngs_recognise_the_new_shapes(tmp_path):
    keep = T.thumb_png_path(tmp_path, "p.tif", "x/p.tif", 5, 6,
                            single="inferno", labels_mtime_ns=2)
    old = T.thumb_png_path(tmp_path, "p.tif", "x/p.tif", 5, 6,
                           single="inferno", labels_mtime_ns=1)
    for p in (keep, old):
        p.write_bytes(b"x")
    got = T.stale_pngs(tmp_path, "p.tif", "x/p.tif", keep)
    assert [p.name for p in got] == [old.name]


# -- render_thumb -----------------------------------------------------------

def test_render_is_inferno_for_one_channel_and_grey_by_default(qapp, tmp_path):
    p = _patch(tmp_path)
    ok, _png, data = T.render_thumb(_req(p), tmp_path / "c", single="inferno")
    assert ok
    rgb = _png_rgb(data)
    assert not (rgb[..., 0] == rgb[..., 2]).all()
    ok4, _p4, data4 = T.render_thumb(_req(p), tmp_path / "c4")
    g = _png_rgb(data4)
    assert ok4 and (g[..., 0] == g[..., 2]).all()          # Step 4: grey


def test_labels_overlay_is_painted_over_the_picture(qapp, tmp_path):
    p, lab = _patch(tmp_path), _labels(tmp_path)
    ok, png, data = T.render_thumb(_req(p, lab), tmp_path / "c",
                                   single="inferno")
    assert ok and ".l" in os.path.basename(png)
    _ok, _p2, data2 = T.render_thumb(_req(p), tmp_path / "c2",
                                     single="inferno")
    # the inferno ramp has a few near-orange pixels of its own: compare
    assert _overlay_diff(_png_rgb(data), _png_rgb(data2)) > 8


def test_labels_read_only_the_mid_plane(tmp_path):
    lab = np.zeros((7, 20, 20), np.uint16)
    lab[3] = 5
    path = tmp_path / "l.tif"
    tifffile.imwrite(str(path), lab)
    assert (T.read_labels_midz(str(path)) == 5).all()


def test_labels_from_a_zarr_store(tmp_path):
    zarr = pytest.importorskip("zarr")
    lab = np.zeros((7, 20, 20), np.uint16)
    lab[3] = 7
    store = tmp_path / "labels_3d.zarr"
    zarr.save(str(store), lab)
    assert (T.read_labels_midz(str(store)) == 7).all()


def test_unreadable_labels_give_the_raw_picture(qapp, tmp_path, capsys):
    p = _patch(tmp_path)
    bad = tmp_path / "labels_3d.tif"
    bad.write_bytes(b"not a tiff")
    ok, png, data = T.render_thumb(_req(p, bad), tmp_path / "c",
                                   single="inferno")
    assert ok and data
    assert "labels unreadable" in capsys.readouterr().out
    plain = _png_rgb(T.render_thumb(_req(p), tmp_path / 'pl', single='inferno')[2])
    assert _overlay_diff(_png_rgb(data), plain) == 0
    assert ".l" not in Path(png).name


def test_labels_mtime_is_part_of_the_key(qapp, tmp_path):
    p, lab = _patch(tmp_path), _labels(tmp_path)
    r1 = _req(p, lab)
    r2 = r1._replace(labels_mtime_ns=r1.labels_mtime_ns + 5)
    _ok, a, _d = T.render_thumb(r1, tmp_path / "c", single="inferno")
    _ok, b, _d = T.render_thumb(r2, tmp_path / "c", single="inferno")
    assert a != b and not os.path.exists(a) and os.path.exists(b)


# -- request builder --------------------------------------------------------

def test_requests_carry_labels_from_labels_for(tmp_path):
    _patch(tmp_path)
    lab = _labels(tmp_path)
    got = S3.patch_thumb_requests(tmp_path, tmp_path, ["p.tif"],
                                  lambda n: "ZYX",
                                  labels_for=lambda n: str(lab))
    assert got[0].labels_path == str(lab)
    assert got[0].labels_mtime_ns == os.stat(lab).st_mtime_ns


def test_labels_for_none_missing_or_raising_gives_no_outline(tmp_path):
    _patch(tmp_path)
    for fn in (None, lambda n: None, lambda n: str(tmp_path / "gone.tif"),
               lambda n: 1 / 0):
        got = S3.patch_thumb_requests(tmp_path, tmp_path, ["p.tif"],
                                      lambda n: "ZYX", labels_for=fn)
        assert got[0].labels_path == "" and got[0].labels_mtime_ns == 0


# -- the real worker ---------------------------------------------------------

def test_the_real_worker_draws_inferno_and_outline_on_a_real_qthread(
        qapp, tmp_path):
    p, lab = _patch(tmp_path), _labels(tmp_path)
    got, fin = [], []

    class Sink:
        def ready(self, name, path, png, data):
            got.append((name, data))

        def failed(self, *a):
            got.append(("failed", a))

        def finished(self):
            fin.append(1)

    sink = Sink()
    state = S3.start_step3_thumbs(None, [_req(p, lab)], tmp_path / "cache",
                                  sink.ready, sink.failed,
                                  on_finished=sink.finished)
    thread = state.thread
    t0 = time.monotonic()
    while time.monotonic() - t0 < 20 and not (fin and not state.running()):
        qapp.processEvents()
        time.sleep(0.01)
    assert got and got[0][0] == "p.tif", got
    rgb = _png_rgb(got[0][1])
    assert not (rgb[..., 0] == rgb[..., 2]).all()          # inferno
    plain = _png_rgb(T.render_thumb(_req(p), tmp_path / 'pl',
                                    single='inferno')[2])
    assert _overlay_diff(rgb, plain) > 8
    assert thread.quit_called
    S3.stop_step3_thumbs(state)


def test_a_cache_hit_also_sweeps_the_stale_outlined_png(qapp, tmp_path):
    """Audit LOW-2: after Reset the raw PNG is on disk; an older outlined
    one beside it must go when the raw one is a cache hit."""
    p, lab = _patch(tmp_path), _labels(tmp_path)
    cache = tmp_path / "c"
    _ok, outlined, _d = T.render_thumb(_req(p, lab), cache, single="inferno")
    assert os.path.exists(outlined)
    # the raw picture is made (and the outlined one is swept by its write)...
    _ok, raw, _d = T.render_thumb(_req(p), cache, single="inferno")
    assert os.path.exists(raw) and not os.path.exists(outlined)
    # ...bring the old outlined one back: a later HIT on the raw one sweeps it
    Path(outlined).write_bytes(b"x")
    ok, again, _d = T.render_thumb(_req(p), cache, single="inferno")
    assert ok and again == raw
    assert not os.path.exists(outlined) and os.path.exists(raw)
