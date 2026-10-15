"""Brick A: the slice half of Step 2's thumbnail engine -- the request keys, the pair read rule (ONE short
``read_bytes()``, decode from bytes, a retry on ``PermissionError``), the
orientation, the label count, the in-flight set, the trailing debounce, the
slices sweep.  Qt-free except ``test_a_batch_on_a_real_thread`` (offscreen
Qt; run this file on its own: ``-o addopts=""``)."""
import os
import time
from pathlib import Path

import numpy as np
import pytest
import tifffile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from napari_epicycle.hitl_training_annotation import _step2_thumbs as TH
from napari_epicycle.ui_shell import _stack_thumbs as S


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _write_pair(d, sid="P_XY_3", *, img_shape=(2, 40, 60), ids=(1, 2, 3),
                mask_shape=None):
    """A pair like the controller writes: ``{sid}_img.tif`` float32
    (C, Y, X) and ``{sid}_mask.tif`` int32 (Y, X)."""
    d = Path(d)
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(1)
    img = rng.random(img_shape).astype(np.float32)
    ms = mask_shape or img_shape[-2:]
    mask = np.zeros(ms, np.int32)
    for k, i in enumerate(ids):
        mask[2 + 5 * k:6 + 5 * k, 3:12] = i
    ip, mp = d / f"{sid}_img.tif", d / f"{sid}_mask.tif"
    tifffile.imwrite(str(ip), img)
    tifffile.imwrite(str(mp), mask)
    return ip, mp


def _record(ip, mp, *, view="XY", index=3, status="curating", **kw):
    r = {"view": view, "slice_index": index, "status": status,
         "patch_id": "P", "img_uri": str(ip) if ip else "",
         "mask_uri": str(mp) if mp else ""}
    r.update(kw)
    return r


def _png_names(d):
    return sorted(p.name for p in Path(d).glob("*.png"))


def _png_size(data):
    from qtpy.QtGui import QImage

    img = QImage.fromData(bytes(data), "PNG")
    assert not img.isNull()
    return img.width(), img.height()


def _bump(path, ns=5_000_000_000):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + ns))


# ---------------------------------------------------------------------------
# keys
# ---------------------------------------------------------------------------

class TestKeys:
    def test_the_key_holds_both_mtimes_and_the_size(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc")
        req = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        assert req.img_path == str(ip) and req.labels_path == str(mp)
        assert req.mtime_ns == os.stat(ip).st_mtime_ns
        assert req.st_size == os.stat(ip).st_size
        assert req.labels_mtime_ns == os.stat(mp).st_mtime_ns
        name = TH.slice_png_path(tmp_path / "c", req).name
        assert f".{req.mtime_ns}.{req.st_size}." in name
        assert f".l{req.labels_mtime_ns}." in name
        assert ".s56." in name and name.startswith("P_XY_3.")

    def test_a_recurated_slice_is_a_new_png(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc")
        before = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        _bump(mp)
        after = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        assert before.labels_mtime_ns != after.labels_mtime_ns
        assert TH.slice_png_path(tmp_path, before) != \
            TH.slice_png_path(tmp_path, after)
        assert TH.cache_key_for(before) != TH.cache_key_for(after)

    def test_an_image_change_is_a_new_png(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc")
        a = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        _bump(ip)
        b = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        assert TH.slice_png_path(tmp_path, a) != TH.slice_png_path(tmp_path, b)

    def test_a_run_switch_is_a_new_png(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "run_a")
        _, mp2 = _write_pair(tmp_path / "run_b")
        one = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        # the same image, the mask of another run's folder
        two = TH.slice_thumb_request("P_XY_3", _record(ip, mp2))
        assert TH.slice_png_path(tmp_path, one) != \
            TH.slice_png_path(tmp_path, two)
        # a whole pair in another folder (accepted -> staged) too
        ip3, mp3 = _write_pair(tmp_path / "staged")
        three = TH.slice_thumb_request("P_XY_3", _record(ip3, mp3))
        assert TH.slice_png_path(tmp_path, one) != \
            TH.slice_png_path(tmp_path, three)

    def test_img_uri_is_preferred_over_the_stack_cut(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc")
        stack = tmp_path / "P.tif"
        tifffile.imwrite(str(stack), np.zeros((5, 20, 30), np.uint16))
        req = TH.slice_thumb_request("P_XY_3", _record(ip, mp),
                                     stack_paths={"P": str(stack)})
        assert req.img_path == str(ip) and req.stack_path == ""

    def test_the_stack_is_the_fallback_when_img_uri_is_none(self, tmp_path):
        _, mp = _write_pair(tmp_path / "acc")
        stack = tmp_path / "P.tif"
        tifffile.imwrite(str(stack), np.zeros((5, 20, 30), np.uint16))
        rec = _record(None, mp)
        rec["img_uri"] = None
        req = TH.slice_thumb_request("P_XY_3", rec,
                                     stack_paths={"P": str(stack)})
        assert req.img_path == "" and req.stack_path == str(stack)
        assert "::XY:3" in req.path
        other = TH.slice_thumb_request(
            "P_XY_4", dict(rec, slice_index=4),
            stack_paths={"P": str(stack)})
        assert other.path != req.path           # another cut, another key

    def test_nothing_to_draw_is_none(self, tmp_path):
        assert TH.slice_thumb_request("P_XY_3", _record(None, None)) is None
        assert TH.slice_thumb_request("nonsense", {}) is None
        assert TH.slice_thumb_request(
            "P_XY_3", _record(tmp_path / "gone.tif", None)) is None

    def test_a_missing_mask_is_no_labels_part(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc")
        os.unlink(mp)
        req = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        assert req.labels_path == "" and req.labels_mtime_ns == 0
        assert ".l" not in TH.slice_png_path(tmp_path, req).name.split(
            ".s56")[1]


# ---------------------------------------------------------------------------
# rendering: sources, the read rule, the file handles
# ---------------------------------------------------------------------------

class TestRender:
    def test_a_pair_draws_with_labels_and_counts_them(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc", ids=(5, 7, 1000))
        req = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        state, png, data, meta = TH.render_slice_thumb(req, tmp_path / "c")
        assert state == "ok" and Path(png).is_file() and data
        assert meta.n_labels == 3 and meta.has_mask
        assert meta.source == "pair" and meta.n_channels == 2
        assert meta.shape == (40, 60)
        assert TH.count_labels(np.array([[0, 0, 4], [4, 9, 0]])) == 2
        assert _png_size(data)[0] == 56

    def test_a_png_on_disk_is_a_hit_that_still_counts_labels(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc", ids=(1, 2))
        req = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        cache = tmp_path / "c"
        first = TH.render_slice_thumb(req, cache)
        reads = []
        orig = Path.read_bytes

        def spy(self):
            reads.append(self.name)
            return orig(self)

        Path.read_bytes = spy
        try:
            second = TH.render_slice_thumb(req, cache)
        finally:
            Path.read_bytes = orig
        assert second[0] == "ok" and second[2] == first[2]
        assert second[3].n_labels == 2
        # brick A integration: a hit still reports the plane's shape and
        # channels (the staged row's "240 x 240 . 2 ch"), by ONE short read
        # of the image file -- no re-draw, no re-encode
        assert second[3].shape == (40, 60) and second[3].n_channels == 2
        assert [n for n in reads if n.endswith("_img.tif")] == [ip.name]

    def test_a_recurated_mask_makes_a_new_png_and_drops_the_old(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc")
        cache = tmp_path / "c"
        r1 = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        p1 = TH.render_slice_thumb(r1, cache)[1]
        tifffile.imwrite(str(mp), np.full((40, 60), 9, np.int32))
        _bump(mp)
        r2 = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        st, p2, _, meta = TH.render_slice_thumb(r2, cache)
        assert st == "ok" and p2 != p1 and meta.n_labels == 1
        assert not Path(p1).exists() and Path(p2).is_file()
        assert len(_png_names(cache)) == 1

    def test_a_pair_that_moved_folder_leaves_one_png(self, tmp_path):
        cache = tmp_path / "c"
        ip, mp = _write_pair(tmp_path / "accepted")
        TH.render_slice_thumb(
            TH.slice_thumb_request("P_XY_3", _record(ip, mp)), cache)
        ip2, mp2 = _write_pair(tmp_path / "staged")
        TH.render_slice_thumb(
            TH.slice_thumb_request("P_XY_3", _record(ip2, mp2)), cache)
        assert len(_png_names(cache)) == 1

    def test_img_uri_not_the_stack_is_drawn_when_both_exist(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc", img_shape=(1, 40, 60))
        stack = tmp_path / "P.tif"
        tifffile.imwrite(str(stack), np.zeros((5, 20, 30), np.uint16))
        req = TH.slice_thumb_request("P_XY_3", _record(ip, mp),
                                     stack_paths={"P": str(stack)})
        meta = TH.render_slice_thumb(req, tmp_path / "c")[3]
        assert meta.source == "pair" and meta.shape == (40, 60)

    def test_the_stack_cut_is_drawn_when_img_uri_is_none(self, tmp_path):
        stack = tmp_path / "P.tif"
        tifffile.imwrite(str(stack),
                         (np.arange(5 * 20 * 30) % 200).astype(np.uint16)
                         .reshape(5, 20, 30), metadata={"axes": "ZYX"})
        rec = {"view": "XY", "slice_index": 2, "status": "predicted",
               "patch_id": "P", "img_uri": None}
        req = TH.slice_thumb_request("P_XY_2", rec,
                                     stack_paths={"P": str(stack)})
        state, png, data, meta = TH.render_slice_thumb(req, tmp_path / "c")
        assert state == "ok" and meta.source == "stack"
        assert meta.shape == (20, 30) and meta.n_labels == 0
        assert not meta.has_mask

    def test_a_predicted_slice_with_no_mask_is_the_raw_plane(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc")
        os.unlink(mp)
        before = sorted(p.name for p in (tmp_path / "acc").iterdir())
        reads = []
        orig = Path.read_bytes

        def spy(self):
            reads.append(self.name)
            return orig(self)

        req = TH.slice_thumb_request("P_XY_3",
                                     _record(ip, mp, status="predicted"))
        Path.read_bytes = spy
        try:
            state, png, data, meta = TH.render_slice_thumb(req, tmp_path / "c")
        finally:
            Path.read_bytes = orig
        assert state == "ok" and not meta.has_mask and meta.n_labels == 0
        assert reads == ["P_XY_3_img.tif"]          # no mask read
        assert sorted(p.name for p in (tmp_path / "acc").iterdir()) == before
        assert ".l" not in Path(png).name.split(".s56")[1]

    def test_each_file_is_read_once_by_read_bytes(self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc")
        reads = []
        orig = Path.read_bytes

        def spy(self):
            reads.append(self.name)
            return orig(self)

        req = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        Path.read_bytes = spy
        try:
            TH.render_slice_thumb(req, tmp_path / "c")
        finally:
            Path.read_bytes = orig
        assert sorted(reads) == ["P_XY_3_img.tif", "P_XY_3_mask.tif"]

    @pytest.mark.skipif(os.name != "nt", reason="Windows file sharing")
    def test_replace_and_unlink_succeed_while_a_render_decodes(self, tmp_path):
        """Spec section 6, ``test_step2_win_replace`` semantics: no handle is
        held while the worker decodes, so the controller's ``os.replace`` of
        the mask and Discard's ``unlink`` both succeed mid-render."""
        d = tmp_path / "acc"
        ip, mp = _write_pair(d)
        req = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        real = tifffile.imread
        seen = {"n": 0}

        def swap_then_read(arg, *a, **k):
            seen["n"] += 1
            for p in (mp, ip):
                if not p.exists():
                    continue
                tmp = p.with_name(p.name + ".new")
                tifffile.imwrite(str(tmp), np.asarray(real(str(p))))
                os.replace(str(tmp), str(p))          # the atomic write
            if seen["n"] == 2:
                os.unlink(str(mp))                    # Discard
            return real(arg, *a, **k)

        tifffile.imread = swap_then_read
        try:
            state = TH.render_slice_thumb(req, tmp_path / "c")[0]
        finally:
            tifffile.imread = real
        assert state == "ok" and seen["n"] == 2
        assert not mp.exists()

    def test_permission_error_retries_once_after_150_ms_then_skips(
            self, tmp_path, monkeypatch):
        ip, mp = _write_pair(tmp_path / "acc")
        cache = tmp_path / "c"
        cache.mkdir()
        old = cache / "P_XY_3.deadbeef.1.1.c2i.s56.png"
        old.write_bytes(b"old")
        req = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        sleeps, calls = [], []
        orig = Path.read_bytes

        def locked(self):
            if self.name.endswith("_mask.tif"):
                calls.append(self.name)
                raise PermissionError(13, "in use")
            return orig(self)

        monkeypatch.setattr(Path, "read_bytes", locked)
        monkeypatch.setattr(TH, "_sleep", sleeps.append)
        state, payload, data, meta = TH.render_slice_thumb(req, cache)
        assert state == "busy" and data == b"" and meta.busy
        assert calls == ["P_XY_3_mask.tif"] * 2        # one retry, no more
        assert sleeps == [0.15]
        assert old.read_bytes() == b"old"              # the old PNG is kept
        assert _png_names(cache) == [old.name]         # nothing new written

    def test_a_retry_that_succeeds_is_a_normal_render(self, tmp_path,
                                                      monkeypatch):
        ip, mp = _write_pair(tmp_path / "acc")
        req = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        sleeps, n = [], {"k": 0}
        orig = Path.read_bytes

        def flaky(self):
            if self.name.endswith("_mask.tif"):
                n["k"] += 1
                if n["k"] == 1:
                    raise PermissionError(13, "in use")
            return orig(self)

        monkeypatch.setattr(Path, "read_bytes", flaky)
        monkeypatch.setattr(TH, "_sleep", sleeps.append)
        state, _, _, meta = TH.render_slice_thumb(req, tmp_path / "c")
        assert state == "ok" and sleeps == [TH.RETRY_DELAY_S] and \
            meta.n_labels == 3

    def test_a_vanished_mask_is_the_raw_picture_named_without_labels(
            self, tmp_path):
        ip, mp = _write_pair(tmp_path / "acc")
        req = TH.slice_thumb_request("P_XY_3", _record(ip, mp))
        os.unlink(mp)                       # Discard between stat and read
        state, png, _, meta = TH.render_slice_thumb(req, tmp_path / "c")
        assert state == "ok" and not meta.has_mask
        assert ".l" not in Path(png).name.split(".s56")[1]

    def test_a_missing_stack_image_is_a_blank_tile_with_a_reason(
            self, tmp_path):
        req = TH.SliceThumbRequest("P_XY_3", "x", "", 1, 1, "", 0, "XY", 3,
                                   "predicted", str(tmp_path / "no.tif"), "")
        state, reason, data, meta = TH.render_slice_thumb(req, tmp_path / "c")
        assert state == "failed" and reason == S.THUMB_FAIL_MISSING
        assert data == b"" and meta is None


# ---------------------------------------------------------------------------
# orientation
# ---------------------------------------------------------------------------

def _asym_volume(path, shape=(4, 6, 8)):
    vol = (np.arange(int(np.prod(shape))) % 251).astype(np.uint16) \
        .reshape(shape)
    tifffile.imwrite(str(path), vol, metadata={"axes": "ZYX"})
    return vol


class TestOrientation:
    """The annotator's own rule: ``viewer.dims.order`` XZ = (1, 0, 2), YZ =
    (2, 0, 1) -> the displayed plane is ``np.transpose(vol, order)[idx]``
    (controller ``apply_active_view_to_viewer``): Z on the rows."""

    @pytest.mark.parametrize("view,order", [("XY", (0, 1, 2)),
                                            ("XZ", (1, 0, 2)),
                                            ("YZ", (2, 0, 1))])
    def test_the_stack_cut_is_what_napari_displays(self, tmp_path, view,
                                                  order):
        vol = _asym_volume(tmp_path / "v.tif")
        cut = TH.read_stack_cut(tmp_path / "v.tif", view, 2)
        shown = np.transpose(vol, order)[2]
        assert cut.shape == (1,) + shown.shape
        assert np.array_equal(cut[0], shown)

    def test_the_rule_is_the_writers_slice(self, tmp_path):
        vol = _asym_volume(tmp_path / "v.tif")
        assert np.array_equal(TH.read_stack_cut(tmp_path / "v.tif", "XZ",
                                                1)[0], vol[:, 1, :])
        assert np.array_equal(TH.read_stack_cut(tmp_path / "v.tif", "YZ",
                                                5)[0], vol[:, :, 5])

    def test_a_pair_plane_is_drawn_as_stored(self):
        plane = np.arange(12).reshape(3, 4)
        assert TH.orient_plane(plane, "XZ") is plane
        assert TH.orient_plane(plane, "YZ") is plane

    def test_an_axial_tile_is_letterboxed_not_cropped(self, tmp_path):
        # a Z x X strip: 30 deep, 120 wide
        ip, mp = _write_pair(tmp_path / "acc", sid="P_XZ_3",
                             img_shape=(1, 30, 120))
        req = TH.slice_thumb_request("P_XZ_3",
                                     _record(ip, mp, view="XZ"))
        _, _, data, meta = TH.render_slice_thumb(req, tmp_path / "c")
        w, h = _png_size(data)
        assert w == 56 and h < w and meta.shape == (30, 120)
        assert abs(h / w - 30 / 120) < 0.12

    def test_a_cut_outside_the_stack_fails_with_a_reason(self, tmp_path):
        _asym_volume(tmp_path / "v.tif")
        with pytest.raises(IndexError):
            TH.read_stack_cut(tmp_path / "v.tif", "XY", 99)

    def test_odd_axes_are_refused(self, tmp_path):
        tifffile.imwrite(str(tmp_path / "t.tif"),
                         np.zeros((3, 4, 5, 6), np.uint16),
                         metadata={"axes": "TZYX"})
        with pytest.raises(ValueError):
            TH.read_stack_cut(tmp_path / "t.tif", "XY", 0)


# ---------------------------------------------------------------------------
# the in-flight set, the debounce, the sweeps
# ---------------------------------------------------------------------------

class _FakeSignal:
    def __init__(self):
        self.fns = []

    def connect(self, fn):
        self.fns.append(fn)


class _FakeTimer:
    def __init__(self):
        self.timeout = _FakeSignal()
        self.interval = None
        self.running = False
        self.starts = 0

    def setInterval(self, ms):
        self.interval = ms

    def start(self):
        self.running = True
        self.starts += 1

    def stop(self):
        self.running = False

    def fire(self):
        self.running = False
        for fn in self.timeout.fns:
            fn()


class TestInFlight:
    def test_a_duplicate_is_dropped(self):
        f = TH.InFlight()
        assert f.add("a.png") is True
        assert f.add("a.png") is False
        assert "a.png" in f and len(f) == 1
        f.discard("a.png")
        assert f.add("a.png") is True
        f.clear()
        assert len(f) == 0


class TestDebounce:
    def test_a_burst_of_writes_is_one_request(self):
        got, timers = [], []

        def factory():
            t = _FakeTimer()
            timers.append(t)
            return t

        class Owner:
            def on_fire(self, keys):
                got.append(keys)

        o = Owner()
        d = TH.TrailingDebounce(o.on_fire, timer_factory=factory)
        for sid in ("P_XY_3", "P_XY_3", "P_XY_4", "P_XY_3"):
            d.poke(sid)
        assert got == [] and timers[0].interval == 1500
        assert timers[0].starts == 4 and timers[0].running
        timers[0].fire()
        assert got == [{"P_XY_3", "P_XY_4"}]
        assert d.pending == set()
        timers[0].fire()                    # nothing new: no second call
        assert len(got) == 1

    def test_cancel_drops_the_burst(self):
        got = []
        t = _FakeTimer()

        class Owner:
            def on_fire(self, keys):
                got.append(keys)

        o = Owner()
        d = TH.TrailingDebounce(o.on_fire, timer_factory=lambda: t)
        d.poke("a")
        d.cancel()
        t.fire()
        assert got == [] and not t.running

    def test_the_wait_is_1500_ms(self):
        assert TH.SLICE_DEBOUNCE_MS == 1500
        assert TH.RETRY_DELAY_S == 0.15


def _png(d, name, key=(1, 2)):
    p = Path(d) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    return p


class TestSweeps:
    def _ws(self, tmp_path):
        ws = tmp_path / "ws"
        patches, slices = TH.patch_thumbs_dir(ws), TH.slice_thumbs_dir(ws)
        pp = _png(patches, "P.0badf00d.1.2.c2i.s128.png")
        a = _png(slices, "P_XY_3.0badf00d.1.2.c2i.s56.l9.ofo45.png")
        b = _png(slices, "Q_XZ_7.0badf00d.1.2.c2i.s56.png")
        return ws, pp, a, b

    def test_the_slices_sweep_never_touches_the_patches_folder(self, tmp_path):
        ws, pp, a, b = self._ws(tmp_path)
        n = TH.sweep_unlisted_slices(ws, ["P_XY_3"])
        assert n == 1 and a.exists() and not b.exists()
        assert pp.exists()
        assert TH.sweep_unlisted_slices(ws, []) == 1 and not a.exists()
        assert pp.exists()

    def test_the_patch_sweep_never_touches_the_slices_folder(self, tmp_path):
        ws, pp, a, b = self._ws(tmp_path)
        assert TH.sweep_unlisted_patches(ws, []) == 1
        assert not pp.exists() and a.exists() and b.exists()

    def test_forget_removes_the_samples_of_that_patch_only(self, tmp_path):
        ws = tmp_path / "ws"
        d = TH.slice_thumbs_dir(ws)
        keep1 = _png(d, "a_b_XY_1.0badf00d.1.2.c2i.s56.png")
        gone1 = _png(d, "a_XY_1.0badf00d.1.2.c2i.s56.png")
        gone2 = _png(d, "a_YZ_20.0badf00d.1.2.c2i.s56.png")
        pp = _png(TH.patch_thumbs_dir(ws), "a.0badf00d.1.2.c2i.s128.png")
        assert TH.remove_patch_slice_pngs(ws, "a") == 2
        assert keep1.exists() and not gone1.exists() and not gone2.exists()
        assert pp.exists()

    def test_the_sweeps_do_nothing_without_a_workspace_or_folder(
            self, tmp_path):
        assert TH.sweep_unlisted_slices("", []) == 0
        assert TH.sweep_unlisted_slices(tmp_path / "none", []) == 0
        assert TH.remove_patch_slice_pngs(tmp_path / "none", "a") == 0

    def test_a_name_that_is_not_a_worker_png_is_left_alone(self, tmp_path):
        ws = tmp_path / "ws"
        keep = _png(TH.slice_thumbs_dir(ws), "notes.txt")
        assert TH.sweep_unlisted_slices(ws, []) == 0 and keep.exists()


# ---------------------------------------------------------------------------
# the meta payload and the worker on a real thread
# ---------------------------------------------------------------------------

class TestMeta:
    def test_json_round_trip(self):
        m = TH.SliceMeta("P_XY_3", 57, 2, (240, 240), True, "pair", False)
        assert TH.SliceMeta.from_json(m.to_json()) == m
        b = TH.SliceMeta("x", -1, 1, (), True, "stack", True)
        assert TH.SliceMeta.from_json(b.to_json()).busy is True


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


class _Sink:
    def __init__(self):
        self.ready, self.failed, self.meta = {}, {}, {}
        self.finished = False

    def on_ready(self, name, path, png, data):
        self.ready[name] = (path, png, bytes(data))

    def on_failed(self, name, path, reason):
        self.failed[name] = reason

    def on_meta(self, name, path, text):
        self.meta[name] = TH.SliceMeta.from_json(text)

    def on_finished(self):
        self.finished = True


def _wait(app, cond, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end and not cond():
        app.processEvents()
        time.sleep(0.01)
    return cond()


def test_a_batch_on_a_real_thread(qapp, tmp_path):
    ip, mp = _write_pair(tmp_path / "acc", sid="P_XY_3", ids=(1, 2))
    ip2, mp2 = _write_pair(tmp_path / "acc", sid="P_XY_4", ids=(1, 2, 3, 4))
    reqs = [TH.slice_thumb_request("P_XY_3", _record(ip, mp)),
            TH.slice_thumb_request("P_XY_4", _record(ip2, mp2, index=4))]
    sink = _Sink()
    cache = tmp_path / "slices"
    state = TH.start_slice_thumbs(reqs, cache, sink.on_ready, sink.on_failed,
                                  sink.on_meta, on_finished=sink.on_finished)
    try:
        assert _wait(qapp, lambda: sink.finished)
        assert set(sink.ready) == {"P_XY_3", "P_XY_4"} and not sink.failed
        assert sink.meta["P_XY_3"].n_labels == 2
        assert sink.meta["P_XY_4"].n_labels == 4
        path, png, data = sink.ready["P_XY_3"]
        assert path == reqs[0].path and Path(png).is_file()
        assert len(_png_names(cache)) == 2
    finally:
        assert TH.stop_thumbs(state)


def test_submit_adds_to_a_running_batch_or_starts_one(qapp, tmp_path):
    sink = _Sink()
    cache = tmp_path / "slices"
    state = TH.Step2ThumbState()
    reqs = []
    for i in (1, 2, 3):
        ip, mp = _write_pair(tmp_path / "acc", sid=f"P_XY_{i}")
        reqs.append(TH.slice_thumb_request(f"P_XY_{i}",
                                           _record(ip, mp, index=i)))
    try:
        assert TH.submit_slice_thumbs(reqs[:1], cache, sink.on_ready,
                                      sink.on_failed, sink.on_meta,
                                      state=state,
                                      on_finished=sink.on_finished) == 1
        assert TH.submit_slice_thumbs(reqs[1:], cache, sink.on_ready,
                                      sink.on_failed, sink.on_meta,
                                      state=state,
                                      on_finished=sink.on_finished) == 2
        assert _wait(qapp, lambda: len(sink.ready) == 3)
    finally:
        TH.stop_thumbs(state)
    assert TH.submit_slice_thumbs([], cache, sink.on_ready, sink.on_failed,
                                  sink.on_meta, state=state) == 0
