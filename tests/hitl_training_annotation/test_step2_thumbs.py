"""Step 2's patch thumbnails: requests,
folders, the two-folder sweep rule, the 128-vs-80 rule, the header axes
check.  The worker thread itself is in ``test_step2_thumbs_worker.py``."""
import os
import re

import numpy as np
import pytest
import tifffile

from napari_epicycle.hitl_training_annotation import _step2_thumbs as TH
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T


def _tif(path, shape, axes=None, dtype=np.uint16):
    data = (np.arange(int(np.prod(shape))) % 251).astype(dtype).reshape(shape)
    kw = {"metadata": {"axes": axes}} if axes else {}
    tifffile.imwrite(str(path), data, **kw)
    return path


class TestFolders:
    def test_two_folders_under_one_parent(self, tmp_path):
        ws = tmp_path
        parent = TH.step2_thumbs_dir(ws)
        assert parent == ws / "artifacts" / "step2" / "thumbs"
        assert TH.patch_thumbs_dir(ws) == parent / "patches"
        assert TH.slice_thumbs_dir(ws) == parent / "slices"
        assert TH.patch_thumbs_dir(ws) != TH.slice_thumbs_dir(ws)

    def test_naming_a_folder_creates_nothing(self, tmp_path):
        TH.patch_thumbs_dir(tmp_path)
        assert not (tmp_path / "artifacts").exists()


class TestRequests:
    def test_one_request_per_tiff_in_order_skipping_the_rest(self, tmp_path):
        d = tmp_path / "patches"
        d.mkdir()
        _tif(d / "b.tiff", (4, 8, 8), "ZYX")
        _tif(d / "a.TIF", (4, 8, 8), "ZYX")
        (d / "c.npy").write_bytes(b"x")
        (d / "sub.tif").mkdir()
        reqs = TH.patch_thumb_requests(
            str(tmp_path), str(d),
            ["b.tiff", "a.TIF", "c.npy", "missing.tif", "sub.tif", "b.tiff"])
        assert [r.name for r in reqs] == ["b.tiff", "a.TIF"]
        r = reqs[0]
        assert r.path == str(d / "b.tiff") and r.axes == ""
        st = os.stat(d / "b.tiff")
        assert (r.mtime_ns, r.st_size) == (st.st_mtime_ns, st.st_size)
        assert r.labels_path == "" and r.labels_mtime_ns == 0   # no labels

    def test_a_relative_folder_is_read_under_the_workspace(self, tmp_path):
        (tmp_path / "pp").mkdir()
        _tif(tmp_path / "pp" / "a.tif", (2, 4, 4), "ZYX")
        reqs = TH.patch_thumb_requests(str(tmp_path), "pp", ["a.tif"])
        assert [r.name for r in reqs] == ["a.tif"]

    def test_no_folder_is_no_request(self):
        assert TH.patch_thumb_requests("", "", ["a.tif"]) == []


class TestOneSize:
    def test_the_png_carries_the_128_token_and_never_80(self, tmp_path):
        from napari_epicycle.ui_shell import _stack_thumbs as S

        d = tmp_path / "p"
        d.mkdir()
        _tif(d / "a.tif", (2, 4, 4), "ZYX")
        req = TH.patch_thumb_requests(str(tmp_path), str(d), ["a.tif"])[0]
        png = TH.png_for(tmp_path, req, single=S.SINGLE_INFERNO)
        assert ".s128." in png.name and ".s80" not in png.name
        assert TH.PATCH_THUMB_PX == S.THUMB_WIDE_PX == 128

    def test_an_80_px_png_of_the_same_patch_would_be_stale_so_never_make_one(
            self, tmp_path):
        # why one size: the shared stale rule matches ANY size token
        from napari_epicycle.ui_shell import _stack_thumbs as S

        d = tmp_path / "p"
        d.mkdir()
        _tif(d / "a.tif", (2, 4, 4), "ZYX")
        req = TH.patch_thumb_requests(str(tmp_path), str(d), ["a.tif"])[0]
        cache = tmp_path / "cache"
        cache.mkdir()
        p128 = S.thumb_png_path(cache, req.name, req.path, req.mtime_ns,
                                req.st_size, single=S.SINGLE_INFERNO, size=128)
        p80 = S.thumb_png_path(cache, req.name, req.path, req.mtime_ns,
                               req.st_size, single=S.SINGLE_INFERNO, size=80)
        p80.write_bytes(b"x")
        assert p80 in S.stale_pngs(cache, req.name, req.path, p128)


class TestSweep:
    def _png(self, d, name):
        from napari_epicycle.ui_shell import _stack_thumbs as S

        d.mkdir(parents=True, exist_ok=True)
        p = S.thumb_png_path(d, name, f"/x/{name}", 1, 2,
                             single=S.SINGLE_INFERNO, size=128)
        p.write_bytes(b"png")
        return p

    def test_unlisted_patch_pngs_go_listed_ones_stay(self, tmp_path):
        keep = self._png(TH.patch_thumbs_dir(tmp_path), "keep.tif")
        gone = self._png(TH.patch_thumbs_dir(tmp_path), "gone.tif")
        assert TH.sweep_unlisted_patches(tmp_path, ["keep.tif"]) == 1
        assert keep.exists() and not gone.exists()

    def test_a_patch_sweep_never_touches_the_slices_folder_or_the_parent(
            self, tmp_path):
        slice_png = self._png(TH.slice_thumbs_dir(tmp_path), "WS_XY_31")
        parent_png = self._png(TH.step2_thumbs_dir(tmp_path), "stray")
        self._png(TH.patch_thumbs_dir(tmp_path), "gone.tif")
        TH.sweep_unlisted_patches(tmp_path, [])
        assert slice_png.exists() and parent_png.exists()

    def test_only_worker_shaped_plain_files_are_touched(self, tmp_path):
        d = TH.patch_thumbs_dir(tmp_path)
        d.mkdir(parents=True)
        (d / "notes.txt").write_text("x")
        (d / "a.tmp").write_text("x")
        (d / "sub").mkdir()
        assert TH.sweep_unlisted_patches(tmp_path, []) == 0
        assert (d / "notes.txt").exists() and (d / "sub").is_dir()

    def test_no_workspace_or_folder_is_zero(self, tmp_path):
        assert TH.sweep_unlisted_patches("", []) == 0
        assert TH.sweep_unlisted_patches(tmp_path, []) == 0

    def test_the_name_shape_is_a_copy_of_step_3s(self):
        from napari_epicycle.fusion_reconstruction import _step3_thumbs as S3

        assert TH._PNG_NAME.pattern == S3._PNG_NAME.pattern


class TestAxesCheck:
    @pytest.mark.parametrize("axes,shape", [
        ("ZYX", (3, 8, 8)), ("CZYX", (2, 3, 8, 8)), ("ZCYX", (3, 2, 8, 8))])
    def test_the_three_accepted_headers(self, axes, shape):
        assert TH.axes_reason(axes, shape) is None

    @pytest.mark.parametrize("axes,shape", [
        ("YX", (8, 8)), ("TZYX", (2, 3, 8, 8)), ("QYX", (3, 8, 8)),
        ("CYX", (2, 8, 8)), ("ZYXC", (3, 8, 8, 2)), ("ZYX", (3, 8, 8, 2)),
        ("", ()),
    ])
    def test_anything_else_has_a_reason(self, axes, shape):
        assert TH.axes_reason(axes, shape) == T.THUMB_FAIL_AXES

    def test_the_probe_reads_the_header_of_a_real_file(self, tmp_path):
        p = _tif(tmp_path / "a.tif", (2, 5, 8, 8), "CZYX")
        meta = TH.probe_patch_header(p)
        assert (meta.axes, meta.shape, meta.n_channels, meta.reason) == \
            ("CZYX", (2, 5, 8, 8), 2, "")
        p = _tif(tmp_path / "b.tif", (5, 2, 8, 8), "ZCYX")
        assert TH.probe_patch_header(p).n_channels == 2
        p = _tif(tmp_path / "c.tif", (5, 8, 8), "ZYX")
        m = TH.probe_patch_header(p)
        assert m.n_channels == 1 and m.reason == ""

    def test_a_flat_image_is_a_blank_tile_with_a_reason(self, tmp_path):
        p = _tif(tmp_path / "flat.tif", (8, 8), "YX")
        m = TH.probe_patch_header(p)
        assert m.reason == T.THUMB_FAIL_AXES and m.n_channels == 1

    def test_an_unreadable_file_is_a_reason_not_a_crash(self, tmp_path):
        p = tmp_path / "bad.tif"
        p.write_bytes(b"not a tiff")
        m = TH.probe_patch_header(p)
        assert m.reason == T.THUMB_REASONS["THUMB_FAIL_UNREADABLE"]

    def test_the_metadata_payload_round_trips(self):
        m = TH.PatchMeta("a.tif", "CZYX", (2, 3, 4, 5), 2, "")
        assert TH.PatchMeta.from_json(m.to_json()) == m


class TestTooltip:
    def test_the_shared_reasons_are_reworded_for_a_patch(self):
        from napari_epicycle.ui_shell import _stack_thumbs as S

        tip = TH.thumb_tooltip_for_failure(S.THUMB_FAIL_MISSING)
        assert tip == "No picture: The patch file is missing."
        assert "stack" not in TH.thumb_tooltip_for_failure(
            S.THUMB_FAIL_AXES).lower()

    def test_an_unknown_or_empty_reason(self):
        assert TH.thumb_tooltip_for_failure("odd") == "No picture: odd"
        assert TH.thumb_tooltip_for_failure(None).endswith(
            "could not be read.")

    def test_the_axes_sentence_names_the_three_accepted_headers(self):
        assert re.search(r"ZYX, CZYX or ZCYX", T.THUMB_FAIL_AXES)
