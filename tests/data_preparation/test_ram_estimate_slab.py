"""Brick 4b item 5 — an HONEST peak-RAM model for the convert.

`estimate_peak_ram_bytes` modelled `workers x chunk**3 x itemsize`.  But
`multiscale.convert_to_ome_zarr`'s own docstring says the COMPRESSED-TIFF path
(which is what the real wound data is) cannot read a cubic chunk at a time: it
opens the TIFF as thin-Z pages and must buffer a whole `chunk_z x Y x X` SLAB
per worker to assemble one chunk row.  On a whole-tissue XY field that is orders
of magnitude larger than the chunk term, so the old model under-reported and the
worker clamp never bound where it mattered.
"""
from __future__ import annotations

import pytest

from packages.epicycle_largedata.config import (
    DEFAULT_CHUNK_ZYX,
    estimate_peak_ram_bytes,
    recommend_convert_settings,
)

_GB = 1 << 30
_MB = 1 << 20


# --------------------------------------------------------------------------- #
# the model
# --------------------------------------------------------------------------- #

def test_without_a_shape_it_is_still_the_chunk_bound():
    """No shape known -> the memmap path's bound (one chunk per worker), which
    is what the old model computed and what the dialog shows before a file is
    picked."""
    got = estimate_peak_ram_bytes(4, (32, 128, 128), 2)
    assert got == 4 * 32 * 128 * 128 * 2


# --------------------------------------------------------------------------- #
# fix pass item 5 — the REGIME, not the presence of a shape, selects the model
# --------------------------------------------------------------------------- #

def test_an_uncompressed_source_uses_the_chunk_bound_even_with_a_shape():
    """An UNCOMPRESSED TIFF is memmapped, and _LazyReadProxy reads it one chunk
    at a time -- its real peak is ~one chunk per worker no matter how big the XY
    field is.  Inferring the slab model from `shape is not None` charged it the
    compressed path's cost and clamped a 32 GB box to a single worker for a
    source that would happily run on all of them."""
    shape = (500, 8192, 8192)
    got = estimate_peak_ram_bytes(4, (32, 128, 128), 2, shape=shape,
                                  regime="memmap")
    assert got == 4 * 32 * 128 * 128 * 2


def test_a_compressed_source_uses_the_slab():
    shape = (500, 8192, 8192)
    got = estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape=shape,
                                  regime="compressed")
    assert got >= 32 * 8192 * 8192 * 2


def test_an_unknown_regime_stays_conservative():
    """Unknown regime + a known shape -> assume the compressed (worse) case, so
    we under-promise rather than OOM."""
    shape = (500, 8192, 8192)
    a = estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape=shape)
    b = estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape=shape,
                                regime="compressed")
    assert a == b


def test_regime_is_resolved_from_a_real_uncompressed_tiff(tmp_path):
    from packages.epicycle_largedata.config import resolve_source_regime
    import numpy as _np
    import tifffile

    p = tmp_path / "plain.tif"
    tifffile.imwrite(str(p), _np.zeros((4, 32, 32), dtype=_np.uint16))
    assert resolve_source_regime(p) == "memmap"


def test_regime_is_resolved_from_a_real_compressed_tiff(tmp_path):
    from packages.epicycle_largedata.config import resolve_source_regime
    import numpy as _np
    import tifffile

    p = tmp_path / "deflate.tif"
    tifffile.imwrite(str(p), _np.zeros((4, 32, 32), dtype=_np.uint16),
                     compression="deflate")
    assert resolve_source_regime(p) == "compressed"


def test_regime_resolution_never_raises():
    from packages.epicycle_largedata.config import resolve_source_regime

    assert resolve_source_regime(None) is None
    assert resolve_source_regime("/no/such/file.tif") is None


def test_the_uncompressed_clamp_does_not_bind_on_a_32gb_box():
    """The regression this item fixes."""
    r = recommend_convert_settings(
        "fast", total_ram_bytes=64 * _GB, avail_ram_bytes=32 * _GB,
        cpu_count=16, shape=(500, 8192, 8192), regime="memmap")
    assert r["num_workers"] == 16


def test_with_a_shape_it_models_the_z_slab():
    """The compressed path's real buffer: chunk_z x Y x X per worker."""
    shape = (143, 1024, 1024)
    got = estimate_peak_ram_bytes(2, (32, 128, 128), 2, shape=shape)
    slab = 32 * 1024 * 1024 * 2
    assert got >= 2 * slab


def test_the_slab_model_is_much_bigger_than_the_chunk_model():
    shape = (143, 1024, 1024)
    chunk_only = estimate_peak_ram_bytes(1, (32, 128, 128), 2)
    slab = estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape=shape)
    assert slab > 8 * chunk_only          # 64 MB vs 1 MB on this field


def test_a_czyx_shape_uses_only_the_zyx_axes():
    a = estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape=(3, 143, 1024, 1024))
    b = estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape=(143, 1024, 1024))
    assert a == b


def test_the_slab_scales_with_the_xy_field_not_with_depth():
    """The documented property: the buffer is flat in TOTAL Z (it does not grow
    with depth) but grows with the XY field."""
    thin = estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape=(100, 2048, 2048))
    deep = estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape=(9000, 2048, 2048))
    assert thin == deep
    wide = estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape=(100, 8192, 8192))
    assert wide > 8 * thin


def test_a_thinner_z_chunk_costs_proportionally_less():
    shape = (143, 4096, 4096)
    a = estimate_peak_ram_bytes(1, (16, 128, 128), 2, shape=shape)
    b = estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape=shape)
    assert b > a


def test_degenerate_inputs_never_yield_zero():
    assert estimate_peak_ram_bytes(0, 0, 0) > 0
    assert estimate_peak_ram_bytes(0, 0, 0, shape=(0, 0, 0)) > 0
    assert estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape="nonsense") > 0
    assert estimate_peak_ram_bytes(1, (32, 128, 128), 2, shape=(5,)) > 0


# --------------------------------------------------------------------------- #
# the clamp: it must actually BIND on a small box + a big field
# --------------------------------------------------------------------------- #

def _rec(preset, cpu, avail, shape=None, itemsize=2, regime="compressed"):
    return recommend_convert_settings(
        preset, total_ram_bytes=64 * _GB, avail_ram_bytes=avail,
        cpu_count=cpu, dtype_itemsize=itemsize, shape=shape, regime=regime)


def test_clamp_binds_on_an_8gb_laptop_with_a_whole_tissue_field():
    """8 GB free, a 8192x8192 XY field: one worker's slab is 32*8192*8192*2 =
    4 GiB, well over the 25% budget, so `fast` must fall to a single worker
    instead of confidently launching 16 and OOMing."""
    r = _rec("fast", cpu=16, avail=8 * _GB, shape=(500, 8192, 8192))
    assert r["num_workers"] == 1


def test_clamp_does_not_bind_on_a_400gb_box():
    r = _rec("fast", cpu=16, avail=400 * _GB, shape=(500, 8192, 8192))
    assert r["num_workers"] == 16


def test_clamp_is_gentler_on_a_modest_field():
    """The real wound stack (1024x1024) on the same 8 GB laptop: a slab is
    64 MiB, so several workers still fit -- the clamp must not be a blunt
    always-1."""
    r = _rec("fast", cpu=8, avail=8 * _GB, shape=(143, 1024, 1024))
    assert r["num_workers"] == 8


def test_without_a_shape_the_clamp_behaves_as_before():
    r = _rec("fast", cpu=16, avail=8 * _GB)
    assert r["num_workers"] == 16


def test_clamp_never_goes_below_one():
    r = _rec("fast", cpu=16, avail=1, shape=(500, 8192, 8192))
    assert r["num_workers"] == 1


def test_unknown_ram_still_skips_the_clamp():
    r = _rec("fast", cpu=8, avail=0, shape=(500, 8192, 8192))
    assert r["num_workers"] == 8


def test_itemsize_still_tightens_the_clamp():
    a = _rec("fast", cpu=32, avail=32 * _GB, shape=(500, 4096, 4096), itemsize=2)
    b = _rec("fast", cpu=32, avail=32 * _GB, shape=(500, 4096, 4096), itemsize=4)
    assert b["num_workers"] < a["num_workers"]


# --------------------------------------------------------------------------- #
# the dialog readout follows the model
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def _dlg(shape=None, **kw):
    regime = kw.pop("regime", "compressed")
    from napari_epicycle.data_preparation._largedata_config_dialog import (
        LargeDataConfigDialog)

    return LargeDataConfigDialog(
        None, cpu_count=kw.pop("cpu", 8),
        total_ram_bytes=kw.pop("total", 32 * _GB),
        avail_ram_bytes=kw.pop("avail", 16 * _GB),
        source_shape=shape, source_regime=regime, **kw)


def test_dialog_readout_uses_the_slab_when_a_shape_is_known(qt_app):
    d = _dlg(shape=(143, 1024, 1024))
    try:
        expect = estimate_peak_ram_bytes(
            d.workers_spin.value(), DEFAULT_CHUNK_ZYX, 2,
            shape=(143, 1024, 1024), regime="compressed") / _MB
        assert f"~{expect:.0f} MB" in d.estimate_label.text()
    finally:
        d.setParent(None)


def test_dialog_still_works_without_a_shape(qt_app):
    d = _dlg()
    try:
        assert "Estimated peak memory" in d.estimate_label.text()
    finally:
        d.setParent(None)


def test_dialog_reads_in_GB_once_the_number_is_big(qt_app):
    """A five-digit MB figure is unreadable; switch units past ~1 GB."""
    d = _dlg(shape=(500, 8192, 8192), cpu=4, avail=400 * _GB)
    try:
        txt = d.estimate_label.text()
        assert " GB " in txt and "MB" not in txt
    finally:
        d.setParent(None)


def test_dialog_explains_a_clamped_core_count_in_plain_words(qt_app):
    """A silently reduced worker count reads as the app being slow for no
    reason; say why."""
    d = _dlg(shape=(500, 8192, 8192), cpu=16, avail=8 * _GB)
    try:
        d.preset_combo.setCurrentText("Fast")
        txt = d.estimate_label.text()
        assert "Using 1 of your 16 cores" in txt
        assert "more memory than you have free" in txt
    finally:
        d.setParent(None)


def test_dialog_says_nothing_about_cores_when_none_were_held_back(qt_app):
    d = _dlg(shape=(143, 1024, 1024), cpu=8, avail=400 * _GB)
    try:
        d.preset_combo.setCurrentText("Fast")
        assert "cores" not in d.estimate_label.text()
    finally:
        d.setParent(None)


def test_dialog_copy_is_plain_language(qt_app):
    """Biologist-facing wording: no jargon in anything the user reads."""
    d = _dlg(shape=(143, 1024, 1024))
    try:
        readout = d.estimate_label.text()
        assert "slab" not in readout.lower()
        assert "block covering the whole image width and height" in readout
        tip = d.chunk_combo.toolTip()
        for jargon in ("chunk", "voxel", "Z x Y x X", "decompress"):
            assert jargon.lower() not in tip.lower(), jargon
        assert "depth x height x width" in tip
        labels = [d.chunk_combo.itemText(i) for i in range(d.chunk_combo.count())]
        assert any("32 deep x 128 x 128" == t for t in labels)
        assert any("Older layout: 64 in every direction" == t for t in labels)
        assert not any("cubic" in t.lower() for t in labels)
        assert ("they all write the file the same way"
                in d.preset_combo.toolTip().lower())
    finally:
        d.setParent(None)
