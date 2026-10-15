"""Brick 4b item 1 — ANISOTROPIC store chunks ``(1, cz, cy, cx)`` / ``(cz, cy, cx)``.

The claim: a prepared store's chunk shape is a per-axis TUPLE (default
``(32, 128, 128)`` — thin in Z so a 2D slice decompresses little unread depth,
big in-plane so it touches few chunks), while every OLD cubic store (64 / 128)
keeps opening + streaming unchanged.
"""
from __future__ import annotations

import numpy as np
import pytest

from packages.epicycle_largedata.config import (
    ALLOWED_CHUNKS,
    DEFAULT_CHUNK_ZYX,
    normalize_convert_config,
    recommend_convert_settings,
)
from packages.epicycle_largedata.multiscale import (
    _clamp_chunks_to_shape,
    _cubic_chunks,
    convert_to_ome_zarr,
    open_multiscale,
)

_MACHINE = {"total_ram_bytes": 64 << 30, "avail_ram_bytes": 32 << 30,
            "cpu_count": 8}


# --------------------------------------------------------------------------- #
# _cubic_chunks: int (legacy, cubic) OR 3-tuple (new, anisotropic)
# --------------------------------------------------------------------------- #

def test_cubic_chunks_still_accepts_an_int():
    assert _cubic_chunks("ZYX", 64) == (64, 64, 64)
    assert _cubic_chunks("CZYX", 64) == (1, 64, 64, 64)


def test_cubic_chunks_accepts_a_zyx_tuple():
    assert _cubic_chunks("ZYX", (32, 128, 128)) == (32, 128, 128)
    assert _cubic_chunks("CZYX", (32, 128, 128)) == (1, 32, 128, 128)


def test_cubic_chunks_rejects_a_bad_tuple():
    with pytest.raises(ValueError):
        _cubic_chunks("ZYX", (32, 128))
    with pytest.raises(ValueError):
        _cubic_chunks("ZYX", (32, 0, 128))


# --------------------------------------------------------------------------- #
# per-level clamp: a chunk must never exceed the level shape
# --------------------------------------------------------------------------- #

def test_clamp_chunks_to_shape_clips_each_axis():
    assert _clamp_chunks_to_shape((32, 128, 128), (10, 300, 64)) == (10, 128, 64)


def test_clamp_chunks_to_shape_is_a_noop_when_the_level_is_big():
    assert _clamp_chunks_to_shape((32, 128, 128), (500, 500, 500)) == (32, 128, 128)


def test_clamp_chunks_to_shape_handles_the_channel_axis():
    assert _clamp_chunks_to_shape((1, 32, 128, 128), (3, 10, 300, 64)) == (
        1, 10, 128, 64)


def test_clamp_chunks_never_returns_zero():
    assert _clamp_chunks_to_shape((32, 128, 128), (0, 0, 0)) == (1, 1, 1)


# --------------------------------------------------------------------------- #
# config: chunk_zyx is the new field; chunk (int) stays for old configs
# --------------------------------------------------------------------------- #

def test_default_chunk_zyx_is_thin_z_big_in_plane():
    assert DEFAULT_CHUNK_ZYX == (32, 128, 128)


def test_allowed_chunks_no_longer_offers_256():
    assert 256 not in ALLOWED_CHUNKS


@pytest.mark.parametrize("preset", ["safe", "balanced", "fast"])
def test_every_preset_uses_the_anisotropic_shape(preset):
    cfg = recommend_convert_settings(preset, **_MACHINE)
    assert tuple(cfg["chunk_zyx"]) == (32, 128, 128)


def test_presets_differ_only_in_workers():
    safe = recommend_convert_settings("safe", **_MACHINE)
    fast = recommend_convert_settings("fast", **_MACHINE)
    assert safe["num_workers"] < fast["num_workers"]
    for k in ("chunk_zyx", "downsample", "min_size"):
        assert tuple(np.atleast_1d(safe[k])) == tuple(np.atleast_1d(fast[k]))


def test_a_persisted_int_chunk_still_means_cubic():
    # `custom` = the user typed these numbers, so the cubic derivation stands.
    # (A NAMED preset instead adopts the current default -- see
    # test_a_persisted_named_preset_without_chunk_zyx_gets_the_new_default.)
    old = {"preset": "custom", "num_workers": 2, "chunk": 64,
           "downsample": 2, "min_size": 256}
    out = normalize_convert_config(old)
    assert out is not None
    assert out["chunk"] == 64
    assert tuple(out["chunk_zyx"]) == (64, 64, 64)


def test_a_persisted_chunk_zyx_round_trips():
    cfg = {"preset": "balanced", "num_workers": 2, "chunk": 128,
           "downsample": 2, "min_size": 256, "chunk_zyx": [32, 128, 128]}
    out = normalize_convert_config(cfg)
    assert out is not None
    assert tuple(out["chunk_zyx"]) == (32, 128, 128)


def test_a_corrupt_chunk_zyx_falls_back_to_the_cubic_chunk():
    cfg = {"preset": "custom", "num_workers": 2, "chunk": 64,
           "downsample": 2, "min_size": 256, "chunk_zyx": "nonsense"}
    out = normalize_convert_config(cfg)
    assert out is not None
    assert tuple(out["chunk_zyx"]) == (64, 64, 64)


# --------------------------------------------------------------------------- #
# end-to-end: convert at the anisotropic shape, and old cubic stores still work
# --------------------------------------------------------------------------- #

def _write_tif(tmp_path, shape, name="src.tif"):
    import tifffile

    a = np.arange(int(np.prod(shape)), dtype=np.uint16).reshape(shape)
    p = tmp_path / name
    tifffile.imwrite(str(p), a)
    return p, a


def test_convert_writes_the_anisotropic_chunk_shape(tmp_path):
    src, a = _write_tif(tmp_path, (80, 300, 300))
    out = tmp_path / "aniso.zarr"
    man = convert_to_ome_zarr(src, out, chunk=(32, 128, 128), max_levels=2)
    assert tuple(man["chunks"]) == (32, 128, 128)
    assert tuple(man["chunk_zyx"]) == (32, 128, 128)
    levels, _meta = open_multiscale(out)
    assert levels[0].chunksize == (32, 128, 128)
    np.testing.assert_array_equal(np.asarray(levels[0]), a)


def test_a_coarse_level_chunk_never_exceeds_its_shape(tmp_path):
    # level 1 is (40, 150, 150) -> the Z chunk 32 fits, but a 128 in-plane chunk
    # must not exceed 150 (it does not), while a deeper level would clip.
    src, _a = _write_tif(tmp_path, (80, 300, 300))
    out = tmp_path / "aniso2.zarr"
    convert_to_ome_zarr(src, out, chunk=(32, 128, 128), min_size=32)
    levels, _meta = open_multiscale(out)
    for lvl in levels:
        for c, s in zip(lvl.chunksize, lvl.shape):
            assert c <= s, (lvl.chunksize, lvl.shape)


def test_old_cubic_stores_still_open_and_read_back(tmp_path):
    src, a = _write_tif(tmp_path, (70, 130, 130))
    for edge in (64, 128):
        out = tmp_path / f"cubic{edge}.zarr"
        man = convert_to_ome_zarr(src, out, chunk=edge, max_levels=1)
        levels, _meta = open_multiscale(out)
        np.testing.assert_array_equal(np.asarray(levels[0]), a)
        assert tuple(man["chunk_zyx"]) == (edge, edge, edge)


def test_czyx_source_gets_a_channel_chunk_of_one(tmp_path):
    import tifffile

    a = np.arange(2 * 70 * 200 * 200, dtype=np.uint16).reshape(2, 70, 200, 200)
    src = tmp_path / "c.tif"
    tifffile.imwrite(str(src), a)
    out = tmp_path / "c.zarr"
    man = convert_to_ome_zarr(src, out, chunk=(32, 128, 128),
                              image_axes="CZYX", max_levels=1)
    assert tuple(man["chunks"]) == (1, 32, 128, 128)


# --------------------------------------------------------------------------- #
# streaming: the PROCESSED store inherits the SOURCE store's chunk shape, and
# the B1 write-race invariant (one chunk written by exactly one block) holds.
# --------------------------------------------------------------------------- #

def test_aligned_block_aligns_to_the_largest_chunk_axis():
    from packages.epicycle_preprocessing.streaming import _aligned_block

    # a (32, 128, 128) store -> blocks must be multiples of 128, which is also a
    # multiple of 32, so NO chunk is straddled on ANY axis.
    e = _aligned_block(200, (32, 128, 128))
    assert e == 256
    for c in (32, 128, 128):
        assert e % c == 0


def test_aligned_block_still_accepts_a_cubic_int():
    from packages.epicycle_preprocessing.streaming import _aligned_block

    assert _aligned_block(100, 64) == 128


@pytest.mark.parametrize("chunk_zyx", [(32, 128, 128), (64, 64, 64)])
def test_aligned_block_never_straddles_a_chunk(chunk_zyx):
    from packages.epicycle_preprocessing.streaming import _aligned_block

    for edge in range(1, 400):
        e = _aligned_block(edge, chunk_zyx)
        assert e >= max(chunk_zyx)
        for c in chunk_zyx:
            assert e % c == 0, (edge, e, c)


def _zarr_src(tmp_path, shape, chunks, name="src.zarr"):
    import zarr

    a = (np.arange(int(np.prod(shape))) % 4000).astype(np.uint16).reshape(shape)
    z = zarr.open(str(tmp_path / name), mode="w", shape=shape,
                  chunks=chunks, dtype="uint16")
    z[:] = a
    return z, a


def test_processed_store_inherits_the_source_chunk_shape(tmp_path):
    from packages.epicycle_preprocessing.streaming import stream_process_to_store

    z, _a = _zarr_src(tmp_path, (64, 256, 256), (32, 128, 128))
    out = tmp_path / "proc.zarr"
    man = stream_process_to_store(
        [z], {"channels": [[]]}, (1.0, 1.0, 1.0), str(out), max_levels=1)
    assert tuple(man["chunks"]) == (32, 128, 128)


def test_processed_store_inherits_a_legacy_cubic_source_shape(tmp_path):
    from packages.epicycle_preprocessing.streaming import stream_process_to_store

    z, _a = _zarr_src(tmp_path, (64, 128, 128), (64, 64, 64), name="old.zarr")
    out = tmp_path / "proc_old.zarr"
    man = stream_process_to_store(
        [z], {"channels": [[]]}, (1.0, 1.0, 1.0), str(out), max_levels=1)
    assert tuple(man["chunks"]) == (64, 64, 64)


def test_an_explicit_chunk_still_overrides_the_inheritance(tmp_path):
    from packages.epicycle_preprocessing.streaming import stream_process_to_store

    z, _a = _zarr_src(tmp_path, (64, 256, 256), (32, 128, 128), name="s2.zarr")
    out = tmp_path / "proc2.zarr"
    man = stream_process_to_store(
        [z], {"channels": [[]]}, (1.0, 1.0, 1.0), str(out),
        chunk=64, max_levels=1)
    assert tuple(man["chunks"]) == (64, 64, 64)


def test_the_chunk_shape_never_changes_the_streamed_pixels(tmp_path):
    """The write-race gate: the SAME source streamed at the anisotropic shape
    and at the legacy cubic 64 must produce byte-identical level-0 content."""
    from packages.epicycle_preprocessing.streaming import stream_process_to_store

    z, _a = _zarr_src(tmp_path, (64, 256, 256), (32, 128, 128), name="s3.zarr")
    got = []
    for tag, ck in (("aniso", (32, 128, 128)), ("cubic", 64)):
        out = tmp_path / f"proc3_{tag}.zarr"
        stream_process_to_store([z], {"channels": [[]]}, (1.0, 1.0, 1.0),
                                str(out), chunk=ck, max_levels=1)
        levels, _meta = open_multiscale(out)
        got.append(np.asarray(levels[0]))
    np.testing.assert_array_equal(got[0], got[1])


def test_an_unaligned_block_is_the_write_race_negative_control(tmp_path):
    """B1 control: with an UNALIGNED block two threads RMW the same chunk.  The
    aligned (production) path must agree with a single-threaded reference; this
    asserts the aligned anisotropic path does."""
    from packages.epicycle_preprocessing.streaming import stream_process_to_store

    z, _a = _zarr_src(tmp_path, (64, 256, 256), (32, 128, 128), name="s4.zarr")
    ref_p = tmp_path / "ref.zarr"
    stream_process_to_store([z], {"channels": [[]]}, (1.0, 1.0, 1.0),
                            str(ref_p), block_workers=1, max_levels=1)
    par_p = tmp_path / "par.zarr"
    stream_process_to_store([z], {"channels": [[]]}, (1.0, 1.0, 1.0),
                            str(par_p), block_workers=8, max_levels=1)
    ref, _ = open_multiscale(ref_p)
    par, _ = open_multiscale(par_p)
    np.testing.assert_array_equal(np.asarray(ref[0]), np.asarray(par[0]))


# --------------------------------------------------------------------------- #
# Brick 4b fix pass item 2 — the B1 no-straddle invariant must hold for a
# FOREIGN store whose chunk axes are not powers of two.
# --------------------------------------------------------------------------- #

def test_aligned_block_uses_the_lcm_not_the_max():
    """(24, 100, 100): max-alignment gives blocks that are multiples of 100 --
    100 % 24 != 0, so a block edge STRADDLES a Z chunk and two threads
    read-modify-write the same on-disk chunk (the B1 corruption race).  The lcm
    (600) is the smallest edge that is a whole number of chunks on EVERY axis."""
    from packages.epicycle_preprocessing.streaming import _aligned_block

    e = _aligned_block(200, (24, 100, 100))
    assert e == 600
    for c in (24, 100, 100):
        assert e % c == 0


@pytest.mark.parametrize(
    "chunk_zyx", [(32, 128, 128), (64, 64, 64), (24, 100, 100),
                  (17, 33, 65), (16, 128, 256)])
def test_aligned_block_never_straddles_any_chunk_axis(chunk_zyx):
    from packages.epicycle_preprocessing.streaming import _aligned_block

    for edge in range(1, 300):
        e = _aligned_block(edge, chunk_zyx)
        assert e >= max(chunk_zyx)
        for c in chunk_zyx:
            assert e % c == 0, (chunk_zyx, edge, e, c)


def test_aligned_block_floors_at_one():
    from packages.epicycle_preprocessing.streaming import _aligned_block

    assert _aligned_block(0, (1, 1, 1)) == 1
    assert _aligned_block(-5, 1) == 1


def test_a_foreign_non_power_of_two_store_streams_without_corruption(tmp_path):
    """End-to-end B1 gate on a (24, 100, 100) source: the parallel run must equal
    the single-threaded reference."""
    from packages.epicycle_preprocessing.streaming import stream_process_to_store

    z, _a = _zarr_src(tmp_path, (72, 200, 200), (24, 100, 100), name="odd.zarr")
    ref_p, par_p = tmp_path / "odd_ref.zarr", tmp_path / "odd_par.zarr"
    stream_process_to_store([z], {"channels": [[]]}, (1.0, 1.0, 1.0),
                            str(ref_p), block_workers=1, max_levels=1)
    stream_process_to_store([z], {"channels": [[]]}, (1.0, 1.0, 1.0),
                            str(par_p), block_workers=8, max_levels=1)
    ref, _ = open_multiscale(ref_p)
    par, _ = open_multiscale(par_p)
    np.testing.assert_array_equal(np.asarray(ref[0]), np.asarray(par[0]))
    # and it inherited the foreign shape verbatim
    assert tuple(open_multiscale(par_p)[0][0].chunksize) == (24, 100, 100)


# --------------------------------------------------------------------------- #
# Brick 4b fix pass item 3 — a persisted PRESET config must not lock a workspace
# out of the new shape.
# --------------------------------------------------------------------------- #

def test_a_persisted_named_preset_without_chunk_zyx_gets_the_new_default():
    """Every workspace that ran Prepare before Brick 4b has a saved config with
    `chunk: 64` and NO `chunk_zyx`.  Deriving (64,64,64) from it would pin that
    workspace to the OLD cubic shape forever -- the user would never get the fix
    they can see in the dialog.  A NAMED preset means "whatever the recommender
    currently says", so it must adopt the new default."""
    out = normalize_convert_config(
        {"preset": "balanced", "num_workers": 2, "chunk": 64,
         "downsample": 2, "min_size": 256})
    assert tuple(out["chunk_zyx"]) == (32, 128, 128)


@pytest.mark.parametrize("preset", ["safe", "balanced", "fast"])
def test_every_named_preset_adopts_the_new_default(preset):
    out = normalize_convert_config(
        {"preset": preset, "num_workers": 1, "chunk": 128,
         "downsample": 2, "min_size": 256})
    assert tuple(out["chunk_zyx"]) == (32, 128, 128)


def test_a_custom_config_keeps_its_cubic_shape():
    """`custom` means the user chose those exact numbers -- honour them."""
    out = normalize_convert_config(
        {"preset": "custom", "num_workers": 3, "chunk": 64,
         "downsample": 2, "min_size": 256})
    assert tuple(out["chunk_zyx"]) == (64, 64, 64)


def test_an_explicit_chunk_zyx_always_wins_even_for_a_named_preset():
    out = normalize_convert_config(
        {"preset": "balanced", "num_workers": 2, "chunk": 64,
         "downsample": 2, "min_size": 256, "chunk_zyx": [16, 128, 128]})
    assert tuple(out["chunk_zyx"]) == (16, 128, 128)


def test_a_corrupt_chunk_zyx_on_custom_still_falls_back_to_cubic():
    out = normalize_convert_config(
        {"preset": "custom", "num_workers": 3, "chunk": 64,
         "downsample": 2, "min_size": 256, "chunk_zyx": "nonsense"})
    assert tuple(out["chunk_zyx"]) == (64, 64, 64)


def test_the_legacy_cubic_mirror_is_the_z_axis_not_the_in_plane_edge():
    """The legacy `chunk` int is what an OLDER checkout of this app would read
    and prepare with.  Mirroring the in-plane 128 would make that checkout write
    a CUBIC 128 store -- the worst case for scrolling and the very thing Brick 3c
    reverted.  Mirroring the Z axis (32) degrades to a small, safe cube."""
    rec = recommend_convert_settings("balanced", **_MACHINE)
    assert rec["chunk"] == 32
    assert tuple(rec["chunk_zyx"]) == (32, 128, 128)


# --------------------------------------------------------------------------- #
# Phase 2 item 1 — the multi-channel CZYX path must inherit the source shape too
#
# A CZYX source is handed to the streaming engine as one `_ZarrChannelView` per
# channel.  The view exposed only `.shape`, so `_source_chunk_zyx` found no
# `.chunks` and silently fell back to the DEFAULT -- meaning the processed
# store's layout depended on the CHANNEL COUNT: a 1-channel source inherited
# correctly, the user's 2-channel stack did not.
# --------------------------------------------------------------------------- #

def _czyx_zarr(tmp_path, shape, chunks, name="czyx.zarr"):
    import zarr

    a = (np.arange(int(np.prod(shape))) % 4000).astype(np.uint16).reshape(shape)
    z = zarr.open(str(tmp_path / name), mode="w", shape=shape,
                  chunks=chunks, dtype="uint16")
    z[:] = a
    return z, a


@pytest.mark.parametrize("src_chunks,want", [
    ((1, 64, 64, 64), (64, 64, 64)),           # a legacy cubic prepared store
    ((1, 32, 128, 128), (32, 128, 128)),       # a Brick-4b prepared store
])
def test_a_two_channel_source_inherits_its_chunk_shape(tmp_path, src_chunks,
                                                       want):
    from napari_epicycle.data_preparation.controller import _ZarrChannelView
    from packages.epicycle_preprocessing.streaming import stream_process_to_store

    z, _a = _czyx_zarr(tmp_path, (2, 64, 256, 256), src_chunks)
    views = [_ZarrChannelView(z, ci) for ci in range(2)]
    out = tmp_path / f"proc_{want[0]}.zarr"
    man = stream_process_to_store(
        views, {"channels": [[], []]}, (1.0, 1.0, 1.0), str(out), max_levels=1)
    assert tuple(man["chunk_zyx"]) == want
    assert tuple(man["chunks"]) == (1,) + want   # channel axis chunked at 1


def test_the_channel_view_exposes_the_zyx_chunks(tmp_path):
    from napari_epicycle.data_preparation.controller import _ZarrChannelView

    z, _a = _czyx_zarr(tmp_path, (2, 32, 64, 64), (1, 16, 32, 32), name="v.zarr")
    view = _ZarrChannelView(z, 1)
    assert view.chunks == (16, 32, 32)          # ZYX only, channel axis dropped
    assert view.shape == (32, 64, 64)


def test_the_channel_view_chunks_survive_source_chunk_zyx(tmp_path):
    from napari_epicycle.data_preparation.controller import _ZarrChannelView
    from packages.epicycle_preprocessing.streaming import _source_chunk_zyx

    z, _a = _czyx_zarr(tmp_path, (3, 32, 64, 64), (1, 16, 32, 32), name="w.zarr")
    assert _source_chunk_zyx(_ZarrChannelView(z, 0)) == (16, 32, 32)


# --------------------------------------------------------------------------- #
# Phase 2 item 5 — the processed-store worker must survive a chunk TUPLE
# --------------------------------------------------------------------------- #

def test_the_process_worker_forwards_a_chunk_tuple(tmp_path, monkeypatch):
    """`_struct["chunk"] = int(self._chunk)` raised TypeError the moment a
    caller passed the (32,128,128) shape instead of a cubic int -- and since the
    exception happened inside the worker's run(), it would surface as a failed
    Process rather than anything legible."""
    from napari_epicycle.data_preparation import controller as _ctl

    seen = {}
    monkeypatch.setattr(
        _ctl, "_pp_stream_to_store",
        lambda *a, **kw: (seen.update(kw), {"out_store_path": "x"})[1],
        raising=False)
    w = _ctl._PreprocProcessToStoreWorker(
        {"channels": [[]]}, [object()], (1.0, 1.0, 1.0),
        str(tmp_path / "o.zarr"), chunk=(32, 128, 128))
    w.run()
    assert tuple(seen["chunk"]) == (32, 128, 128)


def test_the_process_worker_still_accepts_a_cubic_int(tmp_path, monkeypatch):
    from napari_epicycle.data_preparation import controller as _ctl

    seen = {}
    monkeypatch.setattr(
        _ctl, "_pp_stream_to_store",
        lambda *a, **kw: (seen.update(kw), {"out_store_path": "x"})[1],
        raising=False)
    w = _ctl._PreprocProcessToStoreWorker(
        {"channels": [[]]}, [object()], (1.0, 1.0, 1.0),
        str(tmp_path / "o.zarr"), chunk=64)
    w.run()
    assert seen["chunk"] == 64


def test_the_process_worker_omits_an_unset_chunk(tmp_path, monkeypatch):
    """None must be OMITTED so the engine's inherit-from-source default stands."""
    from napari_epicycle.data_preparation import controller as _ctl

    seen = {}
    monkeypatch.setattr(
        _ctl, "_pp_stream_to_store",
        lambda *a, **kw: (seen.update(kw), {"out_store_path": "x"})[1],
        raising=False)
    w = _ctl._PreprocProcessToStoreWorker(
        {"channels": [[]]}, [object()], (1.0, 1.0, 1.0),
        str(tmp_path / "o.zarr"))
    w.run()
    assert "chunk" not in seen
