"""TilingStage RAM auto-cap (Phase 1c): shrink the XY tile so one tile's recon
fits this machine's RAM. No-op on a fat node; prevents OOM on a small box."""

import pytest

from packages.epicycle_scaledseg.core.tiling import cap_patch_size_to_ram, TilingStage
import packages.epicycle_scaledseg.core.tiling as _t
import packages.epicycle_scaledseg.utils.tiling as _ut
import packages.epicycle_scaledseg.utils.resources as _res


class TestCapPatchSizeToRam:
    def test_fits_unchanged(self):
        # 512 tile, Z=200, 720 GB -> a few GB peak, well under budget -> unchanged.
        capped, status = cap_patch_size_to_ram(512, 200, 720.0)
        assert capped == 512 and status == "fits"

    def test_shrinks_on_small_box(self):
        # 1024 tile, Z=1000 (~82 GB peak) on a 64 GB box -> must shrink.
        capped, status = cap_patch_size_to_ram(1024, 1000, 64.0)
        assert status == "shrunk"
        assert capped < 1024 and capped % 64 == 0 and capped >= 128
        # the shrunk tile actually fits the (avail-16)*0.8 budget
        from packages.epicycle_scaledseg.utils.resources import estimate_tile_peak_ram_gb
        assert estimate_tile_peak_ram_gb(1000, capped, capped) <= (64 - 16) * 0.8

    def test_min_floor_when_z_too_big(self):
        # Huge Z + tiny RAM -> even the 128 floor exceeds budget -> min-floor.
        capped, status = cap_patch_size_to_ram(512, 5000, 20.0)
        assert status == "min-floor" and capped == 128

    def test_unknown_ram_leaves_request(self):
        assert cap_patch_size_to_ram(1024, 1000, 0.0) == (1024, "unknown-ram")

    def test_budget_zero_floors(self):
        # avail == reserve -> budget 0 -> floor to min(req, 128).
        capped, status = cap_patch_size_to_ram(512, 200, 16.0)
        assert capped == 128 and status == "min-floor"

    def test_never_grows(self):
        # Even with huge RAM, never exceed the requested size.
        capped, _ = cap_patch_size_to_ram(256, 100, 100000.0)
        assert capped == 256


class _FakeTiler:
    def __init__(self, tif_path, output_dir, target_patch_size, **kw):
        _FakeTiler.last_patch = target_patch_size
        self.patch_size = target_patch_size
        self.step_size = target_patch_size

    def split_tiles(self):
        return []


class TestTilingStageAutoCapWiring:
    def _stage(self, tmp_path, **kw):
        inp = tmp_path / "stack.tif"
        inp.write_bytes(b"fake")
        return TilingStage(input_tif=str(inp), output_dir=str(tmp_path / "out"),
                           patch_size=1024, **kw)

    def test_run_shrinks_patch_passed_to_tiler(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_ut, "ZStackTiler", _FakeTiler)
        monkeypatch.setattr(_t, "_input_z_dim", lambda p, ax: 1000)
        monkeypatch.setattr(_res, "available_ram_gb_and_source", lambda: (64.0, "test"))
        self._stage(tmp_path).run([], n_gpus=0)
        assert _FakeTiler.last_patch < 1024 and _FakeTiler.last_patch % 64 == 0

    def test_run_keeps_patch_on_fat_node(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_ut, "ZStackTiler", _FakeTiler)
        monkeypatch.setattr(_t, "_input_z_dim", lambda p, ax: 1000)
        monkeypatch.setattr(_res, "available_ram_gb_and_source", lambda: (720.0, "test"))
        self._stage(tmp_path).run([], n_gpus=0)
        assert _FakeTiler.last_patch == 1024   # fits -> unchanged

    def test_run_disabled_keeps_patch(self, tmp_path, monkeypatch):
        monkeypatch.setattr(_ut, "ZStackTiler", _FakeTiler)
        # auto_ram_cap=False -> never even probes Z/RAM.
        monkeypatch.setattr(_t, "_input_z_dim",
                            lambda p, ax: (_ for _ in ()).throw(AssertionError("probed!")))
        self._stage(tmp_path, auto_ram_cap=False).run([], n_gpus=0)
        assert _FakeTiler.last_patch == 1024
