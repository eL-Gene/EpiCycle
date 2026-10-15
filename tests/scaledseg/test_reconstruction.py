"""Tests for epicycle.core.reconstruction module.

Tests cover the ReconstructionStage class logic (validation, skip-completed,
result collection, directory resolution) without requiring heavy dependencies
like segment3D, torch, or tifffile.
"""

import pytest
import json
import numpy as np
from pathlib import Path

from packages.epicycle_scaledseg.core.reconstruction import (
    ReconstructionStage,
    compute_recon_concurrency,
    RECON_EFFECTIVE_CORES_PER_TILE,
    apply_gd_overrides,
)


class TestGradientDescentOverrides:
    """Per-tile recon n_iter / gradient_decay / momenta overrides (pure, no segment3D)."""

    def test_applies_only_non_none(self):
        base = {"gradient_descent": {"n_iter": 25, "gradient_decay": 0.001, "momenta": 0.98},
                "other": 1}
        out = apply_gd_overrides(base, n_iter=50, gradient_decay=0.01)
        assert out["gradient_descent"]["n_iter"] == 50
        assert out["gradient_descent"]["gradient_decay"] == 0.01
        assert out["gradient_descent"]["momenta"] == 0.98   # None -> unchanged
        assert out["other"] == 1

    def test_non_mutating(self):
        base = {"gradient_descent": {"n_iter": 25}}
        out = apply_gd_overrides(base, n_iter=99)
        assert base["gradient_descent"]["n_iter"] == 25      # original untouched
        assert out["gradient_descent"]["n_iter"] == 99

    def test_all_none_is_passthrough_copy(self):
        base = {"gradient_descent": {"n_iter": 25}}
        out = apply_gd_overrides(base)
        out["gradient_descent"]["n_iter"] = 7
        assert base["gradient_descent"]["n_iter"] == 25      # copy, not alias

    def test_stage_stores_and_reports_overrides(self):
        s = ReconstructionStage(n_iter=40, gradient_decay=0.005, momenta=0.9)
        p = s.get_params()
        assert p["gd_n_iter"] == 40
        assert p["gd_gradient_decay"] == 0.005
        assert p["gd_momenta"] == 0.9


class TestReconConcurrency:
    """M = how many recon tiles to run at once — bounded by CPU + RAM, not GPUs.

    This is the Phase-2 resource-utilization centerpiece: u-Segment3D aggregation
    can only use ~4 effective cores/tile, so a fat node should run many tiles.
    """

    def test_default_cores_per_tile_is_four(self):
        assert RECON_EFFECTIVE_CORES_PER_TILE == 4

    def test_fat_node_small_tiles_cpu_bound(self):
        # 64 cores / 4 = 16; RAM fits 720*0.9/9 = 72; 100 tiles -> CPU caps at 16.
        assert compute_recon_concurrency(64, 720, 9.0, 100) == 16

    def test_fat_node_big_tiles_ram_bound(self):
        # 64//4 = 16, but (720-16)*0.8/75 = 7.5 -> 7 -> RAM (with reserve) caps at 7.
        assert compute_recon_concurrency(64, 720, 75.0, 100) == 7

    def test_small_box_big_tile_floors_at_one(self):
        # 64 GB, 75 GB/tile -> (64-16)*0.8/75 = 0.5 -> 0 -> floored to 1 (one, slowly).
        assert compute_recon_concurrency(64, 64, 75.0, 100) == 1

    def test_few_tiles_caps_workers(self):
        assert compute_recon_concurrency(64, 720, 9.0, 3) == 3   # never > n_tiles

    def test_laptop_profile_cpu_bound(self):
        # 8 cores -> 2; RAM (64-16)*0.8/9 = 4; 50 tiles -> CPU caps at 2.
        assert compute_recon_concurrency(8, 64, 9.0, 50) == 2

    def test_unknown_ram_is_conservative_not_cpu_capped(self):
        # "Can't measure memory" -> run one at a time, NOT 16 (the OOM trap).
        assert compute_recon_concurrency(64, 0.0, 9.0, 100) == 1   # RAM unknown
        assert compute_recon_concurrency(64, 720, 0.0, 100) == 1   # peak unknown

    def test_ram_reserve_carved_out(self):
        # The 16 GB reserve is carved off before the fraction: a 20 GB node has only
        # (20-16)*0.8 = 3.2 GB usable / 1 GB-tile = 3 -> min(cpu 16, 3, 100) = 3.
        assert compute_recon_concurrency(64, 20.0, 1.0, 100) == 3

    def test_one_core_floors_at_one(self):
        assert compute_recon_concurrency(1, 720, 9.0, 100) == 1

    def test_cores_per_tile_override(self):
        # If the plateau were 8 cores: 64//8 = 8.
        assert compute_recon_concurrency(64, 720, 9.0, 100, cores_per_tile=8) == 8


# --- Helpers ---

def _make_seg_tile(tmp_path, tile_name, seg_root=None):
    """Create a fake stage1 segmentation directory for a tile."""
    if seg_root:
        seg_dir = Path(seg_root) / tile_name
    else:
        seg_dir = tmp_path / "stage1_segmentation" / tile_name
    seg_dir.mkdir(parents=True, exist_ok=True)

    (seg_dir / "SEGMENTATION_DONE").touch()

    # Fake config
    cfg = {
        "tile_name": tile_name,
        "truncated_range": [10, 90],
        "original_dimensions": {"z": 100},
    }
    with open(seg_dir / f"{tile_name}_config.json", "w") as f:
        json.dump(cfg, f)

    # Fake flow/prob files (empty npy for path resolution)
    for view in ("xy", "xz", "yz"):
        np.save(seg_dir / f"{tile_name}_{view}_flows.npy", np.zeros((2, 2, 2, 2), dtype=np.float32))
        np.save(seg_dir / f"{tile_name}_{view}_probs.npy", np.zeros((2, 2, 2), dtype=np.float32))

    return seg_dir


def _make_recon_output(tmp_path, tile_name, out_root=None):
    """Create a fake stage2 output (mask.tif) to simulate completed tile."""
    if out_root:
        out_dir = Path(out_root) / tile_name
    else:
        out_dir = tmp_path / "stage2_reconstruction" / tile_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{tile_name}_mask.tif").touch()
    return out_dir


# --- Tests ---

class TestReconstructionStageInit:
    def test_defaults(self):
        stage = ReconstructionStage()
        assert stage.name == "reconstruction"
        assert stage.mode == "sequential"
        assert stage.n_cpu_per_worker is None  # auto-detect
        assert stage.max_workers is None
        assert stage.save_prob3d is True
        assert stage.save_binary is True
        assert stage.binary_erode is True
        assert stage.cleanup_intermediates is False

    def test_max_workers_stored(self):
        stage = ReconstructionStage(max_workers=4)
        assert stage.max_workers == 4

    def test_custom_params(self):
        stage = ReconstructionStage(
            mode="sequential",
            save_flow3d=True,
            binary_kernel_size=5,
            n_cpu_per_worker=16,
        )
        assert stage.mode == "sequential"
        assert stage.save_flow3d is True
        assert stage.binary_kernel_size == 5
        assert stage.n_cpu_per_worker == 16

    def test_pre_smooth_defaults_off(self):
        stage = ReconstructionStage()
        assert stage.pre_smooth_flows is False
        assert stage.pre_smooth_probs is False

    def test_pre_smooth_sigma_defaults(self):
        stage = ReconstructionStage()
        assert stage.pre_smooth_sigma_perp == 3.0
        assert stage.pre_smooth_sigma_inplane == 1.0
        assert stage.flow_mask_threshold == 0.2
        assert stage.flow_soft_weighting is True

    def test_pre_smooth_custom(self):
        stage = ReconstructionStage(
            pre_smooth_flows=True,
            pre_smooth_probs=True,
            pre_smooth_sigma_perp=5.0,
            pre_smooth_sigma_inplane=0.5,
            flow_mask_threshold=0.3,
            flow_soft_weighting=False,
        )
        assert stage.pre_smooth_flows is True
        assert stage.pre_smooth_probs is True
        assert stage.pre_smooth_sigma_perp == 5.0
        assert stage.pre_smooth_sigma_inplane == 0.5
        assert stage.flow_mask_threshold == 0.3
        assert stage.flow_soft_weighting is False


class TestReconParallelDispatch:
    """_run_parallel sizes M from RAM+CPU and DECOUPLES it from GPU count — the
    Phase-2 win. A fake Pool records the worker count + tasks; no real spawn."""

    @staticmethod
    def _fake_pool(monkeypatch):
        import packages.epicycle_scaledseg.core.reconstruction as _r
        rec = {}

        class _Pool:
            def __init__(self, processes=None):
                rec["processes"] = processes

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def starmap(self, fn, tasks):
                rec["tasks"] = list(tasks)
                return [None] * len(rec["tasks"])

        class _Ctx:
            def Pool(self, processes=None):
                return _Pool(processes=processes)

        monkeypatch.setattr(_r, "get_context", lambda method: _Ctx())
        # These tests exercise ESTIMATE-based M sizing; bypass the one-tile
        # calibration (its own test class covers it) so no real subprocess spawns.
        monkeypatch.setattr(ReconstructionStage, "_run_one_tile_for_calibration",
                            lambda self, td, n_gpus, log_dir: None)
        return rec

    @staticmethod
    def _patch_resources(monkeypatch, cpus=64, ram=720.0):
        import packages.epicycle_scaledseg.utils.resources as _res
        monkeypatch.setattr(_res, "get_allocated_cpus", lambda: cpus)
        monkeypatch.setattr(_res, "get_available_ram_gb", lambda: ram)
        monkeypatch.setattr(_res, "available_ram_gb_and_source", lambda: (ram, "test"))

    def _tiles_with_probs(self, tmp_path, seg_root, n, shape=(10, 64, 64)):
        import numpy as np
        tiles = []
        for i in range(n):
            td = tmp_path / f"tile_{i}"
            td.mkdir()
            sd = seg_root / td.name
            sd.mkdir(parents=True)
            np.save(sd / f"{td.name}_xy_probs.npy", np.zeros(shape, dtype=np.float32))
            tiles.append(td)
        return tiles

    def test_M_decoupled_from_gpus_cpu_bound(self, tmp_path, monkeypatch):
        rec = self._fake_pool(monkeypatch)
        self._patch_resources(monkeypatch, cpus=64, ram=720.0)
        seg_root, out_root = tmp_path / "seg", tmp_path / "out"
        tiles = self._tiles_with_probs(tmp_path, seg_root, 20)  # tiny tiles -> CPU bound
        stage = ReconstructionStage(mode="parallel", seg_root=str(seg_root), out_root=str(out_root))
        stage.run(tiles, n_gpus=3)
        assert rec["processes"] == 16                 # 64//4 cpu-bound, NOT n_gpus(3)
        assert rec["processes"] > 3                   # the whole point: decoupled from GPUs
        assert rec["tasks"][0][2]["n_cpu"] == 4        # threads pinned to cores//M = 64//16
        gpu_ids = [t[0] for t in rec["tasks"]]
        assert gpu_ids[:6] == [0, 1, 2, 0, 1, 2]      # GPUs round-robined for erosion

    def test_M_ram_bound_big_tile(self, tmp_path, monkeypatch):
        rec = self._fake_pool(monkeypatch)
        self._patch_resources(monkeypatch, cpus=64, ram=720.0)
        # Patch the estimate to a big per-tile peak: (720-16)*0.8/75 = 7.5 -> 7 (RAM caps).
        monkeypatch.setattr(ReconstructionStage, "_estimate_peak_per_tile",
                            lambda self, todo: 75.0)
        seg_root, out_root = tmp_path / "seg", tmp_path / "out"
        tiles = self._tiles_with_probs(tmp_path, seg_root, 20)
        stage = ReconstructionStage(mode="parallel", seg_root=str(seg_root), out_root=str(out_root))
        stage.run(tiles, n_gpus=3)
        assert rec["processes"] == 7                   # RAM-bound (with reserve), below cpu cap 16
        assert rec["tasks"][0][2]["n_cpu"] == 64 // 7   # threads pinned to cores//M

    def test_unknown_peak_runs_conservatively(self, tmp_path, monkeypatch):
        rec = self._fake_pool(monkeypatch)
        self._patch_resources(monkeypatch, cpus=64, ram=720.0)
        monkeypatch.setattr(ReconstructionStage, "_estimate_peak_per_tile",
                            lambda self, todo: None)
        seg_root, out_root = tmp_path / "seg", tmp_path / "out"
        tiles = self._tiles_with_probs(tmp_path, seg_root, 20)
        stage = ReconstructionStage(mode="parallel", seg_root=str(seg_root), out_root=str(out_root))
        stage.run(tiles, n_gpus=3)
        assert rec["processes"] == 1                   # no estimate -> 1 at a time (safe)

    def test_max_workers_caps_auto(self, tmp_path, monkeypatch):
        rec = self._fake_pool(monkeypatch)
        self._patch_resources(monkeypatch, cpus=64, ram=720.0)
        seg_root, out_root = tmp_path / "seg", tmp_path / "out"
        tiles = self._tiles_with_probs(tmp_path, seg_root, 20)  # auto would be 16
        stage = ReconstructionStage(mode="parallel", max_workers=4,
                                    seg_root=str(seg_root), out_root=str(out_root))
        stage.run(tiles, n_gpus=3)
        assert rec["processes"] == 4                   # explicit cap honored
        assert rec["tasks"][0][2]["n_cpu"] == 16        # 64//4

    def test_node_total_warning_only_under_slurm(self, tmp_path, monkeypatch, caplog):
        """The node-total RAM warning fires only under SLURM (shared node), not on a
        personal box where psutil.total is the real RAM."""
        import logging
        import packages.epicycle_scaledseg.utils.resources as _res
        self._fake_pool(monkeypatch)                   # also no-ops calibration
        monkeypatch.setattr(_res, "get_allocated_cpus", lambda: 64)
        monkeypatch.setattr(_res, "available_ram_gb_and_source",
                            lambda: (64.0, "psutil.total(NODE-may-over-report)"))
        seg_root, out_root = tmp_path / "seg", tmp_path / "out"
        tiles = self._tiles_with_probs(tmp_path, seg_root, 5)
        stage = ReconstructionStage(mode="parallel", seg_root=str(seg_root), out_root=str(out_root))

        monkeypatch.delenv("SLURM_JOB_ID", raising=False)   # personal box
        with caplog.at_level(logging.WARNING, logger="epicycle.reconstruction"):
            stage.run(tiles, n_gpus=1)
        assert not any("over-reports" in r.message for r in caplog.records)

        caplog.clear()
        monkeypatch.setenv("SLURM_JOB_ID", "12345")         # shared SLURM node
        with caplog.at_level(logging.WARNING, logger="epicycle.reconstruction"):
            stage.run(tiles, n_gpus=1)
        assert any("over-reports" in r.message for r in caplog.records)


class TestReconCalibrationHelpers:
    """Measured-RSS calibration sidecar — turns the a-priori RAM model into a
    measurement so M can be sized to reality."""

    def test_write_read_roundtrip(self, tmp_path):
        from packages.epicycle_scaledseg.core.reconstruction import _write_calibration_peak, _read_calibration_peak
        assert _read_calibration_peak(tmp_path) is None
        _write_calibration_peak(tmp_path, 42.0)
        assert _read_calibration_peak(tmp_path) == 42.0

    def test_write_keeps_max(self, tmp_path):
        from packages.epicycle_scaledseg.core.reconstruction import _write_calibration_peak, _read_calibration_peak
        _write_calibration_peak(tmp_path, 30.0)
        _write_calibration_peak(tmp_path, 10.0)   # lower -> ignored (a worker can't undercount)
        assert _read_calibration_peak(tmp_path) == 30.0
        _write_calibration_peak(tmp_path, 50.0)   # higher -> updates
        assert _read_calibration_peak(tmp_path) == 50.0

    def test_none_out_root_is_noop(self):
        from packages.epicycle_scaledseg.core.reconstruction import _write_calibration_peak, _read_calibration_peak
        _write_calibration_peak(None, 42.0)       # no crash, no write
        assert _read_calibration_peak(None) is None

    def test_peak_rss_gb_positive_or_none(self):
        from packages.epicycle_scaledseg.core.reconstruction import _peak_rss_gb
        v = _peak_rss_gb()
        assert v is None or v > 0

    def test_stale_when_apriori_changed(self, tmp_path):
        from packages.epicycle_scaledseg.core.reconstruction import (
            _write_calibration_peak, _stamp_calibration_apriori, _read_calibration_peak)
        _write_calibration_peak(tmp_path, 75.0)
        _stamp_calibration_apriori(tmp_path, 10.0)        # measured for ~10 GB-est tiles
        assert _read_calibration_peak(tmp_path, current_apriori_gb=10.0) == 75.0   # same size
        assert _read_calibration_peak(tmp_path, current_apriori_gb=11.0) == 75.0   # within 20%
        # re-tiled bigger (a-priori estimate jumps) -> stale -> None (force recal).
        assert _read_calibration_peak(tmp_path, current_apriori_gb=50.0) is None

    def test_legacy_sidecar_without_apriori_is_trusted(self, tmp_path):
        import json as _json
        from packages.epicycle_scaledseg.core.reconstruction import _read_calibration_peak, _calibration_path
        _calibration_path(tmp_path).write_text(_json.dumps({"measured_peak_gb": 30.0}))
        assert _read_calibration_peak(tmp_path, current_apriori_gb=5.0) == 30.0   # no signature -> trust

    def test_write_peak_preserves_apriori(self, tmp_path):
        from packages.epicycle_scaledseg.core.reconstruction import (
            _stamp_calibration_apriori, _write_calibration_peak, _read_calibration)
        _stamp_calibration_apriori(tmp_path, 12.0)
        _write_calibration_peak(tmp_path, 40.0)           # must not drop apriori_gb
        d = _read_calibration(tmp_path)
        assert d["measured_peak_gb"] == 40.0 and d["apriori_gb"] == 12.0


class TestReconCalibrationFlow:
    """_run_parallel prefers a MEASURED peak; calibrates one tile when none exists."""

    @staticmethod
    def _fake_pool(monkeypatch):
        import packages.epicycle_scaledseg.core.reconstruction as _r
        rec = {}

        class _Pool:
            def __init__(self, processes=None):
                rec["processes"] = processes

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

            def starmap(self, fn, tasks):
                rec["tasks"] = list(tasks)
                return [None] * len(rec["tasks"])

        class _Ctx:
            def Pool(self, processes=None):
                return _Pool(processes=processes)

        monkeypatch.setattr(_r, "get_context", lambda method: _Ctx())
        return rec

    @staticmethod
    def _patch_resources(monkeypatch, cpus=64, ram=720.0):
        import packages.epicycle_scaledseg.utils.resources as _res
        monkeypatch.setattr(_res, "get_allocated_cpus", lambda: cpus)
        monkeypatch.setattr(_res, "get_available_ram_gb", lambda: ram)
        monkeypatch.setattr(_res, "available_ram_gb_and_source", lambda: (ram, "test"))

    def _tiles(self, tmp_path, seg_root, n, shape=(10, 64, 64)):
        import numpy as np
        out = []
        for i in range(n):
            td = tmp_path / f"tile_{i}"
            td.mkdir()
            sd = seg_root / td.name
            sd.mkdir(parents=True)
            np.save(sd / f"{td.name}_xy_probs.npy", np.zeros(shape, dtype=np.float32))
            out.append(td)
        return out

    def test_uses_measured_peak_when_sidecar_present(self, tmp_path, monkeypatch):
        import json as _json
        rec = self._fake_pool(monkeypatch)
        self._patch_resources(monkeypatch, cpus=64, ram=720.0)
        seg_root, out_root = tmp_path / "seg", tmp_path / "out"
        tiles = self._tiles(tmp_path, seg_root, 20)   # tiny tiles -> est peak ~0 -> M would be 16
        out_root.mkdir(parents=True)
        (out_root / "recon_calibration.json").write_text(_json.dumps({"measured_peak_gb": 75.0}))
        called = {"n": 0}
        monkeypatch.setattr(ReconstructionStage, "_run_one_tile_for_calibration",
                            lambda self, td, n_gpus, log_dir: called.__setitem__("n", called["n"] + 1))
        stage = ReconstructionStage(mode="parallel", seg_root=str(seg_root), out_root=str(out_root))
        stage.run(tiles, n_gpus=3)
        assert called["n"] == 0          # sidecar present -> no calibration run
        assert rec["processes"] == 7     # M from MEASURED 75 GB ((720-16)*0.8/75), not the tiny estimate

    def test_calibrates_one_tile_when_no_sidecar(self, tmp_path, monkeypatch):
        import json as _json
        rec = self._fake_pool(monkeypatch)
        self._patch_resources(monkeypatch, cpus=64, ram=720.0)
        seg_root, out_root = tmp_path / "seg", tmp_path / "out"
        tiles = self._tiles(tmp_path, seg_root, 20)
        out_root.mkdir(parents=True)
        calls = {"n": 0}

        def fake_cal(self, td, n_gpus, log_dir):
            calls["n"] += 1
            (out_root / "recon_calibration.json").write_text(_json.dumps({"measured_peak_gb": 75.0}))

        monkeypatch.setattr(ReconstructionStage, "_run_one_tile_for_calibration", fake_cal)
        stage = ReconstructionStage(mode="parallel", seg_root=str(seg_root), out_root=str(out_root))
        stage.run(tiles, n_gpus=3)
        assert calls["n"] == 1           # exactly one calibration tile
        assert rec["processes"] == 7     # M sized from the measured 75 GB
        assert len(rec["tasks"]) == 19   # calibration tile dropped from the parallel batch

    def test_stale_sidecar_triggers_recalibration(self, tmp_path, monkeypatch):
        import json as _json
        rec = self._fake_pool(monkeypatch)
        self._patch_resources(monkeypatch, cpus=64, ram=720.0)
        seg_root, out_root = tmp_path / "seg", tmp_path / "out"
        tiles = self._tiles(tmp_path, seg_root, 20)   # tiny tiles -> current a-priori ~0
        out_root.mkdir(parents=True)
        # A STALE sidecar: measured 75 GB stamped for a BIG previous tile-size (apriori 50).
        (out_root / "recon_calibration.json").write_text(
            _json.dumps({"measured_peak_gb": 75.0, "apriori_gb": 50.0}))
        calls = {"n": 0}

        def fake_cal(self, td, n_gpus, log_dir):
            calls["n"] += 1
            (out_root / "recon_calibration.json").write_text(
                _json.dumps({"measured_peak_gb": 5.0, "apriori_gb": 50.0}))

        monkeypatch.setattr(ReconstructionStage, "_run_one_tile_for_calibration", fake_cal)
        stage = ReconstructionStage(mode="parallel", seg_root=str(seg_root), out_root=str(out_root))
        stage.run(tiles, n_gpus=3)
        # current a-priori (~0) differs >20% from stored 50 -> stale -> recalibrated,
        # did NOT blindly reuse the too-small 75 GB measurement for the new tiles.
        assert calls["n"] == 1


class TestPinInnerCpuCount:
    """u-Segment3D's inner ThreadPool reads cpu_count() (ignoring OMP/MKL pins);
    this caps it to the allocated thread budget — prevents M-way oversubscription
    and keeps it inside a SLURM CPU allocation. Output-neutral.

    monkeypatch restores BOTH modules' ``cpu_count`` after each test (the pin
    patches them in place; without this the cap leaks for the whole session)."""

    @staticmethod
    def _guard_multiprocess(monkeypatch):
        # the pin patches ``multiprocess`` too whenever it is importable (u-Segment3D installs it)
        try:
            import multiprocess as mp
        except ImportError:
            return
        monkeypatch.setattr(mp, "cpu_count", mp.cpu_count)

    def test_caps_multiprocessing_cpu_count(self, monkeypatch):
        import multiprocessing
        from packages.epicycle_scaledseg.core.reconstruction import _pin_inner_cpu_count
        monkeypatch.setattr(multiprocessing, "cpu_count", multiprocessing.cpu_count)
        self._guard_multiprocess(monkeypatch)
        _pin_inner_cpu_count(8)
        # +1 so u-Segment3D's cpu_count()-1 == n_cpu (8) threads.
        assert multiprocessing.cpu_count() == 9

    def test_caps_multiprocess_cpu_count(self, monkeypatch):
        mp = pytest.importorskip("multiprocess")
        import multiprocessing
        from packages.epicycle_scaledseg.core.reconstruction import _pin_inner_cpu_count
        monkeypatch.setattr(mp, "cpu_count", mp.cpu_count)
        monkeypatch.setattr(multiprocessing, "cpu_count", multiprocessing.cpu_count)
        _pin_inner_cpu_count(4)
        assert mp.cpu_count() == 5

    def test_invalid_values_are_noops(self, monkeypatch):
        import multiprocessing
        from packages.epicycle_scaledseg.core.reconstruction import _pin_inner_cpu_count
        monkeypatch.setattr(multiprocessing, "cpu_count", multiprocessing.cpu_count)
        self._guard_multiprocess(monkeypatch)
        real = multiprocessing.cpu_count()
        _pin_inner_cpu_count(0)
        assert multiprocessing.cpu_count() == real     # 0 -> unchanged
        _pin_inner_cpu_count(None)                       # no crash
        _pin_inner_cpu_count("x")                        # no crash
        assert multiprocessing.cpu_count() == real


class TestReconstructionResourceGuards:
    """n_gpus=0 (CPU-only, now reportable by get_allocated_gpus) must not crash the
    CPU-bound recon paths — aggregation is CPU-only and erosion CPU-falls-back."""

    def test_resolve_n_cpu_parallel_zero_gpus_no_div_by_zero(self, monkeypatch):
        import packages.epicycle_scaledseg.utils.resources as _res
        monkeypatch.setattr(_res, "get_allocated_cpus", lambda: 16)
        stage = ReconstructionStage(mode="parallel")
        n = stage._resolve_n_cpu(0)   # must NOT raise ZeroDivisionError
        assert n >= 1

    def test_resolve_n_cpu_parallel_splits_when_gpus_present(self, monkeypatch):
        import packages.epicycle_scaledseg.utils.resources as _res
        monkeypatch.setattr(_res, "get_allocated_cpus", lambda: 16)
        stage = ReconstructionStage(mode="parallel")
        assert stage._resolve_n_cpu(4) == 4   # 16 // 4, unchanged by the clamp

    def test_resolve_n_cpu_sequential_uses_all(self, monkeypatch):
        import packages.epicycle_scaledseg.utils.resources as _res
        monkeypatch.setattr(_res, "get_allocated_cpus", lambda: 16)
        stage = ReconstructionStage(mode="sequential")
        assert stage._resolve_n_cpu(0) == 16  # sequential ignores n_gpus entirely


class TestReconstructionStageValidation:
    def test_validates_seg_done_marker(self, tmp_path):
        tile_dir = tmp_path / "tile_0"
        tile_dir.mkdir()

        seg_dir = tmp_path / "stage1_segmentation" / "tile_0"
        seg_dir.mkdir(parents=True)
        # No SEGMENTATION_DONE marker

        stage = ReconstructionStage()
        assert stage.validate_inputs([tile_dir]) is False

    def test_validates_seg_dir_exists(self, tmp_path):
        tile_dir = tmp_path / "tile_0"
        tile_dir.mkdir()
        # No stage1_segmentation directory at all

        stage = ReconstructionStage()
        assert stage.validate_inputs([tile_dir]) is False

    def test_passes_when_seg_complete(self, tmp_path):
        tile_dir = tmp_path / "tile_0"
        tile_dir.mkdir()
        _make_seg_tile(tmp_path, "tile_0")

        stage = ReconstructionStage()
        assert stage.validate_inputs([tile_dir]) is True

    def test_custom_seg_root(self, tmp_path):
        tile_dir = tmp_path / "tile_0"
        tile_dir.mkdir()
        seg_root = tmp_path / "my_seg"
        _make_seg_tile(tmp_path, "tile_0", seg_root=str(seg_root))

        stage = ReconstructionStage(seg_root=str(seg_root))
        assert stage.validate_inputs([tile_dir]) is True


class TestReconstructionStageSkipCompleted:
    def test_all_completed_returns_all_success(self, tmp_path):
        """If all tiles already have mask.tif, skip reconstruction."""
        tile_dir = tmp_path / "tile_0"
        tile_dir.mkdir()
        _make_seg_tile(tmp_path, "tile_0")
        _make_recon_output(tmp_path, "tile_0")

        stage = ReconstructionStage()
        result = stage.run([tile_dir], n_gpus=1, log_dir=tmp_path / "logs")
        assert result.n_success == 1
        assert result.n_failed == 0


class TestReconstructionStageDirectoryResolution:
    def test_default_seg_dir(self, tmp_path):
        stage = ReconstructionStage()
        tile_dir = tmp_path / "tile_0"
        assert stage._seg_dir(tile_dir) == tmp_path / "stage1_segmentation" / "tile_0"

    def test_custom_seg_root(self, tmp_path):
        seg_root = tmp_path / "my_seg"
        seg_root.mkdir()
        stage = ReconstructionStage(seg_root=str(seg_root))
        tile_dir = tmp_path / "tile_0"
        assert stage._seg_dir(tile_dir) == seg_root / "tile_0"

    def test_default_out_dir(self, tmp_path):
        stage = ReconstructionStage()
        tile_dir = tmp_path / "tile_0"
        assert stage._out_dir(tile_dir) == tmp_path / "stage2_reconstruction" / "tile_0"

    def test_custom_out_root(self, tmp_path):
        out_root = tmp_path / "my_recon"
        out_root.mkdir()
        stage = ReconstructionStage(out_root=str(out_root))
        tile_dir = tmp_path / "tile_0"
        assert stage._out_dir(tile_dir) == out_root / "tile_0"


class TestReconstructionStageWorkerKwargs:
    def test_build_worker_kwargs_explicit_cpu(self):
        stage = ReconstructionStage(
            save_prob3d=False,
            binary_kernel_size=5,
            n_cpu_per_worker=16,
        )
        stage._resolved_n_cpu = 16
        kw = stage._build_worker_kwargs()
        assert kw["save_prob3d"] is False
        assert kw["binary_kernel_size"] == 5
        assert kw["n_cpu"] == 16
        assert "out_root" in kw
        assert "reconstruct_params" in kw

    def test_pre_smooth_kwargs_present(self):
        stage = ReconstructionStage(
            pre_smooth_flows=True,
            pre_smooth_probs=True,
            pre_smooth_sigma_perp=5.0,
            pre_smooth_sigma_inplane=0.5,
            flow_mask_threshold=0.3,
            flow_soft_weighting=False,
        )
        stage._resolved_n_cpu = 8
        kw = stage._build_worker_kwargs()
        assert kw["pre_smooth_flows"] is True
        assert kw["pre_smooth_probs"] is True
        assert kw["pre_smooth_sigma_perp"] == 5.0
        assert kw["pre_smooth_sigma_inplane"] == 0.5
        assert kw["flow_mask_threshold"] == 0.3
        assert kw["flow_soft_weighting"] is False

    def test_pre_smooth_kwargs_defaults(self):
        stage = ReconstructionStage()
        stage._resolved_n_cpu = 8
        kw = stage._build_worker_kwargs()
        assert kw["pre_smooth_flows"] is False
        assert kw["pre_smooth_probs"] is False


class TestReconstructionStageCpuResolution:
    def test_explicit_override(self):
        stage = ReconstructionStage(n_cpu_per_worker=32)
        assert stage._resolve_n_cpu(n_gpus=8) == 32

    def test_sequential_uses_all_cores(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
        stage = ReconstructionStage(mode="sequential")
        assert stage._resolve_n_cpu(n_gpus=8) == 64

    def test_parallel_splits_cores(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
        stage = ReconstructionStage(mode="parallel")
        assert stage._resolve_n_cpu(n_gpus=8) == 8

    def test_slurm_job_cpus_fallback(self, monkeypatch):
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "48")
        stage = ReconstructionStage(mode="sequential")
        assert stage._resolve_n_cpu(n_gpus=8) == 48


class TestReconstructionStageRepr:
    def test_repr(self):
        stage = ReconstructionStage()
        r = repr(stage)
        assert "reconstruction" in r
