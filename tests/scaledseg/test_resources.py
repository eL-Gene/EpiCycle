"""Tests for epicycle.utils.resources module."""

import os
import pytest
from packages.epicycle_scaledseg.utils.resources import (
    get_allocated_cpus,
    get_allocated_gpus,
    estimate_tile_peak_ram_gb,
    check_reconstruction_memory,
    _USEGMENT3D_OVERHEAD,
)


class TestGetAllocatedCpus:
    def test_slurm_cpus_per_task(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "16")
        assert get_allocated_cpus() == 16

    def test_slurm_job_cpus_per_node(self, monkeypatch):
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        monkeypatch.setenv("SLURM_JOB_CPUS_PER_NODE", "64")
        assert get_allocated_cpus() == 64

    def test_falls_back_to_os(self, monkeypatch):
        monkeypatch.delenv("SLURM_CPUS_PER_TASK", raising=False)
        monkeypatch.delenv("SLURM_JOB_CPUS_PER_NODE", raising=False)
        result = get_allocated_cpus()
        assert result >= 1

    def test_returns_int(self, monkeypatch):
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "32")
        assert isinstance(get_allocated_cpus(), int)


class TestGetAllocatedGpus:
    def test_slurm_gpus_on_node(self, monkeypatch):
        monkeypatch.setenv("SLURM_GPUS_ON_NODE", "8")
        assert get_allocated_gpus() == 8

    def test_cuda_visible_devices(self, monkeypatch):
        monkeypatch.delenv("SLURM_GPUS_ON_NODE", raising=False)
        monkeypatch.delenv("SLURM_GPUS_PER_TASK", raising=False)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1,2,3")
        assert get_allocated_gpus() == 4

    def test_returns_int(self, monkeypatch):
        monkeypatch.setenv("SLURM_GPUS_ON_NODE", "4")
        assert isinstance(get_allocated_gpus(), int)

    @staticmethod
    def _clear(monkeypatch):
        for v in ("SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK", "CUDA_VISIBLE_DEVICES"):
            monkeypatch.delenv(v, raising=False)

    def test_cuda_visible_empty_is_zero(self, monkeypatch):
        """CUDA_VISIBLE_DEVICES="" means hide all GPUs -> 0 (NOT len([""])==1)."""
        self._clear(monkeypatch)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        assert get_allocated_gpus() == 0

    def test_cuda_visible_minus_one_is_zero(self, monkeypatch):
        """CUDA_VISIBLE_DEVICES="-1" means hide all GPUs -> 0 (NOT 1)."""
        self._clear(monkeypatch)
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
        assert get_allocated_gpus() == 0

    def test_no_signals_torch_zero_is_honest_zero(self, monkeypatch):
        """No SLURM/CVD + torch sees no CUDA -> 0, not a coerced 1 (CPU-only fix)."""
        self._clear(monkeypatch)
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
        assert get_allocated_gpus() == 0

    def test_no_signals_torch_one_no_regression(self, monkeypatch):
        """A real 1-GPU box (no SLURM/CVD) still reports 1 via torch."""
        self._clear(monkeypatch)
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
        assert get_allocated_gpus() == 1


class TestEstimateTilePeakRam:
    def test_small_tile(self):
        """512×512×100 tile should be well under 10 GB."""
        gb = estimate_tile_peak_ram_gb(100, 512, 512)
        assert 0 < gb < 10

    def test_large_tile(self):
        """1920×1920×1886 tile ≈ 544 GB at the 1.5x overhead factor."""
        gb = estimate_tile_peak_ram_gb(1886, 1920, 1920)
        assert 450 < gb < 650

    def test_scales_with_size(self):
        small = estimate_tile_peak_ram_gb(100, 512, 512)
        big = estimate_tile_peak_ram_gb(100, 1024, 1024)
        assert big > small * 3  # 4× voxels → ~4× RAM

    def test_returns_float(self):
        assert isinstance(estimate_tile_peak_ram_gb(10, 64, 64), float)

    def test_uses_1_5_overhead_factor(self):
        """The estimator must apply the source-audited 1.5x (not the old 1.3x)."""
        assert _USEGMENT3D_OVERHEAD == 1.5
        z, h, w = 100, 256, 256
        voxels = z * h * w
        # 14 float32-equiv arrays x 4 bytes x 1.5 overhead.
        expected = 14 * voxels * 4 * 1.5 / (1024 ** 3)
        assert estimate_tile_peak_ram_gb(z, h, w) == pytest.approx(expected, rel=1e-9)


class TestAvailableRamSource:
    """RAM detection across SLURM allocation styles (--mem / --mem-per-cpu /
    --mem-per-gpu) + the source string. The per-cpu/per-gpu derivations matter on
    GPU clusters (e.g. UW Hyak) where SLURM_MEM_PER_NODE is unset."""

    _ALL = ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU",
            "SLURM_CPUS_PER_TASK", "SLURM_JOB_CPUS_PER_NODE",
            "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_TASK")

    def _clear(self, mp):
        for v in self._ALL:
            mp.delenv(v, raising=False)

    def test_mem_per_node(self, monkeypatch):
        from packages.epicycle_scaledseg.utils.resources import available_ram_gb_and_source
        self._clear(monkeypatch)
        monkeypatch.setenv("SLURM_MEM_PER_NODE", str(720 * 1024))   # MB
        gb, src = available_ram_gb_and_source()
        assert abs(gb - 720) < 1 and src == "SLURM_MEM_PER_NODE"

    def test_mem_per_cpu_times_cpus(self, monkeypatch):
        from packages.epicycle_scaledseg.utils.resources import available_ram_gb_and_source
        self._clear(monkeypatch)
        monkeypatch.setenv("SLURM_MEM_PER_CPU", str(10 * 1024))     # 10 GB/cpu (MB)
        monkeypatch.setenv("SLURM_CPUS_PER_TASK", "64")
        gb, src = available_ram_gb_and_source()
        assert abs(gb - 640) < 1 and src == "SLURM_MEM_PER_CPU*SLURM_CPUS_PER_TASK"

    def test_mem_per_gpu_times_gpus(self, monkeypatch):
        from packages.epicycle_scaledseg.utils.resources import available_ram_gb_and_source
        self._clear(monkeypatch)
        monkeypatch.setenv("SLURM_MEM_PER_GPU", str(240 * 1024))    # 240 GB/gpu (MB)
        monkeypatch.setenv("SLURM_GPUS_ON_NODE", "3")
        gb, src = available_ram_gb_and_source()
        assert abs(gb - 720) < 1 and src == "SLURM_MEM_PER_GPU*SLURM_GPUS_ON_NODE"

    def test_per_node_takes_precedence(self, monkeypatch):
        from packages.epicycle_scaledseg.utils.resources import available_ram_gb_and_source
        self._clear(monkeypatch)
        monkeypatch.setenv("SLURM_MEM_PER_NODE", str(500 * 1024))
        monkeypatch.setenv("SLURM_MEM_PER_GPU", str(240 * 1024))
        monkeypatch.setenv("SLURM_GPUS_ON_NODE", "3")
        _, src = available_ram_gb_and_source()
        assert src == "SLURM_MEM_PER_NODE"   # explicit node value wins over per-gpu

    def test_env_int_tolerates_non_int(self, monkeypatch):
        from packages.epicycle_scaledseg.utils.resources import _env_int
        monkeypatch.setenv("X_CPUS", "32(x2)")   # SLURM multi-node format
        assert _env_int("X_CPUS") is None
        monkeypatch.setenv("X_CPUS", "64")
        assert _env_int("X_CPUS") == 64

    def test_get_available_ram_gb_wrapper(self, monkeypatch):
        from packages.epicycle_scaledseg.utils.resources import get_available_ram_gb
        self._clear(monkeypatch)
        monkeypatch.setenv("SLURM_MEM_PER_NODE", str(256 * 1024))
        assert abs(get_available_ram_gb() - 256) < 1


class TestCheckReconstructionMemory:
    def test_returns_true_when_fits(self, monkeypatch):
        """Small tile should pass on any system."""
        # Force SLURM_MEM_PER_NODE to 500 GB
        monkeypatch.setenv("SLURM_MEM_PER_NODE", str(500 * 1024))
        assert check_reconstruction_memory(100, 512, 512) is True

    def test_returns_false_when_exceeds(self, monkeypatch):
        """1920×1920×1886 with 240 GB limit should warn."""
        monkeypatch.setenv("SLURM_MEM_PER_NODE", str(240 * 1024))
        assert check_reconstruction_memory(1886, 1920, 1920) is False

    def test_concurrent_multiplier(self, monkeypatch):
        """Multiple concurrent tiles should multiply the estimate."""
        monkeypatch.setenv("SLURM_MEM_PER_NODE", str(500 * 1024))
        # Single tile fits in 500 GB
        assert check_reconstruction_memory(500, 512, 512, n_concurrent=1) is True
        # But many concurrent tiles may not
        result = check_reconstruction_memory(500, 512, 512, n_concurrent=100)
        assert result is False
