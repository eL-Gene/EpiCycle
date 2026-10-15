"""Tests for epicycle.core.segmentation module.

Tests strategy resolution logic and SegmentationStage configuration.
Does NOT test actual CellPose inference (requires GPU + models).
"""

import inspect

import pytest
from pathlib import Path

from packages.epicycle_scaledseg.core.segmentation import (
    _resolve_strategy,
    resolve_channels,
    SegmentationStage,
    _seg_worker_threads,
)
import packages.epicycle_scaledseg.core.segmentation as _seg_mod


class TestSegWorkerThreads:
    """Per-worker CPU-thread budget — prevents OMP/MKL oversubscription.

    Replaces the old hardcoded ``OMP_NUM_THREADS=8`` (which ignored both core count
    and how many workers run at once → N_workers x 8 threads on a small box).
    """

    def test_splits_cores_among_workers(self):
        assert _seg_worker_threads(64, 3) == 21   # 3-GPU fixed: 1 triplet x 3 views
        assert _seg_worker_threads(64, 6) == 10   # 6-GPU fixed: 2 triplets x 3
        assert _seg_worker_threads(12, 6) == 2
        assert _seg_worker_threads(64, 1) == 64   # 1-GPU round_robin: 1 busy worker
        assert _seg_worker_threads(64, 2) == 32   # 2-GPU round_robin: 2 busy workers

    def test_floors_at_one(self):
        assert _seg_worker_threads(4, 8) == 1     # more workers than cores

    def test_guards_zero_workers(self):
        assert _seg_worker_threads(8, 0) == 8     # no ZeroDivisionError

    def test_workers_use_passed_thread_count_not_hardcoded(self):
        for fn in (_seg_mod._seg_view_worker, _seg_mod._seg_gpu_worker):
            src = inspect.getsource(fn)
            assert "str(n_threads)" in src, f"{fn.__name__} must use n_threads"
            assert '= "8"' not in src, f"{fn.__name__} still hardcodes 8 threads"

    def test_workers_honor_use_gpu_flag(self):
        # CellposeModel must be built with gpu=use_gpu, never a hardcoded True.
        for fn in (_seg_mod._seg_view_worker, _seg_mod._segment_single_tile):
            src = inspect.getsource(fn)
            assert "gpu=use_gpu" in src, f"{fn.__name__} must pass gpu=use_gpu"
            assert "gpu=True" not in src, f"{fn.__name__} still hardcodes gpu=True"


class TestSegDispatch:
    """Dispatch-level coverage: spawn counts, the n_threads/use_gpu values the
    workers actually receive, and the n_gpus=0 / fixed-below-3 guards. A fake
    get_context records spawns so nothing real is launched (no GPU/cellpose)."""

    @staticmethod
    def _recorder():
        records = []

        class _P:
            def __init__(self, target=None, args=None):
                records.append({"target": getattr(target, "__name__", None), "args": args})

            def start(self):
                pass

            def join(self):
                pass

        class _Ctx:
            def Process(self, target=None, args=None):
                return _P(target=target, args=args)

        return (lambda method: _Ctx()), records

    @staticmethod
    def _tiles(tmp_path, n):
        out = []
        for i in range(n):
            d = tmp_path / f"tile_{i}"
            d.mkdir()
            out.append(d)
        return out

    def _stage(self, tmp_path, **kw):
        return SegmentationStage(
            lateral_model="lat.pth", axial_model="ax.pth",
            out_root=str(tmp_path / "seg_out"), **kw,
        )

    def test_3gpu_fixed_spawns_3_views_with_pinned_threads(self, tmp_path, monkeypatch):
        fake_ctx, records = self._recorder()
        monkeypatch.setattr(_seg_mod, "get_context", fake_ctx)
        monkeypatch.setattr(_seg_mod, "get_allocated_cpus", lambda: 12)
        self._stage(tmp_path).run(self._tiles(tmp_path, 1), n_gpus=3, log_dir=tmp_path)
        assert len(records) == 3                          # 1 triplet x 3 views
        assert all(r["target"] == "_seg_view_worker" for r in records)
        for r in records:
            n_threads, use_gpu = r["args"][-2], r["args"][-1]
            assert use_gpu is True
            assert n_threads == 12 // 3                    # cores // (len(triplets)*3)

    def test_1gpu_round_robin_spawns_one_with_all_cores(self, tmp_path, monkeypatch):
        fake_ctx, records = self._recorder()
        monkeypatch.setattr(_seg_mod, "get_context", fake_ctx)
        monkeypatch.setattr(_seg_mod, "get_allocated_cpus", lambda: 12)
        self._stage(tmp_path).run(self._tiles(tmp_path, 2), n_gpus=1, log_dir=tmp_path)
        assert len(records) == 1                          # 1 busy GPU -> 1 worker
        assert records[0]["target"] == "_seg_gpu_worker"
        n_threads, use_gpu = records[0]["args"][-2], records[0]["args"][-1]
        assert use_gpu is True
        assert n_threads == 12                            # cores // 1 busy worker

    def test_0gpu_spawns_nothing_and_fails_clearly(self, tmp_path, monkeypatch):
        """The honest-0 GPU return must NOT silently no-op (the B1 cliff)."""
        fake_ctx, records = self._recorder()
        monkeypatch.setattr(_seg_mod, "get_context", fake_ctx)
        result = self._stage(tmp_path).run(self._tiles(tmp_path, 2), n_gpus=0, log_dir=tmp_path)
        assert len(records) == 0                          # no silent zero-worker spawn
        assert result.n_failed == 2 and result.n_success == 0
        assert all("no GPU" in f["error"] for f in result.failures)

    def test_explicit_fixed_below_3_falls_back_to_round_robin(self, tmp_path, monkeypatch):
        """gpu_strategy='fixed' on <3 GPUs must fall back, not crash (the L4 cliff)."""
        fake_ctx, records = self._recorder()
        monkeypatch.setattr(_seg_mod, "get_context", fake_ctx)
        monkeypatch.setattr(_seg_mod, "get_allocated_cpus", lambda: 8)
        stage = self._stage(tmp_path, gpu_strategy="fixed")
        stage.run(self._tiles(tmp_path, 2), n_gpus=2, log_dir=tmp_path)
        assert len(records) == 2                          # 2 busy GPUs via round_robin
        assert all(r["target"] == "_seg_gpu_worker" for r in records)


class TestResolveChannels:
    """Grayscale / single-channel channel auto-derivation (follow-up #8)."""

    def test_single_channel_forces_grayscale(self):
        # The default [signal, nucleus] indices are invalid for a 1-channel tile.
        assert resolve_channels([2, 1], 1) == [0, 0]

    def test_zero_channels_defensive(self):
        assert resolve_channels([2, 1], 0) == [0, 0]

    def test_two_channels_unchanged(self):
        assert resolve_channels([2, 1], 2) == [2, 1]

    def test_three_channels_unchanged(self):
        assert resolve_channels([2, 1], 3) == [2, 1]

    def test_already_grayscale_single_channel(self):
        assert resolve_channels([0, 0], 1) == [0, 0]

    def test_configured_grayscale_on_multichannel_respected(self):
        # A user can grayscale-segment one channel of a multi-channel stack.
        assert resolve_channels([0, 0], 3) == [0, 0]

    def test_worker_uses_resolved_channels(self):
        """The view worker must pass the RESOLVED channels to model.eval, not the
        raw configured channels — else a C=1 tile still gets the invalid [2,1]."""
        src = inspect.getsource(_seg_mod._run_view_segmentation)
        assert "resolve_channels(" in src
        assert "channels=eff_channels" in src
        # And it must NOT pass the raw config channels straight to eval.
        assert "channels=segment_params['channels']" not in src


class TestResolveStrategy:
    """Test the auto-fallback logic for GPU strategy selection."""

    def test_auto_uses_fixed_with_3_gpus(self):
        assert _resolve_strategy('auto', 3) == 'fixed'

    def test_auto_uses_fixed_with_6_gpus(self):
        assert _resolve_strategy('auto', 6) == 'fixed'

    def test_auto_uses_fixed_with_8_gpus(self):
        assert _resolve_strategy('auto', 8) == 'fixed'

    def test_auto_uses_round_robin_with_2_gpus(self):
        assert _resolve_strategy('auto', 2) == 'round_robin'

    def test_auto_uses_round_robin_with_1_gpu(self):
        assert _resolve_strategy('auto', 1) == 'round_robin'

    def test_explicit_fixed_always_fixed(self):
        assert _resolve_strategy('fixed', 1) == 'fixed'
        assert _resolve_strategy('fixed', 8) == 'fixed'

    def test_explicit_round_robin_always_round_robin(self):
        assert _resolve_strategy('round_robin', 8) == 'round_robin'
        assert _resolve_strategy('round_robin', 1) == 'round_robin'


class TestSegmentationStageInit:
    def test_default_strategy_is_auto(self, tmp_path):
        # Create dummy model files
        lat = tmp_path / "lat.model"
        lat.touch()
        ax = tmp_path / "ax.model"
        ax.touch()
        stage = SegmentationStage(str(lat), str(ax))
        assert stage.gpu_strategy_name == 'auto'

    def test_custom_strategy(self, tmp_path):
        lat = tmp_path / "lat.model"
        lat.touch()
        ax = tmp_path / "ax.model"
        ax.touch()
        stage = SegmentationStage(str(lat), str(ax), gpu_strategy='round_robin')
        assert stage.gpu_strategy_name == 'round_robin'

    def test_stage_name(self, tmp_path):
        lat = tmp_path / "lat.model"
        lat.touch()
        ax = tmp_path / "ax.model"
        ax.touch()
        stage = SegmentationStage(str(lat), str(ax))
        assert stage.name == 'segmentation'


class TestSegmentationStageValidation:
    def test_validate_missing_lateral_model(self, tmp_path):
        ax = tmp_path / "ax.model"
        ax.touch()
        stage = SegmentationStage("/nonexistent/lat.model", str(ax))
        tile = tmp_path / "tile_0"
        tile.mkdir()
        assert stage.validate_inputs([tile]) is False

    def test_validate_missing_axial_model(self, tmp_path):
        lat = tmp_path / "lat.model"
        lat.touch()
        stage = SegmentationStage(str(lat), "/nonexistent/ax.model")
        tile = tmp_path / "tile_0"
        tile.mkdir()
        assert stage.validate_inputs([tile]) is False

    def test_validate_success(self, tmp_path):
        lat = tmp_path / "lat.model"
        lat.touch()
        ax = tmp_path / "ax.model"
        ax.touch()
        stage = SegmentationStage(str(lat), str(ax))
        tile = tmp_path / "tile_0"
        tile.mkdir()
        assert stage.validate_inputs([tile]) is True


class TestSegmentationStageSkipCompleted:
    def test_all_completed_returns_all_success(self, tmp_path):
        """If all tiles already have SEGMENTATION_DONE, skip work."""
        lat = tmp_path / "lat.model"
        lat.touch()
        ax = tmp_path / "ax.model"
        ax.touch()
        stage = SegmentationStage(str(lat), str(ax))

        tiles = []
        for i in range(3):
            tile = tmp_path / f"tile_{i}"
            tile.mkdir()
            tiles.append(tile)
            # Create stage1 dir with done marker
            s1 = tmp_path / "stage1_segmentation" / f"tile_{i}"
            s1.mkdir(parents=True)
            (s1 / "SEGMENTATION_DONE").touch()

        result = stage.run(tiles, n_gpus=8, log_dir=tmp_path / "logs")
        assert result.n_success == 3
        assert result.n_failed == 0


class TestTripletFormation:
    """Test that the fixed strategy correctly forms GPU triplets."""

    def test_8_gpus_gives_2_triplets(self):
        gpu_ids = list(range(8))
        triplets = [
            tuple(gpu_ids[i:i + 3])
            for i in range(0, len(gpu_ids) - 2, 3)
        ]
        assert triplets == [(0, 1, 2), (3, 4, 5)]

    def test_6_gpus_gives_2_triplets(self):
        gpu_ids = list(range(6))
        triplets = [
            tuple(gpu_ids[i:i + 3])
            for i in range(0, len(gpu_ids) - 2, 3)
        ]
        assert triplets == [(0, 1, 2), (3, 4, 5)]

    def test_9_gpus_gives_3_triplets(self):
        gpu_ids = list(range(9))
        triplets = [
            tuple(gpu_ids[i:i + 3])
            for i in range(0, len(gpu_ids) - 2, 3)
        ]
        assert triplets == [(0, 1, 2), (3, 4, 5), (6, 7, 8)]

    def test_3_gpus_gives_1_triplet(self):
        gpu_ids = list(range(3))
        triplets = [
            tuple(gpu_ids[i:i + 3])
            for i in range(0, len(gpu_ids) - 2, 3)
        ]
        assert triplets == [(0, 1, 2)]

    def test_4_gpus_gives_1_triplet(self):
        """4 GPUs: 1 triplet (0,1,2), GPU 3 idle."""
        gpu_ids = list(range(4))
        triplets = [
            tuple(gpu_ids[i:i + 3])
            for i in range(0, len(gpu_ids) - 2, 3)
        ]
        assert triplets == [(0, 1, 2)]


class TestStage1DirResolution:
    def test_default_stage1_dir(self, tmp_path):
        lat = tmp_path / "lat.model"
        lat.touch()
        ax = tmp_path / "ax.model"
        ax.touch()
        stage = SegmentationStage(str(lat), str(ax))
        tile = tmp_path / "tile_0"
        assert stage._stage1_dir(tile) == tmp_path / "stage1_segmentation" / "tile_0"

    def test_custom_out_root(self, tmp_path):
        lat = tmp_path / "lat.model"
        lat.touch()
        ax = tmp_path / "ax.model"
        ax.touch()
        out_root = tmp_path / "my_seg"
        out_root.mkdir()
        stage = SegmentationStage(str(lat), str(ax), out_root=str(out_root))
        tile = tmp_path / "tile_0"
        assert stage._stage1_dir(tile) == out_root / "tile_0"
