"""Phase B.3 + B.3.1 (2026-05-25) — Cellpose training-backend adapter
integration tests.

These tests close the integration-test gap surfaced by B.3 audit 3
MEDIUM finding: the `build_step2_model_manifest` data-layer tests
exercised the manifest builder directly with hand-crafted string
values; they NEVER invoked the adapter's `_backend()` function.
Result: the CRITICAL `datetime.now(timezone)` bug (audit 1) went
undetected by tests despite "all green" gate.

This file exercises `_backend()` via stubbed CellposeTrainer +
tifffile so the timing capture, env capture, slice-label
reconstruction, and result-dict shape are all verified.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def stub_cellpose_modules(monkeypatch):
    """Insert stub `cellpose.models` + `cellpose.train` modules so the
    adapter's `import cellpose` + version-check at function entry
    succeeds.  Also stubs `cellpose` top-level with a `version` attr
    so `env_cellpose_version` capture works.
    """
    import types as _types
    fake_cellpose = _types.ModuleType("cellpose")
    fake_cellpose.version = "3.99.fake"
    fake_models = _types.ModuleType("cellpose.models")
    fake_train = _types.ModuleType("cellpose.train")
    monkeypatch.setitem(sys.modules, "cellpose", fake_cellpose)
    monkeypatch.setitem(sys.modules, "cellpose.models", fake_models)
    monkeypatch.setitem(sys.modules, "cellpose.train", fake_train)
    return fake_cellpose


@pytest.fixture
def stub_trainer(monkeypatch):
    """Stub `CellposeTrainer` so `_backend()` can be called without
    real Cellpose.  Captures fit() args + simulates train_loss/val_loss
    arrays + model_path.
    """
    class _StubTrainer:
        def __init__(self, config=None, use_gpu=False):
            self.config = config
            self.use_gpu = use_gpu
            self.train_loss = np.array([0.9, 0.7, 0.5])
            self.val_loss = np.array([0.95, 0.75])
            self.model_path = None
            self.fit_calls = []

        def fit(self, *, train_data, train_labels, model_name, save_path,
                pretrained_model=None, base_model=None, epoch_callback=None):
            # Simulate Cellpose writing weights to `save_path/models/{name}`
            models_subdir = save_path / "models"
            models_subdir.mkdir(parents=True, exist_ok=True)
            weights = models_subdir / model_name
            weights.write_bytes(b"fake weights")
            (models_subdir / f"{model_name}_train_loss.npy").write_bytes(b"x")
            (models_subdir / f"{model_name}_val_loss.npy").write_bytes(b"x")
            (models_subdir / f"{model_name}_training_config.json").write_text("{}")
            self.model_path = weights
            self.fit_calls.append({
                "model_name": model_name,
                "save_path": str(save_path),
                "pretrained_model": pretrained_model,
                "base_model": base_model,
            })

    from packages.epicycle_training import training_backend_adapter as _adapter
    monkeypatch.setattr(_adapter, "CellposeTrainer", _StubTrainer)
    return _StubTrainer


@pytest.fixture
def planted_samples(tmp_path):
    """Plant 3 training samples on disk (img + mask TIFFs) — adapter's
    `tifffile.imread` needs real files."""
    import tifffile
    accepted = tmp_path / "ws" / "accepted"
    accepted.mkdir(parents=True)
    samples = []
    for idx, view in [(5, "XY"), (42, "XY"), (103, "XZ")]:
        stack_id = "stkA"
        img_p = accepted / f"{stack_id}_{view}_z{idx:04d}_img.tif"
        mask_p = accepted / f"{stack_id}_{view}_z{idx:04d}_mask.tif"
        tifffile.imwrite(str(img_p),
                         np.random.rand(2, 16, 16).astype(np.float32))
        tifffile.imwrite(str(mask_p),
                         np.ones((16, 16), dtype=np.uint16))
        samples.append({
            "stack_id": stack_id,
            "view": view,
            "slice_index": idx,
            "img_uri": str(img_p),
            "mask_uri": str(mask_p),
        })
    return samples


class TestB31BackendTimingCapture:
    """B.3.1 CRITICAL regression test (datetime.now(timezone) bug).
    Pre-fix this would raise `TypeError: tzinfo argument must be None
    or of a tzinfo subclass, not type 'type'`."""

    def test_backend_completes_without_timing_typeerror(
        self, tmp_path, stub_cellpose_modules, stub_trainer, planted_samples
    ):
        from packages.epicycle_training.training_backend_adapter import (
            build_epicycle_training_backend,
        )
        backend = build_epicycle_training_backend()
        assert backend is not None, "stub cellpose modules should enable backend"

        save_path = tmp_path / "ws" / "model"
        result = backend(
            round_id="round_00",
            config={"epochs": 3, "batch_size": 2, "device": "CPU"},
            working_dir=str(tmp_path / "ws"),
            training_samples=planted_samples,
            view_type="spatial",
            save_path_override=str(save_path),
            weights_filename="testrun_lateral_round_00",
        )

        # Pre-fix this assertion never ran — the function raised inside
        # `_dt.now(_tz).isoformat()` at the very top of the body.
        # Post-fix: result dict is populated.
        assert isinstance(result, dict)
        assert "timing_started_at" in result
        assert "timing_finished_at" in result
        assert result["timing_started_at"] is not None
        assert result["timing_finished_at"] is not None
        # Duration is a small float ≥ 0
        assert isinstance(result["timing_duration_seconds"], float)
        assert result["timing_duration_seconds"] >= 0


class TestB31AdapterReproducibilityFields:
    """B.3.1 audit 3 MEDIUM gap fix: integration test for the adapter's
    schema 2.1 field population."""

    def test_all_b3_fields_in_result(
        self, tmp_path, stub_cellpose_modules, stub_trainer, planted_samples
    ):
        from packages.epicycle_training.training_backend_adapter import (
            build_epicycle_training_backend,
        )
        backend = build_epicycle_training_backend()
        save_path = tmp_path / "ws" / "model"
        result = backend(
            round_id="round_00",
            config={"epochs": 3, "device": "CPU"},
            working_dir=str(tmp_path / "ws"),
            training_samples=planted_samples,
            view_type="spatial",
            save_path_override=str(save_path),
            weights_filename="testrun_lateral_round_00",
        )

        # All 10 B.3 keys present in result dict
        for key in (
            "train_pair_labels", "train_pair_uris",
            "timing_started_at", "timing_finished_at",
            "timing_duration_seconds",
            "env_cellpose_version", "env_torch_version",
            "env_python_version", "env_device", "env_platform",
        ):
            assert key in result, (
                f"Adapter result dict missing schema 2.1 key {key!r}; "
                f"this would silently default to None in the manifest "
                f"via result.get() in the controller."
            )

    def test_train_pair_labels_use_z_prefix(
        self, tmp_path, stub_cellpose_modules, stub_trainer, planted_samples
    ):
        """B.3.1 audit 1 HIGH fix: slice-label canonical format is
        `{stack}_{VIEW}_z{idx:04d}` (z-prefix matches disk filename).
        Pre-fix wrote `{stack}_{VIEW}_{idx:04d}` (no z) — provenance
        label didn't match the on-disk file."""
        from packages.epicycle_training.training_backend_adapter import (
            build_epicycle_training_backend,
        )
        backend = build_epicycle_training_backend()
        save_path = tmp_path / "ws" / "model"
        result = backend(
            round_id="round_00",
            config={"epochs": 3, "device": "CPU"},
            working_dir=str(tmp_path / "ws"),
            training_samples=planted_samples,
            view_type="spatial",
            save_path_override=str(save_path),
            weights_filename="testrun_lateral_round_00",
        )

        labels = result["train_pair_labels"]
        assert len(labels) == 3
        # Each label uses z-prefix matching the on-disk filename format
        assert "stkA_XY_z0005" in labels
        assert "stkA_XY_z0042" in labels
        assert "stkA_XZ_z0103" in labels

    def test_train_pair_uris_are_basenames_not_absolute(
        self, tmp_path, stub_cellpose_modules, stub_trainer, planted_samples
    ):
        """B.3.1 audit 3 HIGH fix: URIs stored as filenames (relative
        to `accepted/`), NOT absolute paths.  Pre-fix absolute paths
        would go stale when workspace moves."""
        from packages.epicycle_training.training_backend_adapter import (
            build_epicycle_training_backend,
        )
        backend = build_epicycle_training_backend()
        save_path = tmp_path / "ws" / "model"
        result = backend(
            round_id="round_00",
            config={"epochs": 3, "device": "CPU"},
            working_dir=str(tmp_path / "ws"),
            training_samples=planted_samples,
            view_type="spatial",
            save_path_override=str(save_path),
            weights_filename="testrun_lateral_round_00",
        )

        uris = result["train_pair_uris"]
        for entry in uris:
            img = entry["img_uri"]
            mask = entry["mask_uri"]
            # No drive letter, no leading slash, no parent dirs
            assert "/" not in img and "\\" not in img, \
                f"img_uri must be basename only; got {img!r}"
            assert "/" not in mask and "\\" not in mask, \
                f"mask_uri must be basename only; got {mask!r}"
            assert img.endswith("_img.tif")
            assert mask.endswith("_mask.tif")


class TestB31AdapterEnvCapture:
    """B.3.1 audit 1 LOW fix: env_device should be None (not "cpu")
    when torch import fails — that's honest provenance."""

    def test_cellpose_version_captured(
        self, tmp_path, stub_cellpose_modules, stub_trainer, planted_samples
    ):
        from packages.epicycle_training.training_backend_adapter import (
            build_epicycle_training_backend,
        )
        backend = build_epicycle_training_backend()
        save_path = tmp_path / "ws" / "model"
        result = backend(
            round_id="round_00",
            config={"epochs": 3, "device": "CPU"},
            working_dir=str(tmp_path / "ws"),
            training_samples=planted_samples,
            view_type="spatial",
            save_path_override=str(save_path),
            weights_filename="testrun_lateral_round_00",
        )
        assert result["env_cellpose_version"] == "3.99.fake"

    def test_python_version_format(
        self, tmp_path, stub_cellpose_modules, stub_trainer, planted_samples
    ):
        from packages.epicycle_training.training_backend_adapter import (
            build_epicycle_training_backend,
        )
        backend = build_epicycle_training_backend()
        save_path = tmp_path / "ws" / "model"
        result = backend(
            round_id="round_00",
            config={"epochs": 3, "device": "CPU"},
            working_dir=str(tmp_path / "ws"),
            training_samples=planted_samples,
            view_type="spatial",
            save_path_override=str(save_path),
            weights_filename="testrun_lateral_round_00",
        )
        # Format: "X.Y.Z" (major.minor.patch from sys.version_info[:3])
        pyv = result["env_python_version"]
        parts = pyv.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts), (
            f"env_python_version should be X.Y.Z digits; got {pyv!r}"
        )
