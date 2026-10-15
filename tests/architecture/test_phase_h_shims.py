from packages.epicycle_reconstruction import (
    ReconstructionPipelineService as CanonicalReconstructionPipelineService,
    ReconstructionRunStore as CanonicalReconstructionRunStore,
)
from packages.epicycle_training import (
    TrainingManager as CanonicalTrainingManager,
    build_epicycle_inference_backend as canonical_build_epicycle_inference_backend,
    build_epicycle_training_backend as canonical_build_epicycle_training_backend,
)

from napari_epicycle.fusion_reconstruction.logic import ReconstructionPipelineService
from napari_epicycle.fusion_reconstruction.io import ReconstructionRunStore
from napari_epicycle.hitl_training_annotation.controller.training_backend_adapter import (
    build_epicycle_inference_backend,
    build_epicycle_training_backend,
)
from napari_epicycle.hitl_training_annotation.controller.training_manager import TrainingManager


def test_training_shims_export_canonical_services():
    assert TrainingManager is CanonicalTrainingManager
    assert build_epicycle_training_backend is canonical_build_epicycle_training_backend
    assert build_epicycle_inference_backend is canonical_build_epicycle_inference_backend


def test_reconstruction_shims_export_canonical_services():
    assert ReconstructionPipelineService is CanonicalReconstructionPipelineService
    assert ReconstructionRunStore is CanonicalReconstructionRunStore


def test_reconstruction_new_exports():
    from packages.epicycle_reconstruction import (
        ReconstructionConfig,
        ReconstructionResult,
        segment_views,
        get_default_reconstruction_config,
    )
    assert ReconstructionConfig is not None
    assert ReconstructionResult is not None
    assert callable(segment_views)
    assert callable(get_default_reconstruction_config)

