from packages.epicycle_domain import (
    ReconstructionJobSpec as CanonicalReconstructionJobSpec,
    RoundMetadata as CanonicalRoundMetadata,
)
from packages.epicycle_io import (
    FileArtifactRegistry as CanonicalFileArtifactRegistry,
    HITLPersistenceService as CanonicalHITLPersistenceService,
)

from napari_epicycle.domain import ReconstructionJobSpec, RoundMetadata
from napari_epicycle.io_registry import FileArtifactRegistry, HITLPersistenceService


def test_domain_shim_exports_canonical_contracts():
    assert RoundMetadata is CanonicalRoundMetadata
    assert ReconstructionJobSpec is CanonicalReconstructionJobSpec


def test_io_shim_exports_canonical_services():
    assert FileArtifactRegistry is CanonicalFileArtifactRegistry
    assert HITLPersistenceService is CanonicalHITLPersistenceService


def test_io_shim_registry_roundtrip_still_works(tmp_path):
    registry = FileArtifactRegistry()
    spec = ReconstructionJobSpec(
        schema_version="1.0",
        input_predictions=["predictions/r01/xy"],
        fusion_params={"method": "placeholder"},
        output_target="artifacts/reconstruction",
        run_mode="local",
    )
    path = registry.save_reconstruction_job_spec(str(tmp_path), spec)
    assert path.exists()
