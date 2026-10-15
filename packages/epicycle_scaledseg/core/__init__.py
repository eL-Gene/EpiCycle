"""Core pipeline stages for EpiCycle ScaledSegmentation.

Re-exports the pipeline primitives + every stage so callers can
``from packages.epicycle_scaledseg.core import EpiCyclePipeline, IoUStitchStage, ...``. See the
stage module docstrings for constructor contracts and the on-disk output schemas.
"""

from .pipeline import Stage, StageResult, PipelineResult, EpiCyclePipeline
from .tiling import TilingStage
from .segmentation import SegmentationStage
from .reconstruction import ReconstructionStage
# Mode A (coarse): downsample-then-resegment stitch
from .assembly import AssemblyStage
from .global_reconstruction import GlobalReconstructionStage
# Mode B (full): full-resolution instance-preserving stitch + post-stitch cleanup
from .iou_stitch_stage import IoUStitchStage
from .postprocess_stage import PostprocessStage
# Feature extraction (Wave 1): per-cell QC / morphometry on the cleaned global labels
from .feature_stage import FeatureStage

__all__ = [
    "Stage", "StageResult", "PipelineResult", "EpiCyclePipeline",
    "TilingStage", "SegmentationStage", "ReconstructionStage",
    "AssemblyStage", "GlobalReconstructionStage",
    "IoUStitchStage", "PostprocessStage", "FeatureStage",
]
