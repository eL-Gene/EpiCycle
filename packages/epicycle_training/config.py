"""
Configuration dataclasses for EpiCycle Training.

Copied from epicycle-training/epicycle_local/config.py to consolidate
into the EpiCycle monorepo.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class TrainingConfig:
    """Configuration for Cellpose model training.

    Attributes:
        n_epochs: Number of training epochs
        learning_rate: Initial learning rate
        batch_size: Training batch size
        weight_decay: L2 regularization weight
        momentum: SGD momentum
        min_train_masks: Minimum masks required per image
        normalize: Whether to normalize input images
        rescale: Whether to rescale images (True learns scale invariance)
        SGD: Use SGD optimizer (vs Adam)
        compute_flows: Compute flow fields during training
        channels: Channel configuration [cytoplasm, nucleus] e.g. [2, 1]
        seed: Random seed for reproducibility
    """
    n_epochs: int = 500
    learning_rate: float = 1e-2
    batch_size: int = 16
    weight_decay: float = 1e-5
    momentum: float = 0.9
    min_train_masks: int = 0
    normalize: bool = True
    rescale: bool = True
    SGD: bool = True
    compute_flows: bool = True
    channels: List[int] = field(default_factory=lambda: [2, 1])
    seed: Optional[int] = None

    def get_lr_for_round(self, round_number: int, use_dynamic: bool = False) -> float:
        """Get learning rate, optionally with dynamic schedule."""
        if use_dynamic:
            return self.learning_rate if round_number < 5 else self.learning_rate / 10
        return self.learning_rate


@dataclass
class SegmentationConfig:
    """Configuration for Cellpose inference.

    Attributes:
        channels: Channel configuration [cytoplasm, nucleus]
        normalize: Whether to normalize input
        rescale: Whether to rescale (use same as training)
        cellprob_threshold: Cell probability threshold
        flow_threshold: Flow error threshold
        diameter: Cell diameter (None for auto)
        batch_size: Inference batch size
        compute_masks: Whether to compute instance masks (needed for indirect method)
    """
    channels: List[int] = field(default_factory=lambda: [2, 1])
    normalize: bool = True
    rescale: bool = True
    cellprob_threshold: float = 0.0
    flow_threshold: float = 0.4
    diameter: Optional[float] = None
    batch_size: int = 128
    compute_masks: bool = True
