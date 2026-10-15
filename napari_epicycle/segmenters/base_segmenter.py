from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple

class BaseSegmenter(ABC):
    @abstractmethod
    def segment(self, image: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Abstract method to segment a multi-channel 2D image.

        Args:
            image (np.ndarray): Input image with shape (C, H, W)
            config (dict): Configuration parameters

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - masks: segmentation labels (H, W)
                - flows: flow predictions (2, H, W)
                - cell_probs: cell probability map (H, W)
        """
        pass
