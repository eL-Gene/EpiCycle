from napari_epicycle.sam_assist.sam_segmenter import SAMSegmenter
import numpy as np
from typing import Dict, Tuple


class InteractiveSegmentationManager:
    def __init__(self):
        self.segmenter = None

    def initialize_segmenter(self, model_type: str, device: str = "cuda"):
        self.segmenter = SAMSegmenter(model_type=model_type, device=device)

    def set_image(self, image: np.ndarray):
        if self.segmenter is None:
            raise RuntimeError("SAM Segmenter not initialized.")
        self.segmenter.set_image(image)

    def reset(self):
        if self.segmenter:
            self.segmenter.reset_logits()

    def cleanup_prompt_layers(self, viewer):
        """Safely remove prompt-related layers and disconnect events."""
        for lname in ["sam_points", "sam_boxes", "sam_pred_masks"]:
            if lname in viewer.layers:
                layer = viewer.layers[lname]

                # Try to disconnect events
                try:
                    layer.events.data.disconnect()
                except Exception:
                    pass

                # Hide before remove (vispy safety).
                layer.visible = False
                viewer.layers.remove(layer)
        print("[SAM] Cleaned up prompt layers.")



