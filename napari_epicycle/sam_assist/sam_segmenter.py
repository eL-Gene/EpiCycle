import numpy as np
from segment_anything import sam_model_registry, SamPredictor
import torch
import os
import urllib.request

SAM_MODEL_URLS = {
    "vit_h": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
    "vit_l": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}

class SAMSegmenter:
    def __init__(self, model_type: str, device: str = "cuda", checkpoint_dir: str = "~/.sam_checkpoints"):
        
        self.model_type = model_type
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.checkpoint_dir = os.path.expanduser(checkpoint_dir)
        self.checkpoint_path = self._download_checkpoint(model_type)
        self._load_predictor()

        self.last_mask_logits = None
        self.last_image = None

    def _download_checkpoint(self, model_type: str) -> str:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        url = SAM_MODEL_URLS[model_type]
        path = os.path.join(self.checkpoint_dir, f"sam_{model_type}.pth")
        if not os.path.exists(path):
            print(f"[SAM] Downloading checkpoint: {model_type}")
            urllib.request.urlretrieve(url, path)
        return path

    def _load_predictor(self):
        sam = sam_model_registry[self.model_type](checkpoint=self.checkpoint_path)
        sam.to(self.device)
        self.predictor = SamPredictor(sam)

    def set_image(self, image: np.ndarray):
        self.predictor.set_image(image)
        self.last_mask_logits = None
        self.last_image = image

    def predict(self, input_points=None, input_labels=None, input_boxes=None, multimask_output=True):
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_boxes[0] if input_boxes is not None else None,
            multimask_output=multimask_output,
            mask_input=None,
            return_logits=True
        )
        self.last_mask_logits = logits
        return masks, scores, logits

    def predict_with_logits(self, input_points=None, input_labels=None, input_boxes=None, multimask_output=True):
        masks, scores, logits = self.predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            box=input_boxes[0] if input_boxes is not None else None,
            multimask_output=multimask_output,
            mask_input=self.last_mask_logits,
            return_logits=True
        )
        self.last_mask_logits = logits
        return masks, scores

    def reset_logits(self):
        self.last_mask_logits = None
        self.last_image = None


