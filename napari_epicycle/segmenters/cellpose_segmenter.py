# from cellpose import models
# from .base_segmenter import BaseSegmenter
# from typing import Tuple
# import numpy as np


# class CellposeSegmenter(BaseSegmenter):
#     def __init__(self, config: dict, working_dir: str = None):
#         self.config = config
#         self.use_gpu = config.get("device", "GPU") == "GPU"
#         self.model_type = config["model_type"]
#         self._init_model(working_dir)

#     def _init_model(self, working_dir: str = None):
#         if self.model_type == "scratch":
#             self.model = models.CellposeModel(gpu=self.use_gpu, pretrained_model=None)
#         elif self.model_type == "CPx":
#             self.model = models.CellposeModel(gpu=self.use_gpu, pretrained_model="CPx")
#         else:
#             self.model = models.Cellpose(model_type=self.model_type, gpu=self.use_gpu)

#     def segment(self, image: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#         from cellpose import plot  # ensure plot is imported

#         def parse_channel(cstr):
#             return int(cstr.split(":")[0].strip())


#         print("Parsing Channels")
#         channels = [
#             [parse_channel(config["channel1"]),
#             parse_channel(config["channel2"])]
#         ]

#         # Run segmentation
#         if isinstance(self.model, models.CellposeModel):
#             print("Using Cellpose Model")
#             # CellposeModel returns 3 values
#             masks, flows_raw, _styles = self.model.eval(
#                 image,
#                 channels=channels,
#                 normalize=config["normalize"],
#                 flow_threshold=config["flow_threshold"],
#                 cellprob_threshold=config["cellprob_threshold"],
#                 resample=config["rescale"],
#             )
#         else:
#             print("Using Cellpose")
#             # Cellpose returns 4 values
#             masks, flows_raw, _styles, _diams = self.model.eval(
#                 image,
#                 channels=channels,
#                 normalize=config["normalize"],
#                 flow_threshold=config["flow_threshold"],
#                 cellprob_threshold=config["cellprob_threshold"],
#                 resample=config["rescale"],
#             )

#         print("Segmented image")

#         # flows_raw = [flow_vis (HSV), flow_xy, cell_probs, final_pixels]
#         # We return the HSV flow visualization (flows_raw[0]) for easy rendering
#         flow_combined = flows_raw[0]
#         cell_probs = flows_raw[2]

#         # ========== Sanity checks ==========
#         print(f"[INFO] masks.shape: {masks.shape}, dtype: {masks.dtype}")
#         print(f"[INFO] flow_combined.shape: {flow_combined.shape}, dtype: {flow_combined.dtype}")
#         print(f"[INFO] cell_probs.shape: {cell_probs.shape}, dtype: {cell_probs.dtype}")

#         if masks.ndim != 2:
#             raise ValueError(f"[ERROR] masks should be 2D but got shape {masks.shape}")

#         if flow_combined.ndim != 3:
#             raise ValueError(f"[ERROR] flow visualization should be 3D (e.g. RGB) but got shape {flow_combined.shape}")

#         if cell_probs.ndim != 2:
#             raise ValueError(f"[ERROR] cell_probs should be 2D but got shape {cell_probs.shape}")

#         if masks.shape != cell_probs.shape or masks.shape != flow_combined.shape[:2]:
#             raise ValueError(f"[ERROR] Shape mismatch. masks: {masks.shape}, flow_combined: {flow_combined.shape}, cell_probs: {cell_probs.shape}")
#         # ===================================

#         return masks, flow_combined, cell_probs

from cellpose import models
from .base_segmenter import BaseSegmenter
from typing import Tuple
import numpy as np
from scipy import ndimage


class CellposeSegmenter(BaseSegmenter):
    def __init__(self, config: dict, working_dir: str = None, use_high_level: bool = False):
        self.config = config
        self.use_gpu = config.get("device", "GPU") == "GPU"
        self.model_type = config["model_type"]
        self.use_high_level = use_high_level
        self._init_model(working_dir)

    def _init_model(self, working_dir: str = None):
        if self.use_high_level:
            print("[INFO] Initializing high-level Cellpose interface")
            self.model = models.Cellpose(model_type=self.model_type, gpu=self.use_gpu)
        else:
            print("[INFO] Initializing low-level CellposeModel interface")
            pretrained_override = str(self.config.get("pretrained_model_path", "")).strip()
            if pretrained_override:
                print(f"[INFO] Using pretrained checkpoint override: {pretrained_override}")
                self.model = models.CellposeModel(gpu=self.use_gpu, pretrained_model=pretrained_override)
            elif self.model_type == "scratch":
                self.model = models.CellposeModel(gpu=self.use_gpu, pretrained_model=None)
            elif self.model_type == "CPx":
                self.model = models.CellposeModel(gpu=self.use_gpu, pretrained_model="CPx")
            else:
                self.model = models.CellposeModel(gpu=self.use_gpu, pretrained_model=self.model_type)

    def segment(self, image: np.ndarray, config: dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        from cellpose import plot  # Needed for flow visualization

        def parse_channel(cstr):
            return int(cstr.split(":")[0].strip())

        print("Parsing Channels")
        channels = [
            [parse_channel(config["channel1"]),
             parse_channel(config["channel2"])]
        ]
        diameter = int(config.get("diameter", 0) or 0)
        diameter = None if diameter <= 0 else diameter
        invert = bool(config.get("invert", False))
        # 2026-05-27: niter (dynamics iterations) — Cellpose's
        # post-prediction simulation step.  0 / None = auto (~2× diameter).
        # Moved from Step 2 Advanced Parameters (where it was an orphan
        # widget never reaching this call) to Inference Parameters with
        # proper wiring through the config dict.
        niter_raw = int(config.get("n_iterations", 0) or 0)
        niter = None if niter_raw <= 0 else niter_raw
        eval_kwargs = dict(
            channels=channels,
            normalize=config["normalize"],
            flow_threshold=config["flow_threshold"],
            cellprob_threshold=config["cellprob_threshold"],
            resample=config["rescale"],
            invert=invert,
            diameter=diameter,
        )
        # Some Cellpose versions accept `niter`; older ones don't.  Pass
        # it only when the user has overridden the auto default so we
        # don't break on the older API.
        if niter is not None:
            eval_kwargs["niter"] = niter

        # Run segmentation using selected API
        if self.use_high_level:
            print("[INFO] Using high-level Cellpose API")
            masks, flows_raw, _styles, _diams = self.model.eval(image, **eval_kwargs)
        else:
            print("[INFO] Using low-level CellposeModel API")
            masks, flows_raw, _styles = self.model.eval(image, **eval_kwargs)

        print("Segmented image")

        # flows_raw = [flow_vis (HSV), flow_xy, cell_probs, final_pixels]
        flow_combined = flows_raw[0]
        cell_probs = flows_raw[2]

        # === Sanity Checks ===
        print(f"[INFO] masks.shape: {masks.shape}, dtype: {masks.dtype}")
        print(f"[INFO] flow_combined.shape: {flow_combined.shape}, dtype: {flow_combined.dtype}")
        print(f"[INFO] cell_probs.shape: {cell_probs.shape}, dtype: {cell_probs.dtype}")

        if masks.ndim != 2:
            raise ValueError(f"[ERROR] masks should be 2D but got shape {masks.shape}")

        if flow_combined.ndim != 3:
            raise ValueError(f"[ERROR] flow visualization should be 3D (RGB) but got shape {flow_combined.shape}")

        if cell_probs.ndim != 2:
            raise ValueError(f"[ERROR] cell_probs should be 2D but got shape {cell_probs.shape}")

        # Cellpose may return flow/cellprob at a resized shape when rescale/resample is active.
        # Align auxiliary outputs to mask shape so downstream overlay + caching stays consistent.
        if masks.shape != cell_probs.shape:
            print(
                f"[WARN] Resizing cell_probs from {cell_probs.shape} to {masks.shape} "
                "to match mask shape."
            )
            cell_probs = self._resize_2d_like(cell_probs, masks.shape, order=1)

        if masks.shape != flow_combined.shape[:2]:
            print(
                f"[WARN] Resizing flow_combined from {flow_combined.shape[:2]} to {masks.shape} "
                "to match mask shape."
            )
            flow_combined = self._resize_rgb_like(flow_combined, masks.shape)

        return masks, flow_combined, cell_probs

    def _resize_2d_like(self, arr: np.ndarray, target_hw: Tuple[int, int], order: int = 1) -> np.ndarray:
        src = np.asarray(arr)
        if src.ndim != 2:
            raise ValueError(f"[ERROR] Expected 2D array for resize, got shape {src.shape}")
        if tuple(src.shape) == tuple(target_hw):
            return src
        zoom = (
            float(target_hw[0]) / float(src.shape[0]),
            float(target_hw[1]) / float(src.shape[1]),
        )
        out = ndimage.zoom(src, zoom=zoom, order=order)
        out = out[: target_hw[0], : target_hw[1]]
        if out.shape != tuple(target_hw):
            padded = np.zeros(target_hw, dtype=out.dtype)
            h = min(target_hw[0], out.shape[0])
            w = min(target_hw[1], out.shape[1])
            padded[:h, :w] = out[:h, :w]
            out = padded
        return out.astype(src.dtype, copy=False)

    def _resize_rgb_like(self, arr: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
        src = np.asarray(arr)
        if src.ndim != 3 or src.shape[-1] not in {3, 4}:
            raise ValueError(f"[ERROR] Expected RGB/RGBA array, got shape {src.shape}")
        if tuple(src.shape[:2]) == tuple(target_hw):
            return src
        zoom = (
            float(target_hw[0]) / float(src.shape[0]),
            float(target_hw[1]) / float(src.shape[1]),
            1.0,
        )
        out = ndimage.zoom(src, zoom=zoom, order=1)
        out = out[: target_hw[0], : target_hw[1], :]
        if tuple(out.shape[:2]) != tuple(target_hw):
            padded = np.zeros((target_hw[0], target_hw[1], src.shape[-1]), dtype=out.dtype)
            h = min(target_hw[0], out.shape[0])
            w = min(target_hw[1], out.shape[1])
            padded[:h, :w, :] = out[:h, :w, :]
            out = padded
        return out.astype(src.dtype, copy=False)
