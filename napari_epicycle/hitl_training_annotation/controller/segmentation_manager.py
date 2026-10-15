from napari_epicycle.segmenters.cellpose_segmenter import CellposeSegmenter
import numpy as np

class SegmentationManager:
    def __init__(self):
        self.segmenter = None

        self.data_names = {}
        self.data_inputs = {}
        self.data_masks = {}
        self.data_flows = {}
        self.data_cellprobs = {}

        # Associated RGB image to be segmented
        self.data_inputs_rgb = {}

        # Row un modified predicitons -- accessed during the curation 
        self.data_masks_raw = {}

    # use the config to set the preference
    def set_model_preference(self, config: dict, model_type: str = "cellpose", working_dir: str = None):
        if model_type == "cellpose":
            self.segmenter = CellposeSegmenter(config, working_dir=working_dir)
        else:
            raise NotImplementedError(f"Model type '{model_type}' is not supported yet.")

    # def segment_image(self, image: np.ndarray, slice_label: str):
    def segment_image(self, image: np.ndarray, config: dict, slice_label: str):
        if self.segmenter is None:
            raise RuntimeError("Segmentation model not set. Call `set_model_preference()` first.")

        # masks, flows, cell_probs = self.segmenter.segment(image)
        masks, flows, cell_probs = self.segmenter.segment(image, config)

        # keep track of the slice that is being segmented
        self.data_names[slice_label] = True
        self.data_inputs[slice_label] = image
        self.data_masks[slice_label] = masks
        self.data_flows[slice_label] = flows
        self.data_cellprobs[slice_label] = cell_probs
        return masks, flows, cell_probs

    def get_input_stack(self, slice_label: str):
        return self.data_inputs.get(slice_label, None)

    def get_masks(self, slice_label: str):
        return self.data_masks.get(slice_label, None)

    def get_flows(self, slice_label: str):
        return self.data_flows.get(slice_label, None)

    def get_cell_probs(self, slice_label: str):
        return self.data_cellprobs.get(slice_label, None)
    
    def get_rgb_input_stack(self, slice_label: str):
        return self.data_inputs_rgb.get(slice_label, None)

    # new addition -- to work with samn prompt segmetnation, should change in place 
    # the current mask that is beingredined -- keep this in mind, might wanna keep this separate
    def save_interactive_mask(self, slice_label, mask):
        self.data_masks[slice_label] = mask

    def save_model(self, path: str):
        pass

    def clear(self):
        # probably needsto implement a line that deltes the model
        # for the next round
        self.data_inputs.clear()
        self.data_names.clear()
        self.data_masks.clear()
        self.data_flows.clear()
        self.data_cellprobs.clear()
        self.data_inputs_rgb.clear()
