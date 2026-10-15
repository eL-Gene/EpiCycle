"""A bare annotator fake for the Refine-recipe tests.

The ``test_refine_cross_method_chain.py`` pattern (the REAL ``_preview_smooth``
and its tails bound onto a bare object) with the REAL
``AdvancedFeaturesWidget`` as the panel, so every value a step runs on is
read off real widgets, and the recipe runner's own methods bound too.
"""

from __future__ import annotations

import numpy as np


def K():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


BOUND = (
    "_preview_smooth", "_finish_refine_preview", "_reset_smooth_state",
    "_fire_refine_lock", "_refine_breadcrumb_text", "_refine_spacing",
    "_refine_spacing_known", "_abandon_refine_worker",
    "_refine_thread_cleanup", "_drop_refine_walls",
    "_drop_refine_walls_layer", "_accept_smooth", "_reject_smooth",
    "_refuse_refine_step", "_on_refine_method_combo_changed",
    "_refresh_refine_preview_gate", "_launch_refine_method",
    "_launch_refine_settle", "_push_still_staged",
    "_refine_still_staged_text", "_settle_roi_margin",
    "_contour_roi_margin", "_launch_refine_contour", "_refine_contour_call",
    "_refine_room", "_refine_step_margin", "_grow_refine_region",
    "_finish_settle_result", "_refine_staged_status", "_refine_s1_text",
    "_update_smooth_ui", "_cancel_refine_preview", "_on_refine_stopped",
    "_on_refine_thread_finished", "_teardown_refine_for_switch",
    "_launch_refine_worker", "_refine_is_stale", "_on_refine_finished",
    "_on_refine_error", "_on_refine_progress", "_on_refine_message",
    "_result_token", "_signal_sender", "_refine_band_um",
)

#: The runner's own methods (bound when the class has them).
RECIPE_BOUND = (
    "_recipe_start", "_recipe_preflight", "_recipe_next",
    "_recipe_on_worker_done", "_recipe_after_step", "_recipe_end",
    "_recipe_stage_text", "_recipe_active", "_recipe_on_stopped",
    "_recipe_availability", "_recipe_load_store", "_recipe_mode",
    "_on_refine_preview_clicked", "_on_refine_accept_clicked",
    "_on_refine_discard_clicked", "_on_recipes_menu_about_to_show",
    "_on_recipe_run", "_on_recipe_load", "_on_recipe_save",
    "_on_recipe_delete", "_recipe_selected_object", "_recipe_busy_sentence",
    "_recipe_stale_reason", "_recipe_session_holds",
    "_recipe_schedule_judge", "_recipe_clear_stage_line",
    "_recipe_refuse_click", "_recipe_honour_stop", "_recipe_stopped_stage",
    "_recipe_status_kind",
)
CLASS_TABLES = ("_REFINE_NOTHING_MOVED", "_REFINE_STEP_WORD",
                "_REFINE_METHOD_WORD")


class _Sync:
    def __init__(self, vol):
        self.shared_labels = {"labels": vol}


class _Layer:
    def __init__(self, sel=7):
        self.selected_label = sel


class _Viewer:
    layers = ()


class Fake:
    def __init__(self, vol, adv, *, prob=None, channel_roi=None,
                 spacing=(3.0, 1.0, 1.0)):
        self._active_labels_name = "labels"
        self.sync_manager = _Sync(vol)
        self.viewer = _Viewer()
        self.adv = adv
        self.layer = _Layer(7)
        self.lock_calls = []
        self.smoothed = []
        self.draw_calls = []
        self.ctx = {"patch_name": "p", "run_epoch": 0,
                    "spacing": tuple(spacing)}
        self.prob = prob
        self.channel_roi = channel_roi
        self.confirm_answer = True
        self.confirms = []
        self.confirm_buttons = []
        self.name_answer = ("tidy", True)
        self.store = {}
        self.store_problem = None
        self.saved = []

        self._split_running = False
        self._split_fragment_refine = False
        self._batch_staged = False
        self._batch_running = False
        self._smooth_running = False
        self._smooth_snapshot = None
        self._smooth_roi = None
        self._smooth_label_id = None
        self._smooth_chain_count = 0
        self._refine_steps = []
        self._refine_step_records = []
        self._refine_pending_record = None
        self._refine_chain_recipe_names = []
        self._refine_staged_method = None
        self._refine_staged_s5 = None
        self._refine_running = False
        self._refine_lock_fired = False
        self._refine_stopping = False
        self._refine_worker = None
        self._refine_thread = None
        self._refine_token = 0
        self._refine_busy_kind = None
        self._refine_elevation = None
        self._refine_elevation_roi = None
        self._refine_pending_elevation = None
        self._refine_walls_wanted = False
        self._refine_parked_threads = []
        self._refine_live_messages = []
        self._refine_sync_busy = False
        self._refine_recipe_run = None
        self._refine_recipe_token = 0
        self._temp_guide_layers = {}
        self._interp_refine_mode = False
        self._load_locked = False
        self._prob_volume = None
        self._prob_volume_path = None

        k = K()
        for name in BOUND + RECIPE_BOUND:
            fn = getattr(k, name, None)
            if fn is not None:
                setattr(self, name, fn.__get__(self, Fake))
        for name in CLASS_TABLES:
            setattr(self, name, getattr(k, name))
        self._snapshot_label_roi = k._snapshot_label_roi
        self._label_bbox_roi = k._label_bbox_roi
        self._refine_lock_cb = self.lock_calls.append
        self._on_label_smoothed_cb = self._smoothed_cb
        # the controller's three recipe callbacks, against `self.store`
        self._get_saved_refine_recipes_cb = (
            lambda: (dict(self.store), self.store_problem))
        self._save_refine_recipe_cb = self._save_cb
        self._delete_refine_recipe_cb = self._delete_cb
        # the two dialogs, answered by the test
        self._recipe_confirm = (
            lambda text, confirm_text="Continue", cancel_text="Cancel":
            (self.confirms.append(text), self.confirm_buttons.append(
                (confirm_text, cancel_text)), self.confirm_answer)[2])
        self._recipe_ask_name = lambda n, suggested: self.name_answer

    # --- controller seams ------------------------------------------------
    def _smoothed_cb(self, lid, roi_slices=None, old_roi_data=None, **kw):
        self.smoothed.append(dict(label=lid, roi=roi_slices, **kw))

    def _save_cb(self, name, obj):
        self.saved.append((name, obj))
        self.store[name] = obj
        return None

    def _delete_cb(self, name):
        self.store.pop(name, None)
        return None

    # --- seams -----------------------------------------------------------
    def _get_advanced_features_widget(self):
        return self.adv

    def _ensure_shared_data_aligned(self, name):
        return self.sync_manager.shared_labels.get(name)

    def _layer_by_name(self, layers, name):
        return self.layer

    def _get_selected_label(self):
        return int(self.layer.selected_label)

    def _get_prob_roi(self, roi):
        return None if self.prob is None else self.prob[roi].astype(
            np.float64)

    def _has_prob_volume(self):
        return self.prob is not None

    def _get_refine_channel_roi(self, roi):
        if self.channel_roi is None:
            return None
        return np.array(self.channel_roi[roi], dtype=np.float64)

    def _refresh_refine_object_row(self):
        pass

    def _sync_all_views_after_operation(self, roi=None, label_id=None):
        pass

    def _refresh_refine_walls_toggle(self):
        pass

    def _lock_subviewers_for_preview(self):
        self.draw_calls.append("lock")

    def _restore_ready_to_draw_state(self):
        self.draw_calls.append("restore")

    def _refresh_merge_button_state(self):
        pass

    def _refresh_label_action_buttons_state(self):
        pass

    def _remove_temp_layer(self, key):
        self._temp_guide_layers.pop(key, None)

    def _reshape_context(self):
        return dict(self.ctx)

    def _remove_prob_overlay(self):
        pass

    def _clear_draw_snapshot(self):
        pass

    def _is_auto_advance_enabled(self):
        return False

    # --- vol ---------------------------------------------------------------
    def vol(self):
        return self.sync_manager.shared_labels["labels"]


def vol_with_notch():
    v = np.zeros((28, 40, 40), np.int32)
    v[6:22, 10:30, 10:30] = 7
    v[6:22, 10:13, 10:13] = 0            # a notch, so Smooth has work to do
    v[12:16, 18:22, 18:22] = 0           # an enclosed cavity for Fill Holes
    return v


def make_panel():
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    return AdvancedFeaturesWidget(mode="3d_curation")
