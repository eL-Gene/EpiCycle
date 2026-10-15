"""I-2 / I-3: the two params getters are FROZEN and the setters are additive.

``get_split_params()`` (17 keys, 22 after Brick 2a) and
``get_add_params()`` (12 keys) are the
only things the orchestrator reads off this widget before building a
recipe, and every combo in them is read BY INDEX -- item order is the wire
format, item text is presentation.  So the panel rework may
relabel anything it likes, and these goldens say so: same keys, same
defaults, same index -> engine-token mapping.

The four ``update_*`` setters are additive-only: every keyword that exists
today must still exist, with the same default of None.  Brick 1 adds
``review_active`` and ``accept_text`` to the Split setter and nothing else.

Pattern 3 (offscreen QApplication) .
"""

from __future__ import annotations

import inspect

import pytest

pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        try:
            app = QApplication([])
        except Exception as exc:  # pragma: no cover - CI without Qt
            pytest.skip("cannot create a QApplication: {}".format(exc))
    return app


@pytest.fixture
def widget(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    yield w
    try:
        w.setParent(None)
    except Exception:
        pass


def _W():
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    return AdvancedFeaturesWidget


# --------------------------------------------------------------------------- #
# the frozen dicts
# --------------------------------------------------------------------------- #

#: Byte-for-byte what get_split_params() returned on a freshly built
#: widget BEFORE Brick 2a.  The key set may only
#: grow, and every one of these values is unchanged for every
#: pre-existing combination.
SPLIT_PARAMS_PRE_BRICK2A = {
    "seeds_from": 0,
    "walls_from": 0,
    "guided": False,
    # a layer NAME, not an index; empty until the viewer fills the combo
    "wall_channel": "",
    "clip": 0.01,
    "ridge": True,
    "arbiter": False,
    "arbiter_percentile": 75.0,
    "solver": "watershed",
    "rw_beta": 130.0,
    "compactness": 0.0,
    "h": 0.15,
    "min_distance_um": 5.0,
    "refine_guided": False,
    "refine_merge_small": False,
    "min_size": 20,
    "target_count": None,
}

#: Exactly the keys Brick 2a ADDS, and their defaults.
SPLIT_PARAMS_BRICK2A_ADDITIONS = {
    # the index of the six image detectors; the recipe maps it
    "seed_detector": 0,
    # a layer NAME, blank until the viewer fills the combo
    "seed_channel": "",
    # the marker channel's preparation chain -- local contrast is the
    # one tick on by default
    "seed_preproc": "clahe",
    # the wall chain; nothing ticked, and CLAHE is never carried here
    "wall_preproc": "none",
    # read at Start by _snapshot_label_roi; 15 is today's hard-coded value
    "search_padding": 15,
}

#: Exactly the keys Brick 2b ADDS, and their defaults.  Every one of them
#: is a knob `reshape.spec.DEFAULT_PARAMS` has always had and the panel
#: could not reach, so every default here is the ENGINE's own -- a user
#: who never opens Advanced gets the answer they got before.
SPLIT_PARAMS_BRICK2B_ADDITIONS = {
    "seed_smooth": 2.0,
    "max_sigma_um": 6.0,
    # the spin box's minimum shows "automatic"; the engine wants None
    "seed_thresh": None,
    "seed_clip": 0.02,
    # which token the "Lift dim signal" tick puts in the chain
    "lift_mode": "gentle",
    "seed_gamma": 0.5,
    "seed_bg": 15,
    "seed_denoise": 2,
    "smooth": 1.0,
    "refine_radius": 4,
    "refine_eps": 0.01,
}

#: THE DECLARED EXCEPTION to rule 7, and the
#: only one there is: the sandbox ALWAYS applied MIN_SIZE=20, and the
#: owner asked for that behaviour back.  Written as a one-entry dict, so
#: "which pre-existing value moved" is answered by reading it rather than
#: by diffing two goldens.
SPLIT_PARAMS_BRICK2B_EXCEPTIONS = {
    "refine_merge_small": True,
}

SPLIT_PARAMS_GOLDEN = dict(SPLIT_PARAMS_PRE_BRICK2A)
SPLIT_PARAMS_GOLDEN.update(SPLIT_PARAMS_BRICK2A_ADDITIONS)
SPLIT_PARAMS_GOLDEN.update(SPLIT_PARAMS_BRICK2B_ADDITIONS)
SPLIT_PARAMS_GOLDEN.update(SPLIT_PARAMS_BRICK2B_EXCEPTIONS)

#: Every key added after the pre-2a freeze, in one set.
SPLIT_PARAMS_ADDITIONS = dict(SPLIT_PARAMS_BRICK2A_ADDITIONS)
SPLIT_PARAMS_ADDITIONS.update(SPLIT_PARAMS_BRICK2B_ADDITIONS)

ADD_PARAMS_GOLDEN = {
    "channel": "",
    "clahe": False,
    "clahe_clip": 0.01,
    "min_size": 0,
    "separate": True,
    "min_distance": 3,
    "clear_border": True,
    "edge_margin": 3,
    "auto_smooth": True,
    "smooth_sigma": 1.5,
    # "connectivity" (collected, never passed
    # on) and "marker" (the Boundary stub) are GONE; the rebuild's keys,
    # at their defaults, joined.
    "preset": "nuclei",
    "click_radius_um": 2.0,
    "core_erode_um": 1.0,
    "dark_clearance_um": 2.0,
    "dark_otsu_factor": 0.5,
    "nuclei_threshold": 0.0,
    "dark_outside": True,
}


class TestSplitParamsGolden:
    def test_key_set_is_frozen(self, widget):
        assert set(widget.get_split_params()) == set(SPLIT_PARAMS_GOLDEN)

    def test_the_key_set_grew_by_EXACTLY_the_named_additions(self, widget):
        """Rule 7.  A combo one item ahead of the recipe is a
        KeyError; one behind is a silently wrong algorithm.  The same
        applies to the dict between them."""
        keys = set(widget.get_split_params())
        assert set(SPLIT_PARAMS_PRE_BRICK2A) <= keys, (
            "a pre-existing key was dropped: {}".format(
                sorted(set(SPLIT_PARAMS_PRE_BRICK2A) - keys))
        )
        assert keys - set(SPLIT_PARAMS_PRE_BRICK2A) == set(
            SPLIT_PARAMS_ADDITIONS), (
            "unexpected new key(s): {}".format(
                sorted(keys - set(SPLIT_PARAMS_GOLDEN)))
        )

    def test_no_pre_existing_value_moved_for_any_pre_existing_setting(
            self, widget):
        """Rule 7's other half: the pre-2a keys keep their values across
        every combination that existed before 2a -- the four seed
        sources x the two wall kinds x the three solvers.

        ONE declared exception, and only one: `refine_merge_small`.  It is applied from `SPLIT_PARAMS_BRICK2B_EXCEPTIONS`,
        so this sweep proves the flip is the only value that moved and
        that it moved for EVERY combination, not just the default one.
        """
        from napari_epicycle.annotator import advanced_features_widget as M

        for seeds in range(4):
            for walls in range(2):
                for solver in range(len(M.SPLIT_SOLVERS)):
                    widget.split_seeds_from_combo.setCurrentIndex(seeds)
                    widget.split_walls_from_combo.setCurrentIndex(walls)
                    widget.split_solver_combo.setCurrentIndex(solver)
                    p = widget.get_split_params()
                    expected = dict(SPLIT_PARAMS_PRE_BRICK2A)
                    expected.update(SPLIT_PARAMS_BRICK2B_EXCEPTIONS)
                    expected["seeds_from"] = seeds
                    expected["walls_from"] = walls
                    expected["solver"] = M.SPLIT_SOLVERS[solver]
                    expected["guided"] = (
                        seeds != M.SEEDS_FROM_PAINTED
                        or walls != M.WALLS_FROM_SHAPE
                    )
                    for key, want in expected.items():
                        assert p[key] == want, (
                            "{!r} moved at seeds={} walls={} solver={}: "
                            "{!r} != {!r}".format(
                                key, seeds, walls, solver, p[key], want)
                        )

    def test_absorb_slivers_is_the_ONE_declared_exception(self, widget):
        """The sandbox ALWAYS applied MIN_SIZE=20, and the owner
        asked for that back.  This is the only pre-existing value Brick 2b
        is allowed to move, and it moved in the commit that flipped it."""
        assert SPLIT_PARAMS_PRE_BRICK2A["refine_merge_small"] is False
        assert SPLIT_PARAMS_BRICK2B_EXCEPTIONS == {
            "refine_merge_small": True}
        p = widget.get_split_params()
        assert p["refine_merge_small"] is True
        assert p["min_size"] == 20

    def test_nothing_else_pre_existing_moved(self, widget):
        """The rest of rule 7, stated once more against the whole dict."""
        p = widget.get_split_params()
        for key, want in SPLIT_PARAMS_PRE_BRICK2A.items():
            if key in SPLIT_PARAMS_BRICK2B_EXCEPTIONS:
                continue
            assert p[key] == want, key

    def test_default_values_are_frozen(self, widget):
        p = widget.get_split_params()
        for key, expected in SPLIT_PARAMS_GOLDEN.items():
            got = p[key]
            if isinstance(expected, float):
                assert got == pytest.approx(expected), key
            else:
                assert got == expected, key
                assert type(got) is type(expected), key

    def test_every_control_combination_round_trips(self, widget):
        """The sweep: move each control, read the dict, put it back.

        The re-layout re-parents these widgets; a knob that silently stops
        reaching ``get_split_params`` (renamed, dropped from a container it
        is looked up through) would still look fine on screen.
        """
        cases = [
            ("split_seeds_from_combo", "index", 3, "seeds_from", 3),
            ("split_walls_from_combo", "index", 1, "walls_from", 1),
            ("split_solver_combo", "index", 1, "solver", "random_walker"),
            ("split_solver_combo", "index", 2, "solver", "label_diffusion"),
            ("split_wall_clip_sb", "value", 0.05, "clip", 0.05),
            ("split_wall_ridge_cb", "check", False, "ridge", False),
            ("split_arbiter_pct_sb", "value", 42.0, "arbiter_percentile", 42.0),
            ("split_rw_beta_sb", "value", 200.0, "rw_beta", 200.0),
            # Brick 2b narrowed Balance to its real range (parity row
            # 47); 3.0 would silently clamp and the sweep would read a
            # value nobody set.
            ("split_compactness_sb", "value", 0.5, "compactness", 0.5),
            ("split_seed_h_sb", "value", 0.4, "h", 0.4),
            ("split_min_distance_sb", "value", 9.0, "min_distance_um", 9.0),
            ("split_refine_guided_cb", "check", True, "refine_guided", True),
            ("split_refine_merge_small_cb", "check", True,
             "refine_merge_small", True),
            ("split_min_size_sb", "value", 77, "min_size", 77),
            # The TICK is the off switch now, so it is what the sweep
            # moves; the spin box's own round trip is pinned in
            # test_reshape_split_widget (it needs the tick on first).
            ("split_target_count_cb", "check", True, "target_count", 2),
            # --- Brick 2a ---
            ("split_seeds_from_combo", "index", 4, "seeds_from", 4),
            ("split_walls_from_combo", "index", 8, "walls_from", 8),
            ("split_seed_detector_combo", "index", 5, "seed_detector", 5),
            ("split_search_padding_sb", "value", 7, "search_padding", 7),
            ("split_seed_prep_clahe_cb", "check", False,
             "seed_preproc", "none"),
            ("split_wall_prep_denoise_cb", "check", True,
             "wall_preproc", "denoise"),
            # --- Brick 2b: the eleven knobs the engine always had ---
            ("split_seed_smooth_sb", "value", 3.5, "seed_smooth", 3.5),
            ("split_max_sigma_sb", "value", 9.0, "max_sigma_um", 9.0),
            ("split_seed_thresh_sb", "value", 0.35, "seed_thresh", 0.35),
            ("split_seed_clip_sb", "value", 0.05, "seed_clip", 0.05),
            ("split_lift_mode_combo", "index", 1, "lift_mode", "strong"),
            ("split_seed_gamma_sb", "value", 0.8, "seed_gamma", 0.8),
            ("split_seed_bg_sb", "value", 30, "seed_bg", 30),
            ("split_seed_denoise_sb", "value", 4, "seed_denoise", 4),
            ("split_wall_smooth_sb", "value", 2.5, "smooth", 2.5),
            ("split_refine_radius_sb", "value", 9, "refine_radius", 9),
            ("split_refine_eps_sb", "value", 0.05, "refine_eps", 0.05),
        ]
        for attr, kind, set_to, key, expected in cases:
            w = getattr(widget, attr)
            if kind == "index":
                before = w.currentIndex()
                w.setCurrentIndex(set_to)
            elif kind == "value":
                before = w.value()
                w.setValue(set_to)
            else:
                before = w.isChecked()
                w.setChecked(set_to)
            got = widget.get_split_params()[key]
            if isinstance(expected, float):
                assert got == pytest.approx(expected), (attr, key)
            else:
                assert got == expected, (attr, key)
            if kind == "index":
                w.setCurrentIndex(before)
            elif kind == "value":
                w.setValue(before)
            else:
                w.setChecked(before)

    def test_the_wall_channel_is_carried_by_NAME(self, widget):
        """`wall_channel` is the one params value that is a layer NAME, not
        an index -- and it is the value `_reshape_recipe` resolves against
        the viewer.  It was the only key the sweep did not cover."""
        widget.set_wall_channels(["membrane", "dapi"])
        widget.split_wall_channel_combo.setCurrentIndex(1)
        assert widget.get_split_params()["wall_channel"] == "dapi"
        widget.split_wall_channel_combo.setCurrentIndex(0)
        assert widget.get_split_params()["wall_channel"] == "membrane"

    def test_an_unresolvable_channel_is_an_EMPTY_name_not_a_wrong_one(
            self, widget):
        """After a refresh miss the combo sits at index -1, and the engine
        gets "" -- which `_reshape_recipe` refuses loudly -- rather than
        whatever happened to be first."""
        widget.set_wall_channels(["membrane", "dapi"])
        widget.split_wall_channel_combo.setCurrentIndex(1)
        widget.set_wall_channels(["membrane", "brightfield"])
        assert widget.get_split_params()["wall_channel"] == ""

    def test_the_arbiter_checkbox_reaches_the_dict(self, widget):
        """It is the one control the re-layout moves OUT of Advanced."""
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        widget.split_arbiter_cb.setChecked(True)
        assert widget.get_split_params()["arbiter"] is True

    def test_guided_flips_only_off_the_two_pickers(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        assert widget.get_split_params()["guided"] is False
        widget.split_solver_combo.setCurrentIndex(2)
        assert widget.get_split_params()["guided"] is False
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_RUN_LABELS)
        assert widget.get_split_params()["guided"] is True


class TestAddParamsGolden:
    def test_key_set_is_frozen(self, widget):
        assert set(widget.get_add_params()) == set(ADD_PARAMS_GOLDEN)

    def test_default_values_are_frozen(self, widget):
        p = widget.get_add_params()
        for key, expected in ADD_PARAMS_GOLDEN.items():
            got = p[key]
            if isinstance(expected, float):
                assert got == pytest.approx(expected), key
            else:
                assert got == expected, key

    def test_every_control_combination_round_trips(self, widget):
        cases = [
            ("add_clahe_cb", "check", True, "clahe", True),
            ("add_clahe_clip_sb", "value", 0.5, "clahe_clip", 0.5),
            ("add_min_size_sb", "value", 120, "min_size", 120),
            ("add_separate_cb", "check", False, "separate", False),
            ("add_min_distance_sb", "value", 11, "min_distance", 11),
            ("add_clear_border_cb", "check", False, "clear_border", False),
            ("add_edge_margin_sb", "value", 7, "edge_margin", 7),
            ("add_auto_smooth_cb", "check", False, "auto_smooth", False),
            ("add_smooth_sigma_sb", "value", 2.5, "smooth_sigma", 2.5),
            ("add_preset_combo", "index", 1, "preset", "cells"),
            ("add_click_radius_sb", "value", 3.5, "click_radius_um", 3.5),
            ("add_core_erode_sb", "value", 2.5, "core_erode_um", 2.5),
            ("add_dark_clearance_sb", "value", 4.0, "dark_clearance_um", 4.0),
            ("add_dark_factor_sb", "value", 0.35, "dark_otsu_factor", 0.35),
            ("add_nuclei_threshold_sb", "value", 12.5, "nuclei_threshold",
             12.5),
            ("add_dark_outside_cb", "check", False, "dark_outside", False),
        ]
        for attr, kind, set_to, key, expected in cases:
            w = getattr(widget, attr)
            if kind == "index":
                before = w.currentIndex()
                w.setCurrentIndex(set_to)
            elif kind == "value":
                before = w.value()
                w.setValue(set_to)
            else:
                before = w.isChecked()
                w.setChecked(set_to)
            got = widget.get_add_params()[key]
            if isinstance(expected, float):
                assert got == pytest.approx(expected), (attr, key)
            else:
                assert got == expected, (attr, key)
            if kind == "index":
                w.setCurrentIndex(before)
            elif kind == "value":
                w.setValue(before)
            else:
                w.setChecked(before)


# --------------------------------------------------------------------------- #
# I-2 -- the setters are additive-only
# --------------------------------------------------------------------------- #

SETTER_KEYWORDS = {
    # NOTE: this is the pre-rework baseline, asserted as a SUBSET -- the
    # setters are additive-only, so Brick 2a's marker_* keywords do not
    # belong here.
    "update_split_ui_state": {
        "seed_count", "seed_ids", "active_seed_id", "split_active",
        "start_enabled", "next_seed_enabled", "undo_seed_enabled",
        "refine_seeds_enabled", "preview_enabled", "accept_enabled",
        "reject_enabled", "reject_text", "status_text", "edt_enabled",
        "refine_further_enabled", "fragment_refine_active", "fragment_ids",
        "active_fragment_id", "detect_enabled", "detect_visible",
        "detect_histogram_data", "detect_threshold", "detect_seed_count",
    },
    "update_split_guided_ui": {
        "pieces", "piece_sizes", "piece_flags", "caption", "arbiter_message",
        "split_anyway_visible", "engine_messages", "guide_note",
        "guide_loaded", "busy", "busy_text", "busy_progress",
        "blocked_reason",
    },
    "update_add_ui_state": {
        "box_active", "box_shape", "box_center", "box_size", "histogram_data",
        "threshold_value", "results_text", "preview_enabled",
        "accept_enabled", "accept_count", "reject_enabled", "start_enabled",
        "status_text", "review_active", "piece_ids", "piece_sizes",
        "candidate_ids", "candidate_sizes", "candidate_edge_ids",
        "marker_enabled",
    },
    "update_smooth_ui_state": {
        "preview_enabled", "accept_enabled", "reject_enabled", "status_text",
    },
    "set_split_callbacks": {
        "on_start", "on_next_seed", "on_undo_seed", "on_refine_seeds",
        "on_select_seed", "on_preview", "on_accept", "on_reject",
        "on_edt_toggle", "on_refine_further", "on_select_fragment",
        "on_detect_seeds", "on_threshold_changed", "on_seeds_from_changed",
        "on_walls_from_changed", "on_guide_browse", "on_guide_load",
        "on_guide_show_toggled", "on_piece_check_changed",
        "on_cancel_preview", "on_split_anyway",
    },
    "set_add_callbacks": {
        "on_start", "on_box_changed", "on_recenter", "on_size_preset",
        "on_clahe_toggled", "on_threshold_changed", "on_otsu_reset",
        "on_preview", "on_accept", "on_reject", "on_advanced_changed",
        "on_select_piece", "on_refine_piece", "on_merge_pieces",
        "on_batch_refine", "on_smooth_pieces", "on_candidate_changed",
        "on_add_another", "on_undo_all",
    },
    "set_smooth_callbacks": {
        "on_preview", "on_accept", "on_reject", "on_prob_toggle",
        # Refine recipes, additive
        "on_recipes_menu", "on_recipe_run", "on_recipe_load",
        "on_recipe_save", "on_recipe_delete",
    },
}


class TestSettersAreAdditiveOnly:
    @pytest.mark.parametrize("method", sorted(SETTER_KEYWORDS))
    def test_todays_keyword_set_is_a_subset_of_the_shipped_one(self, method):
        sig = inspect.signature(getattr(_W(), method))
        params = set(sig.parameters) - {"self"}
        missing = SETTER_KEYWORDS[method] - params
        assert not missing, "{} lost keyword(s): {}".format(
            method, sorted(missing))

    @pytest.mark.parametrize("method", sorted(SETTER_KEYWORDS))
    def test_every_keyword_defaults_to_None_and_is_keyword_only(self, method):
        sig = inspect.signature(getattr(_W(), method))
        for name, p in sig.parameters.items():
            if name == "self":
                continue
            assert p.kind is p.KEYWORD_ONLY, "{}.{} must be keyword-only".format(
                method, name)
            assert p.default is None, (
                "{}.{} must default to None so partial updates stay cheap"
                .format(method, name)
            )

    def test_calling_a_setter_with_no_keywords_is_a_no_op(self, widget):
        """Every one is None-guarded; a bare call must not raise or change
        anything (the orchestrator nudges one field at a time)."""
        before = (
            widget.split_reject_btn.text(),
            widget.split_accept_btn.text(),
            widget.smooth_status_label.text(),
        )
        widget.update_split_ui_state()
        widget.update_split_guided_ui()
        widget.update_add_ui_state()
        widget.update_smooth_ui_state()
        assert (
            widget.split_reject_btn.text(),
            widget.split_accept_btn.text(),
            widget.smooth_status_label.text(),
        ) == before
