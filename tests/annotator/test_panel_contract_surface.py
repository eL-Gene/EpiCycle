"""The frozen contract surface of the curation tools panel.

Nearly every consumer reaches this widget through ``hasattr``/``getattr``,
so a renamed widget does NOT
raise -- the button simply stops working.  Eleven Advanced knobs are
resolved by *string literal* inside ``get_split_params`` and three Refine
image combos arrive as literals from the annotator, so no static tool
protects them either (Appendix A.8).

Hence: a RUNTIME table test.  It instantiates the real widget offscreen
and asserts, for every name any consumer reaches, that the attribute
exists and is still the same Qt type.

Pattern 3 (offscreen QApplication) .
"""

from __future__ import annotations

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


# --------------------------------------------------------------------------- #
# I-1 -- every attribute a consumer reaches, and its Qt type
# --------------------------------------------------------------------------- #

#: name -> Qt class name.  Sourced from Appendix A (A.1 callbacks, A.2 Add,
#: A.3 smooth, A.6 raw reads, A.8 getattr-by-literal) plus 2.1b's slot table.
CONTRACT_WIDGETS = {
    # --- Reshape (Split) page: the 17 wired in set_split_callbacks ---------
    "split_start_btn": "QPushButton",
    "split_next_seed_btn": "QPushButton",
    "split_undo_seed_btn": "QPushButton",
    "split_refine_seeds_btn": "QPushButton",
    "split_preview_btn": "QPushButton",
    "split_accept_btn": "QPushButton",
    "split_reject_btn": "QPushButton",
    "split_detect_seeds_btn": "QPushButton",
    "split_refine_further_btn": "QPushButton",
    "split_fragment_list": "QListWidget",
    "split_guide_browse_btn": "QPushButton",
    "split_guide_load_btn": "QPushButton",
    "split_cancel_preview_btn": "QPushButton",
    "split_anyway_btn": "QPushButton",
    "split_guide_show_cb": "QCheckBox",
    "split_edt_cb": "QCheckBox",
    # --- the pickers + the 11 Advanced knobs resolved BY STRING LITERAL ----
    "split_seeds_from_combo": "QComboBox",
    "split_walls_from_combo": "QComboBox",
    "split_guide_source_combo": "QComboBox",
    "split_wall_channel_combo": "QComboBox",
    "split_guided_summary_btn": "QPushButton",
    "split_advanced_btn": "QPushButton",
    "split_solver_combo": "QComboBox",
    "split_arbiter_cb": "QCheckBox",
    "split_arbiter_pct_sb": "QDoubleSpinBox",
    "split_rw_beta_sb": "QDoubleSpinBox",
    "split_compactness_sb": "QDoubleSpinBox",
    "split_seed_h_sb": "QDoubleSpinBox",
    "split_min_distance_sb": "QDoubleSpinBox",
    "split_refine_guided_cb": "QCheckBox",
    "split_refine_merge_small_cb": "QCheckBox",
    "split_min_size_sb": "QSpinBox",
    "split_target_count_sb": "QSpinBox",
    "split_wall_clip_sb": "QDoubleSpinBox",
    "split_wall_ridge_cb": "QCheckBox",
    # --- Brick U2 (UX round 3): the outline tick, in S2 TARGET ------------
    "split_contour_cb": "QCheckBox",
    # --- Brick U2 (UX round 4): the description above S1 ------------------
    "split_about_btn": "QPushButton",
    # --- Brick U3 (UX round 5): the wall picture, on S5's header row ------
    # It rides the RESULT card's title row rather than taking one of its
    # own: review-with-markers-open had 4 px of its 753 px ceiling left,
    # and a stacked row is 15.  The legend below it is on screen only
    # while the box is ticked.
    "split_show_walls_cb": "QCheckBox",
    "split_walls_legend": "QLabel",
    # --- the auto-smooth trio (C3D reads) ---------------------------------
    # The legacy manual-split pair (`split_method_combo` /
    # `split_image_combo`) and its `_split_image_container` were RETIRED
    # in Brick 2b: they asked the same question as "Walls
    # from" one chevron above, had been hidden since Brick 1, and their
    # orchestrator branch was unreachable.  Three entries left this
    # table in the commit that removed them.
    "split_auto_smooth_cb": "QCheckBox",
    "smooth_sigma_z_sb": "QDoubleSpinBox",
    "smooth_sigma_xy_sb": "QDoubleSpinBox",
    "split_detect_min_dist_sb": "QSpinBox",
    "split_detect_bins_sb": "QSpinBox",
    "split_auto_advance_cb": "QCheckBox",
    # --- lists, labels, bar, containers -----------------------------------
    "split_piece_list": "QListWidget",
    # Brick U1 (UX round 2.2): `split_seed_list` is RETIRED and
    # `split_marker_list` is THE list for all five marker sources --
    # `seed_ids=` / `active_seed_id=` point at it now.
    "split_marker_list": "QListWidget",
    "split_marker_caption": "QLabel",
    # `split_marker_hint` RETIRED, Brick U4 (audit H2): its 14 px
    # were the difference between the tallest state fitting its
    # ceiling and not, and since U2b the object itself answers the
    # question it asked.  The sentence lives on the list tooltip.
    "split_piece_card_caption": "QLabel",

    "split_seed_label": "QLabel",
    "split_status_label": "QLabel",
    "split_piece_caption": "QLabel",
    "split_arbiter_msg_label": "QLabel",
    "split_engine_msg_label": "QLabel",
    "split_guide_note_label": "QLabel",
    "split_busy_bar": "QProgressBar",
    "_detect_container": "QWidget",
    "_detect_count_label": "QLabel",
    "_split_busy_container": "QWidget",
    "_split_guided_container": "QWidget",
    "_split_guide_container": "QWidget",
    "_split_walls_container": "QWidget",
    "_split_advanced_container": "QWidget",
    "_split_cut_image_container": "QWidget",
    "_smooth_sigma_container": "QWidget",
    # --- Add page ----------------------------------------------------------
    "add_channel_combo": "QComboBox",
    "add_start_btn": "QPushButton",
    "add_recenter_btn": "QPushButton",
    "add_clahe_cb": "QCheckBox",
    "add_clahe_clip_sb": "QDoubleSpinBox",
    "add_otsu_btn": "QPushButton",
    "add_preview_btn": "QPushButton",
    "add_accept_btn": "QPushButton",
    "add_reject_btn": "QPushButton",
    "add_candidate_list": "QListWidget",
    "add_pieces_list": "QListWidget",
    "add_refine_piece_btn": "QPushButton",
    "add_merge_pieces_btn": "QPushButton",
    "add_again_btn": "QPushButton",
    "add_undo_all_btn": "QPushButton",
    # Add_results_label merged into S1, and the
    # "Boundary -- soon" marker combo and the never-passed connectivity
    # combo are removed.
    "add_status_label": "QLabel",
    "add_size_s_btn": "QPushButton",
    "add_size_m_btn": "QPushButton",
    "add_size_l_btn": "QPushButton",
    # the 10 Advanced knobs re-run Preview on change
    "add_separate_cb": "QCheckBox",
    "add_min_distance_sb": "QSpinBox",
    # --- ---
    "add_preset_combo": "QComboBox",
    "add_marker_source_combo": "QComboBox",
    "add_nuclei_channel_combo": "QComboBox",
    "add_wall_channel_combo": "QComboBox",
    "add_wall_kind_combo": "QComboBox",
    "add_threshold_sb": "QDoubleSpinBox",
    "add_nuclei_threshold_sb": "QDoubleSpinBox",
    "add_nuclei_auto_btn": "QPushButton",
    "add_numbers_btn": "QPushButton",
    "add_place_markers_btn": "QPushButton",
    "add_click_undo_btn": "QPushButton",
    "add_click_clear_btn": "QPushButton",
    "add_stop_btn": "QPushButton",
    "add_all_btn": "QPushButton",
    "add_none_btn": "QPushButton",
    "add_show_walls_cb": "QCheckBox",
    "add_dark_outside_cb": "QCheckBox",
    "add_click_radius_sb": "QDoubleSpinBox",
    "add_core_erode_sb": "QDoubleSpinBox",
    "add_dark_clearance_sb": "QDoubleSpinBox",
    "add_dark_factor_sb": "QDoubleSpinBox",
    "_add_s4_container": "QWidget",
    "add_min_size_sb": "QSpinBox",
    "add_clear_border_cb": "QCheckBox",
    "add_edge_margin_sb": "QSpinBox",
    "add_auto_smooth_cb": "QCheckBox",
    "add_smooth_sigma_sb": "QDoubleSpinBox",
    "_add_tune_container": "QWidget",
    "_add_review_container": "QWidget",
    "_add_box_container": "QWidget",
    "_add_advanced_container": "QWidget",
    # --- Refine page -------------------------------------------------------
    "refine_method_combo": "QComboBox",
    # B2: a QStackedWidget subclass that is as tall as the page it SHOWS
    # (a plain stack asks for its tallest page, which B2 made the contour).
    "_refine_params_stack": "_CurrentPageStack",
    "smooth_standalone_sigma_z_sb": "QDoubleSpinBox",
    "smooth_standalone_sigma_xy_sb": "QDoubleSpinBox",
    "smooth_preserve_topology_cb": "QCheckBox",
    # B1: the three per-method image combos are RETIRED
    # and replaced by ONE shared channel picker, whose blank state is an
    # explicit refusal rather than a fallback to the first image layer.
    "refine_channel_combo": "QComboBox",
    "refine_prep_clahe_cb": "QCheckBox",
    "refine_prep_zcorrect_cb": "QCheckBox",
    "refine_prep_bgsub_cb": "QCheckBox",
    "refine_prep_lift_cb": "QCheckBox",
    "refine_prep_denoise_cb": "QCheckBox",
    # B1: Settle onto the wall signal.
    "refine_wall_kind_combo": "QComboBox",
    "refine_solver_combo": "QComboBox",
    "refine_settle_band_sb": "QDoubleSpinBox",
    "refine_settle_iter_sb": "QSpinBox",
    "refine_settle_clamp_sb": "QDoubleSpinBox",
    "refine_settle_compactness_sb": "QDoubleSpinBox",
    # B1: the pinned S4 strip and the S5 result card.
    "refine_cancel_preview_btn": "QPushButton",
    "refine_busy_bar": "QProgressBar",
    "refine_stage_label": "QLabel",
    # Refine recipes: one menu button in
    # the pinned strip's existing row.
    "refine_recipes_btn": "QPushButton",
    "refine_recipes_menu": "QMenu",
    "refine_show_walls_cb": "QCheckBox",
    "refine_walls_legend": "QLabel",
    "refine_breadcrumb": "QLabel",
    "refine_result_note": "QLabel",
    "refine_result_warning": "QLabel",
    "refine_object_value": "QLabel",
    "refine_about_btn": "QPushButton",
    "refine_advanced_btn": "QPushButton",
    "_refine_s4_container": "QWidget",
    "_refine_result_container": "QWidget",
    "_refine_channel_container": "QWidget",
    "_refine_advanced_container": "QWidget",
    # B2: the merged Active contour.  RETIRED here: the
    # two pairs of Steps/Tidy boxes and `refine_balloon_sb` -- one Steps,
    # one push (`refine_contour_balloon_sb`) whose gate is now on the
    # page, and the tidy passes fixed at 2.  RETIRED in B1:
    # `refine_distance_clamp_sb` / `refine_acwe_distance_clamp_sb` (a
    # ONE-SIDED limit) and `refine_alpha_sb` (a measured no-op,
    # ).
    "refine_contour_mode_combo": "QComboBox",
    "refine_contour_axis_combo": "QComboBox",
    "refine_contour_slices_combo": "QComboBox",
    "refine_contour_share_sb": "QSpinBox",
    "refine_contour_nslices_sb": "QSpinBox",
    "refine_contour_limit_sb": "QDoubleSpinBox",
    "refine_contour_iterations_sb": "QSpinBox",
    "refine_contour_balloon_sb": "QDoubleSpinBox",
    "refine_contour_push_line": "QLabel",
    "refine_contour_gate_sb": "QDoubleSpinBox",
    "refine_contour_blur_sb": "QDoubleSpinBox",
    "refine_contour_sharpness_sb": "QSpinBox",
    "refine_contour_rebuild_um_sb": "QDoubleSpinBox",
    "refine_contour_rebuild_readout": "QLabel",
    "_refine_contour_lower": "QWidget",
    "refine_lambda1_sb": "QDoubleSpinBox",
    "refine_lambda2_sb": "QDoubleSpinBox",
    "_refine_wall_container": "QWidget",
    "refine_de_radius_sb": "QSpinBox",
    "refine_de_axis_combo": "QComboBox",
    "refine_gg_max_radius_sb": "QSpinBox",
    "refine_gg_prob_threshold_sb": "QDoubleSpinBox",
    "refine_gg_axis_combo": "QComboBox",
    "refine_fh_axis_combo": "QComboBox",
    "refine_ff_axis_combo": "QComboBox",
    "refine_ff_max_size_sb": "QSpinBox",
    "refine_prob_overlay_cb": "QCheckBox",
    "refine_prob_cmap_combo": "QComboBox",
    "smooth_preview_btn": "QPushButton",
    "smooth_accept_btn": "QPushButton",
    "smooth_reject_btn": "QPushButton",
    "smooth_status_label": "QLabel",
    # --- the tab strip itself ---------------------------------------------
    "_tool_tabs": "QTabWidget",
}


class TestContractSurface:
    @pytest.mark.parametrize("name", sorted(CONTRACT_WIDGETS))
    def test_attribute_exists_with_its_type(self, widget, name):
        assert hasattr(widget, name), (
            "{} is reached by a consumer (Appendix A) and is gone".format(name)
        )
        obj = getattr(widget, name)
        assert type(obj).__name__ == CONTRACT_WIDGETS[name], (
            "{} changed Qt type: {} -> {}".format(
                name, CONTRACT_WIDGETS[name], type(obj).__name__)
        )

    def test_every_contract_widget_is_a_descendant_of_the_panel(self, widget):
        """I-7: 20 existing assertions use isVisibleTo(widget), which needs it."""
        for name in sorted(CONTRACT_WIDGETS):
            obj = getattr(widget, name)
            node = obj
            seen = False
            while node is not None:
                if node is widget:
                    seen = True
                    break
                node = node.parentWidget()
            assert seen, "{} is not parented under the panel".format(name)


# --------------------------------------------------------------------------- #
# I-9 -- tab order, index 5, and the name-based focus_tab lookup
# --------------------------------------------------------------------------- #
class TestTabsAndFocus:
    def test_tab_construction_order_is_unchanged(self, widget):
        names = [
            widget._tool_tabs.tabText(i)
            for i in range(widget._tool_tabs.count())
        ]
        assert names[:4] == ["SAM", "Membrane", "Scribble", "Histogram"]
        assert names[4] == "Reshape"
        assert names[5] == "Refine"
        assert names[6] == "Add"
        assert len(names) == 7

    def test_the_four_stubs_stay_hidden(self, widget):
        for i in range(4):
            assert widget._tool_tabs.isTabVisible(i) is False

    def test_focus_tab_finds_a_tab_by_name(self, widget):
        assert widget.focus_tab("Refine") is True
        assert widget._tool_tabs.currentIndex() == 5
        assert widget.focus_tab("Add") is True
        assert widget._tool_tabs.tabText(
            widget._tool_tabs.currentIndex()) == "Add"

    def test_the_reshape_tab_is_the_one_that_opens(self, widget):
        assert widget._tool_tabs.tabText(
            widget._tool_tabs.currentIndex()) == "Reshape"

    def test_focus_tab_returns_False_for_an_unknown_name(self, widget):
        before = widget._tool_tabs.currentIndex()
        assert widget.focus_tab("Derive") is False
        assert widget.focus_tab("Split") is False
        assert widget._tool_tabs.currentIndex() == before

    def test_focus_tab_survives_a_missing_tab_strip(self, widget):
        """It is called from a getattr-guarded orchestrator; never raise."""
        del widget._tool_tabs
        assert widget.focus_tab("Refine") is False


#: Every control that carries a tooltip.  I-15: the re-layout may not drop
#: one, and the two the owner found confusing -- the footer's Accept and
#: its secondary -- were the ONLY two of the Split page's 45 controls with
#: no tooltip until this brick.  Generated from the shipped widget, frozen.
TOOLTIP_GOLDEN = (
    "add_again_btn", "add_auto_smooth_cb", "add_candidate_list",
    "add_channel_combo", "add_clahe_cb", "add_clahe_clip_sb",
    "add_clear_border_cb", "add_edge_margin_sb",
    "add_merge_pieces_btn", "add_min_distance_sb",
    "add_min_size_sb", "add_otsu_btn", "add_pieces_list", "add_preview_btn",
    "add_recenter_btn", "add_refine_piece_btn", "add_separate_cb",
    "add_size_l_btn", "add_size_m_btn", "add_size_s_btn",
    "add_smooth_sigma_sb", "add_start_btn", "add_undo_all_btn",
    # Brick B1 RETIRED seven of these:
    # the three per-method image combos, the Guided Grow source picker, the
    # "Image weight" alpha and the two one-sided distance clamps.
    # B2 retired the two Steps/Tidy pairs and the old push box.
    "refine_de_axis_combo",
    "refine_de_radius_sb", "refine_ff_axis_combo",
    "refine_ff_max_size_sb", "refine_fh_axis_combo",
    "refine_gg_axis_combo", "refine_gg_max_radius_sb",
    "refine_gg_prob_threshold_sb",
    "refine_lambda1_sb", "refine_lambda2_sb",
    "refine_prob_cmap_combo", "refine_prob_overlay_cb",
    # ...and added these, which explain themselves the same way.
    "refine_about_btn", "refine_cancel_preview_btn", "refine_channel_combo",
    "refine_contour_mode_combo", "refine_settle_band_sb",
    "refine_settle_clamp_sb", "refine_settle_compactness_sb",
    "refine_settle_iter_sb", "refine_solver_combo", "refine_wall_kind_combo",
    # ...and B2's contour page, every control of it.
    "refine_contour_axis_combo", "refine_contour_slices_combo",
    "refine_contour_share_sb", "refine_contour_nslices_sb",
    "refine_contour_limit_sb", "refine_contour_iterations_sb",
    "refine_contour_balloon_sb", "refine_contour_gate_sb",
    "refine_contour_blur_sb", "refine_contour_sharpness_sb",
    "refine_contour_rebuild_um_sb",
    "smooth_preserve_topology_cb", "smooth_preview_btn", "smooth_sigma_xy_sb",
    "smooth_sigma_z_sb", "smooth_standalone_sigma_xy_sb",
    "smooth_standalone_sigma_z_sb",
    "split_about_btn",
    "split_accept_btn", "split_advanced_btn", "split_anyway_btn",
    "split_arbiter_cb", "split_arbiter_pct_sb", "split_auto_advance_cb",
    "split_auto_smooth_cb", "split_cancel_preview_btn",
    "split_compactness_sb", "split_contour_cb", "split_detect_bins_sb",
    "split_detect_min_dist_sb", "split_detect_seeds_btn", "split_edt_cb",
    "split_fragment_list", "split_guide_browse_btn", "split_guide_load_btn",
    "split_guide_show_cb", "split_guide_source_combo",
    "split_guided_summary_btn",
    "split_min_distance_sb", "split_min_size_sb", "split_next_seed_btn",
    "split_piece_list", "split_preview_btn", "split_refine_further_btn",
    "split_refine_guided_cb", "split_refine_merge_small_cb",
    "split_refine_seeds_btn", "split_reject_btn", "split_rw_beta_sb",
    "split_seed_h_sb", "split_seeds_from_combo",
    "split_solver_combo", "split_start_btn", "split_target_count_sb",
    "split_undo_seed_btn", "split_wall_channel_combo", "split_wall_clip_sb",
    "split_wall_ridge_cb", "split_walls_from_combo",
    # --- Brick 2b: the promoted tick and the eleven new knobs ---------
    "split_target_count_cb", "split_seed_smooth_sb", "split_max_sigma_sb",
    "split_seed_thresh_sb", "split_seed_clip_sb", "split_lift_mode_combo",
    "split_seed_gamma_sb", "split_seed_bg_sb", "split_seed_denoise_sb",
    "split_wall_smooth_sb", "split_refine_radius_sb", "split_refine_eps_sb",
    # --- Brick U3: "Show what the tool sees".  Its sentence is a
    # TOOLTIP and deliberately not a HELP_LINES entry -- an entry there
    # is rendered as a grey line on the page, and this page has 4 px of
    # its ceiling left (UX round 1.4).  Same reasoning, same precedent,
    # as `split_about_btn`.
    "split_show_walls_cb",
)


class TestTooltips:
    @pytest.mark.parametrize("name", TOOLTIP_GOLDEN)
    def test_the_control_still_explains_itself(self, widget, name):
        assert getattr(widget, name).toolTip().strip(), (
            "{} lost its tooltip".format(name)
        )

    def test_the_footer_buttons_say_what_they_do_TO_THE_CELL(self, widget):
        """The two the owner found confusing."""
        accept = widget.split_accept_btn.toolTip().lower()
        reject = widget.split_reject_btn.toolTip().lower()
        assert "ticked" in accept and "never deleted" in accept
        assert "back" in reject and "volume" in reject

    def test_the_refine_fragments_tooltip_no_longer_promises_a_tab_switch(
            self, widget):
        """The old text said "open the Refine tab"; the method contains no
        tab switch at all -- it fills the fragment list in place."""
        tip = widget.split_refine_further_btn.toolTip().lower()
        assert "refine tab" not in tip
        assert "one at a time" in tip

    def test_the_solver_tooltip_says_label_diffusion_is_not_smoothing(
            self, widget):
        tip = widget.split_solver_combo.toolTip().lower()
        assert "not smoothing" in tip
        assert "u-segment3d" in tip


class TestShippedLabels:
    def test_the_solve_with_row_uses_the_plain_labels(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        combo = widget.split_solver_combo
        assert combo.count() == len(M.SPLIT_SOLVERS)
        assert [combo.itemText(i) for i in range(combo.count())] == [
            "Sharp cut (watershed)",
            "Soft cut (random walker)",
            # The wall channel can be ANY channel, so the
            # ACTION stops naming the membrane.  The wall TYPE
            # "Membrane sheets" keeps its name (see WALLS_FROM_LABELS).
            "Settle onto the wall signal (label diffusion)",
        ]

    def test_relabelling_did_not_move_a_single_index(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        for i, token in enumerate(M.SPLIT_SOLVERS):
            widget.split_solver_combo.setCurrentIndex(i)
            assert widget.get_split_params()["solver"] == token

    def test_the_arbiter_says_cut_not_split(self, widget):
        assert widget.split_arbiter_cb.text() == (
            "Only cut where I can see a wall"
        )

    def test_both_disclosures_carry_a_chevron(self, widget):
        """Finding 10: they rendered as ordinary buttons whose label
        happened to be a summary, with no affordance at all."""
        from napari_epicycle.annotator import advanced_features_widget as M

        assert widget.split_guided_summary_btn.text().startswith(
            M._CHEVRON_CLOSED)
        assert widget.split_advanced_btn.text().startswith(M._CHEVRON_CLOSED)
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_advanced_btn.setChecked(True)
        assert widget.split_guided_summary_btn.text().startswith(
            M._CHEVRON_OPEN)
        assert widget.split_advanced_btn.text().startswith(M._CHEVRON_OPEN)

    def test_the_page_description_is_one_line_now(self, widget):
        """Finding 5: a 10-line essay headed the page, always visible."""
        import inspect

        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        src = inspect.getsource(
            AdvancedFeaturesWidget._build_assisted_split_section)
        assert "1. Select a merged label" not in src
        assert "Euclidean Distance Transform of the label shape" not in src
        # ...and Brick U2 (UX round 4) put an explanation BACK -- behind a
        # chevron.  The visible line is one line; the body is hidden at
        # birth.  That is the whole difference from the 2024 essay, and it
        # is what these two assertions pin.
        from napari_epicycle.annotator import advanced_features_widget as M

        assert widget._split_about_line.text() == M.SPLIT_ABOUT_LINE
        assert widget._split_about_body.isVisibleTo(widget) is False


# --------------------------------------------------------------------------- #
# Brick U2 (UX round 4) -- the description above S1
# --------------------------------------------------------------------------- #
class TestThePageDescription:
    """One sentence you cannot miss, four paragraphs you can.

    It REPLACES `_split_desc_label` in its own slot in the header block
    above S1, so the S1..S6 order does not move and nothing is appended
    to the page after construction -- `test_panel_skeleton.py` owns those
    two assertions and they are untouched by this.
    """

    def test_the_chevron_exists_and_is_closed_at_birth(self, widget):
        from qtpy.QtWidgets import QPushButton

        from napari_epicycle.annotator import advanced_features_widget as M

        assert isinstance(widget.split_about_btn, QPushButton)
        assert widget.split_about_btn.isCheckable() is True
        assert widget.split_about_btn.isChecked() is False
        assert widget.split_about_btn.text() == M._CHEVRON_CLOSED

    def test_the_visible_line_wraps_and_is_never_elided(self, widget):
        """It is the only sentence a first-time user is guaranteed to
        read, so it may not be cut off at the dock's width."""
        assert widget._split_about_line.wordWrap() is True
        assert widget._split_about_line.property(
            "_epicycle_elide_title") is not True
        assert "\u2026" not in widget._split_about_line.text()

    def test_the_line_and_the_body_are_parented(self, widget):
        assert widget.split_about_btn.parentWidget() is not None
        assert widget._split_about_line.parentWidget() is not None
        assert widget._split_about_body.parentWidget() is not None

    def test_only_a_LEFT_click_opens_it(self, widget):
        """A right-click belongs to the context menu, not to us
        (audit M3)."""
        from qtpy.QtCore import QEvent, QPointF, Qt
        from qtpy.QtGui import QMouseEvent

        ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(2, 2),
                         Qt.RightButton, Qt.RightButton, Qt.NoModifier)
        widget._split_about_line.mousePressEvent(ev)
        assert widget.split_about_btn.isChecked() is False

    def test_the_sentence_looks_clickable(self, widget):
        """It is a QLabel pretending to be a link; without a hover cue
        nothing says so (audit M3)."""
        css = widget._split_about_line.styleSheet()
        assert "hover" in css
        assert "underline" in css

    def test_clicking_the_SENTENCE_opens_it_too(self, widget):
        """The chevron is 22 px; the sentence is the click target
        anybody will actually aim at."""
        from qtpy.QtCore import QEvent, QPointF, Qt
        from qtpy.QtGui import QMouseEvent

        ev = QMouseEvent(QEvent.MouseButtonPress, QPointF(2, 2),
                         Qt.LeftButton, Qt.LeftButton, Qt.NoModifier)
        widget._split_about_line.mousePressEvent(ev)
        assert widget.split_about_btn.isChecked() is True

    def test_opening_it_shows_the_body_and_turns_the_chevron(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.split_about_btn.setChecked(True)
        assert widget._split_about_body.isVisibleTo(widget) is True
        assert widget.split_about_btn.text() == M._CHEVRON_OPEN
        widget.split_about_btn.setChecked(False)
        assert widget._split_about_body.isVisibleTo(widget) is False
        assert widget.split_about_btn.text() == M._CHEVRON_CLOSED

    def test_the_body_is_four_paragraphs(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        paras = [t for t in M.SPLIT_ABOUT_BODY.split("\n\n") if t.strip()]
        assert len(paras) == 4, paras

    def test_the_body_wraps(self, widget):
        assert widget._split_about_body.wordWrap() is True

    def test_the_two_inputs_are_BOLD_not_shouted(self, widget):
        """The mockup emphasises them; the first build shipped
        "MARKERS:" and "WALLS:" in capitals, which reads as shouting in
        a four-paragraph explainer (audit M2)."""
        from qtpy.QtCore import Qt

        from napari_epicycle.annotator import advanced_features_widget as M

        assert widget._split_about_body.textFormat() == Qt.RichText
        assert "<b>Markers</b>" in M.SPLIT_ABOUT_BODY
        assert "<b>Walls</b>" in M.SPLIT_ABOUT_BODY
        assert "MARKERS:" not in M.SPLIT_ABOUT_BODY
        assert "WALLS:" not in M.SPLIT_ABOUT_BODY

    def test_it_uses_em_dashes_not_hyphens_for_asides(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        assert "&#8212;" in M.SPLIT_ABOUT_BODY
        assert " - " not in M.SPLIT_ABOUT_BODY

    @pytest.mark.parametrize("word", ["cell", "cells", "Cell", "Cells"])
    def test_neither_string_says_cell(self, word):
        """UX round 0a: this tab reshapes nuclei too."""
        from napari_epicycle.annotator import advanced_features_widget as M

        def words(text):
            out = text
            for ch in ",.;:-\n":
                out = out.replace(ch, " ")
            return out.split()

        assert word not in words(M.SPLIT_ABOUT_LINE)
        assert word not in words(M.SPLIT_ABOUT_BODY)

    def test_it_says_what_the_two_inputs_are(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        low = M.SPLIT_ABOUT_LINE.lower()
        assert "marker" in low and "wall" in low


class TestTheDescriptionGoesAwayDuringASession:
    """Section 4: both halves are hidden at Start.  They are 100+ px of a
    243 px screen, and the page's tallest state has ONE pixel in hand
    (`test_panel_skeleton.py::REVIEW_MARKERS_SCREENS`)."""

    def test_all_three_are_gone_at_start(self, widget):
        widget.update_split_ui_state(split_active=True)
        for attr in ("split_about_btn", "_split_about_line",
                     "_split_about_body"):
            assert getattr(widget, attr).isVisibleTo(widget) is False, attr

    def test_the_line_comes_back_when_the_session_ends(self, widget):
        widget.update_split_ui_state(split_active=True)
        widget.update_split_ui_state(split_active=False)
        assert widget.split_about_btn.isVisibleTo(widget) is True
        assert widget._split_about_line.isVisibleTo(widget) is True

    def test_the_body_comes_back_CLOSED_if_it_was_closed(self, widget):
        """The bug this pins: a loop that sets every hidden widget back
        to visible pops the four paragraphs open on Discard, under a
        chevron the user never touched."""
        assert widget.split_about_btn.isChecked() is False
        widget.update_split_ui_state(split_active=True)
        widget.update_split_ui_state(split_active=False)
        assert widget._split_about_body.isVisibleTo(widget) is False, (
            "the body reopened itself when the session ended"
        )

    def test_the_body_comes_back_OPEN_if_it_was_open(self, widget):
        widget.split_about_btn.setChecked(True)
        widget.update_split_ui_state(split_active=True)
        assert widget._split_about_body.isVisibleTo(widget) is False
        widget.update_split_ui_state(split_active=False)
        assert widget._split_about_body.isVisibleTo(widget) is True

    def test_opening_the_chevron_MID_SESSION_shows_nothing(self, widget):
        """Both halves are hidden, so a stray toggle must not put four
        paragraphs back on the page while the object is punched out."""
        widget.update_split_ui_state(split_active=True)
        widget.split_about_btn.setChecked(True)
        assert widget._split_about_body.isVisibleTo(widget) is False

    def test_the_other_two_tabs_got_the_same_treatment(self, widget):
        """Add carried a 4-step workflow (106 px) and Refine a seven-method
        essay (226 px), both always visible above everything else."""
        import inspect

        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        add_src = inspect.getsource(
            AdvancedFeaturesWidget._build_add_denovo_section)
        assert "1. Pick the image channel" not in add_src
        assert "4. Accept" not in add_src
        ref_src = inspect.getsource(
            AdvancedFeaturesWidget._build_refine_section)
        assert "Preview Refine \u2192 Accept" not in ref_src
        # the PER-METHOD descriptions stay: they are what the user needs
        # once a method is chosen, and only one is on screen at a time.
        assert AdvancedFeaturesWidget._REFINE_METHOD_DESCRIPTIONS[0]


class TestGuideSubPickerBranches:
    """Finding 9: index 2 (a run) and index 3 (a file) revealed the same
    block, so the run picker stayed on screen for an external file."""

    def test_a_run_source_shows_the_run_picker_only(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.split_guided_summary_btn.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_RUN_LABELS)
        assert widget._split_guide_run_row.isVisibleTo(widget) is True
        assert widget._split_guide_file_row.isVisibleTo(widget) is False
        # Load applies to BOTH sources -- it reads whichever was chosen.
        assert widget.split_guide_load_btn.isVisibleTo(widget) is True

    def test_a_file_source_shows_the_file_row_only(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.split_guided_summary_btn.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(
            M.SEEDS_FROM_EXTERNAL_MASK)
        assert widget._split_guide_run_row.isVisibleTo(widget) is False
        assert widget._split_guide_file_row.isVisibleTo(widget) is True
        assert widget.split_guide_load_btn.isVisibleTo(widget) is True

    def test_detect_is_source_0s_finder_and_ONLY_source_0s(self, widget):
        """One finder per card (UX round 2.2).

        Index 1 lost Detect in Brick U1: it is a finder in its own
        right, and showing both said the two did the same thing.  It
        gains Find markers instead, which is what cures bug B."""
        from napari_epicycle.annotator import advanced_features_widget as M

        # Brick 2b: the painting block waits for a session (every control
        # in it is disabled before Start), so the source rule is read
        # inside one.
        widget.update_split_ui_state(split_active=True)
        for idx in range(widget.split_seeds_from_combo.count()):
            widget.split_seeds_from_combo.setCurrentIndex(idx)
            assert widget.split_detect_seeds_btn.isVisibleTo(widget) is (
                idx in M.DETECT_SEED_SOURCES), idx


class TestTheAddSecondary:
    """State 3: add_reject_btn has had ONE label its whole life
    ("Cancel", zero setText calls repo-wide).  It gains exactly one
    transition -- with candidates staged it throws a RESULT away."""

    def test_it_starts_as_cancel(self, widget):
        assert widget.add_reject_btn.text() == "Cancel"

    def test_staged_candidates_make_it_discard(self, widget):
        widget.update_add_ui_state(
            candidate_ids=[3, 4], candidate_sizes={3: 900, 4: 800})
        assert widget.add_reject_btn.text() == "Discard"

    def test_an_empty_result_puts_it_back_to_cancel(self, widget):
        widget.update_add_ui_state(
            candidate_ids=[3], candidate_sizes={3: 900})
        widget.update_add_ui_state(candidate_ids=[])
        assert widget.add_reject_btn.text() == "Cancel"

    def test_the_review_swap_retires_Discard(self, widget):
        """Discard means "throw the staged result away"; in REVIEW the
        result is already written, so the label becomes "Done" (the
        REVIEW-only third value) and never stays "Discard"."""
        widget.update_add_ui_state(
            candidate_ids=[3], candidate_sizes={3: 900})
        assert widget.add_reject_btn.text() == "Discard"
        widget.update_add_ui_state(review_active=True, piece_ids=[3])
        assert widget.add_reject_btn.text() == "Done"
        widget.update_add_ui_state(review_active=False)
        assert widget.add_reject_btn.text() == "Cancel"

    def test_REVIEW_offers_a_live_Done_and_a_dead_Accept(self, widget):
        """State 4 / I-5's third value.

        In REVIEW the cells are already written: `_accept_add` returns at
        its `_add_running` guard, so an enabled primary is a button that
        does nothing, and the secondary -- which runs `_reject_add`, a
        clean exit that keeps the committed work -- was labelled "Cancel"
        while it actually meant "done".
        """
        widget.update_add_ui_state(
            candidate_ids=[3, 4], candidate_sizes={3: 900, 4: 800})
        widget.update_add_ui_state(review_active=True, piece_ids=[3, 4])
        assert widget.add_accept_btn.isEnabled() is False
        assert widget.add_accept_btn.text() == "Accept"
        assert widget.add_reject_btn.isEnabled() is True
        assert widget.add_reject_btn.text() == "Done"

    def test_Done_is_a_REVIEW_only_value(self, widget):
        widget.update_add_ui_state(review_active=True, piece_ids=[3])
        assert widget.add_reject_btn.text() == "Done"
        widget.update_add_ui_state(review_active=False)
        assert widget.add_reject_btn.text() == "Cancel"

    def test_a_late_reject_enabled_does_not_undo_the_REVIEW_labels(self, widget):
        """The orchestrator's REVIEW push carries neither flag, and the
        swap must win over whatever the previous stage left."""
        widget.update_add_ui_state(reject_enabled=True, accept_enabled=True)
        widget.update_add_ui_state(review_active=True, piece_ids=[3])
        assert widget.add_reject_btn.text() == "Done"
        assert widget.add_accept_btn.isEnabled() is False

    def test_the_primary_stops_counting_a_committed_result(self, widget):
        widget.update_add_ui_state(
            candidate_ids=[3, 4], candidate_sizes={3: 900, 4: 800})
        assert widget.add_accept_btn.text() == "Accept (2)"
        widget.update_add_ui_state(review_active=True, piece_ids=[3, 4])
        assert widget.add_accept_btn.text() == "Accept"


class TestOrchestratorUsesFocusTab:
    """The two index-5 sites must be name-based before the tab text moves."""

    def test_no_hardcoded_tab_index_remains(self):
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        for name in ("_refine_add_piece", "_enter_interp_refine_mode"):
            src = inspect.getsource(getattr(Curation3DAnnotator, name))
            assert "setCurrentIndex(5)" not in src, (
                "{} still hard-codes the Refine tab index".format(name)
            )
            assert 'focus_tab("Refine")' in src, (
                "{} must reach the Refine tab by name".format(name)
            )
