"""Contract tests for the guided-Split pickers in the Split tab (Reshape R1).

These instantiate the REAL widget against an offscreen QApplication rather
than inspecting its source, because the thing that must not break is
*behaviour*: the defaults, the reveal logic, the params dict, and the piece
list.  Source-inspection tests rot the moment a helper is extracted
(the stale-test class under `tests/ui_shell/`) and would not have
caught a single one of the bugs this file exists to catch.

The invariant everything else hangs off:

    a user who never opens the collapsed row must see NO change at all.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")


@pytest.fixture(scope="module")
def qapp():
    """An offscreen QApplication, or a clean skip when Qt cannot start."""
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


def _mod():
    from napari_epicycle.annotator import advanced_features_widget as M

    return M



def _is_inside(widget, container) -> bool:
    """True when *widget* is somewhere under *container* in the widget tree.

    Read as "which card is this control in" -- the question a promotion
    out of Advanced has to answer, and the one `findChildren` answers
    without caring how many layouts deep the row sits.
    """
    node = widget
    while node is not None:
        if node is container:
            return True
        node = node.parentWidget()
    return False


# --------------------------------------------------------------------------- #
# THE gate: defaults are today's behaviour
# --------------------------------------------------------------------------- #
class TestDefaultsAreTodaysBehaviour:
    def test_both_pickers_open_on_todays_behaviour(self, widget):
        M = _mod()
        assert widget.split_seeds_from_combo.currentIndex() == M.SEEDS_FROM_PAINTED
        assert widget.split_walls_from_combo.currentIndex() == M.WALLS_FROM_SHAPE

    def test_is_guided_split_is_False_at_the_defaults(self, widget):
        """While this is False the orchestrator MUST run the legacy path."""
        assert widget.is_guided_split() is False

    def test_the_guided_row_is_collapsed(self, widget):
        assert widget.split_guided_summary_btn.isChecked() is False

    def test_the_advanced_row_is_collapsed(self, widget):
        assert widget.split_advanced_btn.isChecked() is False

    def test_the_arbiter_is_OFF_by_default(self, widget):
        """Decided 2026-09-08: a cut you asked for is not second-guessed."""
        assert widget.split_arbiter_cb.isChecked() is False

    def test_the_guide_and_wall_panels_start_hidden(self, widget):
        assert widget._split_guide_container.isVisibleTo(widget) is False
        assert widget._split_walls_container.isVisibleTo(widget) is False

    def test_the_piece_list_and_busy_bar_start_hidden(self, widget):
        assert widget.split_piece_list.isVisibleTo(widget) is False
        assert widget._split_busy_container.isVisibleTo(widget) is False
        assert widget.split_anyway_btn.isVisibleTo(widget) is False

    def test_guided_becomes_True_when_EITHER_picker_moves(self, widget):
        M = _mod()
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_EXTERNAL_MASK)
        assert widget.is_guided_split() is True
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_PAINTED)
        assert widget.is_guided_split() is False
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        assert widget.is_guided_split() is True


# --------------------------------------------------------------------------- #
# reveal logic
# --------------------------------------------------------------------------- #
class TestRevealLogic:
    def test_a_guided_seed_source_reveals_the_guide_picker(self, widget):
        M = _mod()
        widget.split_guided_summary_btn.setChecked(True)
        for idx in sorted(M.GUIDED_SEED_SOURCES):
            widget.split_seeds_from_combo.setCurrentIndex(idx)
            assert widget._split_guide_container.isVisibleTo(widget) is True

    def test_a_legacy_seed_source_hides_the_guide_picker(self, widget):
        M = _mod()
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_EXTERNAL_MASK)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_DETECTED)
        assert widget._split_guide_container.isVisibleTo(widget) is False

    def test_membrane_walls_reveal_the_channel_knobs_and_the_arbiter(self, widget):
        M = _mod()
        widget.split_guided_summary_btn.setChecked(True)
        # Brick 2a: the wall knobs live inside the Walls sub-section, and
        # only one of the three is open at a time.  Brick 2b moved the
        # ARBITER into How-to-cut, beside the solver it modifies, so the
        # two halves of this rule are now read in two cards.
        widget.split_walls_btn.setChecked(True)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        assert widget._split_walls_container.isVisibleTo(widget) is True
        widget.split_solve_btn.setChecked(True)
        assert widget.split_arbiter_cb.isVisibleTo(widget) is True

    def test_leaving_membrane_walls_also_UNCHECKS_the_arbiter(self, widget):
        """The engine REFUSES arbiter=True with a shape-derived elevation,
        so a checked arbiter must never survive behind a hidden
        checkbox -- that would be a ValueError from the spec constructor."""
        M = _mod()
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        widget.split_arbiter_cb.setChecked(True)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_SHAPE)
        assert widget.split_arbiter_cb.isChecked() is False
        assert widget.get_split_params()["arbiter"] is False

    def test_the_summary_line_names_both_current_choices(self, widget):
        """The line is ELIDED to the button width now, so the
        whole of it lives in the tooltip -- which is where a reader who
        cannot see the end of the line goes looking."""
        M = _mod()
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_RUN_LABELS)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        full = widget.split_guided_summary_btn.toolTip()
        assert M.SEEDS_FROM_LABELS[M.SEEDS_FROM_RUN_LABELS] in full
        assert M.WALLS_FROM_LABELS[M.WALLS_FROM_MEMBRANE] in full
        assert "Markers:" in full and "Walls:" in full

    def test_the_summary_toggles_the_guided_container(self, widget):
        assert widget._split_guided_container.isVisibleTo(widget) is False
        widget.split_guided_summary_btn.setChecked(True)
        assert widget._split_guided_container.isVisibleTo(widget) is True
        widget.split_guided_summary_btn.setChecked(False)
        assert widget._split_guided_container.isVisibleTo(widget) is False


# --------------------------------------------------------------------------- #
# get_split_params -- the recipe as a dict
# --------------------------------------------------------------------------- #
class TestSplitParams:
    def test_defaults_match_the_validated_sandbox_recipe(self, widget):
        p = widget.get_split_params()
        assert p["guided"] is False
        assert p["clip"] == pytest.approx(0.01)
        assert p["ridge"] is True
        assert p["h"] == pytest.approx(0.15)
        assert p["min_size"] == 20
        assert p["solver"] == "watershed"
        assert p["arbiter"] is False

    def test_the_params_dict_is_json_serialisable(self, widget):
        """A preset has to be savable and a derived run has to record it."""
        import json

        json.loads(json.dumps(widget.get_split_params()))

    def test_a_zero_piece_count_means_None_not_zero(self, widget):
        """An UNTICKED "I expect exactly n cells" means "do not force a
        count"; the engine wants None.  Passing 0 through would ask for
        zero pieces.  (Brick 2b moved the off switch from a 0 in the spin
        box onto its own tick -- "0 cells" was never a thing to want.)"""
        assert widget.get_split_params()["target_count"] is None
        widget.split_target_count_sb.setValue(3)
        assert widget.get_split_params()["target_count"] is None
        widget.split_target_count_cb.setChecked(True)
        assert widget.get_split_params()["target_count"] == 3

    def test_the_solver_combo_maps_onto_the_engine_enum(self, widget):
        from packages.epicycle_annotator.reshape import SOLVERS

        M = _mod()
        assert set(M.SPLIT_SOLVERS) <= set(SOLVERS)
        for i, name in enumerate(M.SPLIT_SOLVERS):
            widget.split_solver_combo.setCurrentIndex(i)
            assert widget.get_split_params()["solver"] == name

    def test_every_params_key_is_consumable_by_the_engine(self, widget):
        """A knob that reaches nothing must not exist (asymmetry b).

        Every param this dict carries either drives a ReshapeSpec field or is
        a ReshapeSpec param name -- nothing is collected and dropped, which is
        exactly the latent dead `connectivity` knob in the Add tab.
        """
        from packages.epicycle_annotator.reshape.spec import DEFAULT_PARAMS

        p = widget.get_split_params()
        # keys the orchestrator turns into ReshapeSpec FIELDS or into UI state
        structural = {
            "seeds_from", "walls_from", "guided", "solver", "arbiter",
            "refine_guided", "refine_merge_small",
            # Brick 2a: an INDEX the recipe maps to a seed token, and a
            # margin read at Start by `_snapshot_label_roi` -- neither is
            # a ReshapeSpec param, and both are consumed.
            "seed_detector", "search_padding",
            # Brick 2b: which kind of lift the "Lift dim signal" tick
            # runs.  It is not a param -- it picks the `gamma` / `log`
            # TOKEN inside `seed_preproc` / `wall_preproc`, which are.
            "lift_mode",
        }
        for key in p:
            assert key in structural or key in DEFAULT_PARAMS, (
                "get_split_params key {!r} reaches neither a ReshapeSpec "
                "field nor a ReshapeSpec param".format(key)
            )

    def test_params_survive_a_missing_widget(self, widget):
        """Defensive: the getter must never raise mid-Preview."""
        del widget.split_solver_combo
        assert widget.get_split_params()["solver"] == "watershed"


# --------------------------------------------------------------------------- #
# the checkable piece list
# --------------------------------------------------------------------------- #
class TestPieceList:
    def test_pieces_populate_ticked_and_report_sizes(self, widget):
        widget.update_split_guided_ui(
            pieces=[11, 12], piece_sizes={11: 4000, 12: 3800},
            caption="unticked pieces stay part of cell 412",
        )
        assert widget.split_piece_list.count() == 2
        assert widget.get_checked_split_pieces() == [11, 12]
        assert "4,000 vx" in widget.split_piece_list.item(0).text()

    def test_a_sliver_starts_UNTICKED_as_a_recommendation(self, widget):
        """Brick 2b moved the 0.2 x median arithmetic into
        `reshape.quality` -- ONE rule, not the three copies that lived in
        this widget ((c)).  The recommendation is unchanged; it
        arrives as a note now instead of being recomputed here, and the
        note is what the panel acts on."""
        from packages.epicycle_annotator.reshape import quality_warnings

        result = type("R", (), {})()
        result.status = "ok"
        result.n_seeds = 3
        result.n_pieces = 3
        result.coverage = 1.0
        result.piece_sizes = [4000, 3800, 100]
        result.piece_stats = [
            {"id": 1, "voxels": 4000}, {"id": 2, "voxels": 3800},
            {"id": 3, "voxels": 100},
        ]
        widget.update_split_guided_ui(
            pieces=[1, 2, 3], piece_sizes={1: 4000, 2: 3800, 3: 100},
            quality_notes=quality_warnings(result, target_id=9),
        )
        assert widget.get_checked_split_pieces() == [1, 2]

    def test_the_caption_says_where_unticked_voxels_go(self, widget):
        """Unticking piece 3 means 'not a separate cell', NEVER 'delete that
        tissue' -- and the panel must say so."""
        widget.update_split_guided_ui(
            pieces=[1, 2], piece_sizes={1: 10, 2: 10},
            caption="unticked pieces stay part of cell 412",
        )
        assert widget.split_piece_caption.isVisibleTo(widget)
        assert "stay part of cell 412" in widget.split_piece_caption.text()

    def test_ticking_retitles_accept_with_the_count(self, widget):
        from qtpy.QtCore import Qt

        widget.update_split_guided_ui(
            pieces=[1, 2], piece_sizes={1: 10, 2: 10})
        assert widget.split_accept_btn.text() == "Accept (2)"
        widget.split_piece_list.item(0).setCheckState(Qt.Unchecked)
        assert widget.split_accept_btn.text() == "Accept (1)"

    def test_unticking_fires_the_orchestrator_callback_once(self, widget):
        from qtpy.QtCore import Qt

        fired = []
        widget.set_split_callbacks(on_piece_check_changed=lambda: fired.append(1))
        widget.update_split_guided_ui(
            pieces=[1, 2], piece_sizes={1: 10, 2: 10})
        fired.clear()
        widget.split_piece_list.item(0).setCheckState(Qt.Unchecked)
        assert len(fired) == 1

    def test_a_flag_is_shown_on_the_piece_row(self, widget):
        widget.update_split_guided_ui(
            pieces=[1], piece_sizes={1: 500}, piece_flags={1: "dividing"})
        assert "dividing" in widget.split_piece_list.item(0).text()

    def test_clear_returns_everything_to_idle(self, widget):
        widget.update_split_guided_ui(
            pieces=[1, 2], piece_sizes={1: 10, 2: 10},
            caption="x", arbiter_message="y", split_anyway_visible=True,
            busy=True,
        )
        widget.clear_split_guided_ui()
        assert widget.split_piece_list.count() == 0
        assert widget.split_piece_list.isVisibleTo(widget) is False
        assert widget.split_anyway_btn.isVisibleTo(widget) is False
        assert widget._split_busy_container.isVisibleTo(widget) is False
        assert widget.split_accept_btn.text() == "Accept"


# --------------------------------------------------------------------------- #
# the arbiter's verdict + Split anyway
# --------------------------------------------------------------------------- #
class TestArbiterVerdictUI:
    def test_a_dividing_verdict_shows_its_message_and_the_override(self, widget):
        widget.update_split_guided_ui(
            arbiter_message=(
                "kept whole - no membrane between the nuclei "
                "(possibly dividing)"
            ),
            split_anyway_visible=True,
        )
        assert widget.split_arbiter_msg_label.isVisibleTo(widget)
        assert "possibly dividing" in widget.split_arbiter_msg_label.text()
        assert widget.split_anyway_btn.isVisibleTo(widget)

    def test_split_anyway_is_wired_to_its_callback(self, widget):
        fired = []
        widget.set_split_callbacks(on_split_anyway=lambda: fired.append(1))
        widget.split_anyway_btn.click()
        assert fired == [1]

    def test_engine_messages_are_surfaced_verbatim(self, widget):
        """The six carved-out guard prints are the ONLY signal that a guard
        fired; losing them is worse than not having them."""
        widget.update_split_guided_ui(
            engine_messages=["u-Segment3D is not installed - used watershed instead"]
        )
        assert widget.split_engine_msg_label.isVisibleTo(widget)
        assert "u-Segment3D" in widget.split_engine_msg_label.text()


# --------------------------------------------------------------------------- #
# the busy bar (the guided Preview is threaded)
# --------------------------------------------------------------------------- #
class TestBusyBar:
    def test_busy_shows_the_bar_and_the_stop_button(self, widget):
        widget.update_split_guided_ui(busy=True)
        assert widget._split_busy_container.isVisibleTo(widget) is True
        assert widget.split_cancel_preview_btn.isVisibleTo(widget) is True

    def test_progress_makes_the_bar_determinate(self, widget):
        widget.update_split_guided_ui(busy=True, busy_progress=(3, 8))
        assert widget.split_busy_bar.maximum() == 8
        assert widget.split_busy_bar.value() == 3

    def test_zero_total_falls_back_to_a_marquee(self, widget):
        widget.update_split_guided_ui(busy=True, busy_progress=(0, 0))
        assert widget.split_busy_bar.maximum() == 0

    def test_the_stop_button_is_wired(self, widget):
        fired = []
        widget.set_split_callbacks(on_cancel_preview=lambda: fired.append(1))
        widget.split_cancel_preview_btn.click()
        assert fired == [1]

    def test_the_stop_button_says_abandon_not_stop_the_computation(self, widget):
        """Cancel is polled BETWEEN stages, so a stage already entered runs to
        completion.  A button that promises to stop the maths would lie."""
        tip = widget.split_cancel_preview_btn.toolTip()
        assert "thrown away" in tip or "throw" in tip
        assert widget.split_cancel_preview_btn.text() == "Stop waiting"


# --------------------------------------------------------------------------- #
# the guide picker
# --------------------------------------------------------------------------- #
class TestGuidePicker:
    def test_an_empty_source_list_says_so_and_disables_the_combo(self, widget):
        widget.set_guide_sources([])
        assert widget.split_guide_source_combo.isEnabled() is False
        assert widget.selected_guide_source() is None
        assert "No other run" in widget.split_guide_source_combo.itemText(0)

    def test_a_source_round_trips_through_UserRole(self, widget):
        class _Src:
            label = "Nuclei run - reconstructed - you are trusting these labels."

        src = _Src()
        widget.set_guide_sources([src])
        assert widget.split_guide_source_combo.isEnabled() is True
        assert widget.selected_guide_source() is src
        assert widget.split_guide_source_combo.itemText(0) == src.label

    def test_show_guide_is_disabled_until_a_mask_is_loaded(self, widget):
        assert widget.split_guide_show_cb.isEnabled() is False
        widget.update_split_guided_ui(guide_loaded=True)
        assert widget.split_guide_show_cb.isEnabled() is True

    def test_the_guide_note_is_shown_only_when_there_is_one(self, widget):
        # Checked against its OWN parent: the whole guide panel is hidden
        # until a guided seed source is picked, so isVisibleTo(widget) would
        # be False either way and prove nothing.
        note = widget.split_guide_note_label
        parent = note.parentWidget()
        assert note.isVisibleTo(parent) is False
        widget.update_split_guided_ui(
            guide_note="This session is showing the patch at half size.")
        assert note.isVisibleTo(parent) is True
        assert "half size" in note.text()

    def test_browse_and_load_are_wired(self, widget):
        fired = []
        widget.set_split_callbacks(
            on_guide_browse=lambda: fired.append("browse"),
            on_guide_load=lambda: fired.append("load"),
            on_guide_show_toggled=lambda v: fired.append(("show", v)),
        )
        widget.split_guide_browse_btn.click()
        widget.split_guide_load_btn.click()
        widget.split_guide_show_cb.setChecked(False)
        assert fired == ["browse", "load", ("show", False)]

    def test_guided_needs_guide_mask_only_for_the_two_mask_sources(self, widget):
        M = _mod()
        for idx, expected in (
            (M.SEEDS_FROM_PAINTED, False),
            (M.SEEDS_FROM_DETECTED, False),
            (M.SEEDS_FROM_RUN_LABELS, True),
            (M.SEEDS_FROM_EXTERNAL_MASK, True),
        ):
            widget.split_seeds_from_combo.setCurrentIndex(idx)
            assert widget.guided_needs_guide_mask() is expected


# --------------------------------------------------------------------------- #
# help + housekeeping
# --------------------------------------------------------------------------- #
class TestHelpAndHousekeeping:
    def test_the_reshape_tab_has_a_help_button_and_its_guide_exists(self):
        from napari_epicycle.annotator import advanced_features_widget as M

        guide = M._GUIDES_DIR / "annotator_3d_reshape.md"
        assert guide.exists(), "the Reshape tab's ? points at a missing guide"

    def test_refine_still_holds_tab_index_5(self, qapp):
        """The pieces-to-Refine handoff hard-references index 5."""
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        w = AdvancedFeaturesWidget(mode="3d_curation")
        try:
            assert w._tool_tabs.tabText(5) == "Refine"
            assert w._tool_tabs.tabText(4) == "Reshape"
        finally:
            w.setParent(None)

    def test_no_new_tab_was_added(self, qapp):
        """Owner decision Q3/Q8: no new panel, no new tab."""
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        w = AdvancedFeaturesWidget(mode="3d_curation")
        try:
            names = [w._tool_tabs.tabText(i) for i in range(w._tool_tabs.count())]
            assert names == [
                "SAM", "Membrane", "Scribble", "Histogram",
                "Reshape", "Refine", "Add",
            ]
            # The tab text is the ONLY thing the rename moved: no tab was
            # added (owner decision Q5 -- no Derive placeholder) and none
            # was reordered.
            assert not any(n == "Split" for n in names)
        finally:
            w.setParent(None)

    def test_the_R1_methods_stay_cp1252_safe(self):
        """Only what this brick ADDED is held to it -- the file's older copy
        legitimately uses typographic dashes and arrows.  Checked per method
        via inspect, so it cannot silently widen to unrelated code."""
        import inspect

        from napari_epicycle.annotator import advanced_features_widget as M

        targets = [
            M.SEEDS_FROM_LABELS, M.WALLS_FROM_LABELS, M.SPLIT_SOLVERS,
            # Brick 2a
            M.SEED_DETECTORS, M.SEED_DETECTOR_LABELS, M.WALL_KINDS,
            tuple(M.PREP_TICK_LABELS), tuple(M.PREP_TICK_LABELS.values()),
            tuple(M.HELP_LINES), tuple(M.HELP_LINES.values()),
            tuple(M.ITEM_HELP),
            tuple(t for texts in M.ITEM_HELP.values() for t in texts),
            (M._PREP_ORDER_SENTENCE,),
        ]
        for tup in targets:
            for value in tup:
                value.encode("cp1252")

        W = M.AdvancedFeaturesWidget
        for name in (
            "is_guided_split", "guided_needs_guide_mask",
            "_refresh_split_guided_summary", "_on_split_seeds_from_changed",
            "_on_split_walls_from_changed", "_on_split_piece_check_changed",
            "get_checked_split_pieces", "set_guide_sources",
            "selected_guide_source", "get_split_params", "set_wall_channels",
            "update_split_guided_ui", "clear_split_guided_ui",
            # --- Brick 2a ---
            "set_seed_channels", "get_checked_marker_ids",
            "_populate_marker_list", "_set_marker_busy",
            "_on_split_subsection_toggled", "_on_split_prep_toggled",
            "_on_marker_knob_changed", "_refresh_split_prep_chips",
            "_refresh_wall_item_availability", "_prep_ticks",
            "_marker_prep_chain", "_wall_prep_chain", "_chain_summary",
            "_subsection_chevron", "_refresh_chevron_text",
            "_set_elided_button_text", "_helper_line", "_item_help_label",
            "_refresh_item_help", "_refresh_marker_count_line",
            "_clear_marker_list", "_set_all_marker_checks",
        ):
            src = inspect.getsource(getattr(W, name))
            for lineno, line in enumerate(src.splitlines(), 1):
                try:
                    line.encode("cp1252")
                except UnicodeEncodeError as exc:  # pragma: no cover
                    pytest.fail("{} line {} is not cp1252-safe: {}".format(
                        name, lineno, exc))


# --------------------------------------------------------------------------- #
# Brick 2a -- the wire format
# --------------------------------------------------------------------------- #
class TestItemOrderIsTheWireFormat:
    def test_no_existing_index_moved(self, widget):
        M = _mod()
        assert M.SEEDS_FROM_PAINTED == 0
        assert M.SEEDS_FROM_DETECTED == 1
        assert M.SEEDS_FROM_RUN_LABELS == 2
        assert M.SEEDS_FROM_EXTERNAL_MASK == 3
        assert M.SEEDS_FROM_IMAGE == 4
        assert M.WALLS_FROM_SHAPE == 0
        assert M.WALLS_FROM_MEMBRANE == 1
        assert M.WALL_KINDS[0] == "distance"
        assert M.WALL_KINDS[1] == "membrane"

    def test_every_combo_has_one_label_per_item(self, widget):
        M = _mod()
        assert widget.split_seeds_from_combo.count() == len(M.SEEDS_FROM_LABELS)
        assert widget.split_walls_from_combo.count() == len(M.WALLS_FROM_LABELS)
        assert len(M.WALLS_FROM_LABELS) == len(M.WALL_KINDS) == 9
        assert widget.split_seed_detector_combo.count() == len(M.SEED_DETECTORS)
        assert len(M.SEED_DETECTOR_LABELS) == len(M.SEED_DETECTORS) == 6

    def test_the_detectors_are_in_the_agreed_order(self, widget):
        """Commonest first -- and the order IS the wire format."""
        M = _mod()
        assert M.SEED_DETECTORS == (
            "peaks", "hmax", "blobs_log", "blobs_dog", "threshcc", "hmin")

    def test_the_engine_knows_every_wall_kind_the_combo_offers(self, widget):
        from packages.epicycle_annotator.reshape.spec import WALLS

        M = _mod()
        for kind in M.WALL_KINDS:
            assert kind in WALLS
        assert len(set(M.WALL_KINDS)) == len(M.WALL_KINDS)

    def test_the_engine_knows_every_detector_the_combo_offers(self, widget):
        from packages.epicycle_annotator.reshape.spec import (
            IMAGE_DERIVED_SEEDS,
        )

        M = _mod()
        for token in M.SEED_DETECTORS:
            assert token in IMAGE_DERIVED_SEEDS

    def test_the_image_derived_wall_indices_match_the_engines_set(self, widget):
        from packages.epicycle_annotator.reshape.spec import (
            IMAGE_DERIVED_WALLS,
        )

        M = _mod()
        mine = {M.WALL_KINDS[i] for i in M.IMAGE_DERIVED_WALL_INDICES}
        assert mine == set(IMAGE_DERIVED_WALLS)

    def test_is_guided_split_covers_index_4_and_every_new_wall(self, widget):
        M = _mod()
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
        assert widget.is_guided_split() is True
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_PAINTED)
        for idx in range(1, len(M.WALL_KINDS)):
            widget.split_walls_from_combo.setCurrentIndex(idx)
            assert widget.is_guided_split() is True, idx
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_SHAPE)
        assert widget.is_guided_split() is False

    def test_the_params_guided_flag_keeps_its_own_divergent_meaning(self, widget):
        """Rule 3: `get_split_params()["guided"]` is in the frozen
        dict and pinned by the golden test.  Routing is `is_guided_split()`
        only.  Written down so nobody "fixes" the divergence."""
        M = _mod()
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_DETECTED)
        assert widget.get_split_params()["guided"] is True
        assert widget.is_guided_split() is False


class TestDetectIsHiddenForEverySelfFindingSource:
    def test_detect_is_hidden_for_all_three(self, widget):
        M = _mod()
        for idx in sorted(M.IMAGE_DETECTOR_SOURCES):
            widget.split_seeds_from_combo.setCurrentIndex(idx)
            assert widget.split_detect_seeds_btn.isVisibleTo(widget) is False, idx

    def test_detect_comes_back_for_the_PAINTED_source_only(self, widget):
        """One finder per card (UX round 2.2).  Index 1 lost Detect in
        Brick U1 and gained Find markers, because "Shape thickness
        peaks" IS a finder -- and showing the painting finder beside it
        is what let bug B discard everything the user painted."""
        M = _mod()
        # Brick 2b: the painting block waits for a session, because
        # every control in it is disabled before Start.
        widget.update_split_ui_state(split_active=True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
        for idx in range(widget.split_seeds_from_combo.count()):
            widget.split_seeds_from_combo.setCurrentIndex(idx)
            assert widget.split_detect_seeds_btn.isVisibleTo(widget) is (
                idx in M.DETECT_SEED_SOURCES), idx

    def test_detect_is_not_on_screen_before_the_session_starts(self, widget):
        """It cannot run until the cell is held out of the volume, and it
        is disabled until then -- so it is not 98 px of the viewport."""
        assert widget.split_detect_seeds_btn.isVisibleTo(widget) is False

    def test_the_image_rows_appear_ONLY_for_index_4(self, widget):
        M = _mod()
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(True)
        for idx in range(widget.split_seeds_from_combo.count()):
            widget.split_seeds_from_combo.setCurrentIndex(idx)
            shown = widget._split_image_seed_container.isVisibleTo(widget)
            assert shown is (idx == M.SEEDS_FROM_IMAGE), idx

    def test_find_markers_appears_wherever_detect_does_not(self, widget):
        """Sources 1-4 since Brick U1 -- index 1 joined them (bug B)."""
        M = _mod()
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(True)
        for idx in range(widget.split_seeds_from_combo.count()):
            widget.split_seeds_from_combo.setCurrentIndex(idx)
            shown = widget._split_find_container.isVisibleTo(widget)
            assert shown is (idx in M.FIND_MARKER_SOURCES), idx

    def test_the_two_finder_sets_partition_the_five_sources(self, widget):
        """One finder per card, stated as a set law: every source has
        exactly one, and no source has both."""
        M = _mod()
        n = widget.split_seeds_from_combo.count()
        assert set(M.DETECT_SEED_SOURCES) | set(M.FIND_MARKER_SOURCES) == (
            set(range(n)))
        assert not (set(M.DETECT_SEED_SOURCES)
                    & set(M.FIND_MARKER_SOURCES))

    def test_the_marker_work_rows_are_keyed_on_the_SESSION(self, widget):
        """Add marker / Undo / Edit markers belong to the session, not
        to the source: every source can gain a marker by hand, and the
        sources whose finder most often misses one used to be exactly
        the sources where you could not add it (UX round 2.2)."""
        M = _mod()
        widget.update_split_ui_state(split_active=True)
        for idx in range(widget.split_seeds_from_combo.count()):
            widget.split_seeds_from_combo.setCurrentIndex(idx)
            assert widget._split_paint_rows_container.isVisibleTo(
                widget) is True, idx
        assert M is not None

    def test_the_buttons_say_marker(self, widget):
        """UX round 2.2: "seed" is an implementation word."""
        assert widget.split_next_seed_btn.text() == "Add marker"
        assert widget.split_undo_seed_btn.text() == "Undo"
        assert widget.split_refine_seeds_btn.text() == "Edit markers"


class TestArbiterVisibilityFollowsTheWallKind:
    def test_every_image_derived_wall_shows_the_channel_and_the_arbiter(
            self, widget):
        M = _mod()
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_walls_btn.setChecked(True)
        for idx in range(widget.split_walls_from_combo.count()):
            widget.split_walls_from_combo.setCurrentIndex(idx)
            shown = widget._split_walls_container.isVisibleTo(widget)
            assert shown is (idx in M.IMAGE_DERIVED_WALL_INDICES), (
                "wall kind {!r} (index {}) shows the channel row: {}".format(
                    M.WALL_KINDS[idx], idx, shown)
            )

    def test_leaving_an_image_wall_unchecks_the_arbiter(self, widget):
        M = _mod()
        widget.split_walls_from_combo.setCurrentIndex(
            M.WALL_KINDS.index("hybrid"))
        widget.split_arbiter_cb.setChecked(True)
        widget.split_walls_from_combo.setCurrentIndex(
            M.WALL_KINDS.index("skeleton"))
        assert widget.split_arbiter_cb.isChecked() is False


class TestShapeMarkersNeedAShapeWall:
    """Rule 6, measured: the `distance` detector reads the WALL."""

    def _model_enabled(self, widget, idx):
        return widget.split_walls_from_combo.model().item(idx).isEnabled()

    def test_no_wall_and_between_the_spines_are_disabled(self, widget):
        M = _mod()
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_DETECTED)
        for idx in sorted(M.SHAPE_MARKER_FORBIDDEN_WALLS):
            assert self._model_enabled(widget, idx) is False, M.WALL_KINDS[idx]

    def test_the_reason_is_on_the_item(self, widget):
        M = _mod()
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_DETECTED)
        idx = M.WALL_KINDS.index("none")
        tip = widget.split_walls_from_combo.model().item(idx).toolTip()
        assert "shape wall" in tip

    def test_every_other_source_leaves_them_available(self, widget):
        M = _mod()
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_DETECTED)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
        for idx in sorted(M.SHAPE_MARKER_FORBIDDEN_WALLS):
            assert self._model_enabled(widget, idx) is True, M.WALL_KINDS[idx]

    def test_a_forbidden_pair_already_on_screen_is_moved_off_it(self, widget):
        M = _mod()
        widget.split_walls_from_combo.setCurrentIndex(
            M.WALL_KINDS.index("none"))
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_DETECTED)
        assert widget.split_walls_from_combo.currentIndex() == M.WALLS_FROM_SHAPE


# --------------------------------------------------------------------------- #
# Brick 2a -- S3's three sibling sub-sections
# --------------------------------------------------------------------------- #
class TestTheThreeSubSections:
    def test_markers_opens_first(self, widget):
        assert widget.split_markers_btn.isChecked() is True
        assert widget.split_walls_btn.isChecked() is False
        assert widget.split_solve_btn.isChecked() is False

    def test_only_one_is_open_at_a_time(self, widget):
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_walls_btn.setChecked(True)
        assert widget.split_markers_btn.isChecked() is False
        assert widget._split_markers_container.isVisibleTo(widget) is False
        assert widget._split_walls_group_container.isVisibleTo(widget) is True
        widget.split_solve_btn.setChecked(True)
        assert widget.split_walls_btn.isChecked() is False
        assert widget._split_solve_container.isVisibleTo(widget) is True

    def test_closing_the_open_one_leaves_all_three_closed(self, widget):
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(False)
        for attr, box in widget._SPLIT_SUBSECTIONS:
            assert getattr(widget, attr).isChecked() is False
            assert getattr(widget, box).isVisibleTo(widget) is False

    def test_each_chevron_shows_its_state(self, widget):
        from napari_epicycle.annotator import advanced_features_widget as M

        widget.split_walls_btn.setChecked(True)
        assert widget.split_walls_btn.text().startswith(M._CHEVRON_OPEN)
        assert widget.split_markers_btn.text().startswith(M._CHEVRON_CLOSED)

    def test_collapsing_SOURCES_hides_all_three(self, widget):
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_guided_summary_btn.setChecked(False)
        for _attr, box in widget._SPLIT_SUBSECTIONS:
            assert getattr(widget, box).isVisibleTo(widget) is False


class TestTheElidedSummary:
    def test_it_elides_rather_than_clipping(self, widget, qapp):
        """A clipped line silently drops the half that names the walls.

        The button shows a SHORT form that fits the 307 px it has; this
        squeezes it narrower still, where even the short form cannot fit,
        and asserts it is shortened with an ellipsis rather than cut."""
        from qtpy.QtWidgets import QApplication

        M = _mod()
        widget.resize(200, 350)
        widget.show()
        QApplication.processEvents()
        try:
            widget.split_seeds_from_combo.setCurrentIndex(
                M.SEEDS_FROM_RUN_LABELS)
            QApplication.processEvents()
            btn = widget.split_guided_summary_btn
            full = str(btn.property("_epicycle_full_text"))
            assert len(full) > 20
            assert btn.text() != full, "the summary was not elided"
            assert len(btn.text()) < len(full)
        finally:
            widget.hide()

    def test_the_tooltip_carries_the_REAL_names_not_the_shorthand(self, widget):
        """The button says "markers: run mask"; the tooltip has to say what
        that is, or the shorthand is just a riddle."""
        M = _mod()
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_RUN_LABELS)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        tip = widget.split_guided_summary_btn.toolTip()
        assert M.SEEDS_FROM_LABELS[M.SEEDS_FROM_RUN_LABELS] in tip
        assert M.WALLS_FROM_LABELS[M.WALLS_FROM_MEMBRANE] in tip

    def test_the_button_says_SOURCES(self, widget):
        full = str(widget.split_guided_summary_btn.property(
            "_epicycle_full_text"))
        assert "Sources" in full

    def test_the_tooltip_keeps_its_sentence_after_the_line(self, widget):
        tip = widget.split_guided_summary_btn.toolTip()
        assert "Markers:" in tip
        assert "defaults" in tip


# --------------------------------------------------------------------------- #
# Brick 2a -- preparing the channel
# --------------------------------------------------------------------------- #
class TestPrepareTheChannel:
    def test_local_contrast_is_the_only_marker_tick_on_by_default(self, widget):
        assert widget.split_seed_prep_clahe_cb.isChecked() is True
        for attr in ("split_seed_prep_zcorrect_cb", "split_seed_prep_bgsub_cb",
                     "split_seed_prep_lift_cb", "split_seed_prep_denoise_cb"):
            assert getattr(widget, attr).isChecked() is False
        assert widget.get_split_params()["seed_preproc"] == "clahe"

    def test_the_chain_is_emitted_in_the_order_the_engine_runs_it(self, widget):
        widget.split_seed_prep_denoise_cb.setChecked(True)
        widget.split_seed_prep_zcorrect_cb.setChecked(True)
        widget.split_seed_prep_bgsub_cb.setChecked(True)
        widget.split_seed_prep_lift_cb.setChecked(True)
        assert widget.get_split_params()["seed_preproc"] == (
            "zcorrect+bgsub+gamma+denoise+clahe")

    def test_nothing_ticked_is_the_engines_own_passthrough_word(self, widget):
        widget.split_seed_prep_clahe_cb.setChecked(False)
        assert widget.get_split_params()["seed_preproc"] == "none"

    def test_every_chain_token_is_one_the_engine_understands(self, widget):
        M = _mod()
        widget.split_seed_prep_zcorrect_cb.setChecked(True)
        widget.split_seed_prep_bgsub_cb.setChecked(True)
        widget.split_seed_prep_lift_cb.setChecked(True)
        widget.split_seed_prep_denoise_cb.setChecked(True)
        allowed = {"zcorrect", "bgsub", "gamma", "log", "denoise", "clahe"}
        for key in ("seed_preproc", "wall_preproc"):
            chain = widget.get_split_params()[key]
            if chain == "none":
                continue
            assert set(chain.split("+")) <= allowed, (key, chain)
        assert M.chain_from_ticks({}, ) == "none"

    def test_the_wall_chain_never_carries_local_contrast(self, widget):
        """The wall stage applies its own through `membrane clip`; a second
        one would double-apply it."""
        widget.split_wall_prep_same_cb.setChecked(True)
        assert "clahe" not in widget.get_split_params()["wall_preproc"]

    def test_same_as_markers_MIRRORS_and_greys_the_four_below(self, widget):
        widget.split_seed_prep_zcorrect_cb.setChecked(True)
        widget.split_wall_prep_same_cb.setChecked(True)
        assert widget.split_wall_prep_zcorrect_cb.isChecked() is True
        assert widget.split_wall_prep_zcorrect_cb.isEnabled() is False
        assert widget.get_split_params()["wall_preproc"] == "zcorrect"

    def test_unticking_same_as_markers_hands_the_ticks_back(self, widget):
        """The TRUE edge BORROWS the four rows.  Unticking must give the
        user back what they had, not leave them holding the marker chain
        as though they had chosen it."""
        widget.split_wall_prep_bgsub_cb.setChecked(True)
        widget.split_wall_prep_denoise_cb.setChecked(False)
        # the markers are set to something else entirely
        widget.split_seed_prep_zcorrect_cb.setChecked(True)
        widget.split_seed_prep_denoise_cb.setChecked(True)

        widget.split_wall_prep_same_cb.setChecked(True)
        assert widget.split_wall_prep_zcorrect_cb.isChecked() is True
        assert widget.split_wall_prep_bgsub_cb.isChecked() is False
        assert widget.get_split_params()["wall_preproc"] == "zcorrect+denoise"

        widget.split_wall_prep_same_cb.setChecked(False)
        assert widget.split_wall_prep_zcorrect_cb.isEnabled() is True
        assert widget.split_wall_prep_bgsub_cb.isChecked() is True, (
            "the borrowed rows were not given back"
        )
        assert widget.split_wall_prep_zcorrect_cb.isChecked() is False
        assert widget.split_wall_prep_denoise_cb.isChecked() is False
        assert widget.get_split_params()["wall_preproc"] == "bgsub"

    def test_turning_it_on_twice_does_not_lose_the_saved_ticks(self, widget):
        widget.split_wall_prep_bgsub_cb.setChecked(True)
        widget.split_seed_prep_zcorrect_cb.setChecked(True)
        widget.split_wall_prep_same_cb.setChecked(True)
        widget._refresh_split_prep_chips()          # an idempotent re-run
        widget.split_wall_prep_same_cb.setChecked(False)
        assert widget.split_wall_prep_bgsub_cb.isChecked() is True

    def test_the_chip_names_what_would_actually_run(self, widget):
        """The chip is a summary and is elided like the S3 line, so the full
        words live in its stored text (and its tooltip)."""
        def _full(btn):
            return str(btn.property("_epicycle_full_text") or btn.text())

        assert "local contrast" in _full(widget.split_seed_prep_btn)
        widget.split_seed_prep_clahe_cb.setChecked(False)
        assert _full(widget.split_seed_prep_btn).endswith("none")
        assert _full(widget.split_wall_prep_btn).endswith("none")

    def test_the_chip_says_every_step_it_would_run(self, widget):
        widget.split_seed_prep_zcorrect_cb.setChecked(True)
        widget.split_seed_prep_denoise_cb.setChecked(True)
        full = str(widget.split_seed_prep_btn.property("_epicycle_full_text"))
        for words in ("even out depth", "despeckle", "local contrast"):
            assert words in full, (words, full)

    def test_the_order_sentence_matches_the_engine(self, widget):
        """A run-order sentence that lies is worse than none at all."""
        import inspect

        from packages.epicycle_annotator.reshape import preproc

        M = _mod()
        src = inspect.getsource(preproc.apply_preproc)
        order = [tok for tok in ("zcorrect", "bgsub", "gamma", "log",
                                 "denoise", "clahe")
                 if '"{}" in tok'.format(tok) in src]
        assert order == ["zcorrect", "bgsub", "gamma", "log", "denoise",
                         "clahe"]
        said = M._PREP_ORDER_SENTENCE.lower()
        positions = [said.index(word) for word in (
            "even out depth", "remove haze", "lift dim signal", "despeckle",
            "local contrast")]
        assert positions == sorted(positions)

    def test_the_prep_chips_open_and_close(self, widget):
        M = _mod()
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_markers_btn.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
        widget.split_seed_prep_btn.setChecked(True)
        assert widget._split_seed_prep_container.isVisibleTo(widget) is True
        widget.split_seed_prep_btn.setChecked(False)
        assert widget._split_seed_prep_container.isVisibleTo(widget) is False


# --------------------------------------------------------------------------- #
# Brick 2a -- the marker picker
# --------------------------------------------------------------------------- #
class TestTheMarkerPicker:
    def test_nothing_found_means_None_not_an_empty_pick(self, widget):
        """None = no Find happened (Preview finds them fresh); [] = the user
        unticked every one (Preview refuses).  Two different answers."""
        assert widget.get_checked_marker_ids() is None

    def test_a_find_fills_the_list_all_ticked(self, widget):
        widget.update_split_ui_state(
            marker_ids=[4, 5, 6], marker_stats={4: 900, 5: 120, 6: 40})
        assert widget.split_marker_list.count() == 3
        assert widget.get_checked_marker_ids() == [4, 5, 6]
        assert "Found 3 markers" in widget.split_marker_count_label.text()

    def test_the_rows_say_Marker_n_and_the_size(self, widget):
        widget.update_split_ui_state(
            marker_ids=[4], marker_stats={4: 1204})
        text = widget.split_marker_list.item(0).text()
        assert text.startswith("Marker 1")
        assert "1,204" in text

    def test_a_small_marker_is_NOT_unticked_for_the_user(self, widget):
        """The sliver rule is about PIECES, which have a size distribution.
        A marker is a blob; small does not mean spurious."""
        widget.update_split_ui_state(
            marker_ids=[1, 2], marker_stats={1: 5000, 2: 3})
        assert widget.get_checked_marker_ids() == [1, 2]

    def test_unticking_updates_the_count_line(self, widget):
        from qtpy.QtCore import Qt

        widget.update_split_ui_state(marker_ids=[1, 2, 3])
        widget.split_marker_list.item(1).setCheckState(Qt.Unchecked)
        assert widget.get_checked_marker_ids() == [1, 3]
        assert "2 of 3 ticked" in widget.split_marker_count_label.text()

    def test_tick_none_leaves_an_EMPTY_pick_not_a_cleared_list(self, widget):
        widget.update_split_ui_state(marker_ids=[1, 2])
        widget.split_marker_none_btn.click()
        assert widget.get_checked_marker_ids() == []
        widget.split_marker_all_btn.click()
        assert widget.get_checked_marker_ids() == [1, 2]

    def test_the_summary_carries_the_ticked_count(self, widget):
        widget.update_split_ui_state(marker_ids=[1, 2, 3])
        full = str(widget.split_guided_summary_btn.property(
            "_epicycle_full_text"))
        assert "(3)" in full
        assert "(3 ticked)" in widget.split_guided_summary_btn.toolTip()

    def test_find_markers_is_wired(self, widget):
        fired = []
        widget.set_split_callbacks(on_find_markers=lambda: fired.append(1))
        widget.split_find_markers_btn.click()
        assert fired == [1]

    def test_a_refusal_lands_on_the_count_line(self, widget):
        widget.update_split_ui_state(marker_note="No markers found in this cell")
        assert "No markers found" in widget.split_marker_count_label.text()


class TestMarkerKnobsInvalidateTheList:
    """The ids are assigned by enumeration order over the crop, so ANY knob
    renumbers them -- and the ticks would then select the wrong markers."""

    CASES = (
        ("split_seed_prep_clahe_cb", "check"),
        ("split_seed_prep_zcorrect_cb", "check"),
        ("split_seed_prep_bgsub_cb", "check"),
        ("split_seed_prep_lift_cb", "check"),
        ("split_seed_prep_denoise_cb", "check"),
        ("split_seed_h_sb", "value"),
        ("split_min_distance_sb", "value"),
        ("split_seed_detector_combo", "index"),
        ("split_seeds_from_combo", "index"),
    )

    @pytest.mark.parametrize("attr,kind", CASES)
    def test_every_marker_knob_clears_the_list(self, widget, attr, kind):
        widget.update_split_ui_state(marker_ids=[1, 2])
        w = getattr(widget, attr)
        if kind == "check":
            w.setChecked(not w.isChecked())
        elif kind == "value":
            w.setValue(w.value() + 1)
        else:
            w.setCurrentIndex((w.currentIndex() + 1) % w.count())
        assert widget.get_checked_marker_ids() is None, attr
        assert widget.split_marker_list.count() == 0

    def test_the_channel_combo_counts_as_a_marker_knob(self, widget):
        widget.set_seed_channels(["nuclei", "membrane"])
        widget.update_split_ui_state(marker_ids=[1, 2])
        widget.split_seed_channel_combo.setCurrentIndex(1)
        assert widget.get_checked_marker_ids() is None

    def test_the_stale_sentence_goes_on_the_COUNT_LINE_not_on_S1(self, widget):
        before = widget.split_status_label.text()
        widget.update_split_ui_state(marker_ids=[1, 2])
        widget.split_seed_h_sb.setValue(0.4)
        assert "Find again" in widget.split_marker_count_label.text()
        assert widget.split_status_label.text() == before

    def test_a_knob_moved_with_no_list_says_nothing(self, widget):
        widget.split_seed_h_sb.setValue(0.31)
        assert widget.split_marker_count_label.text() == ""


class TestTheFindBusyStateDoesNotCollapseSources:
    def test_sources_stays_open_while_a_find_runs(self, widget):
        """state 5a: a Find that hid its own button and its list would be
        absurd.  The SOLVE's busy state still collapses S3."""
        widget.split_guided_summary_btn.setChecked(True)
        widget.update_split_ui_state(marker_busy=True)
        assert widget.split_guided_summary_btn.isEnabled() is True
        assert widget._split_guided_container.isVisibleTo(widget) is True
        assert widget._split_markers_container.isVisibleTo(widget) is True

    def test_the_find_button_and_the_knobs_go_grey(self, widget):
        widget.update_split_ui_state(marker_busy=True)
        assert widget.split_find_markers_btn.isEnabled() is False
        assert widget.split_seed_h_sb.isEnabled() is False
        assert widget.split_preview_btn.isEnabled() is False

    def test_the_bar_and_the_stop_button_appear(self, widget):
        widget.update_split_ui_state(marker_busy=True)
        assert widget._split_busy_container.isVisibleTo(widget) is True
        assert widget.split_cancel_preview_btn.isVisibleTo(widget) is True
        assert "Finding" in widget.split_marker_count_label.text()

    def test_it_hands_everything_back_when_it_finishes(self, widget):
        widget.update_split_ui_state(marker_busy=True)
        widget.update_split_ui_state(marker_busy=False)
        assert widget.split_find_markers_btn.isEnabled() is True
        assert widget.split_seed_h_sb.isEnabled() is True
        assert widget._split_busy_container.isVisibleTo(widget) is False


class TestTheMarkerChannelCombo:
    """The fifth channel combo, on the shared channel policy verbatim."""

    def test_a_restore_miss_blanks_it(self, widget):
        widget.set_seed_channels(["nuclei", "dapi"])
        widget.split_seed_channel_combo.setCurrentIndex(1)
        widget.set_seed_channels(["nuclei", "brightfield"])
        assert widget.split_seed_channel_combo.currentIndex() == -1
        assert widget.get_split_params()["seed_channel"] == ""

    def test_the_blank_state_does_not_heal_itself(self, widget):
        widget.set_seed_channels(["old"])
        widget.split_seed_channel_combo.setCurrentIndex(0)
        widget.set_seed_channels(["new"])
        widget.set_seed_channels(["new"])
        assert widget.split_seed_channel_combo.currentIndex() == -1

    def test_renaming_back_restores_the_choice(self, widget):
        widget.set_seed_channels(["old", "dapi"])
        widget.split_seed_channel_combo.setCurrentIndex(0)
        widget.set_seed_channels(["new", "dapi"])
        widget.set_seed_channels(["old", "dapi"])
        assert widget.split_seed_channel_combo.currentText() == "old"

    def test_an_explicit_current_wins(self, widget):
        widget.set_seed_channels(["nuclei", "dapi"], current="dapi")
        assert widget.split_seed_channel_combo.currentText() == "dapi"

    def test_the_first_fill_still_selects_something(self, widget):
        widget.set_seed_channels(["nuclei", "dapi"])
        assert widget.split_seed_channel_combo.currentIndex() == 0


class TestClearAlsoClearsTheMarkers:
    def test_clear_split_guided_ui_empties_the_picker(self, widget):
        """A Discard on cell 139 must not pass stale ids to cell 140."""
        widget.update_split_ui_state(marker_ids=[1, 2, 3])
        widget.clear_split_guided_ui()
        assert widget.get_checked_marker_ids() is None
        assert widget.split_marker_list.isVisibleTo(widget) is False
        assert widget.split_marker_caption.isVisibleTo(widget) is False
        assert widget.split_marker_count_label.text() == ""


# --------------------------------------------------------------------------- #
# Fix pass -- widget-side findings
# --------------------------------------------------------------------------- #
class TestTheMarkerBusyStateLeavesTheSolvesWidgetsAlone:
    def test_releasing_a_find_does_not_hide_a_running_solves_bar(self, widget):
        """The bar and Stop belong to whichever job is running; a late Find
        must not take them away from a live solve."""
        widget.update_split_guided_ui(busy=True)          # the SOLVE
        assert widget._split_busy_container.isVisibleTo(widget) is True
        widget.update_split_ui_state(marker_busy=False)   # a late FIND
        assert widget._split_busy_container.isVisibleTo(widget) is True
        assert widget.split_cancel_preview_btn.isVisibleTo(widget) is True

    def test_releasing_a_find_with_no_solve_running_does_hide_them(self, widget):
        widget.update_split_ui_state(marker_busy=True)
        widget.update_split_ui_state(marker_busy=False)
        assert widget._split_busy_container.isVisibleTo(widget) is False


class TestMarkersInvalidatedReachesTheOrchestrator:
    def test_the_callback_fires_once_per_knob_change(self, widget):
        fired = []
        widget.set_split_callbacks(
            on_markers_invalidated=lambda: fired.append(1))
        widget.update_split_ui_state(marker_ids=[1, 2])
        widget.split_seed_h_sb.setValue(0.42)
        assert fired == [1]

    def test_it_does_NOT_fire_when_there_was_nothing_to_invalidate(self, widget):
        fired = []
        widget.set_split_callbacks(
            on_markers_invalidated=lambda: fired.append(1))
        widget.split_seed_h_sb.setValue(0.31)
        assert fired == []

    def test_a_raising_callback_never_reaches_the_user(self, widget):
        def _boom():
            raise RuntimeError("nope")

        widget.set_split_callbacks(on_markers_invalidated=_boom)
        widget.update_split_ui_state(marker_ids=[1])
        widget.split_seed_h_sb.setValue(0.5)      # must not raise


class TestMembraneInteriorsPreSelectsTheWallChannel:
    """`hmin` reads the MEMBRANE image by design, so choosing it when a wall
    channel is already picked should point the marker channel at it rather
    than leaving the user to discover the trick."""

    def test_choosing_it_follows_the_wall_channel(self, widget):
        M = _mod()
        widget.set_wall_channels(["membrane", "nuclei"])
        widget.set_seed_channels(["membrane", "nuclei"])
        widget.split_wall_channel_combo.setCurrentIndex(0)
        widget.split_seed_channel_combo.setCurrentIndex(1)
        widget.split_seed_detector_combo.setCurrentIndex(
            M.SEED_DETECTORS.index("hmin"))
        assert widget.split_seed_channel_combo.currentText() == "membrane"

    def test_every_other_detector_leaves_the_choice_alone(self, widget):
        M = _mod()
        widget.set_wall_channels(["membrane", "nuclei"])
        widget.set_seed_channels(["membrane", "nuclei"])
        widget.split_wall_channel_combo.setCurrentIndex(0)
        widget.split_seed_channel_combo.setCurrentIndex(1)
        widget.split_seed_detector_combo.setCurrentIndex(
            M.SEED_DETECTORS.index("peaks"))
        assert widget.split_seed_channel_combo.currentText() == "nuclei"

    def test_with_no_wall_channel_chosen_it_changes_nothing(self, widget):
        M = _mod()
        widget.set_seed_channels(["nuclei"])
        widget.split_seed_channel_combo.setCurrentIndex(0)
        widget.split_seed_detector_combo.setCurrentIndex(
            M.SEED_DETECTORS.index("hmin"))
        assert widget.split_seed_channel_combo.currentText() == "nuclei"


class TestTheDistanceHeatmapGoesWithThePaintingRows:
    def test_it_is_unchecked_when_the_painting_rows_are_hidden(self, widget):
        """Same discipline as the arbiter: never leave a checked box behind
        an invisible one -- its overlay would stay on screen."""
        M = _mod()
        widget.split_edt_cb.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_IMAGE)
        assert widget.split_edt_cb.isChecked() is False

    def test_it_is_left_alone_for_a_painting_source(self, widget):
        M = _mod()
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_DETECTED)
        widget.split_edt_cb.setChecked(True)
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_PAINTED)
        assert widget.split_edt_cb.isChecked() is True


# --------------------------------------------------------------------------- #
# The marker row has to say something a user can act on
# --------------------------------------------------------------------------- #
class TestAMarkerRowSaysWhereTheMarkerIs:
    """Found by a real-button shots run, 2026-09-18: every row read
    "Marker 1 - 1".  The detectors return SINGLE-VOXEL peaks, so the size
    is 1 for all of them -- a column of 1s that distinguishes nothing.
    A position does: it is how you find the marker in the viewer."""

    POS = {"voxels": 1, "centroid_zyx": (12.0, 240.4, 88.6)}

    def test_a_single_voxel_marker_shows_its_position(self, widget):
        widget.update_split_ui_state(marker_ids=[4], marker_stats={4: self.POS})
        text = widget.split_marker_list.item(0).text()
        assert text.startswith("Marker 1")
        assert "z 12" in text and "y 240" in text and "x 89" in text
        from napari_epicycle.annotator import advanced_features_widget as M

        assert not text.endswith("{}  1".format(M._DOT_SEP)), (
            "the useless voxel count is still there: {!r}".format(text)
        )

    def test_a_real_sized_marker_still_shows_its_size(self, widget):
        widget.update_split_ui_state(
            marker_ids=[4],
            marker_stats={4: {"voxels": 1204, "centroid_zyx": (1, 2, 3)}})
        text = widget.split_marker_list.item(0).text()
        assert "1,204" in text

    def test_a_plain_integer_still_works(self, widget):
        """The old shape -- id -> voxels -- must keep working."""
        widget.update_split_ui_state(marker_ids=[4], marker_stats={4: 900})
        assert "900" in widget.split_marker_list.item(0).text()

    def test_a_marker_with_nothing_known_is_just_numbered(self, widget):
        widget.update_split_ui_state(marker_ids=[4])
        assert widget.split_marker_list.item(0).text() == "Marker 1"

    def test_the_tooltip_keeps_BOTH_numbers(self, widget):
        widget.update_split_ui_state(marker_ids=[4], marker_stats={4: self.POS})
        tip = widget.split_marker_list.item(0).toolTip()
        assert "id 4" in tip
        assert "1 voxel" in tip
        assert "z 12" in tip

    def test_the_position_is_rounded_to_whole_voxels(self, widget):
        """A centroid of 240.4 is not a place you can navigate to."""
        widget.update_split_ui_state(marker_ids=[4], marker_stats={4: self.POS})
        text = widget.split_marker_list.item(0).text()
        assert "240.4" not in text and "88.6" not in text

    def test_the_orchestrator_sends_the_position_too(self):
        """The widget cannot invent it -- `_on_seed_find_finished` has to
        pass `centroid_zyx` through from the report."""
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        src = inspect.getsource(Curation3DAnnotator._on_seed_find_finished)
        assert "centroid_zyx" in src


# --------------------------------------------------------------------------- #
# Brick 2b -- the How-to-cut card
# --------------------------------------------------------------------------- #
class TestTheHowToCutCard:
    def _open(self, widget):
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_solve_btn.setChecked(True)

    def test_the_three_promoted_rows_live_in_How_to_cut(self, widget):
        """Snap, the arbiter and the expected count are promoted OUT of
        Advanced.  A control the user needs on every cut, behind a
        disclosure they never open, is a control that does not exist."""
        self._open(widget)
        card = widget._split_solve_container
        for attr in ("split_refine_guided_cb", "split_arbiter_cb",
                     "split_target_count_cb", "split_target_count_sb"):
            w = getattr(widget, attr)
            assert _is_inside(w, card), (
                "{} is not in the How-to-cut card".format(attr)
            )

    def test_none_of_them_is_left_in_Advanced(self, widget):
        for attr in ("split_refine_guided_cb", "split_arbiter_cb",
                     "split_target_count_cb", "split_target_count_sb"):
            w = getattr(widget, attr)
            assert not _is_inside(w, widget._split_advanced_container), attr

    def test_the_snap_and_arbiter_rows_are_image_wall_only(self, widget):
        M = _mod()
        self._open(widget)
        for idx in range(widget.split_walls_from_combo.count()):
            widget.split_walls_from_combo.setCurrentIndex(idx)
            shown = widget._split_cut_image_container.isVisibleTo(widget)
            assert shown is (idx in M.IMAGE_DERIVED_WALL_INDICES), (
                "{!r} (index {}) shows the snap/arbiter rows: {}".format(
                    M.WALL_KINDS[idx], idx, shown)
            )

    def test_the_wall_prepare_chip_is_image_wall_only_too(self, widget):
        """M3.  "Prepare the channel first" prepares the CHANNEL, and a
        shape wall has none -- the chip and its five ticks were the only
        thing left on screen offering to clean an image nothing reads."""
        M = _mod()
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_walls_btn.setChecked(True)
        for idx in range(widget.split_walls_from_combo.count()):
            widget.split_walls_from_combo.setCurrentIndex(idx)
            want = idx in M.IMAGE_DERIVED_WALL_INDICES
            assert widget.split_wall_prep_btn.isVisibleTo(widget) is want, (
                M.WALL_KINDS[idx])

    def test_leaving_an_image_wall_closes_the_prepare_chip(self, widget):
        """A chip left OPEN behind an invisible one comes back open on
        the next image wall, showing ticks the user did not re-open."""
        M = _mod()
        widget.split_guided_summary_btn.setChecked(True)
        widget.split_walls_btn.setChecked(True)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        widget.split_wall_prep_btn.setChecked(True)
        assert widget._split_wall_prep_container.isVisibleTo(widget) is True
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_SHAPE)
        assert widget.split_wall_prep_btn.isChecked() is False
        assert widget._split_wall_prep_container.isVisibleTo(widget) is False

    def test_the_wall_chain_itself_is_untouched_by_hiding_the_chip(
            self, widget):
        """Hiding is not unticking: a shape wall reads no chain at all,
        and the user's ticks have to be there when they come back."""
        M = _mod()
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        widget.split_wall_prep_bgsub_cb.setChecked(True)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_SHAPE)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_MEMBRANE)
        assert widget.split_wall_prep_bgsub_cb.isChecked() is True
        assert "bgsub" in widget.get_split_params()["wall_preproc"]

    def test_leaving_an_image_wall_unchecks_BOTH_of_them(self, widget):
        """Same discipline the arbiter has always had: a checked box
        behind an invisible one is a setting with no way to turn it off."""
        M = _mod()
        widget.split_walls_from_combo.setCurrentIndex(
            M.WALL_KINDS.index("membrane"))
        widget.split_arbiter_cb.setChecked(True)
        widget.split_refine_guided_cb.setChecked(True)
        widget.split_walls_from_combo.setCurrentIndex(M.WALLS_FROM_SHAPE)
        assert widget.split_arbiter_cb.isChecked() is False
        assert widget.split_refine_guided_cb.isChecked() is False

    def test_the_expected_count_row_is_shown_for_every_wall_kind(self, widget):
        """It is about the answer, not about the image."""
        self._open(widget)
        for idx in range(widget.split_walls_from_combo.count()):
            widget.split_walls_from_combo.setCurrentIndex(idx)
            assert widget.split_target_count_cb.isVisibleTo(widget) is True

    def test_the_count_spin_only_answers_while_its_tick_is_on(self, widget):
        widget.split_target_count_cb.setChecked(False)
        widget.split_target_count_sb.setValue(4)
        assert widget.get_split_params()["target_count"] is None
        widget.split_target_count_cb.setChecked(True)
        assert widget.get_split_params()["target_count"] == 4

    def test_the_count_spin_is_enabled_by_its_tick(self, widget):
        assert widget.split_target_count_sb.isEnabled() is False
        widget.split_target_count_cb.setChecked(True)
        assert widget.split_target_count_sb.isEnabled() is True

    def test_the_count_can_never_be_a_nonsense_number(self, widget):
        """0 and 1 used to mean "off"; the tick means that now, so the
        spin's own range starts at the smallest count worth asking for."""
        assert widget.split_target_count_sb.minimum() == 2


class TestTheOwnerWording:
    """The wall channel can be any channel, so the ACTIONS stop
    saying "membrane".  The wall TYPE "Membrane sheets" keeps its name --
    it names the sheet-ridge filter, not the channel."""

    def test_snap_says_wall_signal(self, widget):
        assert widget.split_refine_guided_cb.text() == (
            "Snap to the wall signal")

    def test_the_arbiter_says_wall(self, widget):
        assert widget.split_arbiter_cb.text() == (
            "Only cut where I can see a wall")

    def test_the_expected_count_row_says_what_it_means(self, widget):
        """Draws it as "I expect exactly [ n ] cells", so the
        words sit either side of the number rather than all in the tick
        (which also cost 262 px of a 316 px viewport -- H1)."""
        assert widget.split_target_count_cb.text() == "I expect exactly"
        # "objects", not "cells": the tab is used on nuclei too, so the
        # thing being reshaped is an OBJECT (UX round 0a).
        assert widget.split_target_count_unit.text() == "objects"

    def test_the_solver_item_says_wall_signal(self, widget):
        M = _mod()
        assert M.SPLIT_SOLVER_LABELS[2] == (
            "Settle onto the wall signal (label diffusion)")

    def test_the_membrane_sheets_wall_type_keeps_its_name(self, widget):
        M = _mod()
        assert M.WALLS_FROM_LABELS[M.WALL_KINDS.index("membrane")] == (
            "Membrane sheets")

    #: The two places "membrane" is still the right word, and why.
    #: `Membrane sheets` and `Membrane sheets plus shape` name the
    #: SHEET-RIDGE FILTER, not the channel -- that is what the filter
    #: looks for, whatever channel it is pointed at.  The detector
    #: `Membrane-enclosed interiors` names the geometry it looks for
    #: (a basin with a rim all the way round), for the same reason.
    #: Everything else on this page is an ACTION and says "wall".
    MEMBRANE_EXEMPT = (
        "split_walls_from_combo",       # items 1 and 2 + their ITEM_HELP
        "split_seed_detector_combo",    # item 5 + its ITEM_HELP
    )

    def test_no_promoted_row_still_says_membrane(self, widget):
        for attr in ("split_refine_guided_cb", "split_arbiter_cb",
                     "split_target_count_cb"):
            w = getattr(widget, attr)
            assert "membrane" not in w.text().lower(), attr
            assert "membrane" not in w.toolTip().lower(), attr

    def test_NOTHING_on_the_reshape_page_says_membrane_but_the_two_names(
            self, widget):
        """The sweep behind the three rows above (M6).

        Seven tooltips survived the 2026-09-18 rename sitting under
        labels that had already been changed -- "Wall clip" explained as
        "before looking for the membrane", "Wall required (%)" as a
        percentile, "Soft cut: wall resistance" as "Random walker only".
        A tooltip is where a user goes when the label is not enough, so
        a tooltip in the old vocabulary is worse than none.
        """
        from qtpy.QtWidgets import (
            QCheckBox, QComboBox, QLabel, QPushButton, QSpinBox,
            QDoubleSpinBox,
        )

        offenders = []
        for attr in dir(widget):
            if not attr.startswith("split_"):
                continue
            if attr in self.MEMBRANE_EXEMPT:
                continue
            w = getattr(widget, attr, None)
            if not isinstance(w, (QCheckBox, QComboBox, QLabel, QPushButton,
                                  QSpinBox, QDoubleSpinBox)):
                continue
            for what, text in (("text", getattr(w, "text", lambda: "")()),
                               ("tooltip", w.toolTip())):
                if "membrane" in str(text).lower():
                    offenders.append((attr, what))
        assert not offenders, (
            "still in the old vocabulary: {}".format(sorted(offenders)))

    def test_the_marker_count_row_says_markers(self, widget):
        """Fix pass M3: "Seeds: 0" was the birth text of the line that
        `seed_count=` later rewrites as "Markers: n" -- so the panel
        said "seed" exactly once, at the only moment a first-time user
        is guaranteed to be reading it."""
        assert widget.split_seed_label.text() == "Markers: 0"

    def test_the_shape_finder_is_named_after_what_it_looks_at(self, widget):
        """"Detect Seeds" was the last button on the page in the old
        vocabulary, and it names the ALGORITHM's input rather than the
        user's.  It is source 0's finder and it looks at the object's
        own shape."""
        assert widget.split_detect_seeds_btn.text() == "Find from shape"

    def test_the_found_count_says_markers_too(self, widget):
        assert widget._detect_count_label.text() == "Found: 0 markers"
        widget.update_split_ui_state(detect_seed_count=3)
        assert widget._detect_count_label.text() == "Found: 3 markers"
        widget.update_split_ui_state(detect_seed_count=1)
        assert widget._detect_count_label.text() == "Found: 1 marker"

    def test_NOTHING_on_the_reshape_page_says_seed(self, widget):
        """The sweep behind the three rows above.

        "Markers", never "seeds", in anything the user reads (spec
        preamble).  ATTRIBUTE names keep `seed` -- they are the wire
        format `get_split_params()` reads by index and renaming them
        would move the contract -- so this sweeps TEXT and TOOLTIPS
        only.
        """
        from qtpy.QtWidgets import (
            QCheckBox, QComboBox, QLabel, QPushButton, QSpinBox,
            QDoubleSpinBox,
        )

        offenders = []
        for attr in dir(widget):
            if not (attr.startswith("split_") or attr.startswith("_detect")):
                continue
            w = getattr(widget, attr, None)
            if not isinstance(w, (QCheckBox, QComboBox, QLabel, QPushButton,
                                  QSpinBox, QDoubleSpinBox)):
                continue
            for what, text in (("text", getattr(w, "text", lambda: "")()),
                               ("tooltip", w.toolTip())):
                if "seed" in str(text).lower():
                    offenders.append((attr, what))
        assert not offenders, (
            "still in the old vocabulary: {}".format(sorted(offenders)))

    def test_no_tooltip_on_the_page_names_an_algorithm_either(self, widget):
        """The same sweep for the words that name the machinery rather
        than the thing: "Random walker", "watershed", "percentile"."""
        from qtpy.QtWidgets import (
            QCheckBox, QComboBox, QPushButton, QSpinBox, QDoubleSpinBox,
        )

        # `split_solver_combo`'s ITEMS carry the algorithm in brackets on
        # purpose -- "Sharp cut (watershed)" is the shipped label (spec
        # 2.6), so a reader can look the method up.  Its TOOLTIP is
        # swept like everything else.
        banned = ("random walker", "watershed", "percentile")
        offenders = []
        for attr in dir(widget):
            if not attr.startswith("split_"):
                continue
            w = getattr(widget, attr, None)
            if not isinstance(w, (QCheckBox, QComboBox, QPushButton,
                                  QSpinBox, QDoubleSpinBox)):
                continue
            tip = str(w.toolTip()).lower()
            for word in banned:
                if word in tip:
                    offenders.append((attr, word))
        assert not offenders, (
            "tooltips naming the machinery: {}".format(sorted(offenders)))

    def test_the_hybrid_wall_says_which_membrane_thing_it_means(self, widget):
        """M6: "Membrane plus shape" read as "the membrane channel plus
        the shape".  It is the sheet-ridge FILTER plus the shape, and the
        carve-out above is only honest if the name says so."""
        M = _mod()
        assert M.WALLS_FROM_LABELS[M.WALL_KINDS.index("hybrid")] == (
            "Membrane sheets plus shape")

    def test_the_wall_channel_row_does_not_promise_a_membrane(self, widget):
        assert "membrane" not in (
            widget.split_wall_channel_combo.toolTip().lower())


class TestAbsorbSliversIsOnByDefault:
    """The ONE declared exception to the golden's rule 7: the
    sandbox ALWAYS applied MIN_SIZE=20 and the owner wants that back."""

    def test_the_tick_is_on(self, widget):
        assert widget.split_refine_merge_small_cb.isChecked() is True

    def test_the_size_is_20(self, widget):
        assert widget.split_min_size_sb.value() == 20

    def test_the_params_say_so(self, widget):
        p = widget.get_split_params()
        assert p["refine_merge_small"] is True
        assert p["min_size"] == 20


# --------------------------------------------------------------------------- #
# Brick 2b -- the rest of the engine's knobs (table)
# --------------------------------------------------------------------------- #
#: attribute -> (params key, default).  Every one of these is a knob the
#: engine has had since R0 and the panel could not reach.
NEW_KNOBS = {
    "split_seed_smooth_sb": ("seed_smooth", 2.0),
    "split_max_sigma_sb": ("max_sigma_um", 6.0),
    "split_seed_clip_sb": ("seed_clip", 0.02),
    "split_seed_gamma_sb": ("seed_gamma", 0.5),
    "split_seed_bg_sb": ("seed_bg", 15),
    "split_seed_denoise_sb": ("seed_denoise", 2),
    "split_wall_smooth_sb": ("smooth", 1.0),
    "split_refine_radius_sb": ("refine_radius", 4),
    "split_refine_eps_sb": ("refine_eps", 0.01),
}


class TestTheRemainingAdvancedKnobs:
    @pytest.mark.parametrize("attr", sorted(NEW_KNOBS))
    def test_the_knob_exists_and_lives_in_Advanced(self, widget, attr):
        w = getattr(widget, attr, None)
        assert w is not None, attr
        assert _is_inside(w, widget._split_advanced_container)

    @pytest.mark.parametrize("attr", sorted(NEW_KNOBS))
    def test_the_knob_reaches_the_params_dict(self, widget, attr):
        key, default = NEW_KNOBS[attr]
        assert widget.get_split_params()[key] == pytest.approx(default)

    @pytest.mark.parametrize("attr", sorted(NEW_KNOBS))
    def test_turning_the_knob_moves_its_key(self, widget, attr):
        key, default = NEW_KNOBS[attr]
        box = getattr(widget, attr)
        box.setValue(box.value() + box.singleStep())
        assert widget.get_split_params()[key] != pytest.approx(default)

    def test_every_new_key_is_one_the_engine_knows(self):
        from packages.epicycle_annotator.reshape.spec import DEFAULT_PARAMS

        for key, _default in NEW_KNOBS.values():
            assert key in DEFAULT_PARAMS, key

    def test_the_defaults_match_the_engines_own(self, widget):
        """A panel default that differs from ``DEFAULT_PARAMS`` silently
        changes the answer for every user who never opens Advanced."""
        from packages.epicycle_annotator.reshape.spec import DEFAULT_PARAMS

        p = widget.get_split_params()
        for _attr, (key, _d) in NEW_KNOBS.items():
            assert p[key] == pytest.approx(DEFAULT_PARAMS[key]), key


class TestTheMarkerThreshold:
    def test_blank_means_automatic(self, widget):
        assert widget.get_split_params()["seed_thresh"] is None

    def test_the_spin_says_automatic_at_its_minimum(self, widget):
        box = widget.split_seed_thresh_sb
        assert box.value() == box.minimum()
        assert box.specialValueText().strip()

    def test_a_real_number_is_passed_through(self, widget):
        widget.split_seed_thresh_sb.setValue(0.4)
        assert widget.get_split_params()["seed_thresh"] == pytest.approx(0.4)

    def test_it_is_a_readable_width_even_with_its_special_text(self, widget):
        """The word it shows instead of a number is wider than 1.00, and
        a spin box narrower than what it shows is the bug the owner
        screenshotted on the Detect row."""
        box = widget.split_seed_thresh_sb
        fm = box.fontMetrics()
        assert box.minimumWidth() >= fm.horizontalAdvance(
            box.specialValueText())


class TestLiftDimSignalGentleOrStrong:
    def test_gentle_is_the_default(self, widget):
        assert widget.get_split_params()["lift_mode"] == "gentle"

    def test_the_combo_has_exactly_the_two_choices(self, widget):
        combo = widget.split_lift_mode_combo
        assert [combo.itemText(i) for i in range(combo.count())] == [
            "gentle", "strong"]

    def test_strong_changes_the_chain_token(self, widget):
        widget.split_seed_prep_lift_cb.setChecked(True)
        assert "gamma" in widget.get_split_params()["seed_preproc"]
        widget.split_lift_mode_combo.setCurrentIndex(1)
        chain = widget.get_split_params()["seed_preproc"]
        assert "log" in chain and "gamma" not in chain

    def test_it_reaches_the_wall_chain_too(self, widget):
        widget.split_wall_prep_lift_cb.setChecked(True)
        widget.split_lift_mode_combo.setCurrentIndex(1)
        assert "log" in widget.get_split_params()["wall_preproc"]

    def test_the_token_is_one_the_engine_accepts(self):
        """RUN it, rather than grep its source: a token the engine has
        stopped honouring still appears in the file that used to."""
        import numpy as np

        from packages.epicycle_annotator.reshape.preproc import apply_preproc

        img = np.linspace(0.0, 1.0, 8 * 8 * 8,
                          dtype=np.float32).reshape(8, 8, 8)
        plain = apply_preproc(img, "none")
        for token in ("gamma", "log"):
            out = apply_preproc(img, token)
            assert out.shape == img.shape, token
            assert not np.allclose(out, plain), (
                "{!r} did nothing -- the chain token is not honoured "
                "any more".format(token)
            )
        assert not np.allclose(apply_preproc(img, "gamma"),
                               apply_preproc(img, "log")), (
            "gentle and strong are the same lift"
        )


# --------------------------------------------------------------------------- #
# Brick 2b -- the quality block
# --------------------------------------------------------------------------- #
class TestTheQualityBlock:
    def _notes(self, *specs):
        from packages.epicycle_annotator.reshape import QualityNote

        return [QualityNote(k, text, ids) for k, text, ids in specs]

    def test_the_block_sits_above_the_piece_list(self, widget):
        """The mockup's RESULT card: what went wrong, then the pieces,
        then what an unticked one means."""
        card = widget._split_result_container
        lay = card.layout()
        order = []
        for i in range(lay.count()):
            w = lay.itemAt(i).widget()
            if w is None:
                continue
            if w is widget.split_quality_label:
                order.append("quality")
            elif _is_inside(widget.split_piece_list, w):
                order.append("pieces")
            elif w is widget.split_piece_caption:
                order.append("caption")
        assert order.index("quality") < order.index("pieces")
        assert order.index("pieces") < order.index("caption")

    def test_it_is_amber(self, widget):
        from napari_epicycle.ui_shell.theme import PANEL_WARN_STYLE

        assert widget.split_quality_label.styleSheet() == PANEL_WARN_STYLE

    def test_nothing_to_say_means_nothing_on_screen(self, widget):
        widget.update_split_guided_ui(pieces=[1, 2], piece_sizes={1: 9, 2: 9},
                                      quality_notes=[])
        assert widget.split_quality_label.isVisibleTo(widget) is False
        assert widget.split_quality_label.text() == ""

    def test_the_sentences_are_shown_one_per_line(self, widget):
        notes = self._notes(
            ("coverage", "6 % of the object was not given to any piece.", ()),
            ("sliver", "Piece 3 is very small (410 voxels).", (13,)),
        )
        widget.update_split_guided_ui(
            pieces=[11, 12, 13], piece_sizes={11: 5000, 12: 4800, 13: 410},
            quality_notes=notes)
        assert widget.split_quality_label.isVisibleTo(widget) is True
        assert widget.split_quality_label.text().split("\n") == [
            n.text for n in notes]

    def test_a_sliver_note_unticks_exactly_the_piece_it_names(self, widget):
        notes = self._notes(
            ("sliver", "Piece 3 is very small (410 voxels).", (13,)))
        widget.update_split_guided_ui(
            pieces=[11, 12, 13], piece_sizes={11: 5000, 12: 4800, 13: 410},
            quality_notes=notes)
        assert widget.get_checked_split_pieces() == [11, 12]

    def test_no_note_no_untick(self, widget):
        """The rule lives in `quality.py` now -- ONE rule, not two.  The widget's own 0.2 x median copy is gone, so a tiny
        piece with no note about it stays ticked."""
        widget.update_split_guided_ui(
            pieces=[11, 12, 13], piece_sizes={11: 5000, 12: 4800, 13: 1},
            quality_notes=[])
        assert widget.get_checked_split_pieces() == [11, 12, 13]

    def test_the_named_piece_is_marked_small_in_the_row(self, widget):
        notes = self._notes(
            ("sliver", "Piece 3 is very small (410 voxels).", (13,)))
        widget.update_split_guided_ui(
            pieces=[11, 12, 13], piece_sizes={11: 5000, 12: 4800, 13: 410},
            quality_notes=notes)
        texts = [widget.split_piece_list.item(i).text()
                 for i in range(widget.split_piece_list.count())]
        assert "(small)" in texts[2]
        assert "(small)" not in texts[0]

    def test_it_is_cleared_with_the_rest_of_the_result(self, widget):
        widget.update_split_guided_ui(
            pieces=[11], piece_sizes={11: 5},
            quality_notes=self._notes(("coverage", "Six percent.", ())))
        widget.clear_split_guided_ui()
        assert widget.split_quality_label.text() == ""
        assert widget.split_quality_label.isVisibleTo(widget) is False

    def test_the_block_alone_keeps_the_result_slot_on_screen(self, widget):
        """M4.  S5's visibility is recomputed from what is left showing;
        the quality label was not in that chain, so a state where it is
        the only thing left would hide the card it sits in."""
        widget.update_split_guided_ui(
            quality_notes=[{"kind": "coverage", "text": "Six percent.",
                            "piece_ids": []}])
        assert widget._split_result_container.isVisibleTo(widget) is True

    def test_a_caller_may_pass_plain_dicts(self, widget):
        """The orchestrator hands over QualityNote objects; a test rig or
        a future caller may hand over the same thing as data."""
        widget.update_split_guided_ui(
            pieces=[11, 12], piece_sizes={11: 5000, 12: 20},
            quality_notes=[{"kind": "sliver", "text": "Piece 2 is small.",
                            "piece_ids": [12]}])
        assert widget.split_quality_label.text() == "Piece 2 is small."
        assert widget.get_checked_split_pieces() == [11]


class TestARefusalIsRed:
    """M8.  A refusal is not a warning: the tool did NOT do what was
    asked, and the sentence says what to change.  It had the same grey
    as the "Wall contrast" line sitting under it."""

    def test_a_refusal_shows_in_its_own_label(self, widget):
        widget.update_split_guided_ui(
            engine_error="Pick the image channel the walls are in.")
        assert widget.split_engine_error_label.isVisibleTo(widget) is True
        assert widget.split_engine_error_label.text() == (
            "Pick the image channel the walls are in.")

    def test_it_keeps_the_result_slot_on_screen_by_itself(self, widget):
        """It is often the ONLY thing there -- a refusal has no pieces."""
        widget.update_split_guided_ui(engine_error="No markers found.")
        assert widget._split_result_container.isVisibleTo(widget) is True

    def test_an_empty_string_takes_it_away(self, widget):
        widget.update_split_guided_ui(engine_error="No markers found.")
        widget.update_split_guided_ui(engine_error="")
        assert widget.split_engine_error_label.isVisibleTo(widget) is False

    def test_clearing_the_result_clears_it(self, widget):
        widget.update_split_guided_ui(engine_error="No markers found.")
        widget.clear_split_guided_ui()
        assert widget.split_engine_error_label.text() == ""
        assert widget.split_engine_error_label.isVisibleTo(widget) is False

    def test_it_sits_apart_from_the_grey_message_line(self, widget):
        """Two labels, not one: the engine's informational messages keep
        their grey, and a refusal is not mixed in among them."""
        widget.update_split_guided_ui(
            engine_messages=["Wall contrast 0.12-0.81."],
            engine_error="No markers found.")
        assert widget.split_engine_msg_label.text() == (
            "Wall contrast 0.12-0.81.")
        assert widget.split_engine_error_label.text() == "No markers found."

    def test_the_kwarg_is_additive(self, widget):
        """Every existing caller passes None for it and is unaffected."""
        import inspect

        sig = inspect.signature(widget.update_split_guided_ui)
        assert sig.parameters["engine_error"].default is None
        assert sig.parameters["engine_error"].kind is (
            inspect.Parameter.KEYWORD_ONLY)


class TestTheLegacyPairIsGone:
    """`split_method_combo` + `split_image_combo` asked the
    same question as "Walls from", one chevron away, and have been
    hidden since Brick 1.  Their manual branch is UI-unreachable --
    `is_guided_split()` routes every wall index >= 1 to the engine."""

    @pytest.mark.parametrize("attr", [
        "split_method_combo", "split_image_combo",
        "_split_legacy_container", "_split_image_container",
        "refresh_split_image_combo",
    ])
    def test_the_control_and_its_refresher_are_gone(self, widget, attr):
        assert not hasattr(widget, attr), (
            "{} survived the retirement".format(attr))

    def test_is_guided_split_is_the_whole_truth_table(self, widget):
        """The proof the removed branch was unreachable: for EVERY
        combination of the two pickers, the only False cells are the two
        legacy seed sources against shape-only walls -- and neither of
        those reaches the membrane code that was deleted."""
        M = _mod()
        legacy = {M.SEEDS_FROM_PAINTED, M.SEEDS_FROM_DETECTED}
        for seeds in range(widget.split_seeds_from_combo.count()):
            for walls in range(widget.split_walls_from_combo.count()):
                widget.split_seeds_from_combo.setCurrentIndex(seeds)
                widget.split_walls_from_combo.setCurrentIndex(walls)
                # the coupling rule may bounce a forbidden pair back
                walls_now = widget.split_walls_from_combo.currentIndex()
                want = not (seeds in legacy
                            and walls_now == M.WALLS_FROM_SHAPE)
                assert widget.is_guided_split() is want, (seeds, walls_now)

    def test_no_wall_index_above_shape_is_ever_manual(self, widget):
        M = _mod()
        widget.split_seeds_from_combo.setCurrentIndex(M.SEEDS_FROM_PAINTED)
        for walls in range(1, widget.split_walls_from_combo.count()):
            widget.split_walls_from_combo.setCurrentIndex(walls)
            assert widget.is_guided_split() is True, walls
