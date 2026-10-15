"""The Refine tab's wire format, and the six things that hold it.

The index -> engine-key mapping is the one thing on this tab that cannot be
inspected safely: a reorder of the combo would mis-route four methods with a
fully green suite.  After it lives in ONE module constant,
``REFINE_METHOD_KEYS``, and this file's job is to prove that the constant is
not merely equal to itself -- it re-derives the pairs from the constant and
then walks the ANNOTATOR's source for each one.

The docstring the first version of this file carried -- "the map lives in
`_preview_smooth` as a literal ... nothing else pins it" -- was **false even
before the change**: ``method_label``, a second key -> text dict in the same
method, pinned it too.  Both are asserted here.

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


def _mod():
    from napari_epicycle.annotator import advanced_features_widget as M

    return M


def _keys():
    """``(index, combo text, engine key)`` -- RE-DERIVED, not restated.

    Restating the table here would make every assertion below a proof that
    a constant equals itself.  The order is read off the constant and then
    checked against the widget, the stack, the descriptions and the
    annotator's two dispatch dicts, which is where a reorder does damage.
    """
    return tuple(_mod().REFINE_METHOD_KEYS)


#: The seven, in the order fixes them.  This is the ONE place the
#: expected order is written down independently, so that a silent reorder of
#: the constant is still a failure.
EXPECTED_ORDER = (
    "sdf", "contour", "dilate_erode", "guided_grow",
    "fill_holes", "flood_fill", "settle",
)


class TestMethodComboIsTheWireFormat:
    def test_the_constant_holds_the_order_the_spec_fixes(self):
        assert tuple(k for _i, _t, k in _keys()) == EXPECTED_ORDER

    def test_the_indices_are_dense_and_in_order(self):
        assert [i for i, _t, _k in _keys()] == list(range(len(EXPECTED_ORDER)))

    def test_seven_methods_in_this_exact_order(self, widget):
        combo = widget.refine_method_combo
        assert combo.count() == len(_keys())
        for idx, text, key in _keys():
            assert combo.itemText(idx) == text, (
                "index {} is the wire format for {!r}".format(idx, key)
            )

    def test_the_annotator_dispatch_map_matches_the_combo(self):
        """`_preview_smooth` holds the index -> key dict.

        It is not the only thing that pins the order: `method_label`, in
        the same method, maps the KEYS to the words the status line uses,
        and the two must name the same set.
        """
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        src = inspect.getsource(Curation3DAnnotator._preview_smooth)
        for idx, _text, key in _keys():
            assert '{}: "{}"'.format(idx, key) in src, (
                "the refine dispatch no longer maps index {} to {!r}"
                .format(idx, key)
            )

    def test_the_status_line_names_every_method_too(self):
        """The second dict the first draft of this file missed."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        src = inspect.getsource(Curation3DAnnotator._preview_smooth)
        head = src.split("method_label", 1)
        assert len(head) > 1, "`method_label` is gone -- rewrite this test"
        body = head[1]
        for _idx, _text, key in _keys():
            assert '"{}"'.format(key) in body, (
                "`method_label` does not name {!r}".format(key)
            )

    def test_the_two_contour_branches_collapsed_into_one(self):
        """Thing 6: MorphGAC and MorphACWE stop being two
        methods and become two modes of one."""
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        src = inspect.getsource(Curation3DAnnotator._preview_smooth)
        assert 'method == "morphgac"' not in src
        assert 'method == "morphacwe"' not in src
        # The two collapse into ONE key, and that key is one of the two the
        # dispatch sends to a worker -- so it appears in the threaded
        # branch rather than in a branch of its own.
        assert 'method in ("contour", "settle")' in src
        # ...and B2 put the two 2024 bodies behind ONE pure function,
        # `refine.contour_refine`, which the call hands the mode picker's
        # index to.  The bodies are its private halves now.
        call = inspect.getsource(Curation3DAnnotator._refine_contour_call)
        assert "contour_refine" in call
        assert "morphgac_refine" not in call and "morphacwe_refine" not in call
        assert 'p.get("contour_mode"' in call

    def test_every_method_has_its_own_params_page(self, widget):
        assert widget._refine_params_stack.count() == len(_keys())

    def test_picking_a_method_swaps_the_page_and_the_description(self, widget):
        for idx, _text, _key in _keys():
            widget.refine_method_combo.setCurrentIndex(idx)
            assert widget._refine_params_stack.currentIndex() == idx
            desc = widget._refine_method_desc_label.text()
            assert desc, "method {} has no description".format(idx)
            assert desc == widget._REFINE_METHOD_DESCRIPTIONS[idx]

    def test_the_descriptions_are_keyed_by_the_same_indices(self, widget):
        assert set(widget._REFINE_METHOD_DESCRIPTIONS) == {
            i for i, _t, _k in _keys()}

    def test_only_the_selected_methods_params_are_visible(self, widget):
        """The stack is the swap: exactly one page on screen at a time."""
        widget.refine_method_combo.setCurrentIndex(3)
        stack = widget._refine_params_stack
        pages = [stack.widget(i) for i in range(stack.count())]
        # Checked against the STACK, not the panel: the Refine page itself
        # is hidden while another tab is in front, so isVisibleTo(widget)
        # would be False for all seven and prove nothing.
        shown = [i for i, p in enumerate(pages) if p.isVisibleTo(stack)]
        assert shown == [3]

    def test_the_key_accessor_agrees_with_the_combo(self, widget):
        for idx, _text, key in _keys():
            widget.refine_method_combo.setCurrentIndex(idx)
            assert widget.refine_method_key() == key


class TestRefineButtonsAndState:
    def test_the_three_buttons_start_in_their_documented_state(self, widget):
        assert widget.smooth_preview_btn.isEnabled() is True
        assert widget.smooth_accept_btn.isEnabled() is False
        assert widget.smooth_reject_btn.isEnabled() is False

    def test_update_smooth_ui_state_drives_all_four_fields(self, widget):
        widget.update_smooth_ui_state(
            preview_enabled=False, accept_enabled=True, reject_enabled=True,
            status_text="Object 149 - 3 steps staged",
        )
        assert widget.smooth_preview_btn.isEnabled() is False
        assert widget.smooth_accept_btn.isEnabled() is True
        assert widget.smooth_reject_btn.isEnabled() is True
        assert "3 steps" in widget.smooth_status_label.text()

    def test_the_three_buttons_are_wired_through_set_smooth_callbacks(self, widget):
        fired = []
        widget.set_smooth_callbacks(
            on_preview=lambda: fired.append("preview"),
            on_accept=lambda: fired.append("accept"),
            on_reject=lambda: fired.append("reject"),
            on_prob_toggle=lambda v: fired.append(("prob", v)),
        )
        widget.update_smooth_ui_state(accept_enabled=True, reject_enabled=True)
        widget.smooth_preview_btn.click()
        widget.smooth_accept_btn.click()
        widget.smooth_reject_btn.click()
        widget.refine_prob_overlay_cb.setChecked(True)
        assert fired == ["preview", "accept", "reject", ("prob", True)]

    def test_the_three_new_callbacks_are_additive(self, widget):
        """`set_smooth_callbacks` grows three keywords and every
        existing call site stays valid."""
        fired = []
        widget.set_smooth_callbacks(
            on_cancel_preview=lambda: fired.append("stop"),
            on_show_walls_toggled=lambda v: fired.append(("walls", v)),
            on_method_changed=lambda i: fired.append(("method", i)),
        )
        widget.refine_cancel_preview_btn.click()
        widget.refine_show_walls_cb.setEnabled(True)
        widget.refine_show_walls_cb.setChecked(True)
        widget.refine_method_combo.setCurrentIndex(2)
        assert ("walls", True) in fired
        assert "stop" in fired
        assert ("method", 2) in fired

    def test_rewiring_does_not_stack_connections(self, widget):
        """The dock is REUSED across relaunches, so the setter must be
        disconnect-before-connect."""
        fired = []
        for _ in range(3):
            widget.set_smooth_callbacks(on_preview=lambda: fired.append(1))
        widget.smooth_preview_btn.click()
        assert fired == [1]

    def test_the_colormap_combo_reapplies_only_while_the_overlay_is_on(self, widget):
        fired = []
        widget.set_smooth_callbacks(on_prob_toggle=lambda v: fired.append(v))
        widget.refine_prob_cmap_combo.setCurrentIndex(1)
        assert fired == []
        widget.refine_prob_overlay_cb.setChecked(True)
        fired.clear()
        widget.refine_prob_cmap_combo.setCurrentIndex(2)
        assert fired == [True]


class TestTheRefineSecondary:
    """Two words, one meaning each -- the same rule as Reshape, keyed on
    whether there is a staged shape to throw away."""

    def test_it_says_cancel_while_nothing_is_staged(self, widget):
        assert widget.smooth_reject_btn.text() == "Cancel"

    def test_it_says_discard_once_a_result_is_staged(self, widget):
        widget.update_smooth_ui_state(accept_enabled=True, reject_enabled=True)
        assert widget.smooth_reject_btn.text() == "Discard"

    def test_it_goes_back_to_cancel_after_accept_or_reject(self, widget):
        widget.update_smooth_ui_state(accept_enabled=True, reject_enabled=True)
        widget.update_smooth_ui_state(
            accept_enabled=False, reject_enabled=False)
        assert widget.smooth_reject_btn.text() == "Cancel"

    def test_both_footer_buttons_explain_themselves(self, widget):
        # B3 audit: "only ever writes inside this object" was false -- a
        # grow writes into EMPTY space too; what it never touches is a
        # neighbour.
        assert "never a neighbour" in widget.smooth_accept_btn.toolTip()
        assert "back exactly as it was" in widget.smooth_reject_btn.toolTip()

    def test_it_is_never_called_Reject(self, widget):
        for staged in (True, False, True):
            widget.update_smooth_ui_state(reject_enabled=staged)
            assert widget.smooth_reject_btn.text() in ("Cancel", "Discard")


class TestTheOneChannelPicker:
    """`TestRefineImageCombos` replaced.  The three per-method
    combos are gone; there is one, on the shared five-point policy, and its
    blank state is an explicit refusal rather than a fallback."""

    RETIRED = ("refine_gac_image_combo", "refine_acwe_image_combo",
               "refine_gg_image_combo", "refine_gg_source_combo",
               "refine_alpha_sb", "refine_distance_clamp_sb",
               "refine_acwe_distance_clamp_sb")

    @pytest.mark.parametrize("attr", RETIRED)
    def test_the_retired_controls_are_gone(self, widget, attr):
        assert not hasattr(widget, attr), (
            "{} survived the merge -- a control nobody can reach is a "
            "control somebody will wire".format(attr)
        )

    def test_the_old_refresher_is_gone_too(self, widget):
        assert not hasattr(widget, "refresh_refine_image_combos")

    def test_it_is_filled_by_the_refresher(self, widget):
        widget.set_refine_channels(["membrane", "dapi"])
        combo = widget.refine_channel_combo
        assert [combo.itemText(i) for i in range(combo.count())] == [
            "membrane", "dapi"]

    def test_the_current_choice_survives_a_refresh(self, widget):
        widget.set_refine_channels(["membrane", "dapi"])
        widget.refine_channel_combo.setCurrentIndex(1)
        widget.set_refine_channels(["membrane", "dapi", "extra"])
        assert widget.refine_channel_combo.currentText() == "dapi"

    def test_a_blank_combo_names_nothing_rather_than_the_first_layer(
            self, widget):
        widget.set_refine_channels(["membrane", "dapi"])
        widget.refine_channel_combo.setCurrentIndex(1)
        widget.set_refine_channels(["membrane", "brightfield"])
        assert widget.refine_channel_combo.currentIndex() == -1
        assert widget.refine_channel_name() == ""


class TestWhichRowsTheMethodShows:
    """`isHidden()`, not `isVisibleTo(widget)`: the Refine page is behind
    another tab in this fixture, so every descendant answers False to
    `isVisibleTo(widget)` and the assertion would prove nothing.
    `isHidden()` asks the question actually being tested -- was this
    widget explicitly taken off the page.

    Asking for a channel a method will not read is the same
    mistake as silently picking one."""

    def _pick(self, widget, key):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        widget.refine_method_combo.setCurrentIndex(REFINE_METHOD_INDEX[key])

    @pytest.mark.parametrize("key", ["sdf", "dilate_erode", "guided_grow",
                                     "fill_holes", "flood_fill"])
    def test_the_channel_card_is_hidden_for_a_geometry_method(
            self, widget, key):
        self._pick(widget, key)
        assert widget.refine_needs_channel() is False
        assert widget._refine_channel_container.isHidden() is True

    @pytest.mark.parametrize("key", ["contour", "settle"])
    def test_it_is_shown_for_the_two_that_read_one(self, widget, key):
        self._pick(widget, key)
        assert widget.refine_needs_channel() is True
        assert widget._refine_channel_container.isHidden() is False

    def test_the_contour_asks_on_the_SAME_terms_settle_does(self, widget):
        """Audit C-M3: this branch returned True unconditionally, so *Snap
        to edges* on a shape wall asked for a channel it would not read --
        and refused Preview without one."""
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_WALL_KINDS,
        )

        self._pick(widget, "contour")
        widget.refine_contour_mode_combo.setCurrentIndex(0)   # snap to edges
        for kind in ("distance", "skeleton", "probability"):
            widget.refine_wall_kind_combo.setCurrentIndex(
                REFINE_WALL_KINDS.index(kind))
            assert widget.refine_needs_channel() is False, kind
            assert widget._refine_channel_container.isHidden() is True
        widget.refine_wall_kind_combo.setCurrentIndex(
            REFINE_WALL_KINDS.index("membrane"))
        assert widget.refine_needs_channel() is True

    def test_brightness_mode_always_needs_one(self, widget):
        """It reads the prepared channel directly, whatever the wall kind
        says."""
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_WALL_KINDS,
        )

        self._pick(widget, "contour")
        widget.refine_contour_mode_combo.setCurrentIndex(1)
        for kind in ("distance", "probability", "membrane"):
            widget.refine_wall_kind_combo.setCurrentIndex(
                REFINE_WALL_KINDS.index(kind))
            assert widget.refine_needs_channel() is True, kind

    def test_settle_on_a_shape_wall_hides_it_again(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_WALL_KINDS,
        )

        self._pick(widget, "settle")
        widget.refine_wall_kind_combo.setCurrentIndex(
            REFINE_WALL_KINDS.index("distance"))
        assert widget.refine_needs_channel() is False
        assert widget._refine_channel_container.isHidden() is True

    def test_the_probability_wall_says_why_it_needs_no_channel(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_WALL_KINDS,
        )

        self._pick(widget, "settle")
        widget.refine_wall_kind_combo.setCurrentIndex(
            REFINE_WALL_KINDS.index("probability"))
        assert widget.refine_needs_channel() is False
        assert widget._refine_no_channel_line.isHidden() is False
        assert "probability map" in widget._refine_no_channel_line.text()


class TestThePrepareTicklist:
    """ONE ticklist, and the consumer's rule decides which
    chain a method gets.  "Local contrast" is HIDDEN where the tool is
    looking for walls, not silently ignored."""

    def _pick(self, widget, key):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        widget.refine_method_combo.setCurrentIndex(REFINE_METHOD_INDEX[key])

    def test_the_five_labels_are_the_shipped_ones_verbatim(self, widget):
        M = _mod()
        for key, label in M.PREP_TICK_LABELS.items():
            cb = getattr(widget, "refine_prep_{}_cb".format(key))
            assert cb.text() == label

    def test_local_contrast_is_not_offered_where_walls_are_built(self, widget):
        self._pick(widget, "settle")
        cb = widget.refine_prep_clahe_cb
        assert cb.isHidden() is True
        assert "clahe" not in widget.refine_prep_chain()

    def test_it_is_offered_for_the_brightness_contour(self, widget):
        self._pick(widget, "contour")
        widget.refine_contour_mode_combo.setCurrentIndex(1)
        assert widget.refine_prep_clahe_cb.isHidden() is False
        assert "clahe" in widget.refine_prep_chain()

    def test_the_chain_is_the_shared_builder(self, widget):
        M = _mod()
        self._pick(widget, "contour")
        widget.refine_contour_mode_combo.setCurrentIndex(1)
        widget.refine_prep_bgsub_cb.setChecked(True)
        assert widget.refine_prep_chain() == M.chain_from_ticks(
            {"clahe": True, "bgsub": True})

    def test_nothing_ticked_is_the_engines_own_passthrough(self, widget):
        self._pick(widget, "settle")
        for key in ("clahe", "zcorrect", "bgsub", "lift", "denoise"):
            getattr(widget, "refine_prep_{}_cb".format(key)).setChecked(False)
        assert widget.refine_prep_chain() == "none"


class TestTheWallKinds:
    """Nine, with "No wall (midline only)" dropped and the
    reconstruction's probability map appended."""

    def test_there_are_nine(self):
        M = _mod()
        assert len(M.REFINE_WALL_KINDS) == 9
        assert len(M.REFINE_WALLS_FROM_LABELS) == 9

    def test_no_wall_is_dropped(self):
        assert "none" not in _mod().REFINE_WALL_KINDS

    def test_the_probability_map_is_last(self):
        M = _mod()
        assert M.REFINE_WALL_KINDS[-1] == "probability"
        assert "Probability" in M.REFINE_WALLS_FROM_LABELS[-1]

    def test_the_reshape_combo_is_untouched(self):
        M = _mod()
        assert len(M.WALL_KINDS) == 9
        assert M.WALL_KINDS[-1] == "none"

    def test_the_kind_is_read_by_index(self, widget):
        M = _mod()
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        widget.refine_method_combo.setCurrentIndex(REFINE_METHOD_INDEX["settle"])
        for i, kind in enumerate(M.REFINE_WALL_KINDS):
            widget.refine_wall_kind_combo.setCurrentIndex(i)
            assert widget.refine_wall_kind() == kind

    def test_the_default_is_membrane_sheets(self, widget):
        M = _mod()
        assert widget.refine_wall_kind() == "membrane"
        assert widget.refine_wall_kind_combo.currentText() == (
            M.REFINE_WALLS_FROM_LABELS[M.REFINE_WALL_KINDS.index("membrane")])


class TestGetRefineParams:
    def test_every_combo_is_read_by_index_through_a_constant(self, widget):
        M = _mod()
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        widget.refine_method_combo.setCurrentIndex(REFINE_METHOD_INDEX["settle"])
        for i, name in enumerate(M.REFINE_SOLVERS):
            widget.refine_solver_combo.setCurrentIndex(i)
            assert widget.get_refine_params()["solver"] == name

    def test_the_settle_numbers_come_back(self, widget):
        widget.refine_settle_band_sb.setValue(4.5)
        widget.refine_settle_iter_sb.setValue(22)
        widget.refine_settle_clamp_sb.setValue(0.75)
        p = widget.get_refine_params()
        assert p["band_um"] == 4.5
        assert p["n_iter"] == 22
        assert p["clamp"] == 0.75

    def test_it_never_raises_on_a_half_built_widget(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            AdvancedFeaturesWidget,
        )

        class _Half:
            get_refine_params = AdvancedFeaturesWidget.get_refine_params
            refine_method_key = AdvancedFeaturesWidget.refine_method_key
            refine_is_wall_method = AdvancedFeaturesWidget.refine_is_wall_method
            refine_wall_kind = AdvancedFeaturesWidget.refine_wall_kind
            refine_needs_channel = AdvancedFeaturesWidget.refine_needs_channel
            refine_channel_name = AdvancedFeaturesWidget.refine_channel_name
            refine_prep_chain = AdvancedFeaturesWidget.refine_prep_chain
            _prep_ticks = AdvancedFeaturesWidget._prep_ticks

        p = _Half().get_refine_params()
        assert p["method"] == "sdf"
        assert p["channel"] == ""


class TestTheSettleSolverPicker:
    def test_the_engine_names_are_the_reshape_three(self):
        M = _mod()
        assert M.REFINE_SOLVERS == M.SPLIT_SOLVERS
        assert len(M.REFINE_SOLVER_LABELS) == 3

    def test_watershed_is_the_default(self, widget):
        assert widget.get_refine_params()["solver"] == "watershed"

    def test_the_labels_do_not_name_the_algorithm_in_the_action(self):
        """The one that DOES name it is the third, and it names the
        dependency rather than the maths."""
        M = _mod()
        assert "watershed" not in M.REFINE_SOLVER_LABELS[0].lower()
        assert "u-Segment3D" in M.REFINE_SOLVER_LABELS[2]
