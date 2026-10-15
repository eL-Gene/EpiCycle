"""The panel side.

Offscreen, the REAL ``AdvancedFeaturesWidget``: ``get_refine_step_params`` /
``set_refine_step_params`` round trip with NON-default values for the six
methods, the greyed-item and clamp refusals, the prep read-back, the
``step_needs_channel`` parity with ``refine_needs_channel()``, the vocabulary
parity, and the ``get_refine_params()`` key-set golden.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")

from packages.epicycle_annotator import refine_recipe as rr  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def widget(qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    yield w
    w.setParent(None)


def _ticks(**on):
    t = {k: False for k in rr.PREP_TICKS}
    t.update(on)
    return t


#: Non-default values for every method (each differs from the widget's own
#: default, so a setter that silently did nothing fails the round trip).
NON_DEFAULT = {
    "sdf": {"sigma_z": 2.5, "sigma_xy": 0.7, "preserve_topology": False},
    "contour": {
        "mode": "snap", "wall_kind": "frangi", "axis": "XZ",
        "sampling": "count", "share_pct": 40, "n_slices": 9,
        "limit_um": 4.5, "iterations": 150, "balloon": -0.25, "gate": 0.45,
        "blur_um": 1.5, "sharpness": 250, "rebuild_um": 2.5,
        "lambda1": 1.3, "lambda2": 0.7,
        "prep_ticks": _ticks(bgsub=True, lift=True, clahe=True),
        "prep_chain": "bgsub+gamma",          # CLAHE dropped: a wall method
    },
    "dilate_erode": {"radius": -4, "axis": "XZ"},
    "guided_grow": {"max_radius": 9, "prob_threshold": 0.35, "axis": "YZ"},
    "fill_holes": {"axis": "XY"},
    "settle": {
        "wall_kind": "gradient", "solver": "random_walker", "band_um": 3.5,
        "n_iter": 17, "clamp": 0.65, "compactness": 0.2,
        "prep_ticks": _ticks(zcorrect=True, denoise=True),
        "prep_chain": "zcorrect+denoise",
    },
}


class TestRoundTrip:
    @pytest.mark.parametrize("method", rr.RECIPE_METHODS)
    def test_set_then_get_is_the_same_step(self, widget, method):
        ok, why = widget.set_refine_step_params(method, NON_DEFAULT[method])
        assert ok, why
        assert widget.refine_method_key() == method
        assert widget.get_refine_step_params(method) == NON_DEFAULT[method]

    def test_brightness_keeps_clahe_in_its_chain(self, widget):
        p = dict(NON_DEFAULT["contour"], mode="brightness",
                 prep_chain="bgsub+gamma+clahe")
        ok, why = widget.set_refine_step_params("contour", p)
        assert ok, why
        assert widget.get_refine_step_params("contour") == p

    def test_the_method_change_swaps_the_page(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_INDEX,
        )

        widget.set_refine_step_params("settle", NON_DEFAULT["settle"])
        assert widget._refine_params_stack.currentIndex() == (
            REFINE_METHOD_INDEX["settle"])
        assert widget._refine_settle_advanced.isHidden() is False
        widget.set_refine_step_params("fill_holes", NON_DEFAULT["fill_holes"])
        assert widget._refine_params_stack.currentIndex() == (
            REFINE_METHOD_INDEX["fill_holes"])

    def test_the_method_change_fires_the_annotator_handler(self, widget):
        seen = []
        widget.set_smooth_callbacks(on_method_changed=seen.append)
        widget.set_refine_step_params("dilate_erode",
                                      NON_DEFAULT["dilate_erode"])
        assert seen, "the annotator's method handler must hear the change"

    def test_the_channel_combo_is_not_touched(self, widget):
        widget.set_refine_channels(["a", "b"], current="b")
        widget.set_refine_step_params("settle", NON_DEFAULT["settle"])
        assert widget.refine_channel_name() == "b"


class TestRefusals:
    def test_a_greyed_guided_grow_is_refused(self, widget):
        widget.set_refine_guided_grow_available(False)
        ok, why = widget.set_refine_step_params("guided_grow",
                                                NON_DEFAULT["guided_grow"])
        assert not ok and "not available" in why
        assert widget.refine_method_key() != "guided_grow"

    def test_a_greyed_solver_is_refused(self, widget):
        widget.set_refine_solver_available(False, "no u-Segment3D")
        p = dict(NON_DEFAULT["settle"], solver="label_diffusion")
        ok, why = widget.set_refine_step_params("settle", p)
        assert not ok and "solver" in why

    def test_a_value_the_box_would_round_is_refused(self, widget):
        p = dict(NON_DEFAULT["sdf"], sigma_z=2.55)       # one decimal box
        ok, why = widget.set_refine_step_params("sdf", p)
        assert not ok and "decimals" in why
        assert widget.smooth_standalone_sigma_z_sb.value() == 3.0

    def test_a_value_out_of_range_is_refused_before_anything_moves(
            self, widget):
        p = dict(NON_DEFAULT["dilate_erode"], radius=11)
        ok, why = widget.set_refine_step_params("dilate_erode", p)
        assert not ok
        assert widget.refine_method_key() == "sdf"            # untouched

    def test_a_clamped_widget_value_is_refused(self, widget, monkeypatch):
        # a recipe the validator accepts but the widget cannot hold
        widget.refine_settle_band_sb.setMaximum(3.0)
        ok, why = widget.set_refine_step_params("settle",
                                                NON_DEFAULT["settle"])
        assert not ok and "out of range" in why

    def test_an_axis_the_combo_does_not_offer_is_an_error_not_index_0(
            self, widget):
        widget.refine_fh_axis_combo.removeItem(
            widget.refine_fh_axis_combo.findText("XY"))
        ok, why = widget.set_refine_step_params("fill_holes", {"axis": "XY"})
        assert not ok and "orientation" in why

    def test_flood_fill_and_bad_params_never_raise(self, widget):
        ok, _ = widget.set_refine_step_params(
            "flood_fill", {"axis": "3D", "max_size": 5000})
        assert not ok
        ok, _ = widget.set_refine_step_params("sdf", {"sigma_z": 1})
        assert not ok
        ok, _ = widget.set_refine_step_params("nope", {})
        assert not ok

    def test_the_prep_read_back_refuses_a_chain_that_differs(
            self, widget, monkeypatch):
        monkeypatch.setattr(type(widget), "refine_prep_chain",
                            lambda self: "bgsub")
        ok, why = widget.set_refine_step_params("settle",
                                                NON_DEFAULT["settle"])
        assert not ok and why == rr.STAGE_PREP_DIFFERS


class TestParity:
    def test_the_vocabulary_matches_the_panel(self):
        from napari_epicycle.annotator import advanced_features_widget as a
        from packages.epicycle_annotator import refine

        assert rr.WALL_KINDS == a.REFINE_WALL_KINDS
        assert rr.SOLVERS == a.REFINE_SOLVERS
        assert rr.AXES == a.REFINE_CONTOUR_AXES == refine.CONTOUR_ORIENTATIONS
        assert rr.CONTOUR_MODES == refine.CONTOUR_MODES
        assert rr.SLICE_SAMPLINGS == refine.SLICE_SAMPLINGS
        assert set(rr.PREP_TICKS) == set(a.PREP_TICK_LABELS)
        assert rr.IMAGE_WALL_KINDS == {
            a.REFINE_WALL_KINDS[i] for i in a.REFINE_IMAGE_WALL_INDICES}
        assert set(rr.RECIPE_METHODS) | {"flood_fill"} == {
            k for _i, _t, k in a.REFINE_METHOD_KEYS}

    def test_step_needs_channel_matches_the_panel_everywhere(self, widget):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_KEYS,
            REFINE_WALL_KINDS,
        )

        n = 0
        for idx, _t, key in REFINE_METHOD_KEYS:
            widget.refine_method_combo.setCurrentIndex(idx)
            for mode_idx, mode in enumerate(rr.CONTOUR_MODES):
                widget.refine_contour_mode_combo.setCurrentIndex(mode_idx)
                for k_idx, kind in enumerate(REFINE_WALL_KINDS):
                    widget.refine_wall_kind_combo.setCurrentIndex(k_idx)
                    got = rr.step_needs_channel(
                        key, {"mode": mode, "wall_kind": kind})
                    assert got == widget.refine_needs_channel(), (
                        key, mode, kind)
                    n += 1
        assert n == 7 * 2 * 9

    def test_step_word_matches_the_method_list(self):
        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_METHOD_KEYS,
        )

        assert set(rr.STEP_WORD) == {k for _i, _t, k in REFINE_METHOD_KEYS}


#: The ``get_refine_params()`` wire format, frozen ("same keys,
#: same values").  No golden existed before the recipes brick.
REFINE_PARAMS_GOLDEN = {
    "method": "sdf", "channel": "", "prep_chain": "clahe",
    "needs_channel": False, "wall_kind": "membrane", "solver": "watershed",
    "band_um": 2.0, "n_iter": 10, "clamp": 0.5, "compactness": 0.0,
    "contour_mode": 0, "contour_axis": "3D", "contour_sampling": 0,
    "contour_share_pct": 25.0, "contour_n_slices": 12,
    "contour_limit_um": 3.0, "contour_iterations": 100,
    "contour_balloon": 0.0, "contour_gate": 0.3, "contour_blur_um": 2.0,
    "contour_sharpness": 100.0, "contour_rebuild_um": 1.5,
    "contour_lambda1": 1.0, "contour_lambda2": 1.0, "sigma_z": 3.0,
    "sigma_xy": 1.5, "preserve_topology": True,
}


class TestGetRefineParamsGolden:
    def test_key_set_is_frozen(self, widget):
        assert list(widget.get_refine_params()) == list(REFINE_PARAMS_GOLDEN)

    def test_default_values_are_frozen(self, widget):
        assert widget.get_refine_params() == REFINE_PARAMS_GOLDEN

    def test_setting_a_recipe_step_does_not_change_the_format(self, widget):
        widget.set_refine_step_params("settle", NON_DEFAULT["settle"])
        got = widget.get_refine_params()
        assert list(got) == list(REFINE_PARAMS_GOLDEN)
        assert got["solver"] == "random_walker"
        assert got["prep_chain"] == "zcorrect+denoise"


# ---------------------------------------------------------------------------
# -- the button, the menu, the stage line, the footer
# ---------------------------------------------------------------------------


def _actions(menu):
    out = []
    for act in menu.actions():
        if act.isSeparator():
            out.append(("---", False))
        elif act.menu() is not None:
            out.append((act.text(), act.isEnabled(),
                        [(a.text(), a.isEnabled()) for a in
                         act.menu().actions()]))
        else:
            out.append((act.text(), act.isEnabled()))
    return out


class TestTheButton:
    def test_a_menu_button_in_the_pinned_row(self, widget):
        from qtpy.QtWidgets import QMenu, QPushButton

        from napari_epicycle.annotator.advanced_features_widget import (
            REFINE_RECIPES_TIP,
        )

        btn = widget.refine_recipes_btn
        assert type(btn) is QPushButton and btn.text() == "Recipes"
        assert isinstance(btn.menu(), QMenu)
        assert btn.menu() is widget.refine_recipes_menu
        assert btn.toolTip() == REFINE_RECIPES_TIP == rr.BUTTON_TIP
        assert "::menu-indicator" in btn.styleSheet()
        assert btn.parentWidget() is widget.smooth_preview_btn.parentWidget()
        assert widget.refine_recipes_menu.toolTipsVisible()

    def test_reshape_gets_no_button(self, widget):
        from qtpy.QtWidgets import QPushButton

        page = widget._panel_slots["Reshape"]["page"]
        assert not [b for b in page.findChildren(QPushButton)
                    if b.text() == "Recipes"]

    def test_the_idle_footer_names_the_menu(self, widget):
        widget.update_smooth_ui_state(reject_enabled=False)
        assert widget._refine_footer_summary.text() == (
            "Nothing is staged yet. Saved recipes are under Recipes.")


class TestTheMenu:
    def test_it_renders_what_menu_entries_says(self, widget):
        store = {"tidy": rr.recipe_to_json(
            [rr.RecipeStep("sdf", NON_DEFAULT["sdf"]),
             rr.RecipeStep("fill_holes", {"axis": "3D"})],
            created="", spacing_known=True, spacing=()),
            "a&b": [1]}
        entries = rr.menu_entries(store, None, object_id=149,
                                  availability={}, channel="c")
        widget.populate_refine_recipes_menu(entries)
        got = _actions(widget.refine_recipes_menu)
        assert got[0] == ("Run a recipe on object 149", False)
        assert ("    tidy — Smooth -> Fill Holes", True) in got
        assert ("    a&&b — cannot be read: not a Refine recipe",
                False) in got
        assert ("---", False) in got
        assert ("Save as a recipe… (unavailable: nothing is staged yet)",
                False) in got
        load = [x for x in got if x[0] == "Load into the panel"][0]
        assert load[2] == [("tidy", True)]
        delete = [x for x in got if x[0] == "Delete"][0]
        assert [t for t, _e in delete[2]] == [
            "tidy", "a&&b — cannot be read: not a Refine recipe"]

    def test_triggering_calls_back_with_the_name(self, widget):
        seen = []
        widget.set_smooth_callbacks(
            on_recipe_run=lambda n: seen.append(("run", n)),
            on_recipe_load=lambda n: seen.append(("load", n)),
            on_recipe_save=lambda: seen.append(("save", None)),
            on_recipe_delete=lambda n: seen.append(("delete", n)))
        store = {"tidy": rr.recipe_to_json(
            [rr.RecipeStep("fill_holes", {"axis": "3D"})], created="",
            spacing_known=True, spacing=())}
        entries = rr.menu_entries(
            store, None, object_id=7, availability={}, channel="c",
            staged_records=[rr.RecipeStep("fill_holes", {"axis": "3D"})])
        widget.populate_refine_recipes_menu(entries)
        menu = widget.refine_recipes_menu
        for act in menu.actions():
            if act.text().strip().startswith("tidy"):
                act.trigger()
            if act.text().startswith("Save the"):
                act.trigger()
            if act.menu() is not None:
                act.menu().actions()[0].trigger()
        assert sorted(seen) == [("delete", "tidy"), ("load", "tidy"),
                                ("run", "tidy"), ("save", None)]

    def test_about_to_show_is_connected_once(self, widget):
        calls = []
        cb = lambda: calls.append(1)  # noqa: E731
        widget.set_smooth_callbacks(on_recipes_menu=cb)
        widget.set_smooth_callbacks(on_recipes_menu=cb)
        widget.refine_recipes_menu.aboutToShow.emit()
        assert calls == [1]

    def test_the_default_menu_says_it_needs_a_workspace(self, widget):
        got = _actions(widget.refine_recipes_menu)
        assert got == [(rr.NO_WORKSPACE, False)]


class TestTheStageLine:
    def test_it_is_shrinkable_elided_and_the_tooltip_is_whole(self, widget,
                                                              qapp):
        from qtpy.QtWidgets import QSizePolicy

        lb = widget.refine_stage_label
        assert lb.sizePolicy().horizontalPolicy() == QSizePolicy.Ignored
        widget.resize(340, 350)
        widget.show()
        widget.focus_tab("Refine")
        qapp.processEvents()
        try:
            long = ("'" + "x" * 40 + "' stopped at 12/12 · the object "
                    "is empty")
            widget.update_smooth_ui_state(stage_text=long)
            qapp.processEvents()
            assert lb.toolTip() == long
            assert lb.text() != long and lb.text().endswith("…")
            assert lb.fontMetrics().horizontalAdvance(lb.text()) <= (
                lb.contentsRect().width())
            widget.update_smooth_ui_state(stage_text="2/3 · Dilate/Erode")
            qapp.processEvents()
            assert lb.text() == "2/3 · Dilate/Erode"
            widget.update_smooth_ui_state(stage_text="")
            assert lb.isHidden()
        finally:
            widget.hide()


class TestAllOrNothing:
    def test_a_chain_this_method_would_not_give_moves_nothing(self, widget):
        """Audit L2: CLAHE ticked on a brightness contour would run, so a
        stored chain without it is refused BEFORE a widget moves."""
        p = dict(NON_DEFAULT["contour"], mode="brightness",
                 prep_chain="bgsub+gamma")         # ticks say clahe too
        before = (widget.refine_method_key(),
                  widget.get_refine_prep_ticks())
        ok, why = widget.set_refine_step_params("contour", p)
        assert not ok and why == rr.STAGE_PREP_DIFFERS
        assert (widget.refine_method_key(),
                widget.get_refine_prep_ticks()) == before
