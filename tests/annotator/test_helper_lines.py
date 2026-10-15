"""The grey helper lines under the controls.

Two dictionaries in ``advanced_features_widget``:

* ``HELP_LINES`` -- keyed by WIDGET ATTRIBUTE, rendered as one grey line
  directly under that control;
* ``ITEM_HELP`` -- keyed by COMBO attribute and indexed by item, rendered as
  one label under the combo whose text swaps with the selection.

The point of testing them is that a helper line is a promise about
behaviour.  A missing one leaves a knob unexplained; an ORPHAN one is worse,
because it means a control was renamed or dropped and the sentence about it
survived, unreachable, to be copied into the next thing.  So the keys are
held to SET EQUALITY against an explicit list, in both directions.

Brick 2a holds only what Brick 2a added; Brick 2b's Advanced regroup extends
``HELPED_WIDGETS`` in the same commit that extends ``HELP_LINES``.
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


def _mod():
    from napari_epicycle.annotator import advanced_features_widget as M

    return M


#: Every control that carries a HELP_LINES entry after Brick 2a.  Set
#: equality both ways: a missing helper and an orphan entry both fail.
HELPED_WIDGETS = (
    # Brick U2 -- the outline tick in S2 TARGET.  Its grey line is shown
    # at idle only, with the target hint: MEASURED 2026-09-20, leaving it
    # up during a session puts review+markers at 789 px against the
    # 753 px ceiling the UX round's 1.4 says may not be raised again
    # without the owner (hidden: 752 px).  The tooltip carries the same
    # words for the whole session.
    "split_contour_cb",
    # markers.  `split_seed_channel_combo` and `split_find_markers_btn` are
    # NOT here: layout does not draw a grey line under either,
    # and at the dock's 247 px viewport each one cost two lines of the
    # height the ticklist needs.  Their tooltips carry the same words.
    "split_seed_prep_clahe_cb",
    "split_seed_prep_zcorrect_cb",
    "split_seed_prep_bgsub_cb",
    "split_seed_prep_lift_cb",
    "split_seed_prep_denoise_cb",
    # walls
    "split_wall_prep_same_cb",
    "split_wall_prep_zcorrect_cb",
    "split_wall_prep_bgsub_cb",
    "split_wall_prep_lift_cb",
    "split_wall_prep_denoise_cb",
    "split_search_padding_sb",
    # Brick 2b -- the three rows promoted into "How to cut".  Each one
    # asks the user for a decision, so each one says what it decides.
    "split_refine_guided_cb",
    "split_arbiter_cb",
    "split_target_count_cb",
    # Brick 2b -- EVERY Advanced knob (the owner asked for a
    # short line under each; a knob with no sentence is a knob nobody
    # turns).  Set equality both ways, so adding a knob without a line
    # fails HERE rather than shipping unexplained.
    "split_seed_smooth_sb",
    "split_min_distance_sb",
    "split_seed_h_sb",
    "split_max_sigma_sb",
    "split_seed_thresh_sb",
    "split_seed_clip_sb",
    "split_lift_mode_combo",
    "split_seed_gamma_sb",
    "split_seed_bg_sb",
    "split_seed_denoise_sb",
    "split_wall_clip_sb",
    "split_wall_smooth_sb",
    "split_wall_ridge_cb",
    "split_arbiter_pct_sb",
    "split_rw_beta_sb",
    "split_compactness_sb",
    "split_refine_radius_sb",
    "split_refine_eps_sb",
    "split_refine_merge_small_cb",
    "split_min_size_sb",
    "split_auto_smooth_cb",
    "split_auto_advance_cb",
    # Brick 2b -- the Detect Seeds row (the owner's added scope:
    # varying these three is the productive part of that row).  The
    # threshold LINE is drawn by pyqtgraph, so its sentence is keyed on
    # the placeholder the plot is swapped into.
    "split_detect_min_dist_sb",
    "split_detect_bins_sb",
    "_detect_plot_placeholder",
    # --- REFINE, Brick B1.  Only what B1 ADDS: the one shared channel
    # picker and its five prepare ticks, the Settle page, and the
    # contour's mode picker.  The knobs B2 rebuilds keep their tooltips
    # until B2 rewrites them, so naming them here would assert a line
    # against a control that is about to move.
    "refine_channel_combo",
    "refine_prep_clahe_cb",
    "refine_prep_zcorrect_cb",
    "refine_prep_bgsub_cb",
    "refine_prep_lift_cb",
    "refine_prep_denoise_cb",
    "refine_wall_kind_combo",
    "refine_solver_combo",
    "refine_settle_band_sb",
    "refine_settle_iter_sb",
    "refine_settle_clamp_sb",
    "refine_settle_compactness_sb",
    # --- REFINE, Brick B2: the merged Active contour.  NOT here, on
    # purpose: `refine_contour_mode_combo` (its line is PER ITEM, below --
    # trim 1, the page describes only the chosen mode),
    # `refine_contour_axis_combo` (trim 3: the tooltip carries it) and
    # `refine_contour_balloon_sb` (its line is LIVE, computed from the push
    # and the gate -- `TestTheLivePushLine` in test_refine_contour_page.py),
    # `refine_contour_limit_sb` (its line came off the tallest page when
    # the prepare card's sentence went on -- B2 audit UX-M4; the tooltip
    # carries it) and `refine_contour_slices_combo` (PER ITEM, below).
    "refine_contour_gate_sb",
    "refine_contour_blur_sb",
    "refine_contour_sharpness_sb",
    "refine_contour_rebuild_um_sb",
    # --- REFINE, Brick B3: help parity.  A line under every knob on the
    # five short pages (they sit far under both ceilings) and under the
    # probability overlay tick on the Advanced card.  The contour's rows
    # stay tooltips -- see `TOOLTIP_ONLY` in test_refine_help_parity.py,
    # which walks the real page and holds EVERY control to one or the
    # other.
    "smooth_standalone_sigma_z_sb",
    "smooth_standalone_sigma_xy_sb",
    "smooth_preserve_topology_cb",
    "refine_de_radius_sb",
    "refine_de_axis_combo",
    "refine_gg_max_radius_sb",
    "refine_gg_prob_threshold_sb",
    "refine_gg_axis_combo",
    "refine_fh_axis_combo",
    "refine_ff_axis_combo",
    "refine_ff_max_size_sb",
    "refine_prob_overlay_cb",
)

#: The combos that carry a per-item helper, and their item counts -- the four
#: Reshape S3 combos and, since REFINE B2, the contour mode picker.
ITEM_HELPED_COMBOS = (
    # REFINE B2 trim 1: the contour mode's line describes the chosen mode.
    ("refine_contour_mode_combo", 2),
    # ...and the slice sampler's, per sampling (B2 audit UX-L).
    ("refine_contour_slices_combo", 3),
    ("split_seeds_from_combo", 5),
    ("split_seed_detector_combo", 6),
    ("split_walls_from_combo", 9),
    ("split_solver_combo", 3),
)

#: REFINE B3: the combos whose ITEM_HELP is rendered as per-item TOOLTIPS in
#: the open list rather than as a swapping line (Settle and the contour are
#: at their ceilings).  `test_refine_help_parity.py::TestItemHelp` holds
#: each item's tooltip to its sentence.
ITEM_TIPPED_COMBOS = (
    ("refine_method_combo", 7),
    ("refine_wall_kind_combo", 9),
    ("refine_solver_combo", 3),
)

#: Words that name an engine token rather than a thing the user can see.
#: `distance`, `membrane`, `gradient` and friends are ordinary English and
#: are deliberately NOT in this list; these are the ones that are not.
ENGINE_TOKENS = (
    "hmax", "hmin", "blobs_log", "blobs_dog", "threshcc",
    "random_walker", "label_diffusion", "watershed",
    "zcorrect", "bgsub", "clahe", "denoise",
    "merge_small", "merge_to_count",
    "rw_beta", "min_distance_um", "max_sigma_um", "seed_thresh",
    "arbiter_percentile", "keep_seed_ids", "seed_preproc", "wall_preproc",
)


def _all_texts():
    M = _mod()
    out = list(M.HELP_LINES.values())
    for texts in M.ITEM_HELP.values():
        out.extend(texts)
    return out


def _all_user_strings():
    """Every string the panel puts in front of the user from these
    module tables -- the helpers, the item help, the combo LABELS and
    the sentence a stale marker list shows."""
    M = _mod()
    out = list(_all_texts())
    out.extend(M.SEEDS_FROM_LABELS)
    out.extend(M.WALLS_FROM_LABELS)
    out.extend(M.SEED_DETECTOR_LABELS)
    out.extend(M.SEEDS_FROM_SHORT)
    out.extend(M.WALLS_FROM_SHORT)
    out.append(M.AdvancedFeaturesWidget._MARKERS_STALE_TEXT)
    out.append(M._PREP_ORDER_SENTENCE)
    return out


# --------------------------------------------------------------------------- #
# set equality, in both directions
# --------------------------------------------------------------------------- #
class TestHelpLinesCoverExactlyTheHelpedWidgets:
    def test_no_helped_widget_is_missing_its_line(self):
        missing = set(HELPED_WIDGETS) - set(_mod().HELP_LINES)
        assert not missing, "no helper line for: {}".format(sorted(missing))

    def test_no_orphan_entries(self):
        """An orphan means a control was renamed or dropped and its sentence
        survived, unreachable -- ready to be copied onto the next thing."""
        orphans = set(_mod().HELP_LINES) - set(HELPED_WIDGETS)
        assert not orphans, "HELP_LINES entries reach nothing: {}".format(
            sorted(orphans))

    @pytest.mark.parametrize("attr", HELPED_WIDGETS)
    def test_the_widget_it_names_actually_exists(self, widget, attr):
        assert hasattr(widget, attr), (
            "HELP_LINES names {!r}, which is not a control".format(attr)
        )

    @pytest.mark.parametrize("attr", HELPED_WIDGETS)
    def test_the_line_is_on_screen_under_its_control(self, widget, attr):
        """The text has to appear in a QLabel somewhere on the panel, or the
        dictionary is a comment."""
        from qtpy.QtWidgets import QLabel

        text = _mod().HELP_LINES[attr]
        shown = [
            lb for lb in widget.findChildren(QLabel)
            if lb.text() == text
        ]
        assert shown, "{}'s helper line is never rendered".format(attr)


class TestEveryAdvancedKnobIsExplained:
    """The owner's ask in one sentence: "knobs like seed
    prominence, min marker gap, ridge, beta need a short line"."""

    ADVANCED_KNOBS = (
        "split_seed_h_sb", "split_min_distance_sb", "split_wall_ridge_cb",
        "split_rw_beta_sb",
    )

    @pytest.mark.parametrize("attr", ADVANCED_KNOBS)
    def test_the_four_the_owner_named_have_one(self, attr):
        assert _mod().HELP_LINES.get(attr, "").strip()

    def test_no_knob_in_the_advanced_cards_is_left_unexplained(self, widget):
        """The sweep behind the four: every spin box, combo and tick in
        the Advanced container carries a sentence.  The sigma pair is
        the one exemption -- it is ONE knob on two axes and the
        auto-smooth tick above it explains both."""
        from qtpy.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QSpinBox

        M = _mod()
        exempt = {"smooth_sigma_z_sb", "smooth_sigma_xy_sb"}
        by_id = {id(getattr(widget, a)): a for a in dir(widget)
                 if a.startswith(("split_", "smooth_"))
                 and isinstance(getattr(widget, a, None),
                                (QCheckBox, QComboBox, QSpinBox,
                                 QDoubleSpinBox))}
        missing = []
        for cls in (QCheckBox, QComboBox, QSpinBox, QDoubleSpinBox):
            for w in widget._split_advanced_container.findChildren(cls):
                name = by_id.get(id(w))
                if name and name not in exempt and name not in M.HELP_LINES:
                    missing.append(name)
        assert not missing, "unexplained Advanced knob(s): {}".format(
            sorted(set(missing)))


class TestTheSentencesMatchTheCodeTheyDescribe:
    """The class of bug this file exists for, caught live in the Brick 2b
    UX audit: a sentence that reads perfectly and says the wrong thing.
    Silent misinformation is worse than no help at all."""

    def _seed_branches(self, param):
        """Which detector tokens the engine feeds *param* to.

        Read off `engine._build_seeds`, the dispatch that decides it --
        Allows source inspection for exactly this, a dispatch
        table, and nothing here can be asked at runtime without running
        six detectors on real data."""
        import inspect
        import re

        from packages.epicycle_annotator.reshape import engine

        src = inspect.getsource(engine._build_seeds)
        out = set()
        # each branch reads   if kind == "peaks": ... seeds_from_nuclei(
        for chunk in re.split(r'(?=kind == ")', src):
            m = re.match(r'kind == "([a-z_]+)"', chunk)
            if not m:
                continue
            body = chunk.split("elif", 1)[0]
            if param in body:
                out.add(m.group(1))
        return out

    def test_the_marker_threshold_line_names_every_detector_that_reads_it(
            self):
        """M5: it said "Bright-object centres only".  The engine also
        hands it to `peaks` -- which is the DEFAULT detector -- so the
        one knob the default finder has was documented as belonging to
        something else."""
        M = _mod()
        branches = self._seed_branches('p["seed_thresh"]')
        assert branches == {"peaks", "threshcc"}, branches
        text = M.HELP_LINES["split_seed_thresh_sb"]
        labels = dict(zip(M.SEED_DETECTORS, M.SEED_DETECTOR_LABELS))
        for token in branches:
            assert labels[token] in text, (
                "{!r} reads the marker threshold but the line does not "
                "name it: {!r}".format(labels[token], text)
            )

    def test_the_prominence_line_covers_the_blob_finders_too(self):
        """M5: `h` is ALSO the blob finders' detection threshold, where
        it works the other way round (lower finds more).  A user reading
        "how far a peak must stand out" and raising it to find more
        blobs finds fewer."""
        M = _mod()
        branches = self._seed_branches('p["h"]')
        assert "blobs_log" in branches and "blobs_dog" in branches, branches
        text = M.HELP_LINES["split_seed_h_sb"].lower()
        assert "blob" in text, text
        assert "lower" in text, (
            "the line does not say which way it goes for the blob "
            "finders: {!r}".format(text)
        )

    def test_the_auto_smooth_line_does_not_promise_it_happens_at_accept(
            self):
        """H2: it said "when you Accept".  The checkbox is read inside
        `_preview_split`, so the PREVIEW already shows the smoothing --
        and the guided path never reads it at all."""
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        M = _mod()
        preview = inspect.getsource(Curation3DAnnotator._preview_split)
        accept = inspect.getsource(Curation3DAnnotator._accept_split)
        assert "split_auto_smooth_cb" in preview
        assert "split_auto_smooth_cb" not in accept, (
            "it IS read at accept now -- rewrite the sentence, not the test"
        )
        text = M.HELP_LINES["split_auto_smooth_cb"]
        assert "Accept" not in text, text
        assert "preview" in text.lower(), text

    def test_the_auto_smooth_line_says_which_path_it_applies_to(self):
        """The other half of H2: the guided path does not read it, so a
        sentence that does not say "hand-painted" is wrong for every
        guided cut."""
        import inspect

        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        guided = inspect.getsource(Curation3DAnnotator._apply_reshape_result)
        assert "split_auto_smooth_cb" not in guided
        assert "hand-painted" in _mod().HELP_LINES["split_auto_smooth_cb"]


class TestItemHelp:
    @pytest.mark.parametrize("attr,count", ITEM_HELPED_COMBOS)
    def test_one_sentence_per_item(self, widget, attr, count):
        M = _mod()
        assert attr in M.ITEM_HELP, "{} has no per-item help".format(attr)
        assert len(M.ITEM_HELP[attr]) == count
        assert getattr(widget, attr).count() == count, (
            "{} has {} items but {} sentences".format(
                attr, getattr(widget, attr).count(), count)
        )

    @pytest.mark.parametrize("attr,count", ITEM_HELPED_COMBOS)
    def test_the_label_swaps_with_the_selection(self, widget, attr, count):
        M = _mod()
        combo = getattr(widget, attr)
        label = widget._item_help_labels[attr]
        for idx in range(count):
            combo.setCurrentIndex(idx)
            assert label.text() == M.ITEM_HELP[attr][idx], (attr, idx)

    def test_no_combo_carries_help_for_items_it_does_not_have(self, widget):
        M = _mod()
        known = dict(ITEM_HELPED_COMBOS)
        known.update(ITEM_TIPPED_COMBOS)
        for attr in M.ITEM_HELP:
            assert attr in known, (
                "ITEM_HELP names {!r}, which is not one of the item-helped "
                "S3 combos".format(attr)
            )


# --------------------------------------------------------------------------- #
# the sentence lint
# --------------------------------------------------------------------------- #
class TestEveryHelperIsAPlainSentence:
    def test_each_one_ends_in_a_full_stop(self):
        for text in _all_texts():
            assert text.strip().endswith("."), repr(text)

    def test_each_one_starts_with_a_capital(self):
        for text in _all_texts():
            assert text[:1].isupper(), repr(text)

    def test_none_is_longer_than_a_helper_line_should_be(self):
        for text in _all_texts():
            assert len(text) <= 160, "{} chars: {!r}".format(len(text), text)

    def test_no_user_string_names_an_engine_token(self):
        """Widened past the helper lines: the combo LABELS and the
        stale sentence are read far more often than any of them."""
        for text in _all_user_strings():
            low = text.lower()
            for token in ENGINE_TOKENS:
                assert token not in low, (
                    "{!r} names the engine token {!r}".format(text, token)
                )

    def test_every_combo_label_is_short_enough_to_read(self):
        M = _mod()
        for text in (list(M.SEEDS_FROM_LABELS)
                     + list(M.WALLS_FROM_LABELS)
                     + list(M.SEED_DETECTOR_LABELS)):
            assert 0 < len(text) <= 42, "{} chars: {!r}".format(
                len(text), text)

    def test_the_short_forms_line_up_with_the_long_ones(self):
        M = _mod()
        assert len(M.SEEDS_FROM_SHORT) == len(M.SEEDS_FROM_LABELS)
        assert len(M.WALLS_FROM_SHORT) == len(M.WALLS_FROM_LABELS)
        for text in list(M.SEEDS_FROM_SHORT) + list(M.WALLS_FROM_SHORT):
            assert 0 < len(text) <= 16, repr(text)

    def test_none_names_an_engine_token(self):
        """A helper line explains the thing, not the identifier behind it."""
        for text in _all_texts():
            low = text.lower()
            for token in ENGINE_TOKENS:
                assert token not in low, (
                    "helper line names the engine token {!r}: {!r}".format(
                        token, text)
                )

    def test_the_run_order_sentence_is_a_sentence_too(self):
        M = _mod()
        assert M._PREP_ORDER_SENTENCE.strip().endswith(".")
        assert len(M._PREP_ORDER_SENTENCE) <= 200


# --------------------------------------------------------------------------- #
# cp1252 (the 2026-09-07 console hazard class)
# --------------------------------------------------------------------------- #
class TestTheDictionariesStayCp1252Safe:
    def test_every_helper_encodes(self):
        for text in _all_user_strings():
            text.encode("cp1252")

    def test_every_key_encodes(self):
        M = _mod()
        for key in list(M.HELP_LINES) + list(M.ITEM_HELP):
            key.encode("cp1252")

    def test_the_prepare_tick_labels_encode(self):
        M = _mod()
        for key, label in M.PREP_TICK_LABELS.items():
            key.encode("cp1252")
            label.encode("cp1252")
