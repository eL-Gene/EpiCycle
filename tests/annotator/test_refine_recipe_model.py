"""The pure recipe model.

No Qt, no napari: ``packages.epicycle_annotator.refine_recipe`` only.
Invariants 8 (round trip), 10 (Flood Fill never saved) and 11 (every greyed
item carries its reason as text; no two states share a sentence).
"""

from __future__ import annotations

import copy

import pytest

from packages.epicycle_annotator import refine_recipe as rr

TICKS_OFF = {k: False for k in rr.PREP_TICKS}


def _ticks(**on):
    t = dict(TICKS_OFF)
    t.update(on)
    return t


def good_params():
    return {
        "sdf": {"sigma_z": 2.5, "sigma_xy": 1.2, "preserve_topology": False},
        "contour": {
            "mode": "snap", "wall_kind": "membrane", "axis": "XY",
            "sampling": "count", "share_pct": 30, "n_slices": 9,
            "limit_um": 4.5, "iterations": 120, "balloon": -0.2,
            "gate": 0.35, "blur_um": 1.5, "sharpness": 150,
            "rebuild_um": 2.0, "lambda1": 1.2, "lambda2": 0.9,
            "prep_ticks": _ticks(bgsub=True, lift=True),
            "prep_chain": "bgsub+gamma",
        },
        "dilate_erode": {"radius": -3, "axis": "XZ"},
        "guided_grow": {"max_radius": 7, "prob_threshold": 0.4,
                        "axis": "3D"},
        "fill_holes": {"axis": "YZ"},
        "settle": {
            "wall_kind": "probability", "solver": "random_walker",
            "band_um": 3.5, "n_iter": 12, "clamp": 0.6, "compactness": 0.1,
            "prep_ticks": _ticks(clahe=True, zcorrect=True),
            "prep_chain": "zcorrect",
        },
    }


# ---------------------------------------------------------------------------
# -- per method, every key, every range edge
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("method", rr.RECIPE_METHODS)
    def test_every_method_validates_with_all_its_keys(self, method):
        rr.RecipeStep(method, good_params()[method]).validate()

    @pytest.mark.parametrize("method", rr.RECIPE_METHODS)
    def test_a_missing_key_is_refused(self, method):
        for key in good_params()[method]:
            p = good_params()[method]
            del p[key]
            with pytest.raises(rr.RecipeError, match="missing"):
                rr.RecipeStep(method, p).validate()

    @pytest.mark.parametrize("method", rr.RECIPE_METHODS)
    def test_an_unknown_key_is_refused(self, method):
        p = good_params()[method]
        p["surprise"] = 1
        with pytest.raises(rr.RecipeError, match="unknown setting"):
            rr.RecipeStep(method, p).validate()

    @pytest.mark.parametrize("method,key,low,high", [
        ("sdf", "sigma_z", 0.1, 10.0), ("sdf", "sigma_xy", 0.1, 10.0),
        ("contour", "share_pct", 5, 100), ("contour", "n_slices", 3, 60),
        ("contour", "limit_um", 0.5, 30.0),
        ("contour", "iterations", 10, 500),
        ("contour", "balloon", -1.0, 1.0), ("contour", "gate", 0.0, 1.0),
        ("contour", "blur_um", 0.5, 10.0),
        ("contour", "sharpness", 1, 1000),
        ("contour", "rebuild_um", 0.0, 15.0),
        ("contour", "lambda1", 0.1, 10.0), ("contour", "lambda2", 0.1, 10.0),
        ("dilate_erode", "radius", -10, 10),
        ("guided_grow", "max_radius", 1, 15),
        ("guided_grow", "prob_threshold", 0.0, 1.0),
        ("settle", "band_um", 0.5, 20.0), ("settle", "n_iter", 1, 50),
        ("settle", "clamp", 0.05, 0.95),
        ("settle", "compactness", 0.0, 1.0),
    ])
    def test_every_range_edge(self, method, key, low, high):
        is_int = isinstance(low, int)
        step = 1 if is_int else 0.01
        for ok in (low, high):
            p = good_params()[method]
            p[key] = ok
            rr.RecipeStep(method, p).validate()
        for bad in (low - step, high + step):
            p = good_params()[method]
            p[key] = bad
            with pytest.raises(rr.RecipeError, match="out of range"):
                rr.RecipeStep(method, p).validate()

    def test_a_whole_number_knob_refuses_a_fraction_and_a_bool(self):
        p = good_params()["dilate_erode"]
        p["radius"] = 1.5
        with pytest.raises(rr.RecipeError):
            rr.RecipeStep("dilate_erode", p).validate()
        p["radius"] = True
        with pytest.raises(rr.RecipeError):
            rr.RecipeStep("dilate_erode", p).validate()

    @pytest.mark.parametrize("method,key", [
        ("contour", "mode"), ("contour", "wall_kind"), ("contour", "axis"),
        ("contour", "sampling"), ("settle", "wall_kind"),
        ("settle", "solver"), ("dilate_erode", "axis"),
        ("guided_grow", "axis"), ("fill_holes", "axis"),
    ])
    def test_combos_are_TOKENS_never_indices(self, method, key):
        p = good_params()[method]
        p[key] = 0
        with pytest.raises(rr.RecipeError, match="unknown value"):
            rr.RecipeStep(method, p).validate()

    def test_log_is_refused(self):
        p = good_params()["contour"]
        p["prep_chain"] = "bgsub+log"
        with pytest.raises(rr.RecipeError, match="log"):
            rr.RecipeStep("contour", p).validate()

    def test_a_chain_that_contradicts_its_ticks_is_refused(self):
        p = good_params()["settle"]
        p["prep_chain"] = "bgsub"
        with pytest.raises(rr.RecipeError, match="does not match"):
            rr.RecipeStep("settle", p).validate()

    def test_a_wall_method_may_drop_clahe_from_the_chain(self):
        # Settle ran with drop_clahe=True: the tick was on, the chain lacks it.
        rr.RecipeStep("settle", good_params()["settle"]).validate()

    def test_flood_fill_is_refused(self):
        with pytest.raises(rr.RecipeError, match="Flood Fill"):
            rr.RecipeStep("flood_fill",
                          {"axis": "3D", "max_size": 5000}).validate()


# ---------------------------------------------------------------------------
# JSON round trip (invariant 8) and the problems
# ---------------------------------------------------------------------------


class TestJson:
    def _steps(self):
        gp = good_params()
        return [rr.RecipeStep(m, gp[m]) for m in rr.RECIPE_METHODS]

    def test_round_trip(self):
        obj = rr.recipe_to_json(self._steps(), created="2026-09-24T18:02:11",
                                spacing_known=True, spacing=(3.0, 1.0, 1.0))
        rec, problem = rr.recipe_from_json(obj)
        assert problem is None
        assert rec == rr.Recipe(steps=tuple(self._steps()),
                                created="2026-09-24T18:02:11",
                                spacing_known=True, spacing=(3.0, 1.0, 1.0))
        again = rr.recipe_to_json(rec.steps, created=rec.created,
                                  spacing_known=rec.spacing_known,
                                  spacing=rec.spacing)
        assert again == obj
        assert obj["kind"] == "refine_chain" and obj["version"] == 1

    def test_the_stored_shape_is_the_spec_one(self):
        obj = rr.recipe_to_json(self._steps()[:1], created="x",
                                spacing_known=False, spacing=())
        assert set(obj) == {"kind", "version", "created", "spacing_known",
                            "spacing", "steps"}
        assert set(obj["steps"][0]) == {"method", "params"}

    def test_version_2_is_made_by_a_newer_epicycle(self):
        obj = rr.recipe_to_json(self._steps()[:1], created="",
                                spacing_known=False, spacing=())
        obj["version"] = 2
        assert rr.recipe_from_json(obj) == (None, "made by a newer EpiCycle")

    @pytest.mark.parametrize("bad", [[1, 2], "tidy", 3, None,
                                     {"kind": "batch"}])
    def test_a_non_recipe_is_not_a_refine_recipe(self, bad):
        rec, problem = rr.recipe_from_json(bad)
        assert rec is None and problem.startswith("not a Refine recipe")

    def test_a_flood_fill_in_the_file_is_a_problem(self):
        obj = rr.recipe_to_json(self._steps()[:1], created="",
                                spacing_known=False, spacing=())
        obj["steps"].append({"method": "flood_fill",
                             "params": {"axis": "3D", "max_size": 500}})
        rec, problem = rr.recipe_from_json(obj)
        assert rec is None and "step 2" in problem and "Flood Fill" in problem

    def test_saving_flood_fill_raises(self):
        with pytest.raises(rr.RecipeError):
            rr.recipe_to_json([rr.RecipeStep("flood_fill", {"axis": "3D",
                                                            "max_size": 500})],
                              created="", spacing_known=False, spacing=())

    def test_thirteen_steps_are_refused_both_ways(self):
        steps = [rr.RecipeStep("fill_holes", {"axis": "3D"})] * 13
        with pytest.raises(rr.RecipeError):
            rr.recipe_to_json(steps, created="", spacing_known=False,
                              spacing=())
        obj = rr.recipe_to_json(steps[:12], created="", spacing_known=False,
                                spacing=())
        obj["steps"].append(copy.deepcopy(obj["steps"][0]))
        assert rr.recipe_from_json(obj)[0] is None

    def test_from_json_never_raises(self):
        for junk in ({"kind": "refine_chain", "version": 1,
                      "spacing_known": True, "steps": [None]},
                     {"kind": "refine_chain", "version": "1"},
                     {"kind": "refine_chain", "version": 1,
                      "spacing_known": "yes", "steps": []}):
            rec, problem = rr.recipe_from_json(junk)
            assert rec is None and problem


# ---------------------------------------------------------------------------
# Recording -- the record equals the params dict the step ran on
# ---------------------------------------------------------------------------


class TestRecordFromParams:
    PARAMS = {
        "method": "contour", "channel": "c", "prep_chain": "bgsub+gamma",
        "needs_channel": True, "wall_kind": "membrane",
        "solver": "random_walker", "band_um": 3.5, "n_iter": 12,
        "clamp": 0.6, "compactness": 0.1, "contour_mode": 0,
        "contour_axis": "XY", "contour_sampling": 1,
        "contour_share_pct": 30.0, "contour_n_slices": 9,
        "contour_limit_um": 4.5, "contour_iterations": 120,
        "contour_balloon": -0.2, "contour_gate": 0.35,
        "contour_blur_um": 1.5, "contour_sharpness": 150.0,
        "contour_rebuild_um": 2.0, "contour_lambda1": 1.2,
        "contour_lambda2": 0.9, "sigma_z": 2.5, "sigma_xy": 1.2,
        "preserve_topology": False,
    }

    def test_contour_and_settle_and_sdf(self):
        ticks = _ticks(bgsub=True, lift=True)
        c = rr.record_from_params("contour", self.PARAMS, prep_ticks=ticks)
        assert c == rr.RecipeStep("contour", good_params()["contour"])
        c.validate()
        s = rr.record_from_params("settle", self.PARAMS, prep_ticks=ticks)
        assert s.params["solver"] == "random_walker"
        assert s.params["prep_chain"] == "bgsub+gamma"
        s.validate()
        d = rr.record_from_params("sdf", self.PARAMS)
        assert d == rr.RecipeStep("sdf", good_params()["sdf"])

    def test_widget_methods_are_not_read_from_params(self):
        with pytest.raises(rr.RecipeError):
            rr.record_from_params("dilate_erode", self.PARAMS)


# ---------------------------------------------------------------------------
# step_needs_channel -- the pure mirror (panel parity is in the panel file)
# ---------------------------------------------------------------------------


class TestNeedsChannel:
    @pytest.mark.parametrize("kind", rr.WALL_KINDS)
    def test_settle_follows_the_wall_kind(self, kind):
        assert rr.step_needs_channel("settle", {"wall_kind": kind}) == (
            kind in rr.IMAGE_WALL_KINDS)

    @pytest.mark.parametrize("kind", rr.WALL_KINDS)
    def test_contour_brightness_always_needs_one(self, kind):
        assert rr.step_needs_channel(
            "contour", {"mode": "brightness", "wall_kind": kind})
        assert rr.step_needs_channel(
            "contour", {"mode": "snap", "wall_kind": kind}) == (
            kind in rr.IMAGE_WALL_KINDS)

    @pytest.mark.parametrize("m", ["sdf", "dilate_erode", "guided_grow",
                                   "fill_holes", "flood_fill"])
    def test_the_rest_never(self, m):
        assert not rr.step_needs_channel(m, {})


# ---------------------------------------------------------------------------
# Names and summaries
# ---------------------------------------------------------------------------


class TestNamesAndSummaries:
    def test_summary_and_elision(self):
        steps = [rr.RecipeStep(m, {}) for m in ("sdf", "dilate_erode",
                                                "settle")]
        assert rr.steps_summary(steps) == "Smooth -> Dilate/Erode -> Settle"
        long = [rr.RecipeStep(m, {}) for m in (
            "sdf", "settle", "contour", "contour", "dilate_erode", "sdf")]
        s = rr.steps_summary(long)
        assert len(s) <= 45
        assert s.startswith("Smooth -> Settle -> ")
        assert s.endswith("… (6 steps)")

    def test_step_word_is_re_exported_unchanged(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        assert Curation3DAnnotator._REFINE_STEP_WORD == rr.STEP_WORD

    @pytest.mark.parametrize("raw,want", [
        ("  tidy ", "tidy"), ("x" * 40, "x" * 40), ("x" * 41, None),
        ("", None), ("   ", None), ("a\tb", None), ("a\nb", None),
        (None, None),
    ])
    def test_normalise_name(self, raw, want):
        assert rr.normalise_name(raw) == want

    def test_find_existing_is_case_insensitive(self):
        store = {"Tidy": {}, "other": {}}
        assert rr.find_existing(store, "tidy") == "Tidy"
        assert rr.find_existing(store, " TIDY ") == "Tidy"
        assert rr.find_existing(store, "nope") is None


# ---------------------------------------------------------------------------
# The menu -- every sentence of it (invariants 10, 11)
# ---------------------------------------------------------------------------


def _store():
    gp = good_params()

    def rec(*methods, known=True):
        return rr.recipe_to_json([rr.RecipeStep(m, gp[m]) for m in methods],
                                 created="", spacing_known=known,
                                 spacing=(3.0, 1.0, 1.0) if known else ())
    contour = dict(gp["contour"])
    s = {
        "tidy": rec("sdf", "dilate_erode", "fill_holes"),
        "grow": rec("sdf", "guided_grow"),
        "walls": rec("sdf", "dilate_erode",
                     "settle"),          # probability wall
    }
    img = rr.recipe_to_json([rr.RecipeStep("sdf", gp["sdf"]),
                             rr.RecipeStep("contour", contour)],
                            created="", spacing_known=True, spacing=())
    s["img"] = img
    return s


def _texts(entries, kind=None):
    return [e[1] for e in entries if kind is None or e[0] == kind]


class TestMenu:
    AV = {"spacing_known": True, "prob": True, "guided_grow": True,
          "diffusion": True}

    def test_the_happy_menu(self):
        e = rr.menu_entries(_store(), None, staged_records=[], object_id=149,
                            pre_staged=0, availability=self.AV, channel="c",
                            busy=False)
        assert e[0] == ("header", "Run a recipe on object 149", False, None,
                        "")
        runs = [x for x in e if x[0] == "run"]
        assert ("tidy — Smooth -> Dilate/Erode -> Fill Holes"
                in _texts(runs))
        assert all(x[2] for x in runs)
        assert _texts(e, "save") == [
            "Save as a recipe… (unavailable: nothing is staged yet)"]
        assert sorted(_texts(e, "load")) == ["grow", "img", "tidy", "walls"]
        assert "separator" in [x[0] for x in e]

    def test_pre_staged_header(self):
        e = rr.menu_entries(_store(), None, staged_records=[rr.RecipeStep(
            "sdf", good_params()["sdf"])] * 2, object_id=149, pre_staged=2,
            availability=self.AV, channel="c")
        assert e[0][1] == "Add a recipe to the 2 steps staged on object 149"
        assert _texts(e, "save") == ["Save the 2 staged steps as…"]

    def test_every_grey_sentence_is_the_item_text(self):
        seen = {}

        def menu(**kw):
            args = dict(staged_records=[], object_id=149, pre_staged=0,
                        availability=self.AV, channel="c", busy=False)
            args.update(kw)
            store = args.pop("store", _store())
            problem = args.pop("problem", None)
            return rr.menu_entries(store, problem, **args)

        states = {
            "no recipes": (menu(store={}), rr.NO_RECIPES),
            "unreadable file": (menu(problem=rr.FILE_UNREADABLE),
                                rr.FILE_UNREADABLE),
            "no object": (menu(object_id=0), rr.HEADER_NO_OBJECT),
            "other object": (menu(mode="other_object", staged_object=7),
                             "Object 7 has steps staged. Accept or Discard "
                             "them before running a recipe on object 149."),
            "no channel": (menu(channel=""),
                           "img — step 2 (Active contour) needs a "
                           "channel; pick one on this page first"),
            "no prob (grow)": (menu(availability=dict(
                self.AV, prob=False, guided_grow=False)),
                "grow — step 2 (Guided Grow) needs a probability map, "
                "and this patch has none"),
            "no diffusion": (menu(store={"d": _diffusion()},
                                  availability=dict(self.AV,
                                                    diffusion=False)),
                             "d — step 1 (Settle) needs u-Segment3D, "
                             "which is not installed"),
            "spacing known here none": (
                menu(availability=dict(self.AV, spacing_known=False)),
                "tidy — saved with a voxel size; this patch has none"),
            "spacing none here known": (
                menu(store={"u": _unknown()}),
                "u — saved without a voxel size; this patch has one"),
            "fragment": (menu(mode="fragment"), rr.HEADER_FRAGMENT),
            "interp": (menu(mode="interp"), rr.HEADER_INTERP),
            "busy": (menu(busy=True), rr.HEADER_BUSY),
            "busy, a recipe": (menu(busy=rr.S1_RUNNING_BLOCK),
                               rr.S1_RUNNING_BLOCK),
            "busy, batch": (menu(busy="Finish or close Batch Refine first."),
                            "Finish or close Batch Refine first."),
            "nothing staged": (menu(), "Save as a recipe… (unavailable: "
                               "nothing is staged yet)"),
            "flood": (menu(staged_records=[
                rr.RecipeStep("sdf", {}),
                rr.RecipeStep("flood_fill", {"axis": "3D",
                                             "max_size": 500})]),
                "Save as a recipe… (unavailable: Flood Fill, step 2, "
                "works from the crosshair)"),
            "too many": (menu(staged_records=[
                rr.RecipeStep("sdf", {})] * 13),
                "Save as a recipe… (unavailable: a recipe holds at most "
                "12 steps)"),
            "bad entry": (menu(store={"old": [1, 2]}),
                          "old — cannot be read: not a Refine recipe"),
            "no workspace": (menu(mode="no_workspace"), rr.NO_WORKSPACE),
        }
        for label, (entries, sentence) in states.items():
            greyed = [x for x in entries if not x[2] and x[0] != "separator"]
            assert sentence in _texts(greyed), (label, _texts(entries))
            seen.setdefault(sentence, label)
        # no two states share a sentence
        assert len(seen) == len(states), seen

    def test_no_sentence_twice_in_one_menu(self):
        """Header and Save never repeat each other -- except the one
        spec-mandated case, the unreadable file (UX audit LOW)."""
        for kw in ({"busy": rr.S1_RUNNING_BLOCK}, {"mode": "fragment"},
                   {"mode": "interp"}, {"object_id": 0}, {},
                   {"mode": "other_object", "staged_object": 3}):
            args = dict(staged_records=[], object_id=149,
                        availability=self.AV, channel="c")
            args.update(kw)
            # the top level (Load / Delete list the same NAMES by design)
            texts = [x[1] for x in rr.menu_entries(_store(), None, **args)
                     if x[1] and x[0] not in ("load", "delete")]
            assert len(texts) == len(set(texts)), (kw, texts)

    def test_a_blocking_header_greys_every_run_item(self):
        for kw in ({"object_id": 0}, {"mode": "other_object"},
                   {"busy": True}, {"mode": "fragment"}):
            args = dict(staged_records=[], object_id=149,
                        availability=self.AV, channel="c")
            args.update(kw)
            e = rr.menu_entries(_store(), None, **args)
            assert not any(x[2] for x in e if x[0] == "run"), kw

    def test_unreadable_file_offers_no_run_and_refuses_save(self):
        e = rr.menu_entries({"tidy": _store()["tidy"]}, rr.FILE_UNREADABLE,
                            staged_records=[rr.RecipeStep(
                                "sdf", good_params()["sdf"])],
                            object_id=149, availability=self.AV, channel="c")
        assert _texts(e, "run") == []
        assert _texts(e, "save") == [rr.FILE_UNREADABLE]
        assert _texts(e, "delete") == ["tidy"]

    def test_unreadable_entries_are_under_delete_but_not_load(self):
        store = dict(_store(), old=[1, 2])
        e = rr.menu_entries(store, None, object_id=149, availability=self.AV,
                            channel="c")
        assert "old" not in _texts(e, "load")
        assert ("old — cannot be read: not a Refine recipe"
                in _texts(e, "delete"))

    def test_long_summary_is_elided_and_the_tooltip_is_full(self):
        gp = good_params()
        steps = [rr.RecipeStep(m, gp[m]) for m in (
            "sdf", "settle", "sdf", "dilate_erode", "fill_holes", "sdf")]
        store = {"long-one": rr.recipe_to_json(
            steps, created="", spacing_known=True, spacing=())}
        e = rr.menu_entries(store, None, object_id=5, availability=self.AV,
                            channel="c")
        run = [x for x in e if x[0] == "run"][0]
        assert run[1].endswith("(6 steps)")
        assert len(run[1]) <= len("long-one — ") + 45
        assert run[4] == ("Smooth -> Settle -> Smooth -> Dilate/Erode -> "
                          "Fill Holes -> Smooth")

    def test_load_is_greyed_while_busy(self):
        e = rr.menu_entries(_store(), None, object_id=5, availability=self.AV,
                            channel="c", busy=True)
        assert not any(x[2] for x in e if x[0] in ("load", "delete"))


def _diffusion():
    p = good_params()["settle"]
    p["solver"] = "label_diffusion"
    p["wall_kind"] = "distance"
    return rr.recipe_to_json([rr.RecipeStep("settle", p)], created="",
                             spacing_known=True, spacing=())


def _unknown():
    return rr.recipe_to_json([rr.RecipeStep("sdf", good_params()["sdf"])],
                             created="", spacing_known=False, spacing=())


class TestStopTooltipAndBlockers:
    def test_stop_tooltip(self):
        assert rr.stop_tooltip(0).endswith("first staged step.")
        assert "including the 2 steps that were staged" in rr.stop_tooltip(2)
        assert "the 1 step that was staged" in rr.stop_tooltip(1)

    def test_blocker_sentences(self):
        rec, _ = rr.recipe_from_json(_store()["img"])
        b = rr.step_blocker(rec, channel="", availability={})
        assert b == ("channel", 2)
        assert rr.blocker_sentence("img", rec, b) == (
            "The recipe needs a channel for step 2 (Active contour). "
            "Pick one on this page first.")
        assert "img" not in rr.blocker_sentence("img", rec, b)
        rec, _ = rr.recipe_from_json(_store()["walls"])
        assert rr.step_blocker(rec, channel="c",
                               availability={"prob": False}) == ("prob", 3)


class TestTheNameIsNotInS1:
    """DECIDED on the UX audit: no S1 sentence carries a recipe name."""

    def test_no_S1_template_has_a_name_slot(self):
        for attr in dir(rr):
            if attr.startswith("S1_") and isinstance(getattr(rr, attr), str):
                assert "{name}" not in getattr(rr, attr), attr

    def test_the_stage_line_ends_with_the_name(self):
        text = rr.STAGE_STOPPED.format(name="tidy", k=2, n=3,
                                       what="Fill Holes")
        assert text == "Stopped at 2/3 \u00b7 Fill Holes \u2014 'tidy'"

    def test_method_words_are_the_combos(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator,
        )

        assert rr.METHOD_WORD == Curation3DAnnotator._REFINE_METHOD_WORD
