"""Saved Refine chain recipes -- the model, its validation and its menu.

Pure and napari-free: no Qt, no
napari, no import from ``napari_epicycle``.  The panel and the annotator
both read this module; the store (``_batch_recipe_store``) persists what
:func:`recipe_to_json` builds.

A **recipe** is a saved, named list of Refine **steps**.  Each step is one
method and the settings it ran with, every combo stored as its ENGINE TOKEN
(never a combo index), every number checked against the widget's own range.
The name is the store key and is not part of the recipe.

What is NOT stored: the channel (a run uses the channel picked on
the page), the object, the patch, the run, the ROI.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Vocabulary.  Literal copies of the panel's module constants, because a
# package may never import ``napari_epicycle``; a parity test
# in ``tests/annotator/test_refine_recipe_panel.py`` pins each one against
# the panel's.
# ---------------------------------------------------------------------------

#: The methods a recipe may hold.  ``flood_fill`` is absent on purpose: it
#: fills from wherever the crosshair is.
RECIPE_METHODS = ("sdf", "contour", "dilate_erode", "guided_grow",
                  "fill_holes", "settle")

#: The Refine tab's wall kinds (``REFINE_WALL_KINDS`` in the panel).
WALL_KINDS = ("distance", "membrane", "hybrid", "gradient", "intensity",
              "frangi", "meijering", "skeleton", "probability")

#: Of those, the ones built from an image channel.
IMAGE_WALL_KINDS = frozenset(("membrane", "intensity", "frangi", "meijering",
                              "gradient", "hybrid"))

#: Settle's solvers (``REFINE_SOLVERS`` in the panel).
SOLVERS = ("watershed", "random_walker", "label_diffusion")

#: The contour's modes / orientations / slice samplers -- the engine's own.
CONTOUR_MODES = ("snap", "brightness")
AXES = ("3D", "XY", "XZ", "YZ")
SLICE_SAMPLINGS = ("share", "count", "every")

#: The five prepare ticks, and the tokens a Refine chain may contain.
#: ``log`` is refused: the Refine card's "Lift dim signal" always runs
#: ``gamma``, so a ``log`` recipe could not be shown on the page.
PREP_TICKS = ("clahe", "zcorrect", "bgsub", "lift", "denoise")
PREP_TOKENS = ("zcorrect", "bgsub", "gamma", "denoise", "clahe")

MAX_STEPS = 12
MAX_NAME = 40
SUMMARY_CHARS = 45
KIND = "refine_chain"
VERSION = 1

#: The word each step contributes to the S5 breadcrumb (moved here from the
#: annotator's ``_REFINE_STEP_WORD``, which re-exports it).
STEP_WORD = {
    "sdf": "Smooth", "contour": "Active contour",
    "dilate_erode": "Dilate/Erode",
    "guided_grow": "Grow", "fill_holes": "Fill holes",
    "flood_fill": "Flood fill", "settle": "Settle",
}

#: The words the method COMBO uses (the annotator's ``_REFINE_METHOD_WORD``)
#: -- every recipe sentence, summary and stage line names a method the way
#: the page does (UX audit LOW).
METHOD_WORD = {
    "sdf": "Smooth", "contour": "Active contour",
    "dilate_erode": "Dilate/Erode", "guided_grow": "Guided Grow",
    "fill_holes": "Fill Holes", "flood_fill": "Flood Fill",
    "settle": "Settle",
}

# ---------------------------------------------------------------------------
# The params of each method: name -> (kind, low, high) or (kind, tokens).
# Ranges are the widgets' own (``advanced_features_widget.py``).
# ---------------------------------------------------------------------------

_F, _I, _B, _T, _P = "float", "int", "bool", "token", "ticks"

_SPEC: Dict[str, Dict[str, tuple]] = {
    "sdf": {
        "sigma_z": (_F, 0.1, 10.0),
        "sigma_xy": (_F, 0.1, 10.0),
        "preserve_topology": (_B,),
    },
    "contour": {
        "mode": (_T, CONTOUR_MODES),
        "wall_kind": (_T, WALL_KINDS),
        "axis": (_T, AXES),
        "sampling": (_T, SLICE_SAMPLINGS),
        "share_pct": (_I, 5, 100),
        "n_slices": (_I, 3, 60),
        "limit_um": (_F, 0.5, 30.0),
        "iterations": (_I, 10, 500),
        "balloon": (_F, -1.0, 1.0),
        "gate": (_F, 0.0, 1.0),
        "blur_um": (_F, 0.5, 10.0),
        "sharpness": (_I, 1, 1000),
        "rebuild_um": (_F, 0.0, 15.0),
        "lambda1": (_F, 0.1, 10.0),
        "lambda2": (_F, 0.1, 10.0),
        "prep_ticks": (_P,),
        "prep_chain": ("chain",),
    },
    "dilate_erode": {
        "radius": (_I, -10, 10),
        "axis": (_T, AXES),
    },
    "guided_grow": {
        "max_radius": (_I, 1, 15),
        "prob_threshold": (_F, 0.0, 1.0),
        "axis": (_T, AXES),
    },
    "fill_holes": {
        "axis": (_T, AXES),
    },
    "settle": {
        "wall_kind": (_T, WALL_KINDS),
        "solver": (_T, SOLVERS),
        "band_um": (_F, 0.5, 20.0),
        "n_iter": (_I, 1, 50),
        "clamp": (_F, 0.05, 0.95),
        "compactness": (_F, 0.0, 1.0),
        "prep_ticks": (_P,),
        "prep_chain": ("chain",),
    },
    # Recorded (so Save can say why it is greyed), never validated as a
    # recipe step.
    "flood_fill": {
        "axis": (_T, AXES),
        "max_size": (_I, 100, 500000),
    },
}

#: Which params each method has, in a stable order.
PARAM_NAMES = {m: tuple(spec) for m, spec in _SPEC.items()}

# ---------------------------------------------------------------------------
# Sentences.  Every grey reason is ITEM TEXT: a greyed item's
# tooltip is not on screen, its text is.  No two states share a sentence.
# ---------------------------------------------------------------------------

FILE_UNREADABLE = "The recipes file could not be read."
NO_RECIPES = "No recipes saved yet \u2014 stage some steps, then save them here."
NO_WORKSPACE = "Recipes need a workspace \u2014 open this from Step 3."
HEADER_NO_OBJECT = ("Select an object first \u2014 a recipe runs on the "
                    "selected object.")
#: Names both objects (UX audit LOW).
HEADER_OTHER_OBJECT = ("Object {staged} has steps staged. Accept or Discard "
                       "them before running a recipe on object {obj}.")
HEADER_FRAGMENT = "Recipes are off while refining the pieces of a split."
HEADER_INTERP = "Recipes are off while refining an interpolation."
#: The fallback when no busy sentence is passed in; the annotator passes the
#: one `_preview_smooth` would give, so header and S1 say the same (M2).
HEADER_BUSY = "Busy \u2014 wait for the running step to finish."

#: The Save refusals as the S1 BACKSTOP sentence (the handler) ...
SAVE_NOTHING = "Save: nothing is staged yet \u2014 Preview a step first."
SAVE_FLOOD = ("Save: Flood Fill (step {k}) cannot go in a recipe \u2014 it "
              "works from the crosshair.")
SAVE_TOO_MANY = "Save: a recipe holds at most 12 steps."
SAVE_UNRECORDED = ("Save: step {k} was not recorded \u2014 Discard and "
                   "build the chain again.")
#: ... and as the greyed MENU ITEM, whose text is the ACTION with a short
#: reason, so it never repeats the header above it (UX audit LOW).
SAVE_ACTION = "Save as a recipe\u2026 (unavailable: {why})"
SAVE_WHY_NOTHING = "nothing is staged yet"
SAVE_WHY_FLOOD = "Flood Fill, step {k}, works from the crosshair"
SAVE_WHY_TOO_MANY = "a recipe holds at most 12 steps"
SAVE_WHY_UNRECORDED = "step {k} was not recorded"
SAVE_WHY_BUSY = "a step is running"
SAVE_WHY_FRAGMENT = "refining the pieces of a split"
SAVE_WHY_INTERP = "refining an interpolation"
LOAD_TITLE = "Load into the panel"
DELETE_TITLE = "Delete"

#: The runner's S1 sentences.  DECIDED on the
#: UX audit: the recipe's NAME is not in S1 at all -- it rides the stage
#: line, the menu, the confirms and the tooltips, so a long name can neither
#: widen the page nor push a sentence onto a third line.
S1_RUNNING = "Running a recipe ({steps}) on object {obj}."
S1_DONE = "The recipe ran all {steps} on object {obj}."
S1_STOPPED_FRESH = ("Recipe stopped at step {k} of {n}. The object is back "
                    "as it was before the recipe.")
S1_STOPPED_STAGED = ("Recipe stopped at step {k} of {n}. The object is back "
                     "as it was before the first staged step.")
S1_NEEDS_CHANNEL = ("The recipe needs a channel for step {k} ({word}). Pick "
                    "one on this page first.")
S1_NEEDS_PROB = ("The recipe's step {k} ({word}) needs a probability map, "
                 "and this patch has none.")
S1_NEEDS_DIFFUSION = ("The recipe's step {k} ({word}) needs u-Segment3D, "
                      "which is not installed.")
S1_SPACING_KNOWN = ("The recipe was saved with a voxel size; this patch has "
                    "none \u2014 set it in QC Settings, or save the recipe "
                    "again here.")
S1_SPACING_UNKNOWN = ("The recipe was saved without a voxel size; this patch "
                      "has one \u2014 save the recipe again here.")
S1_UNREADABLE = ("The recipe cannot be read \u2014 it was changed by hand or "
                 "by a newer EpiCycle. Delete it and save it again.")
S1_RUNNING_BLOCK = "A recipe is running \u2014 wait, or press Stop."
S1_LOADED = "Loaded step 1 of the recipe ({word}). Nothing has run."
S1_LOAD_FAILED = ("Could not put step 1 of the recipe ({word}) on this page. "
                  "Nothing has changed.")
S1_SAVED = "Saved the recipe ({steps})."
S1_SAVE_FAILED = ("Could not save the recipe: the Step 3 folder could not be "
                  "written to.")
S1_DELETED = "Deleted the recipe."
S1_DELETE_FAILED = ("Could not delete the recipe: the Step 3 folder could "
                    "not be written to.")
S1_BAD_NAME = "A recipe name needs 1 to 40 characters."
#: A step whose settings cannot be put on the page (M3): fixed words, the raw
#: reason goes to the log and the [RECIPE] print.
S1_NOT_SET = ("Recipe stopped at step {k} of {n}: its {word} settings do not "
              "fit this page. The steps before it are still staged.")
S1_NOT_SET_FIRST = ("Recipe stopped at step 1 of {n}: its {word} settings do "
                    "not fit this page. Nothing has changed.")
#: A worker that could not be started (lifecycle L3).
S1_NOT_STARTED = ("Recipe stopped at step {k} of {n}: its {word} step could "
                  "not be started.")



#: Confirms (wording and buttons from the UX audit, M5).
CONFIRM_APPEND = ("Add '{name}' ({steps}) after the {staged} already "
                  "staged on object {obj}? Discard would then undo all "
                  "{total}.")
CONFIRM_REPLACE = ("Replace the saved recipe '{name}' ({summary}) with these "
                   "{steps}?")
CONFIRM_DELETE = "Delete the recipe '{name}'? This cannot be undone."

#: The stage line's endings.
#: Position FIRST, name LAST, so right-elision cuts the name (UX M1).
STAGE_STOPPED = "Stopped at {k}/{n} \u00b7 {what} \u2014 '{name}'"
STAGE_EMPTY = "the object is empty"
STAGE_PREP_DIFFERS = "preparation differs"

#: The Stop tooltip while a recipe runs.
STOP_TIP = ("Stops the recipe and puts the object back as it was before the "
            "first staged step")
STOP_TIP_STAGED = (" — including the {k} step{s} that {verb} staged "
                   "before the recipe.")

#: The button's tooltip.
BUTTON_TIP = ("Save the staged steps as a recipe, or run a saved recipe on "
              "the selected object. The recipe uses the channel picked on "
              "this page.")


def plural(n: int, word: str) -> str:
    return "{} {}{}".format(int(n), word, "" if int(n) == 1 else "s")


# ---------------------------------------------------------------------------
# The step
# ---------------------------------------------------------------------------


class RecipeError(ValueError):
    """A step or a recipe that cannot be used, with a plain reason."""


def _is_number(v) -> bool:
    return (isinstance(v, (int, float)) and not isinstance(v, bool)
            and math.isfinite(float(v)))


def chain_tokens(chain: str) -> Tuple[str, ...]:
    """``"bgsub+gamma"`` -> ``("bgsub", "gamma")``; ``"none"`` -> ``()``."""
    s = str(chain or "").strip()
    if not s or s == "none":
        return ()
    return tuple(t.strip() for t in s.split("+"))


@dataclass(frozen=True)
class RecipeStep:
    """One Refine step: a method and the settings it ran with."""

    method: str
    params: dict

    def validate(self) -> None:
        """Raise :class:`RecipeError` naming the first thing wrong."""
        m = self.method
        if m == "flood_fill":
            raise RecipeError(
                "Flood Fill cannot go in a recipe — it works from the "
                "crosshair")
        if m not in RECIPE_METHODS:
            raise RecipeError("unknown method {!r}".format(m))
        if not isinstance(self.params, dict):
            raise RecipeError("its settings are not a list of names")
        spec = _SPEC[m]
        for key in self.params:
            if key not in spec:
                raise RecipeError("unknown setting {!r}".format(key))
        for key, rule in spec.items():
            if key not in self.params:
                raise RecipeError("missing setting {!r}".format(key))
            _check(key, rule, self.params[key])
        if "prep_chain" in spec:
            # The two must agree on the tokens the ticks can produce: the
            # chain is what RAN, the ticks what was SHOWN.
            ticks = self.params["prep_ticks"]
            toks = chain_tokens(self.params["prep_chain"])
            for tick, tok in (("zcorrect", "zcorrect"), ("bgsub", "bgsub"),
                              ("lift", "gamma"), ("denoise", "denoise")):
                if (tok in toks) != bool(ticks.get(tick)):
                    raise RecipeError(
                        "its preparation does not match its ticks")
            if "clahe" in toks and not ticks.get("clahe"):
                raise RecipeError("its preparation does not match its ticks")

    def to_json(self) -> dict:
        return {"method": str(self.method), "params": _plain(self.params)}


def _plain(obj):
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def _check(key, rule, value) -> None:
    kind = rule[0]
    if kind == _B:
        if not isinstance(value, bool):
            raise RecipeError("{!r} is not on/off".format(key))
    elif kind == _F:
        if not _is_number(value):
            raise RecipeError("{!r} is not a number".format(key))
        if not (rule[1] - 1e-9 <= float(value) <= rule[2] + 1e-9):
            raise RecipeError("{!r} is out of range".format(key))
    elif kind == _I:
        if not (isinstance(value, int) and not isinstance(value, bool)):
            raise RecipeError("{!r} is not a whole number".format(key))
        if not (rule[1] <= int(value) <= rule[2]):
            raise RecipeError("{!r} is out of range".format(key))
    elif kind == _T:
        if value not in rule[1]:
            raise RecipeError("{!r} has an unknown value {!r}".format(
                key, value))
    elif kind == _P:
        if not isinstance(value, dict) or set(value) != set(PREP_TICKS):
            raise RecipeError("{!r} does not name the five ticks".format(key))
        if not all(isinstance(v, bool) for v in value.values()):
            raise RecipeError("{!r} is not on/off".format(key))
    elif kind == "chain":
        if not isinstance(value, str):
            raise RecipeError("{!r} is not a preparation".format(key))
        toks = chain_tokens(value)
        if "log" in toks:
            raise RecipeError(
                "its preparation uses 'log', which this page cannot show")
        for t in toks:
            if t not in PREP_TOKENS:
                raise RecipeError("unknown preparation {!r}".format(t))


# ---------------------------------------------------------------------------
# The recipe
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Recipe:
    steps: tuple
    created: str = ""
    spacing_known: bool = False
    spacing: tuple = ()


def recipe_to_json(steps: Iterable, *, created: str, spacing_known: bool,
                   spacing) -> dict:
    """The JSON object stored under the recipe's name.

    Raises :class:`RecipeError` for a step that cannot be saved, so a bad
    recipe never reaches the file.
    """
    steps = [_as_step(s) for s in steps]
    if not steps:
        raise RecipeError("a recipe needs at least one step")
    if len(steps) > MAX_STEPS:
        raise RecipeError("a recipe holds at most {} steps".format(MAX_STEPS))
    for s in steps:
        s.validate()
    return {
        "kind": KIND,
        "version": VERSION,
        "created": str(created or ""),
        "spacing_known": bool(spacing_known),
        "spacing": [float(v) for v in (spacing or ())],
        "steps": [s.to_json() for s in steps],
    }


def _as_step(s) -> RecipeStep:
    if isinstance(s, RecipeStep):
        return s
    if isinstance(s, dict):
        return RecipeStep(str(s.get("method", "")), s.get("params"))
    raise RecipeError("a step is not a method and its settings")


def recipe_from_json(obj) -> Tuple[Optional[Recipe], Optional[str]]:
    """``(recipe, None)`` or ``(None, problem)``.  Never raises."""
    try:
        if not isinstance(obj, dict) or obj.get("kind") != KIND:
            return None, "not a Refine recipe"
        version = obj.get("version")
        if not isinstance(version, int) or isinstance(version, bool):
            return None, "not a Refine recipe"
        if version > VERSION:
            return None, "made by a newer EpiCycle"
        sk = obj.get("spacing_known")
        if not isinstance(sk, bool):
            return None, "not a Refine recipe"
        raw = obj.get("steps")
        if not isinstance(raw, list) or not raw:
            return None, "it has no steps"
        if len(raw) > MAX_STEPS:
            return None, "it has more than {} steps".format(MAX_STEPS)
        steps = []
        for k, item in enumerate(raw, start=1):
            if not isinstance(item, dict):
                return None, "step {} is not a method and its settings".format(k)
            step = RecipeStep(str(item.get("method", "")),
                              item.get("params"))
            try:
                step.validate()
            except RecipeError as exc:
                return None, "step {}: {}".format(k, exc)
            steps.append(RecipeStep(step.method, _plain(step.params)))
        spacing = obj.get("spacing") or []
        try:
            spacing = tuple(float(v) for v in spacing)
        except Exception:
            spacing = ()
        return Recipe(steps=tuple(steps), created=str(obj.get("created", "")),
                      spacing_known=sk, spacing=spacing), None
    except Exception as exc:  # noqa: BLE001 - never raise into the UI
        return None, "not a Refine recipe ({})".format(type(exc).__name__)


# ---------------------------------------------------------------------------
# Recording
# ---------------------------------------------------------------------------


def _ticks(prep_ticks) -> dict:
    t = dict(prep_ticks or {})
    return {k: bool(t.get(k, False)) for k in PREP_TICKS}


def record_from_params(method: str, params: dict, *,
                       prep_ticks=None) -> RecipeStep:
    """The record of a ``sdf`` / ``contour`` / ``settle`` step, built from
    the ``get_refine_params()`` dict the step actually RAN on.

    ``prep_ticks`` are the five tick states read with it (the params dict
    carries only the chain).  The combos are read back as engine tokens.
    """
    p = dict(params or {})
    if method == "sdf":
        return RecipeStep("sdf", {
            "sigma_z": float(p.get("sigma_z", 3.0)),
            "sigma_xy": float(p.get("sigma_xy", 1.5)),
            "preserve_topology": bool(p.get("preserve_topology", True)),
        })
    if method == "contour":
        mode_idx = int(p.get("contour_mode", 0))
        samp_idx = int(p.get("contour_sampling", 0))
        return RecipeStep("contour", {
            "mode": CONTOUR_MODES[min(max(mode_idx, 0), 1)],
            "wall_kind": str(p.get("wall_kind", "membrane")),
            "axis": str(p.get("contour_axis", "3D")),
            "sampling": SLICE_SAMPLINGS[min(max(samp_idx, 0), 2)],
            "share_pct": int(round(float(p.get("contour_share_pct", 25)))),
            "n_slices": int(p.get("contour_n_slices", 12)),
            "limit_um": float(p.get("contour_limit_um", 3.0)),
            "iterations": int(p.get("contour_iterations", 100)),
            "balloon": float(p.get("contour_balloon", 0.0)),
            "gate": float(p.get("contour_gate", 0.30)),
            "blur_um": float(p.get("contour_blur_um", 2.0)),
            "sharpness": int(round(float(p.get("contour_sharpness", 100)))),
            "rebuild_um": float(p.get("contour_rebuild_um", 1.5)),
            "lambda1": float(p.get("contour_lambda1", 1.0)),
            "lambda2": float(p.get("contour_lambda2", 1.0)),
            "prep_ticks": _ticks(prep_ticks),
            "prep_chain": str(p.get("prep_chain", "none")),
        })
    if method == "settle":
        return RecipeStep("settle", {
            "wall_kind": str(p.get("wall_kind", "membrane")),
            "solver": str(p.get("solver", "watershed")),
            "band_um": float(p.get("band_um", 2.0)),
            "n_iter": int(p.get("n_iter", 10)),
            "clamp": float(p.get("clamp", 0.5)),
            "compactness": float(p.get("compactness", 0.0)),
            "prep_ticks": _ticks(prep_ticks),
            "prep_chain": str(p.get("prep_chain", "none")),
        })
    raise RecipeError("{!r} is read off its widgets, not the params".format(
        method))


def step_needs_channel(method: str, params: dict) -> bool:
    """A pure mirror of the panel's ``refine_needs_channel()``."""
    p = dict(params or {})
    if method == "contour":
        if p.get("mode") == "brightness":
            return True
        return p.get("wall_kind") in IMAGE_WALL_KINDS
    if method == "settle":
        return p.get("wall_kind") in IMAGE_WALL_KINDS
    return False


def step_needs_prob(method: str, params: dict) -> bool:
    """Does this step read the reconstruction's probability map?"""
    p = dict(params or {})
    if method == "guided_grow":
        return True
    if method == "settle":
        return p.get("wall_kind") == "probability"
    if method == "contour":
        return p.get("mode") == "snap" and p.get("wall_kind") == "probability"
    return False


def step_needs_diffusion(method: str, params: dict) -> bool:
    return method == "settle" and dict(params or {}).get(
        "solver") == "label_diffusion"


# ---------------------------------------------------------------------------
# Names and summaries
# ---------------------------------------------------------------------------


def steps_summary(steps: Sequence, max_chars: int = SUMMARY_CHARS) -> str:
    """``"Smooth -> Dilate/Erode -> Settle"``, elided past ``max_chars`` as
    ``"Smooth -> Settle -> … (6 steps)"``."""
    words = [METHOD_WORD.get(_method_of(s), _method_of(s)) for s in steps]
    full = " -> ".join(words)
    if len(full) <= max_chars:
        return full
    tail = "… ({})".format(plural(len(words), "step"))
    kept: List[str] = []
    for w in words:
        trial = " -> ".join(kept + [w, tail])
        if len(trial) > max_chars:
            break
        kept.append(w)
    return " -> ".join(kept + [tail])


def _method_of(s) -> str:
    if isinstance(s, RecipeStep):
        return s.method
    if isinstance(s, dict):
        return str(s.get("method", ""))
    return str(s)


def normalise_name(s) -> Optional[str]:
    """The stripped name, or ``None`` when it cannot be a recipe name:
    empty, a control character, or longer than :data:`MAX_NAME`."""
    if not isinstance(s, str):
        return None
    name = s.strip()
    if not name or len(name) > MAX_NAME:
        return None
    if any(ord(c) < 32 or ord(c) == 127 for c in name):
        return None
    return name


def find_existing(store: dict, name: str) -> Optional[str]:
    """The key in ``store`` that matches ``name`` case-insensitively."""
    want = str(name or "").strip().casefold()
    for key in (store or {}):
        if str(key).casefold() == want:
            return key
    return None


def summary_of(value) -> str:
    """The summary of a stored value, or ``""`` when it cannot be read."""
    rec, _problem = recipe_from_json(value)
    if rec is None:
        return ""
    return steps_summary(rec.steps)


# ---------------------------------------------------------------------------
# Availability -- one question per step, shared by pre-flight and the menu
# ---------------------------------------------------------------------------


def step_blocker(recipe: Recipe, *, channel: str,
                 availability: dict) -> Optional[Tuple[str, int]]:
    """The first reason this recipe could not run on this page, as
    ``(reason_key, step_number)``, or ``None``.

    ``availability`` keys: ``spacing_known`` (bool), ``prob`` (the patch has
    a probability map), ``guided_grow`` (the item is enabled), ``diffusion``
    (the solver item is enabled).  Missing keys mean "available".
    """
    av = dict(availability or {})
    if "spacing_known" in av and bool(av["spacing_known"]) != bool(
            recipe.spacing_known):
        return ("spacing_known" if recipe.spacing_known
                else "spacing_unknown", 0)
    for k, step in enumerate(recipe.steps, start=1):
        m, p = step.method, step.params
        if step_needs_channel(m, p) and not channel:
            return ("channel", k)
        if m == "guided_grow" and not av.get("guided_grow", True):
            return ("prob", k)
        if step_needs_prob(m, p) and not av.get("prob", True):
            return ("prob", k)
        if step_needs_diffusion(m, p) and not av.get("diffusion", True):
            return ("diffusion", k)
    return None


# ---------------------------------------------------------------------------
# Batch Refine: the same recipe over many objects
# ---------------------------------------------------------------------------

#: A 3D Active contour costs about this many seconds per object -- the
#: batch driver measured 21.7 s for ONE ordinary object (12,066 voxels) on
#: the owner's patch (10-13 s on a smaller one).  The
#: sentence below is FORMATTED from it, never typed into prose.
CONTOUR_3D_SECONDS = (15, 25)
#: At most this many objects in one batch whose recipe has a 3D contour.
BATCH_CONTOUR_3D_MAX = 20

#: The batch's refusals -- on the dialog's reason line.  They say
#: "on the Refine tab", never the tab's own "here".
BATCH_NEEDS_CHANNEL = ("Step {k} ({word}) reads an image channel — pick "
                       "one under Channel.")
BATCH_NEEDS_PROB = ("Step {k} ({word}) needs a probability map, and this "
                    "patch has none.")
BATCH_SPACING_KNOWN = ("The recipe was saved with a voxel size; this patch "
                       "has none — set it in QC Settings, or save the "
                       "recipe again on the Refine tab.")
BATCH_SPACING_UNKNOWN = ("The recipe was saved without a voxel size; this "
                         "patch has one — save the recipe again on the "
                         "Refine tab.")
#: The Refine page's own words for the three solvers (UX audit M3): the
#: engine's "label diffusion" / "watershed" never reach the screen.
BATCH_NO_DIFFUSION = ("Step {k} (Settle) uses Diffused edge, which needs too "
                      "much memory to run on many objects at once. Save a "
                      "copy of the recipe with Sharp edge or Soft edge on the "
                      "Refine tab.")
BATCH_CONTOUR_3D = ("Step {k} (Active contour) works in 3D, about {lo}–"
                    "{hi} s per object — at most {cap} objects in one "
                    "batch. Pick fewer, or save the recipe with Orientation "
                    "XY.")
BATCH_PROB_SHAPE = ("The probability map does not match this patch's size "
                    "(it may be from before a downsample), so steps that "
                    "read it cannot run here.")
BATCH_CHANNEL_SHAPE = ("The channel does not match this patch's size, so "
                       "steps that read it cannot run here. Pick another "
                       "channel.")
BATCH_TOO_LARGE = ("The largest object here would need about {gb} GB ({w} "
                   "at a time) — more than a batch may use ({budget} GB). "
                   "Pick smaller objects, or save the recipe with a smaller "
                   "Edge may move.")


class BatchBlock(tuple):
    """``(key, step)`` -- the first reason a batch cannot run -- with the
    on-screen sentence riding on ``.text``.  A tuple, so ``b[:2]`` and an
    unpack read exactly what ``step_blocker`` returns."""

    def __new__(cls, key, step, text):
        obj = tuple.__new__(cls, (str(key), int(step)))
        obj.text = str(text)
        return obj

    @property
    def key(self):
        return self[0]

    @property
    def step(self):
        return self[1]


def _gb(n_bytes) -> str:
    gb = float(n_bytes) / float(1024 ** 3)
    return "{:.1f}".format(gb) if gb < 10 else "{:,}".format(int(round(gb)))


def batch_blocker(recipe: Recipe, *, n_objects: int, channel: str,
                  availability: dict, bboxes, volume_shape, spacing,
                  prob_shape=None, labels_shape=None, channel_shape=None,
                  workers=None) -> Optional[BatchBlock]:
    """The first reason ``recipe`` cannot run as a batch, or ``None``.

    Evaluated twice by the dialog: for **Run** with ``n_objects``
    = the target count and ``bboxes`` = every target's box, and for
    **Preview** with ``n_objects = 1`` and the preview object's box only.
    ``step_blocker`` runs first (``diffusion`` is always "available" there;
    the batch refuses it next with its own sentence), then the diffusion
    solver, the 3D contour cap, the probability map's and the channel's
    size, and the memory budget.
    """
    from . import batch_chain as bc

    av = dict(availability or {})
    av["diffusion"] = True
    av.setdefault("guided_grow", av.get("prob", True))
    first = step_blocker(recipe, channel=channel, availability=av)
    if first is not None:
        key, k = first
        word = METHOD_WORD.get(recipe.steps[k - 1].method, "") if k else ""
        text = {
            "channel": BATCH_NEEDS_CHANNEL.format(k=k, word=word),
            "prob": BATCH_NEEDS_PROB.format(k=k, word=word),
            "spacing_known": BATCH_SPACING_KNOWN,
            "spacing_unknown": BATCH_SPACING_UNKNOWN,
        }.get(key, BATCH_NEEDS_CHANNEL.format(k=k, word=word))
        return BatchBlock(key, k, text)
    for k, step in enumerate(recipe.steps, start=1):
        if step_needs_diffusion(step.method, step.params):
            return BatchBlock("batch_diffusion", k,
                              BATCH_NO_DIFFUSION.format(k=k))
    for k, step in enumerate(recipe.steps, start=1):
        if (step.method == "contour" and step.params.get("axis") == "3D"
                and int(n_objects) > BATCH_CONTOUR_3D_MAX):
            lo, hi = CONTOUR_3D_SECONDS
            return BatchBlock("batch_contour_3d", k, BATCH_CONTOUR_3D.format(
                k=k, lo=lo, hi=hi, cap=BATCH_CONTOUR_3D_MAX))
    shape = tuple(int(n) for n in (labels_shape or volume_shape or ()))
    for k, step in enumerate(recipe.steps, start=1):
        if (step_needs_prob(step.method, step.params)
                and prob_shape is not None
                and tuple(int(n) for n in prob_shape) != shape):
            return BatchBlock("batch_prob_shape", k, BATCH_PROB_SHAPE)
    for k, step in enumerate(recipe.steps, start=1):
        if (step_needs_channel(step.method, step.params)
                and channel_shape is not None
                and tuple(int(n) for n in channel_shape) != shape):
            return BatchBlock("batch_channel_shape", k, BATCH_CHANNEL_SHAPE)
    boxes = [b for b in (bboxes or ()) if b is not None]
    if boxes and volume_shape is not None:
        om = bc.outer_margin(recipe.steps, spacing)
        vshape = tuple(int(n) for n in volume_shape)
        largest = 0
        for b in boxes:
            n = 1
            for s, size in zip(b, vshape):
                lo = max(0, int(s.start) - om)
                hi = min(size, int(s.stop) + om)
                n *= max(0, hi - lo)
            largest = max(largest, n)
        w = bc.chain_worker_count(workers)
        at_once = max(1, min(int(w), int(n_objects) or 1))
        need = bc.estimate_chain_bytes(largest) * at_once
        if need > bc.BATCH_CHAIN_RAM_BUDGET:
            return BatchBlock("batch_too_large", 0, BATCH_TOO_LARGE.format(
                gb=_gb(need), w=at_once,
                budget=_gb(bc.BATCH_CHAIN_RAM_BUDGET)))
    return None


def blocker_sentence(name: str, recipe: Recipe, blocker) -> str:
    """The S1 sentence for a pre-flight refusal.  ``name`` is
    accepted for symmetry and not used: the name is not in S1 (UX M1)."""
    key, k = blocker
    word = ""
    if k:
        word = METHOD_WORD.get(recipe.steps[k - 1].method, "")
    if key == "channel":
        return S1_NEEDS_CHANNEL.format(k=k, word=word)
    if key == "prob":
        return S1_NEEDS_PROB.format(k=k, word=word)
    if key == "diffusion":
        return S1_NEEDS_DIFFUSION.format(k=k, word=word)
    if key == "spacing_known":
        return S1_SPACING_KNOWN
    return S1_SPACING_UNKNOWN


def blocker_item_text(name: str, recipe: Recipe, blocker) -> str:
    """The same reason as the greyed MENU ITEM's text."""
    key, k = blocker
    word = METHOD_WORD.get(recipe.steps[k - 1].method, "") if k else ""
    if key == "channel":
        return ("{} \u2014 step {} ({}) needs a channel; pick one on this "
                "page first".format(name, k, word))
    if key == "prob":
        return ("{} \u2014 step {} ({}) needs a probability map, and this "
                "patch has none".format(name, k, word))
    if key == "diffusion":
        return ("{} \u2014 step {} ({}) needs u-Segment3D, which is not "
                "installed".format(name, k, word))
    if key == "spacing_known":
        return ("{} \u2014 saved with a voxel size; this patch has none"
                .format(name))
    return "{} \u2014 saved without a voxel size; this patch has one".format(
        name)


# ---------------------------------------------------------------------------
# The menu
# ---------------------------------------------------------------------------

#: ``(kind, text, enabled, name, tooltip)``.  ``kind`` is one of
#: ``header`` / ``run`` / ``info`` / ``separator`` / ``save`` / ``load`` /
#: ``delete``; ``name`` is the store key for run / load / delete.
MenuEntry = Tuple[str, str, bool, Optional[str], str]


def save_blocker(staged_records: Sequence) -> Optional[str]:
    """Why Save is refused -- the S1 backstop sentence -- or ``None``."""
    why = save_why(staged_records)
    if why is None:
        return None
    key, k = why
    return {"nothing": SAVE_NOTHING, "flood": SAVE_FLOOD.format(k=k),
            "too_many": SAVE_TOO_MANY,
            "unrecorded": SAVE_UNRECORDED.format(k=k)}[key]


def save_why(staged_records: Sequence):
    """``(key, step)`` for why Save is greyed, or ``None``."""
    recs = list(staged_records or [])
    if not recs:
        return ("nothing", 0)
    for k, r in enumerate(recs, start=1):
        if r is None:
            return ("unrecorded", k)
        if _method_of(r) == "flood_fill":
            return ("flood", k)
    if len(recs) > MAX_STEPS:
        return ("too_many", 0)
    return None


def save_item_text(staged_records: Sequence) -> Optional[str]:
    """The greyed Save item's text (the ACTION + a short reason), or None."""
    why = save_why(staged_records)
    if why is None:
        return None
    key, k = why
    short = {"nothing": SAVE_WHY_NOTHING,
             "flood": SAVE_WHY_FLOOD.format(k=k),
             "too_many": SAVE_WHY_TOO_MANY,
             "unrecorded": SAVE_WHY_UNRECORDED.format(k=k)}[key]
    return SAVE_ACTION.format(why=short)


def save_text(n: int) -> str:
    return ("Save the staged step as\u2026" if int(n) == 1
            else "Save the {} staged steps as\u2026".format(int(n)))


def menu_entries(store: Optional[dict], problem: Optional[str], *,
                 staged_records: Sequence = (), object_id: int = 0,
                 pre_staged: int = 0, availability: Optional[dict] = None,
                 channel: str = "", busy=False, mode: str = "ok",
                 staged_object: int = 0) -> List[MenuEntry]:
    """Everything the Recipes menu shows, as plain tuples.

    ``mode``: ``"ok"``, ``"no_workspace"``, ``"fragment"``, ``"interp"`` or
    ``"other_object"`` (a chain is staged on ``staged_object``).  ``busy``
    is False, True, or the sentence S1 would give (UX M2) -- the header
    then says exactly that.
    """
    if mode == "no_workspace" or store is None:
        return [("info", NO_WORKSPACE, False, None, "")]

    out: List[MenuEntry] = []
    block = None
    if mode == "fragment":
        block = HEADER_FRAGMENT
    elif mode == "interp":
        block = HEADER_INTERP
    elif busy:
        block = busy if isinstance(busy, str) else HEADER_BUSY
    elif not object_id or int(object_id) <= 0:
        block = HEADER_NO_OBJECT
    elif mode == "other_object":
        block = HEADER_OTHER_OBJECT.format(staged=int(staged_object or 0),
                                           obj=int(object_id))

    parsed = {}
    for name in (store or {}):
        parsed[name] = recipe_from_json(store[name])

    if problem:
        out.append(("info", FILE_UNREADABLE, False, None, ""))
    else:
        if block is not None:
            out.append(("header", block, False, None, ""))
        elif int(pre_staged or 0) > 0:
            out.append(("header", "Add a recipe to the {} staged on object {}"
                        .format(plural(pre_staged, "step"), int(object_id)),
                        False, None, ""))
        else:
            out.append(("header", "Run a recipe on object {}".format(
                int(object_id)), False, None, ""))
        if not parsed:
            out.append(("info", NO_RECIPES, False, None, ""))
        for name, (rec, prob) in parsed.items():
            if rec is None:
                out.append(("run", "{} \u2014 cannot be read: {}".format(
                    name, prob), False, name, ""))
                continue
            summary = steps_summary(rec.steps)
            full = " -> ".join(METHOD_WORD.get(s.method, s.method)
                               for s in rec.steps)
            b = step_blocker(rec, channel=channel,
                             availability=availability or {})
            if b is not None:
                out.append(("run", blocker_item_text(name, rec, b), False,
                            name, full))
                continue
            out.append(("run", "{} \u2014 {}".format(name, summary),
                        block is None, name, full))

    out.append(("separator", "", False, None, ""))

    # Save -- its greyed text is the ACTION with a short reason, so it never
    # repeats the header; the unreadable file is the one deliberate
    # exception.
    recs = list(staged_records or [])
    if problem:
        out.append(("save", FILE_UNREADABLE, False, None, ""))
    elif busy:
        out.append(("save", SAVE_ACTION.format(why=SAVE_WHY_BUSY), False,
                    None, ""))
    elif mode in ("fragment", "interp"):
        out.append(("save", SAVE_ACTION.format(
            why=SAVE_WHY_FRAGMENT if mode == "fragment"
            else SAVE_WHY_INTERP), False, None, ""))
    else:
        text = save_item_text(recs)
        if text is None:
            out.append(("save", save_text(len(recs)), True, None, ""))
        else:
            out.append(("save", text, False, None, ""))

    # Load / Delete -- only entries that were read
    for name, (rec, prob) in parsed.items():
        if rec is None:
            continue
        out.append(("load", name, not busy, name,
                    " -> ".join(METHOD_WORD.get(s.method, s.method)
                                for s in rec.steps)))
    for name, (rec, prob) in parsed.items():
        text = name if rec is not None else "{} \u2014 cannot be read: {}" \
            .format(name, prob)
        out.append(("delete", text, not busy, name, ""))
    return out


def stop_tooltip(pre_staged: int) -> str:
    k = int(pre_staged or 0)
    if k <= 0:
        return STOP_TIP + "."
    return STOP_TIP + STOP_TIP_STAGED.format(
        k=k, s="" if k == 1 else "s", verb="was" if k == 1 else "were")


def elide(text: str, max_chars: int) -> str:
    t = str(text or "")
    if len(t) <= max_chars:
        return t
    return t[:max(0, max_chars - 1)] + "…"


__all__ = [
    "RECIPE_METHODS", "WALL_KINDS", "IMAGE_WALL_KINDS", "SOLVERS",
    "CONTOUR_MODES", "AXES", "SLICE_SAMPLINGS", "PREP_TICKS", "PREP_TOKENS",
    "MAX_STEPS", "MAX_NAME", "STEP_WORD", "PARAM_NAMES", "RecipeError",
    "RecipeStep", "Recipe", "recipe_to_json", "recipe_from_json",
    "record_from_params", "step_needs_channel", "step_needs_prob",
    "step_needs_diffusion", "steps_summary", "normalise_name",
    "find_existing", "summary_of", "step_blocker", "blocker_sentence",
    "blocker_item_text", "save_blocker", "save_text", "menu_entries",
    "stop_tooltip", "chain_tokens", "plural", "elide", "METHOD_WORD",
    "save_why", "save_item_text", "CONTOUR_3D_SECONDS",
    "BATCH_CONTOUR_3D_MAX", "BatchBlock", "batch_blocker",
]
