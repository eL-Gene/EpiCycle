"""``ReshapeSpec`` / ``ReshapeResult`` -- the validated, serialisable contract.

Pure dataclasses: **no Qt, no napari**, stdlib only.

The four slots of the reshape model are enums, and every one of them
is validated in :meth:`ReshapeSpec.__post_init__`.  **There is no
"unrecognised value -> fall back to something sensible" path anywhere in this
package.**  The membrane sandbox's ``seeds_auto`` fallthrough is exactly how a
typo turned into a silently different algorithm; an unknown value here is a
``ValueError`` naming the field and the valid options.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from typing import Any, Mapping, Sequence

# --------------------------------------------------------------------------- #
# The allowed values of each slot
# --------------------------------------------------------------------------- #

#: the region we are allowed to write into.
TARGETS = ("label", "region", "layer_label")

#: the markers - one per object we expect to come out.
SEEDS = (
    "paint",       # caller supplies a seed array (hand-painted ids)
    "peaks",       # peak_local_max on a filled signal   (seeds_from_nuclei)
    "hmax",        # h-maxima on a filled signal         (seeds_hmaxima)
    "hmin",        # h-minima basins of a rim signal     (seeds_membrane_minima)
    "blobs_log",   # Laplacian-of-Gaussian blobs
    "blobs_dog",   # Difference-of-Gaussians blobs
    "threshcc",    # threshold + connected-component centroids
    "labels",      # a trusted second label layer: one instance = one seed
    "distance",    # h-maxima of the interior distance transform (geometry-blind)
    # Add rebuild, APPENDED LAST -- item order is a wire format.
    # The Add tab's bright-object pieces, sliver-filtered, as markers for
    # the region target ONLY (refused on `label` in __post_init__).
    "nuclei_masks",
)

#: the elevation the solver floods (high = wall).
WALLS = (
    "membrane",    # CLAHE + gaussian + Sato ridge
    "intensity",   # raw CLAHE'd intensity, no ridge
    "frangi",      # vesselness ridge
    "meijering",   # neuriteness ridge
    "gradient",    # gaussian gradient magnitude
    "distance",    # -EDT of the target footprint (shape only)
    "skeleton",    # distance from the medial axis (shape only)
    "hybrid",      # 0.6 * membrane + 0.4 * (-EDT)
    "none",        # flat elevation -> a pure seed-midline (Voronoi) partition
    # The Refine tab's tenth kind (REFINE).  It is in NEITHER
    # frozenset below, deliberately: it is not IMAGE_DERIVED (it reads the
    # reconstruction's probability map, not a channel, so the arbiter must
    # not weigh it and `IMAGE_DERIVED_WALL_INDICES` stays equal to the
    # panel's nine-item Reshape list) and it is not GEOMETRY_ONLY (it is
    # not built from the footprint).  Out of both, `engine.py`'s
    # "needs a channel" guard refuses it from the engine's own path, which
    # is correct: `probability` is only ever reachable by SUPPLYING the
    # elevation, which is exactly how Refine calls it.
    "probability",
)

#: how the cut is placed.
SOLVERS = ("watershed", "random_walker", "label_diffusion")

#: post-process stages, all optional.
REFINES = ("guided", "merge_small", "merge_to_count")

#: The wall kinds whose elevation is *evidence of a real boundary in the image*.
#: Only these may enable the arbiter: arbitrating a shape-derived
#: cut against its own shape is circular.
IMAGE_DERIVED_WALLS = frozenset(
    {"membrane", "intensity", "frangi", "meijering", "gradient", "hybrid"}
)

#: Wall kinds that need no image channel at all (pure geometry).
GEOMETRY_ONLY_WALLS = frozenset({"distance", "skeleton", "none"})

#: Seed kinds that read an image channel (``params["seed_channel"]``).
IMAGE_DERIVED_SEEDS = frozenset(
    {"peaks", "hmax", "hmin", "blobs_log", "blobs_dog", "threshcc"}
)

#: The wall kinds the Add region target accepts (Add rebuild):
#: every image-derived wall except ``hybrid``, whose shape term is the
#: target's footprint -- which a box has not got.
ADD_WALL_KINDS = IMAGE_DERIVED_WALLS - {"hybrid"}

#: Voxels in an Add box above which the region target refuses (the owner's
#: 4 M limit; the Cells preset is the only caller).
ADD_BOX_BUDGET = 4_000_000

#: What the region target's hidden outside marker is made of.
OUTSIDE_VALUES = ("faces+dark", "faces")

#: How the region target grows new cells from their markers: the seeded
#: flood over the box (the default, as before), the competition with a
#: growth limit (``reshape/compete.py``), or one seeded watershed over the
#: growth space with a background marker far from every marker
#: (``"fill_with_background"``, what fills a whole patch on Cells).
#: Item order is a wire format: new values are appended.
GROWTH_VALUES = ("flood", "compete", "fill_with_background")

#: Where new cells may grow: no outline (as before), a shrink-wrap around
#: the stain (``outline.tissue_outline``) or the outline of the objects the
#: user cleared.  The MODE lives here; the outline itself (a volume) is a
#: run argument, ``outline=`` of the region target's entry points, so a
#: spec stays serialisable and comparable.
OUTLINE_VALUES = ("none", "shrink", "cleared")

#: The marker sources the Add region target reads (Add rebuild):
#: another run's objects, the bright-object pieces, the user's clicks.
#: ``reshape/region.py`` imports THIS tuple; there is no second copy.
REGION_SEEDS = ("labels", "nuclei_masks", "paint")

#: ``random_walker``'s multigrid mode needs ``pyamg``, which is NOT installed in
#: ``epicycle-env-py312``.  Selecting it would raise at solve time, deep inside
#: the worker, so the engine pins the mode and never exposes it.
RANDOM_WALKER_MODE = "cg_j"
FORBIDDEN_RANDOM_WALKER_MODES = frozenset({"cg_mg"})

#: Pre-flight size guard.  ``region_voxels x n_seeds`` is the
#: product that drives ``diffuse_labels3D``'s dense ``(n_voxels, n_labels)``
#: float32 allocation; 2**27 products is 512 MiB of float32.  A cancel button
#: that cannot cancel is not one, so a job over budget is refused up front.
DEFAULT_SIZE_BUDGET = 2 ** 27

#: Default preprocessing knobs.  Every key here is a legal ``params`` key; an
#: unknown key is a ``ValueError`` (a knob that reaches nothing must not exist).
DEFAULT_PARAMS: dict = {
    # channel selection - names into the caller's ``images`` mapping
    "seed_channel": None,
    "wall_channel": None,
    # seed preprocessing chain (apply_preproc token set)
    "seed_preproc": "none",
    "seed_clip": 0.02,
    "seed_gamma": 0.5,
    "seed_bg": 15,
    "seed_denoise": 2,
    # wall preprocessing chain
    "wall_preproc": "none",
    "wall_gamma": 0.5,
    "wall_bg": 15,
    "wall_denoise": 2,
    # wall construction
    "clip": 0.01,
    "smooth": 1.0,
    "ridge": True,
    # seed detection
    "h": 0.15,
    "seed_thresh": None,
    "seed_smooth": 2.0,
    "min_distance_um": 5.0,
    "max_sigma_um": 6.0,
    # solver
    "rw_beta": 130.0,
    "compactness": 0.0,
    # refine
    "refine_radius": 4,
    "refine_eps": 0.01,
    "min_size": 0,
    "target_count": None,
    # arbiter
    "arbiter_percentile": 75.0,
    "arbiter_low_evidence_percentile": 90.0,
    "arbiter_band_um": 1.0,
    # pre-flight guard
    "size_budget": DEFAULT_SIZE_BUDGET,
    # the Add region target (Add rebuild); the label path never
    # reads these six
    "outside": "faces+dark",
    "dark_otsu_factor": 0.5,
    "dark_clearance_um": 2.0,
    "core_erode_um": 1.0,
    "click_radius_um": 2.0,
    "box_budget": ADD_BOX_BUDGET,
    # the region target's growth rule and tissue outline (Derive inside
    # Add); the label path never reads these.  ``wall_cost`` /
    # ``growth_limit`` are read by the competition only, and
    # ``include_membrane`` by the competition and the fill with a
    # background; ``outline`` is the outline's MODE (the outline volume
    # itself is the run argument ``outline=``, never a param);
    # ``whole_patch`` lifts the box budget (the competition, or the fill
    # with a background under its own ceiling)
    "growth": "flood",
    "wall_cost": 40.0,
    "growth_limit": 28.0,
    "include_membrane": True,
    "outline": "none",
    "bridge_um": 10.0,
    "whole_patch": False,
    # the flood's outside marker holds the six box faces (a box is a cut
    # through the tissue); False on a whole patch filled the way a box is
    # (its faces are the edge of the data, not a cut: cells may run to
    # them, and the dark at a face is not eroded back from it)
    "faces_outside": True,
}

#: The inclusive ranges of the region target's numeric knobs.
_REGION_RANGES = {
    "dark_otsu_factor": (0.05, 1.0),
    "dark_clearance_um": (0.0, 10.0),
    "core_erode_um": (0.0, 5.0),
    "click_radius_um": (0.5, 5.0),
    "wall_cost": (0.0, 100.0),
    "growth_limit": (2.0, 40.0),
    "bridge_um": (0.0, 30.0),
}

_VALID_PARAM_KEYS = frozenset(DEFAULT_PARAMS)

_SPEC_FIELDS = frozenset(
    {"target", "seeds", "walls", "solver", "refine", "params", "arbiter"}
)


def _check_growth_params(merged: dict) -> None:
    """The growth rule, the outline mode and their two ticks, in plain
    words.

    A knob the engine would silently ignore is refused: the whole patch
    with the flood (the flood over a whole patch is not built; a whole
    patch grows by the competition or by the fill with a background, and
    the fill is legal on a box too).  The outline
    VOLUME is not a param; whether it is there when the mode needs it is
    checked by the entry points that take it (``outline=``)."""
    if merged["growth"] not in GROWTH_VALUES:
        raise _fail("params['growth']", merged["growth"], GROWTH_VALUES)
    if merged["outline"] not in OUTLINE_VALUES:
        raise _fail("params['outline']", merged["outline"], OUTLINE_VALUES)
    for key in ("include_membrane", "whole_patch", "faces_outside"):
        if not _is_bool(merged[key]):
            raise ValueError(
                "ReshapeSpec.params[{!r}]={!r} must be True or False.".format(
                    key, merged[key]))
        merged[key] = bool(merged[key])
    if merged["whole_patch"] and merged["growth"] not in _WHOLE_PATCH_GROWTH:
        raise ValueError(
            "ReshapeSpec.params['whole_patch']=True needs growth='compete' "
            "or growth='fill_with_background': the flood over the box "
            "('flood') never runs on a whole patch.")


#: The growth rules a whole patch may use.
_WHOLE_PATCH_GROWTH = ("compete", "fill_with_background")


def _is_bool(value: Any) -> bool:
    """True / False, or a numpy boolean scalar (no numpy import here)."""
    if value is True or value is False:
        return True
    dtype = getattr(value, "dtype", None)
    return (dtype is not None and getattr(dtype, "kind", "") == "b"
            and getattr(value, "ndim", 1) == 0)


def _fail(field_name: str, value: Any, allowed: Sequence[str]) -> ValueError:
    return ValueError(
        "ReshapeSpec.{}={!r} is not valid. Allowed: {}".format(
            field_name, value, ", ".join(allowed)
        )
    )


@dataclass(frozen=True)
class ReshapeSpec:
    """The four slots plus refine -- fully serialisable, always validated.

    ``params`` is normalised on construction: unknown keys raise and missing
    keys are filled from :data:`DEFAULT_PARAMS`.  **The wall preprocessing chain
    is left exactly as the caller wrote it.**

    An earlier draft forced ``zdepth_correct`` into that chain whenever the
    arbiter was on.  That rule was **dropped** (owner decision, 2026-09-08):
    ``zdepth_correct`` matches every z-slice's mean to the brightest slice's,
    and a membrane occupying a single z-plane IS the brightest slice -- so the
    mandate absorbed exactly the evidence the arbiter exists to weigh.  It is an
    optional preprocessing step again, as in the original recipe; the engine
    emits a one-line caution when a caller combines it with the arbiter.

    ``arbiter`` defaults to **False**: a cut the user asked for is not
    second-guessed unless they opt in.
    """

    target: str
    seeds: str
    walls: str
    solver: str = "watershed"
    refine: tuple = ()
    params: Mapping[str, Any] = field(default_factory=dict)
    arbiter: bool = False

    def __post_init__(self) -> None:
        if self.target not in TARGETS:
            raise _fail("target", self.target, TARGETS)
        if self.seeds not in SEEDS:
            raise _fail("seeds", self.seeds, SEEDS)
        if self.walls not in WALLS:
            raise _fail("walls", self.walls, WALLS)
        if self.solver not in SOLVERS:
            raise _fail("solver", self.solver, SOLVERS)

        refine = tuple(self.refine or ())
        for stage in refine:
            if stage not in REFINES:
                raise _fail("refine", stage, REFINES)
        if len(set(refine)) != len(refine):
            raise ValueError(
                "ReshapeSpec.refine has duplicate stages: {!r}".format(refine)
            )
        object.__setattr__(self, "refine", refine)

        params = dict(self.params or {})
        unknown = sorted(set(params) - _VALID_PARAM_KEYS)
        if unknown:
            raise ValueError(
                "ReshapeSpec.params has unknown key(s): {}. Allowed: {}".format(
                    ", ".join(unknown), ", ".join(sorted(_VALID_PARAM_KEYS))
                )
            )
        merged = dict(DEFAULT_PARAMS)
        merged.update(params)

        # --- the arbiter's one hard precondition -----------------
        if self.arbiter and self.walls not in IMAGE_DERIVED_WALLS:
            raise ValueError(
                "ReshapeSpec.arbiter cannot be enabled with walls={!r}: the "
                "arbiter needs an image-derived elevation as evidence ({}). A "
                "shape-derived elevation would arbitrate a cut against its own "
                "shape.".format(self.walls, ", ".join(sorted(IMAGE_DERIVED_WALLS)))
            )

        # --- the Add region target (Add rebuild) --------------------
        if self.seeds == "nuclei_masks" and self.target != "region":
            raise ValueError(
                "ReshapeSpec.seeds='nuclei_masks' needs target='region' (got "
                "target={!r}): it is the Add tab's marker source.".format(
                    self.target)
            )
        if self.target == "region" and self.walls not in ADD_WALL_KINDS:
            raise ValueError(
                "ReshapeSpec.walls={!r} is not valid with target='region'. "
                "Allowed: {}".format(self.walls, ", ".join(sorted(ADD_WALL_KINDS)))
            )
        if self.target == "region":
            # The region cut is always one seeded watershed over the box,
            # with no refine and no arbiter; a
            # spec asking for anything else would be silently ignored, so
            # it is refused here, loudly, at construction (B1 audit L-A3).
            if self.seeds not in REGION_SEEDS:
                raise ValueError(
                    "ReshapeSpec.seeds={!r} is not a marker source of "
                    "target='region'. Allowed: {}".format(
                        self.seeds, ", ".join(REGION_SEEDS))
                )
            if self.solver != "watershed":
                raise ValueError(
                    "ReshapeSpec.solver={!r} is not valid with target='region': "
                    "the region cut is always 'watershed'.".format(self.solver)
                )
            if self.refine:
                raise ValueError(
                    "ReshapeSpec.refine={!r} is not valid with target='region': "
                    "the region target has no refine stage.".format(self.refine)
                )
            if self.arbiter:
                raise ValueError(
                    "ReshapeSpec.arbiter=True is not valid with target='region': "
                    "the region target reports faint walls instead "
                    "(faint_pairs)."
                )
        if merged["outside"] not in OUTSIDE_VALUES:
            raise _fail("params['outside']", merged["outside"], OUTSIDE_VALUES)
        for key, (lo, hi) in _REGION_RANGES.items():
            try:
                value = float(merged[key])
            except (TypeError, ValueError):
                value = float("nan")
            if not lo <= value <= hi:
                raise ValueError(
                    "ReshapeSpec.params[{!r}]={!r} is outside {}-{}.".format(
                        key, merged[key], lo, hi)
                )
        try:
            budget_ok = int(merged["box_budget"]) >= 1
        except (TypeError, ValueError):
            budget_ok = False
        if not budget_ok:
            raise ValueError(
                "ReshapeSpec.params['box_budget']={!r} must be at least "
                "1.".format(merged["box_budget"])
            )
        _check_growth_params(merged)

        object.__setattr__(self, "params", merged)

    # -- convenience -------------------------------------------------------- #
    def get(self, key: str, default: Any = None) -> Any:
        """Read a normalised param.  An unknown key is a ``KeyError``."""
        if key not in _VALID_PARAM_KEYS:
            raise KeyError("unknown ReshapeSpec param {!r}".format(key))
        value = self.params.get(key, default)
        return default if value is None and default is not None else value

    @property
    def arbiter_available(self) -> bool:
        """Whether these walls could support an arbiter verdict at all."""
        return self.walls in IMAGE_DERIVED_WALLS

    def with_params(self, **overrides: Any) -> "ReshapeSpec":
        """A copy with ``params`` overridden (re-validated)."""
        merged = dict(self.params)
        merged.update(overrides)
        return replace(self, params=merged)

    # -- serialisation ------------------------------------------------------ #
    def to_dict(self) -> dict:
        data = asdict(self)
        data["refine"] = list(self.refine)
        data["params"] = dict(self.params)
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReshapeSpec":
        data = dict(data)
        unknown = sorted(set(data) - _SPEC_FIELDS)
        if unknown:
            raise ValueError(
                "ReshapeSpec.from_dict got unknown field(s): {}".format(
                    ", ".join(unknown)
                )
            )
        return cls(
            target=data["target"],
            seeds=data["seeds"],
            walls=data["walls"],
            solver=data.get("solver", "watershed"),
            refine=tuple(data.get("refine", ())),
            params=data.get("params", {}),
            arbiter=bool(data.get("arbiter", False)),
        )

    @classmethod
    def from_json(cls, text: str) -> "ReshapeSpec":
        return cls.from_dict(json.loads(text))


@dataclass
class ArbiterPair:
    """One adjacent piece pair's verdict."""

    a: int
    b: int
    interface_voxels: int
    interface_median: float
    threshold: float
    low_evidence_threshold: float
    decision: str          # "split" | "merge"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ArbiterReport:
    """What the membrane decided, and on what evidence."""

    enabled: bool
    percentile: float = 75.0
    threshold: float = 0.0
    pairs: list = field(default_factory=list)          # list[ArbiterPair]
    merged_pairs: list = field(default_factory=list)   # list[tuple[int, int]]
    verdict: str = "not-run"   # "split" | "dividing" | "no-pairs" | "not-run"

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "percentile": self.percentile,
            "threshold": self.threshold,
            "pairs": [p.to_dict() for p in self.pairs],
            "merged_pairs": [list(p) for p in self.merged_pairs],
            "verdict": self.verdict,
        }


@dataclass
class ReshapeResult:
    """Everything the UI needs, and nothing it has to recompute.

    ``status`` is ``"ok"`` or ``"refused"``.  A refusal is an *answer*, not an
    exception: zero seeds in the target, a target that does not exist, or a job
    over the pre-flight size budget all come back as a refused result carrying a
    plain-language reason in :attr:`messages`.  **There is never a fallback to
    an unseeded watershed**.
    """

    status: str = "ok"
    refusal_reason: str | None = None

    roi: tuple | None = None
    region: Any = None
    result: Any = None
    seeds: Any = None
    elevation: Any = None

    n_seeds: int = 0
    n_seeds_found: int = 0
    n_seeds_dropped: int = 0
    n_pieces: int = 0
    piece_sizes: list = field(default_factory=list)
    #: per-piece ``id`` / ``voxels`` / ``centroid_zyx`` / ``bbox``, in
    #: VOLUME coordinates (the ROI offset is already applied) -- what a
    #: caller needs to write df rows without redoing the arithmetic.
    piece_stats: list = field(default_factory=list)
    coverage: float = 0.0
    vox_volume: float = 1.0

    seed_source: str = ""
    #: True when the ``merge_to_count`` stage ACTUALLY merged something.
    #: A caller that needs to know cannot ask the messages: their text is
    #: user-facing prose that gets reworded, and "Merged " appears in
    #: other sentences.  The quality gate's rule (a) is silent when this
    #: is set, because that stage absorbs marker-backed pieces on purpose.
    merged_to_count: bool = False
    #: How many pieces the ``merge_to_count`` stage SAW -- recorded on
    #: every ok result, including the ones where that stage never runs,
    #: which is precisely the case it exists to diagnose (UX round 6).
    #: "The stage was never sent" and "the stage ran and found nothing
    #: to merge" are indistinguishable on screen and have completely
    #: different fixes; with this and ``merged_to_count`` together they
    #: are one glance apart.  Additive and defaulted: nothing reads it
    #: but the orchestrator's diagnostic line.
    #:
    #: Defaults to **None**, not 0 (audit L5): 0 is a real count the
    #: engine could report, so a 0 default made "the engine never got
    #: that far" (a refusal) and "an engine too old to set it"
    #: indistinguishable from a genuine answer.  None says "no number",
    #: and the diagnostic prints it as such.
    n_pieces_before_merge: int | None = None
    flags: dict = field(default_factory=dict)
    arbiter: ArbiterReport | None = None
    messages: list = field(default_factory=list)
    intermediates: dict = field(default_factory=dict)

    # -- the Add region target only (Add rebuild); all defaulted,
    # so every existing construction is unchanged ----------------------- #
    #: automatic markers dropped because >= 50 % of them lay in dark space
    n_dropped_dark: int = 0
    #: run objects the core erosion emptied (no fallback)
    n_dropped_thin: int = 0
    #: nuclei pieces that never reach 1 um from a kept voxel
    n_nuclei_slivers: int = 0
    #: run objects whose whole core lay over kept objects
    n_markers_over_kept: int = 0
    #: basin pieces cut off from their marker by a kept object, dropped
    n_fragments_dropped: int = 0
    #: share of empty, non-dark space left unfilled; None without the dark rule
    unfilled_tissue_fraction: float | None = None
    #: ``(a, b, median)`` touching pairs on a faint wall, weakest first
    faint_pairs: list = field(default_factory=list)
    #: the patch dark threshold used (``{"t", "lo", "hi"}``)
    dark_threshold: dict | None = None
    #: automatic markers dropped because >= 70 % of them lay outside the
    #: tissue outline (counted apart from ``n_dropped_dark``; a marker both
    #: dark and outside is counted once, as dark -- that drop runs first)
    n_outside_outline: int = 0
    #: run markers dropped under the competition because less than half of
    #: their core lay on empty space (a sliver of a kept cell's own nucleus);
    #: a core wholly over kept objects stays in ``n_markers_over_kept``
    n_slivers: int = 0
    #: the tiles the competition ran in (0 = the flood, never tiled)
    n_tiles: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    def add_message(self, text: str) -> None:
        self.messages.append(text)
