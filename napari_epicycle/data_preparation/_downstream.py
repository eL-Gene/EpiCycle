"""Is a Step 1 patch already being used later in the pipeline?

Trimming a patch rewrites its TIFF with a new Z depth, so every Z index Step 2
recorded for it shifts; deleting one leaves every Step 3 artifact keyed on that
patch name pointing at a file that no longer exists.  Neither is recoverable
from Step 1, so both actions ask first — and this module is what they ask.

Scope + guarantees
------------------
* **Read-only.**  Nothing here creates, moves or deletes a file.  It is
  ``os.scandir`` plus two small JSON reads.
* **Never raises.**  A missing, half-built or plain invalid workspace answers
  :meth:`DownstreamUse.empty`, because a scan failure must never be able to
  block a legitimate trim or delete.
* **Never cascades.**  Callers use the answer to warn; deleting the downstream
  work is always the user's own, explicit job.

Where it lives, and why
-----------------------
Not in ``packages/``: the one-way dependency rule forbids ``packages/`` importing
``napari_epicycle``, and the whole point is to read Steps 2 and 3's canonical
path helpers rather than re-deriving their layouts here.
``hitl_training_annotation`` is a namespace package (no ``__init__.py``), so
``._paths`` imports cleanly.  ``fusion_reconstruction``'s ``__init__`` pulls the
18k-line Qt controller, so its ``_paths.py`` is loaded by FILE via
``importlib.util.spec_from_file_location``.

To be precise about what that buys, because an earlier draft of this docstring
overclaimed it: resolving either path module imports **no Qt and never the Step
3 controller**, which is the expensive, side-effect-laden part.  It does NOT
make the module "Qt-free" in the absolute sense — importing anything under
``napari_epicycle`` executes that package's ``__init__``, which pulls qtpy — and
the test asserts the accurate thing (the DELTA in ``sys.modules`` across both
loads is empty), not the overclaim.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Set, Tuple

__all__ = ["DownstreamUse", "downstream_uses", "downstream_used_names"]

#: Step 2's two training buckets (its view routing: XY -> lateral,
#: XZ/YZ -> axial).  Legacy on-disk names are "spatial"/"axial".
_BUCKETS = ("lateral", "axial")
_LEGACY_BUCKETS = ("spatial", "axial")

#: How many ``round_NN`` directories to probe.  The scan checks EVERY index up
#: to this ceiling and never stops at a gap: rounds look dense in a healthy
#: workspace, but a deleted or archived round would otherwise hide every later
#: round from the guard — and reporting "nothing downstream" for a patch Step 2
#: has curated is the one answer this module must never give (Brick 4c item 8).
#: Probing a missing directory is a single failed ``stat``.
_MAX_ROUNDS = 200


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class DownstreamUse:
    """What a Step 1 patch is being used for, later in the pipeline."""

    #: Number of Step 2 curated / staged / predicted slices naming this patch.
    step2_pairs: int = 0
    #: Step 2 run slugs that hold at least one of them.
    step2_runs: Tuple[str, ...] = ()
    #: Step 3 run slugs that hold an artifact for this patch.
    step3_runs: Tuple[str, ...] = ()
    #: Human-readable Step 3 artifact kinds found ("a reconstruction", ...).
    step3_kinds: Tuple[str, ...] = ()
    #: What Step 4 holds for this patch ("prediction", "reconstruction"), read
    #: from its state file.
    step4_kinds: Tuple[str, ...] = ()

    @classmethod
    def empty(cls) -> "DownstreamUse":
        """The "nothing downstream" answer — also the answer on any failure."""
        return cls()

    @property
    def any(self) -> bool:
        """True when anything at all was found."""
        return bool(self.step2_pairs or self.step3_runs or self.step3_kinds
                    or self.step4_kinds)

    def summary(self) -> str:
        """One plain-language clause naming what was found, e.g.

        ``3 slices you already curated in Step 2 (run 'cells') and a Step 3
        reconstruction``.  Empty string when nothing was found."""
        parts = []
        if self.step2_pairs:
            n = self.step2_pairs
            clause = (f"{n} slice{'' if n == 1 else 's'} you already curated "
                      "in Step 2")
            if self.step2_runs:
                named = ", ".join(f"'{r}'" for r in self.step2_runs)
                clause += (f" (run{'' if len(self.step2_runs) == 1 else 's'} "
                           f"{named})")
            parts.append(clause)
        if self.step3_kinds:
            kinds = list(self.step3_kinds)
            clause = "a Step 3 " + kinds[0] if len(kinds) == 1 else (
                "Step 3 " + ", ".join(kinds[:-1]) + " and " + kinds[-1])
            if self.step3_runs and self.step3_runs != ("default",):
                named = ", ".join(f"'{r}'" for r in self.step3_runs)
                clause += (f" (run{'' if len(self.step3_runs) == 1 else 's'} "
                           f"{named})")
            parts.append(clause)
        if self.step4_kinds:
            kinds = list(self.step4_kinds)
            parts.append("a Step 4 " + kinds[0] if len(kinds) == 1 else
                         "Step 4 " + ", ".join(kinds[:-1]) + " and "
                         + kinds[-1])
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        if len(parts) == 2:
            return " and ".join(parts)
        return ", ".join(parts[:-1]) + " and " + parts[-1]


# ---------------------------------------------------------------------------
# Module loading
# ---------------------------------------------------------------------------
_FUSION_PATHS = None


def _load_fusion_paths():
    """``fusion_reconstruction/_paths.py``, loaded by FILE.

    A plain ``from ..fusion_reconstruction import _paths`` would execute that
    package's ``__init__``, which imports ``.controller`` / ``.logic`` / ``.ui``
    and with them Qt — dragging the whole Step 3 controller into a module (and a
    test suite) that has no business importing it.  ``_paths.py`` itself is pure stdlib, so loading the file directly is
    both correct and cheap.  Cached; returns ``None`` if it cannot be loaded."""
    global _FUSION_PATHS
    if _FUSION_PATHS is not None:
        return _FUSION_PATHS
    try:
        import importlib.util

        here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        target = os.path.join(here, "fusion_reconstruction", "_paths.py")
        spec = importlib.util.spec_from_file_location(
            "_epicycle_step3_paths_readonly", target)
        if spec is None or spec.loader is None:
            return None
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _FUSION_PATHS = mod
        return mod
    except Exception:      # noqa: BLE001 — a scan failure is never fatal
        return None


def _load_step2_paths():
    """``hitl_training_annotation/_paths.py`` — a namespace package, so a normal
    relative import is safe (no ``__init__`` to execute)."""
    try:
        from ..hitl_training_annotation import _paths as p2
        return p2
    except Exception:      # noqa: BLE001
        return None


# ---------------------------------------------------------------------------
# Small filesystem helpers
# ---------------------------------------------------------------------------
def _is_dir(path) -> bool:
    try:
        return os.path.isdir(str(path))
    except Exception:      # noqa: BLE001
        return False


def _listdir(path) -> Sequence[str]:
    try:
        return sorted(os.listdir(str(path)))
    except Exception:      # noqa: BLE001
        return ()


def _child_dirs(path) -> Sequence[str]:
    out = []
    try:
        with os.scandir(str(path)) as it:
            for e in it:
                if e.is_dir():
                    out.append(e.name)
    except Exception:      # noqa: BLE001
        return ()
    return sorted(out)


def _read_json(path) -> Optional[dict]:
    try:
        with open(str(path), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except Exception:      # noqa: BLE001
        return None


def _stem_belongs(stem: str, patch: str) -> bool:
    """Stem-keyed match: the patch itself, or the patch plus a suffix.

    ``P1`` must never match ``P10`` — the separator is required."""
    return stem == patch or stem.startswith(patch + "_")


# ---------------------------------------------------------------------------
# Step 2
# ---------------------------------------------------------------------------
def _step2_runs(p2, workspace) -> Sequence[str]:
    """Every Step 2 run slug (its runs root holds one dir per run)."""
    try:
        return _child_dirs(p2.step2_runs_root(workspace))
    except Exception:      # noqa: BLE001
        return ()


def _count_pairs_in(directory, patch: str, p2) -> int:
    """Slice files under *directory* whose stack_id is *patch*.

    ``parse_slice_label`` is the canonical parser (it handles underscored stack
    ids, which a naive ``rsplit("_", 2)`` gets wrong) and it RAISES on anything
    that isn't a slice label, so every entry is guarded individually — one junk
    filename must not abort the scan."""
    if not _is_dir(directory):
        return 0
    n = 0
    for name in _listdir(directory):
        stem = os.path.splitext(name)[0]
        try:
            stack_id, _view, _idx = p2.parse_slice_label(stem)
        except Exception:      # noqa: BLE001 — not a slice label; skip it
            continue
        if stack_id == patch:
            n += 1
    return n


def _step2_uses(workspace, patch: str) -> Tuple[int, Tuple[str, ...]]:
    """``(pair_count, run_slugs)`` for *patch* across every Step 2 run."""
    p2 = _load_step2_paths()
    if p2 is None:
        return 0, ()
    total = 0
    runs: list = []

    for run_id in _step2_runs(p2, workspace):
        found = 0
        for bucket in _BUCKETS:
            for idx in range(1, _MAX_ROUNDS + 1):
                try:
                    rdir = p2.round_dir(workspace, run_id, bucket, idx)
                except Exception:      # noqa: BLE001
                    break
                if not _is_dir(rdir):
                    continue                    # a gap hides nothing (item 8)
                for sub in ("accepted", "staged", "predictions"):
                    found += _count_pairs_in(os.path.join(str(rdir), sub),
                                             patch, p2)
        # The run's own state records which patch is loaded in it.
        try:
            st = _read_json(p2.state_json_path(workspace)) or {}
        except Exception:      # noqa: BLE001
            st = {}
        if found or _state_names_patch(st, run_id, patch):
            runs.append(run_id)
        total += found

    # Pre-multi-run layout: {ws}/training_data/round_NN/{accepted,staged}_pairs/
    legacy = 0
    for idx in range(1, _MAX_ROUNDS + 1):
        try:
            rdir = p2.legacy_round_dir(workspace, idx)
        except Exception:      # noqa: BLE001
            break
        if not _is_dir(rdir):
            continue                            # a gap hides nothing (item 8)
        for bucket in _LEGACY_BUCKETS:
            for fn in (p2.legacy_round_accepted_pairs_dir,
                       p2.legacy_round_staged_pairs_dir):
                try:
                    legacy += _count_pairs_in(fn(workspace, idx, bucket),
                                              patch, p2)
                except Exception:      # noqa: BLE001
                    continue
    if legacy:
        total += legacy
        if "default" not in runs:
            runs.append("default")

    return total, tuple(runs)


def _state_names_patch(state: dict, run_id: str, patch: str) -> bool:
    """Does ``step2_state.json`` record *patch* as the loaded patch (in either
    the flat shape or a v3 runs envelope)?"""
    if not isinstance(state, dict):
        return False
    if state.get("loaded_patch_name") == patch:
        return True
    runs = state.get("runs")
    if isinstance(runs, dict):
        row = runs.get(run_id)
        if isinstance(row, dict) and row.get("loaded_patch_name") == patch:
            return True
    return False


# ---------------------------------------------------------------------------
# Step 3
# ---------------------------------------------------------------------------
def _step3_run_ids(p3, workspace) -> Sequence[str]:
    """Every Step 3 run: ``"default"`` plus each child of ``runs/`` that holds a
    ``run_manifest.json``.

    That manifest test is the rule ``_run_management.list_run_summaries`` uses,
    and it matters here: ``step3_runs_root`` is an ALIAS of ``runs_dir``, which
    in the flat layout ALSO holds the default run's per-patch reconstruction
    dirs.  Listing it naively would report patches as runs."""
    out = ["default"]
    try:
        root = p3.runs_dir(workspace)
    except Exception:      # noqa: BLE001
        return tuple(out)
    for name in _child_dirs(root):
        if os.path.isfile(os.path.join(str(root), name, "run_manifest.json")):
            out.append(name)
    return tuple(out)


#: (path-helper name, needs a patch arg, kind label, is-a-directory)
_STEP3_PATCH_ARTIFACTS = (
    ("patch_run_dir", True, "reconstruction", True),
    ("patch_run_manifest_path", True, "reconstruction", False),
    ("patch_curation_dir", True, "curation", True),
    ("patch_curation_tiers_csv", True, "curation", False),
    ("qc_features_csv", True, "QC analysis", False),
    ("qc_features_raw_csv", True, "QC analysis", False),
    ("qc_adjacency_raw_csv", True, "QC analysis", False),
    ("qc_adjacency_curated_csv", True, "QC analysis", False),
    ("patch_pairs_dir", True, "training pairs", True),
    ("patch_pairs_spatial_dir", True, "training pairs", True),
    ("patch_pairs_axial_dir", True, "training pairs", True),
)


def _exists(path, is_dir: bool) -> bool:
    try:
        return os.path.isdir(str(path)) if is_dir else os.path.isfile(str(path))
    except Exception:      # noqa: BLE001
        return False


def _stem_hit(directory, patch: str, suffix: str = "") -> bool:
    """Any entry under *directory* whose stem belongs to *patch*."""
    if not _is_dir(directory):
        return False
    for name in _listdir(directory):
        stem = os.path.splitext(name)[0]
        if _stem_belongs(stem, patch) and (not suffix or suffix in name):
            return True
    return False


def _step3_uses(workspace, patch: str) -> Tuple[Tuple[str, ...],
                                                Tuple[str, ...]]:
    """``(run_slugs, kinds)`` for *patch* across every Step 3 run."""
    p3 = _load_fusion_paths()
    if p3 is None:
        return (), ()
    runs: list = []
    kinds: list = []

    def _note(kind):
        if kind not in kinds:
            kinds.append(kind)

    for run_id in _step3_run_ids(p3, workspace):
        rid = None if run_id == "default" else run_id
        hit = False
        for fname, _needs_patch, kind, is_dir in _STEP3_PATCH_ARTIFACTS:
            fn = getattr(p3, fname, None)
            if fn is None:
                continue
            try:
                path = fn(workspace, patch, rid)
            except Exception:      # noqa: BLE001
                continue
            if _exists(path, is_dir):
                hit = True
                _note(kind)
        # Stem-keyed caches (`{patch}_...` files, not `{patch}/` dirs).
        try:
            if _stem_hit(p3.prediction_cache_dir(workspace, rid), patch):
                hit = True
                _note("prediction cache")
        except Exception:      # noqa: BLE001
            pass
        # Per-run state: a patch with a status other than "unprocessed" has
        # been worked on even if its artifacts were since cleaned up.
        if _step3_state_names_patch(p3, workspace, run_id, patch):
            hit = True
            _note("run status")
        if hit:
            runs.append(run_id)

    # Workspace-level (not per-run) archives.
    for fname, kind, is_dir in (
            ("patch_training_archive_dir", "training archive", True),):
        fn = getattr(p3, fname, None)
        if fn is None:
            continue
        try:
            if _exists(fn(workspace, patch), is_dir):
                _note(kind)
                if "default" not in runs:
                    runs.append("default")
        except Exception:      # noqa: BLE001
            pass
    try:
        if _stem_hit(p3.snapshots_dir(workspace), patch, "_pre_commit_"):
            _note("pre-commit snapshot")
            if "default" not in runs:
                runs.append("default")
    except Exception:      # noqa: BLE001
        pass

    # Pre-restructure (legacy) roots.
    for getter, kind, keyed in (
            ("legacy_reconstruction_root", "reconstruction", "dir"),
            ("legacy_qc_dir", "QC analysis", "stem"),
            ("legacy_prediction_cache_dir", "prediction cache", "stem"),
            ("legacy_snapshots_dir", "pre-commit snapshot", "stem"),
            ("legacy_refinement_training_data", "training pairs", "dir")):
        fn = getattr(p3, getter, None)
        if fn is None:
            continue
        try:
            root = fn(workspace)
        except Exception:      # noqa: BLE001
            continue
        found = (_exists(os.path.join(str(root), patch), True)
                 if keyed == "dir" else _stem_hit(root, patch))
        if found:
            _note(kind)
            if "default" not in runs:
                runs.append("default")

    return tuple(runs), tuple(kinds)


def _step3_state_names_patch(p3, workspace, run_id: str, patch: str) -> bool:
    """Does ``step3_state.json`` record *patch* as anything but unprocessed?

    Read in BOTH shapes: the v3 runs envelope (``runs.<run_id>.patches[patch]``)
    and the un-upgraded flat file (``patches[patch]``)."""
    try:
        state = _read_json(p3.state_json_path(workspace))
    except Exception:      # noqa: BLE001
        state = None
    if not state:
        return False
    rows: Dict = {}
    runs = state.get("runs")
    if isinstance(runs, dict):
        row = runs.get(run_id)
        if isinstance(row, dict) and isinstance(row.get("patches"), dict):
            rows = row["patches"]
    if not rows and run_id == "default" and isinstance(state.get("patches"), dict):
        rows = state["patches"]
    entry = rows.get(patch) if isinstance(rows, dict) else None
    if not isinstance(entry, dict):
        return False
    return str(entry.get("status") or "unprocessed") != "unprocessed"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
#: Step 4's state file (schema 2), its backup and the legacy file it
#: replaced; read in that order, as Step 4 itself does
#: (K8b).
_STEP4_STATE = ("step4_state.json",)            # the workspace root
#: where it lived before 2026-09-29 (Step 4 moves it on its next load)
_STEP4_OLD_STATE = ("artifacts", "step4", "step4_state.json")
_STEP4_BACKUP = ("artifacts", "step4", "state_backup",
                 "step4_state.json.bak")
_STEP4_LEGACY_STATE = ("stack_segmentation_state.json",)
#: Step 4 statuses that mean a 3D result exists (its status machine;
#: K8c: "curated" is not one of Step 4's statuses).
_STEP4_3D = frozenset({"segmented", "reconstructed", "qc_extracted",
                       "smoothed", "finalized"})


def _norm_path(p) -> str:
    return os.path.normcase(os.path.normpath(os.path.abspath(str(p))))


def _step1_patch_rel(workspace, patch: str) -> Optional[str]:
    """The patch's own file, workspace-relative, from Step 1's registry
    (None when unknown)."""
    try:
        from ._state import all_patches, load_state
        for row in all_patches(load_state(workspace)):
            if row.get("patch_name") == patch and row.get("relative_path"):
                return str(row["relative_path"])
    except Exception:      # noqa: BLE001 -- the guard never blocks
        return None
    return None


def _path_parts(p) -> list:
    norm = os.path.normcase(str(p)).replace("\\", "/")
    return [x for x in norm.split("/") if x and x != "."]


def _same_patch_file(src, workspace, rel) -> bool:
    """Is *src* the patch's file?  Yes when it resolves to
    ``workspace/rel``, or -- for a workspace copied or moved from another
    PC (re-audit M2) -- when its trailing parts ARE the workspace-relative
    path (``.../artifacts/step1/patches/P1.tif``)."""
    if _norm_path(src) == _norm_path(os.path.join(workspace, rel)):
        return True
    tail = _path_parts(rel)
    parts = _path_parts(src)
    return bool(tail) and parts[-len(tail):] == tail


def _stem(p) -> str:
    base = os.path.basename(str(p or "").replace("\\", "/"))
    return os.path.splitext(base)[0]


def _step4_record_names_patch(name: str, record: dict, patch: str,
                              patch_rel: Optional[str] = None,
                              workspace: str = "") -> bool:
    """Is this Step 4 record the patch's (K8a)?  When Step 1 knows the
    patch's file, the record's source path must BE that file -- a stack
    added from another folder with the same stem is not the patch.  The
    stem (exact: ``P1`` never claims ``P10``) is only the fallback, for a
    record with no source path or a patch Step 1 cannot place."""
    src = record.get("source_stack_path")
    if patch_rel and src:
        return _same_patch_file(src, workspace, patch_rel)
    if src:
        return _stem(src) == patch
    if _stem(name) == patch:
        print(f"[STEP1-DOWNSTREAM] Step 4 record {name!r} has no source "
              f"path; matched to {patch!r} by its name only")
        return True
    return False


def _read_step4_state(workspace) -> Optional[dict]:
    """Step 4's own read order (its state manager's ``load``): the main
    file; when the main file exists but cannot be read, its backup then
    the legacy file; when there is no main file, the legacy file."""
    def ok(state):
        return isinstance(state, dict) and isinstance(state.get("stacks"),
                                                      dict)
    main = os.path.join(workspace, *_STEP4_STATE)
    if not os.path.exists(main):
        old = os.path.join(workspace, *_STEP4_OLD_STATE)
        if os.path.exists(old):
            main = old
    if os.path.exists(main):
        state = _read_json(main)
        if ok(state):
            return state
        order = (_STEP4_BACKUP, _STEP4_LEGACY_STATE)
    else:
        order = (_STEP4_LEGACY_STATE,)
    for parts in order:
        state = _read_json(os.path.join(workspace, *parts))
        if ok(state):
            return state
    return None


def _step4_uses(workspace, patch: str) -> Tuple[str, ...]:
    """What Step 4 holds for *patch* (F21): ``("prediction",)``,
    ``("reconstruction",)`` or ``()`` -- Step 3's word for a 3D result.
    Reads the state file, its backup, then the legacy file; a file that
    cannot be parsed answers nothing (the guard never blocks)."""
    state = _read_step4_state(workspace)
    if state is None:
        return ()
    patch_rel = _step1_patch_rel(workspace, patch)
    kinds = []
    for name, rec in state["stacks"].items():
        if not isinstance(rec, dict) or not _step4_record_names_patch(
                str(name), rec, patch, patch_rel, str(workspace)):
            continue
        status = str(rec.get("status") or "unprocessed")
        if status in _STEP4_3D or rec.get("labels_path"):
            kind = "reconstruction"
        elif status == "predicted" or rec.get("predicted_with"):
            kind = "prediction"
        else:
            continue
        if kind not in kinds:
            kinds.append(kind)
    order = ("prediction", "reconstruction")
    return tuple(k for k in order if k in kinds)


def downstream_uses(workspace, patch_name: str) -> DownstreamUse:
    """What *patch_name* is used for in Steps 2, 3 and 4 of *workspace*.

    Returns :meth:`DownstreamUse.empty` — never raises — for a missing or
    invalid workspace, a blank patch name, or any failure mid-scan.  Step 4
    is read from its state file (F21); Step 5 is not scanned: it has no
    Step-1 patch-name reader."""
    try:
        if not workspace or not patch_name or not _is_dir(workspace):
            return DownstreamUse.empty()
        pairs, s2_runs = _step2_uses(workspace, patch_name)
        s3_runs, s3_kinds = _step3_uses(workspace, patch_name)
        try:
            s4_kinds = _step4_uses(workspace, patch_name)
        except Exception:      # noqa: BLE001 -- Step 4's part never blocks
            s4_kinds = ()
        return DownstreamUse(step2_pairs=pairs, step2_runs=s2_runs,
                             step3_runs=s3_runs, step3_kinds=s3_kinds,
                             step4_kinds=s4_kinds)
    except Exception:      # noqa: BLE001 — a scan must never block an action
        return DownstreamUse.empty()


def downstream_used_names(workspace) -> Set[str]:
    """Every Step 1 patch name with downstream use, from ONE call.

    Note the wording: ONE CALL, not one pass.  An earlier draft said "one
    batched pass", which it is not.

    For a caller that needs the whole picture (a ④ list refresh) rather than one
    patch.  The names come from Step 1's own registry, so the answer is always a
    subset of what Step 1 knows about.  Empty set on any failure.

    Honest note on cost: today this sweeps per name and short-circuits on the
    first hit.  Its only planned consumer — the ④ "used in Step 2/3" badge — is
    DEFERRED (the dialogs are the guard), so folding the per-name sweeps into a
    single walk of both trees is the optimisation to make when that badge lands,
    against a real list size.  The API is settled either way."""
    out: Set[str] = set()
    try:
        from ._state import all_patches, load_state

        if not workspace or not _is_dir(workspace):
            return out
        names = [r.get("patch_name") for r in all_patches(load_state(workspace))]
        for name in names:
            if name and downstream_uses(workspace, name).any:
                out.add(name)
    except Exception:      # noqa: BLE001
        return set()
    return out
