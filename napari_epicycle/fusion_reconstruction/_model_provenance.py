"""Step 3 model provenance -- the pure half.

Qt-free and napari-free.  What it gives the R6 wiring:

* **The kind table** -- ``model_kind(path, *, workspace)`` names what made a
  patch, one of ``MODEL_KINDS``:

  ======================  ==================================================
  ``pretrained``          a Cellpose built-in NAME (``cyto3``, ``nuclei``…)
  ``bootstrap``           a Step 2 round model ("2D Model HITL
                          Bootstrapping"): its schema-2.x ``manifest.json``
                          beside the weights, else its path
                          (``artifacts/step2/…`` or the legacy
                          ``…/models/round_NN/<file>``, in this workspace
                          or any other)
  ``domain_specialized``  a Step 2 round model whose manifest names a Step 3
                          refined bundle (``init.mode == "step3_refined"``,
                          ``inherited_from_step3_bundle``, or an init path
                          inside a refined bundle) or ANOTHER run
                          (``init.mode == "inherited"``,
                          ``inherited_from_run``) as its init
  ``refined``             a Step 3 refined bundle: its schema-2.0
                          ``manifest.json`` (``name`` + a ``lateral`` /
                          ``axial`` entry), else its path
                          (``artifacts/step3/[runs/<run>/]refinement/models/``)
  ``external``            an existing file that is none of the above
  ``unknown``             nothing, a name Cellpose does not know, or a file
                          that is gone
  ======================  ==================================================

  The model's own manifest decides first (it travels with a copied or
  browsed bundle); the path decides when there is no manifest or it cannot
  be read.
* **The record** -- ``view_record`` (one view) and ``provenance_block``
  (``{"schema": 1, "recorded_at", "views": {"lateral", "axial"},
  "notes"}``), never raising.
* **The manifests** -- ``read_block`` / ``write_notes`` /
  ``backfill_manifest`` keep EVERY key they did not write, at every level,
  so a later backend adds keys without a migration; writes are atomic
  (``_step3_manifests.write_json_atomic``).
* **The chip** -- ``short_name`` / ``made_by_text`` and ``is_active_model``
  (Step 4's ``same_model`` path rule).

Every word is in ``_step3_pr_text`` (``# --- R6 ---``).
"""
from __future__ import annotations

import copy
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from . import _step3_manifests as _SM
from . import _step3_pr_text as PT
from ..data_preparation._voxel_spacing import _rel_to
from ..ui_shell._model_files import file_fingerprint, read_trained_channels

#: the kinds, in a fixed order
MODEL_KINDS = ("pretrained", "bootstrap", "domain_specialized", "refined",
               "external", "unknown")
#: the block's own version
PROVENANCE_SCHEMA = 1
#: the manifests' key for the block
PROVENANCE_KEY = "model_provenance"
#: the two views, in order
VIEW_NAMES = ("lateral", "axial")
#: a minimal manifest written for an old patch carries this, True
BACKFILLED_KEY = "backfilled"
#: an existing manifest that RECEIVED a backfilled block carries this, True
#: (the manifest itself is real; only the block was added later)
BACKFILLED_BLOCK_KEY = "model_provenance_backfilled"

#: Cellpose's built-in model names (``cellpose.models.MODEL_NAMES`` of
#: 3.1.1.2, plus 4.x's ``cpsam``); copied, so this module never imports
#: Cellpose.  Step 3's catalog (``refinement_registry_dialog.
#: PRETRAINED_MODELS``) is a subset (a test).
CELLPOSE_BUILTIN_NAMES = frozenset({
    "cyto3", "nuclei", "cyto2_cp3", "tissuenet_cp3", "livecell_cp3",
    "yeast_PhC_cp3", "yeast_BF_cp3", "bact_phase_cp3", "bact_fluor_cp3",
    "deepbacs_cp3", "cyto2", "cyto", "CPx", "transformer_cp3",
    "neurips_cellpose_default", "neurips_cellpose_transformer",
    "neurips_grayscale_cyto2", "CP", "TN1", "TN2", "TN3", "LC1", "LC2",
    "LC3", "LC4", "cpsam",
})

_MANIFEST_NAME = "manifest.json"
_STEP2_SCHEMAS = ("2.0", "2.1")
_ROUND_RE = re.compile(r"^round_(\d+)$", re.IGNORECASE)
_STEP2_INHERIT_KEYS = (
    "inherited_from_run", "inherited_from_run_bucket",
    "inherited_from_run_round_id", "inherited_from_bucket",
    "inherited_from_step3_bundle", "inherited_from_step3_bucket",
    "inherited_from_external_path", "inherited_from_external_label",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _text(path) -> str:
    if path is None:
        return ""
    try:
        if isinstance(path, (str, os.PathLike)):
            return str(os.fspath(path)).strip()
    except (TypeError, ValueError):
        return ""
    return ""


def _is_bare_name(text: str) -> bool:
    return bool(text) and "/" not in text and "\\" not in text


def _is_file(p: Path) -> bool:
    try:
        return p.is_file()
    except (OSError, ValueError):
        return False


def _parts(p: Path) -> List[str]:
    try:
        return [s.lower() for s in Path(os.path.normpath(str(p))).parts]
    except (TypeError, ValueError):
        return []


def _seq_at(parts: List[str], seq: Tuple[str, ...]) -> int:
    """The index where *seq* starts in *parts*, or -1."""
    n = len(seq)
    for i in range(len(parts) - n + 1):
        if tuple(parts[i:i + n]) == seq:
            return i
    return -1


def _shown_path(path: str, workspace: str) -> str:
    """Workspace-relative (``/``) when under *workspace*, else as given."""
    if workspace and os.path.isabs(path):
        rel = _rel_to(path, workspace)
        if rel is not None:
            return rel
    return path


def _read_json(path: Path) -> Tuple[Optional[Any], Optional[str]]:
    """``(data, None)`` or ``(None, raw reason)``.  Never raises."""
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except (OSError, ValueError, TypeError) as exc:
        return None, f"{type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# The model's own manifest
# ---------------------------------------------------------------------------


def _manifest_family(data) -> Optional[str]:
    """``"step2"`` / ``"refined"`` for a recognised manifest, else None."""
    if not isinstance(data, dict):
        return None
    if (data.get("bucket") in VIEW_NAMES and "round_number" in data
            and str(data.get("schema_version", "")) in _STEP2_SCHEMAS):
        return "step2"
    if "name" in data and any(isinstance(data.get(b), dict)
                              for b in VIEW_NAMES):
        return "refined"
    return None


def _path_family(p: Path, workspace: str) -> Optional[str]:
    parts = _parts(p)
    s3 = _seq_at(parts, ("artifacts", "step3"))
    if s3 >= 0 and _seq_at(parts[s3:], ("refinement", "models")) >= 0:
        return "refined"
    if _seq_at(parts, ("artifacts", "step2")) >= 0:
        return "step2"
    if workspace:
        rel = _rel_to(str(p), workspace)
        if rel is not None:
            rparts = rel.lower().split("/")
            if len(rparts) >= 3 and rparts[0] == "models" and \
                    _ROUND_RE.match(rparts[1]):
                return "step2"            # the legacy {ws}/models/round_NN/
    # B: a legacy round folder browsed from ANOTHER workspace
    # keeps its kind -- ``.../models/round_NN/<file>`` anywhere
    for i in range(len(parts) - 2):
        if parts[i] == "models" and _ROUND_RE.match(parts[i + 1]):
            return "step2"
    return None


def _init_names_a_refined_bundle(path_text) -> bool:
    if not path_text or not isinstance(path_text, str):
        return False
    parts = _parts(Path(path_text))
    s3 = _seq_at(parts, ("artifacts", "step3"))
    return s3 >= 0 and _seq_at(parts[s3:], ("refinement", "models")) >= 0


def _step2_is_specialized(m: Dict[str, Any]) -> bool:
    init = m.get("init") if isinstance(m.get("init"), dict) else {}
    mode = init.get("mode")
    if mode in ("step3_refined", "inherited"):
        return True
    if m.get("inherited_from_step3_bundle") or m.get("inherited_from_run"):
        return True
    return (_init_names_a_refined_bundle(init.get("path"))
            or _init_names_a_refined_bundle(
                m.get("inherited_from_external_path")))


def _round_from_path(p: Path) -> Optional[int]:
    for seg in reversed(_parts(p)):
        hit = _ROUND_RE.match(seg)
        if hit:
            return int(hit.group(1))
    return None


def _after(p: Path, seq: Tuple[str, ...]) -> Optional[str]:
    """The ORIGINAL-case path segment right after *seq*, or None."""
    try:
        raw = list(Path(os.path.normpath(str(p))).parts)
    except (TypeError, ValueError):
        return None
    low = [s.lower() for s in raw]
    i = _seq_at(low, seq)
    if i < 0 or i + len(seq) >= len(raw):
        return None
    return raw[i + len(seq)]


def _step2_run_id(p: Path, m: Optional[Dict[str, Any]]) -> Optional[str]:
    if isinstance(m, dict) and m.get("run_id"):
        return str(m["run_id"])
    run = _after(p, ("artifacts", "step2", "runs"))
    if run:
        return run
    # the flat default layout under artifacts/step2 IS the default run; a
    # manifest-less LEGACY round folder (path rule only) names no run
    return "default" if _seq_at(_parts(p), ("artifacts", "step2")) >= 0         else None


def _refined_run_id(p: Path) -> str:
    run = _after(p, ("artifacts", "step3", "runs"))
    if run and run.lower() not in ("refinement",):
        return run
    return "default"


def _classify(path) -> Dict[str, Any]:
    """Everything the kind and the record need.  Never raises."""
    out: Dict[str, Any] = {"kind": "unknown", "family": None, "text": "",
                           "path": None, "manifest": None,
                           "manifest_path": None, "error": None}
    text = _text(path)
    out["text"] = text
    if not text:
        out["error"] = PT.ERR_MODEL_MISSING
        return out
    p = Path(text)
    out["path"] = p
    if not _is_file(p):
        if _is_bare_name(text) and text in CELLPOSE_BUILTIN_NAMES:
            out["kind"] = "pretrained"
        else:
            out["error"] = PT.ERR_MODEL_MISSING
        return out
    mpath = p.parent / _MANIFEST_NAME
    family = None
    if _is_file(mpath):
        data, problem = _read_json(mpath)
        family = _manifest_family(data)
        if family is not None:
            out["manifest"] = data
            out["manifest_path"] = mpath
        elif problem is not None:
            out["error"] = PT.ERR_MANIFEST_UNREADABLE
            out["manifest_path"] = mpath
    out["family"] = family
    return out


def _kind_of(info: Dict[str, Any], workspace: str) -> str:
    if info["kind"] == "pretrained" or info["path"] is None:
        return info["kind"]
    if info["error"] == PT.ERR_MODEL_MISSING:
        return "unknown"
    family = info["family"] or _path_family(info["path"], workspace)
    info["family"] = family
    if family == "refined":
        return "refined"
    if family == "step2":
        m = info["manifest"]
        if isinstance(m, dict) and _step2_is_specialized(m):
            return "domain_specialized"
        return "bootstrap"
    return "external"


def model_kind(path, *, workspace: str = "") -> str:
    """The kind of the model at *path* (a weights file or a built-in name),
    one of ``MODEL_KINDS``.  Never raises."""
    try:
        return _kind_of(_classify(path), str(workspace or ""))
    except Exception:      # noqa: BLE001
        return "unknown"


# ---------------------------------------------------------------------------
# The record
# ---------------------------------------------------------------------------


def _pick(d: Any, keys) -> Dict[str, Any]:
    if not isinstance(d, dict):
        return {}
    return {k: d.get(k) for k in keys if k in d}


def _step2_extra(m: Dict[str, Any]) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    for k in ("run_id", "bucket", "round_number", "schema_version",
              "created_at"):
        if k in m:
            extra[k] = m.get(k)
    if isinstance(m.get("init"), dict):
        extra["init"] = _pick(m["init"], ("mode", "path", "init_combo_text"))
    inherited = {k[len("inherited_from_"):]: m.get(k)
                 for k in _STEP2_INHERIT_KEYS if m.get(k) is not None}
    if inherited:
        extra["inherited_from"] = inherited
    if isinstance(m.get("training"), dict):
        extra["training"] = _pick(
            m["training"], ("n_train_pairs", "n_epochs", "channels_used"))
    for k in ("final_loss", "environment"):
        if isinstance(m.get(k), dict):
            extra[k] = dict(m[k])
    return extra


def _refined_entry(m: Dict[str, Any], p: Path) -> Tuple[Optional[str], dict]:
    for b in VIEW_NAMES:
        e = m.get(b)
        if isinstance(e, dict) and str(e.get("weights_path", "")) == p.name:
            return b, e
    b = p.name.lower()
    if b in VIEW_NAMES and isinstance(m.get(b), dict):
        return b, m[b]
    return None, {}


def _refined_extra(m: Dict[str, Any], p: Path) -> Dict[str, Any]:
    bucket, entry = _refined_entry(m, p)
    extra: Dict[str, Any] = {"bundle": m.get("name"), "bucket": bucket}
    for k in ("schema_version", "created_at", "last_updated"):
        if k in m:
            extra[k] = m.get(k)
    if isinstance(entry.get("init"), dict):
        extra["init"] = _pick(entry["init"],
                              ("mode", "path", "init_combo_text"))
    training = _pick(entry, ("n_train_pairs", "n_val_pairs", "n_epochs"))
    if training:
        extra["training"] = training
    if isinstance(entry.get("final_loss"), dict):
        extra["final_loss"] = dict(entry["final_loss"])
    return extra


def view_record(path, *, workspace: str = "",
                backend: str = "cellpose") -> Dict[str, Any]:
    """One view's record: ``{backend, kind, name, path, run_id, round,
    fingerprint, trained_channels, extra}``.  ``path`` is workspace-relative
    when the file sits under *workspace*; ``extra`` carries what the model's
    own manifest knows plus ``model_manifest`` (the manifest it came from),
    and ``error`` when the file is gone or its manifest cannot be read.
    JSON-safe.  Never raises."""
    ws = str(workspace or "")
    rec: Dict[str, Any] = {
        "backend": backend, "kind": "unknown", "name": None, "path": None,
        "run_id": None, "round": None, "fingerprint": None,
        "trained_channels": None, "extra": {},
    }
    try:
        info = _classify(path)
        kind = _kind_of(info, ws)
        rec["kind"] = kind
        text = info["text"]
        p: Optional[Path] = info["path"]
        if kind == "pretrained":
            rec["name"] = text
            rec["path"] = text
            return rec
        rec["path"] = _shown_path(text, ws) if text else None
        rec["name"] = p.name if p is not None else None
        extra: Dict[str, Any] = {}
        if kind != "unknown" and p is not None:
            rec["fingerprint"] = file_fingerprint(p)
            rec["trained_channels"] = read_trained_channels(p)
        m = info["manifest"]
        if kind in ("bootstrap", "domain_specialized") and p is not None:
            rec["run_id"] = _step2_run_id(p, m)
            rnd = m.get("round_number") if isinstance(m, dict) else None
            rec["round"] = rnd if isinstance(rnd, int) else \
                _round_from_path(p)
            if isinstance(m, dict):
                extra.update(_step2_extra(m))
            bucket = (m.get("bucket") if isinstance(m, dict) else None)
            label = [rec["run_id"]] if rec["run_id"] else []
            if bucket:
                label.append(bucket)
            if rec["round"] is not None:
                label.append(f"round_{rec['round']:02d}")
            rec["name"] = (" · ".join(str(s) for s in label) if label
                           else p.name)
        elif kind == "refined" and p is not None:
            rec["run_id"] = _refined_run_id(p)
            if isinstance(m, dict):
                extra.update(_refined_extra(m, p))
                rec["name"] = str(m.get("name") or p.parent.name)
            else:
                rec["name"] = p.parent.name
        if info["manifest_path"] is not None:
            extra["model_manifest"] = _shown_path(str(info["manifest_path"]),
                                                  ws)
        if info["error"]:
            extra["error"] = info["error"]
        rec["extra"] = _SM.json_safe(extra)
    except Exception:      # noqa: BLE001
        rec["kind"] = rec["kind"] if rec["kind"] in MODEL_KINDS else "unknown"
    return rec


def _iso(now) -> str:
    if isinstance(now, datetime):
        return now.isoformat()
    if isinstance(now, str) and now:
        return now
    return datetime.now(timezone.utc).isoformat()


def provenance_block(lateral_path, axial_path, *, workspace: str = "",
                     now=None) -> Dict[str, Any]:
    """The block written into the prediction manifest at Predict, from the
    model paths snapshotted at LAUNCH: ``{"schema": 1, "recorded_at",
    "views": {"lateral", "axial"}, "notes": ""}``.  An empty *axial_path*
    takes the lateral (Step 4's ``model_snapshot`` rule).  *now* is a
    ``datetime`` or an ISO string (default: the clock, UTC)."""
    axial_path = axial_path if _text(axial_path) else lateral_path
    return {
        "schema": PROVENANCE_SCHEMA,
        "recorded_at": _iso(now),
        "views": {
            "lateral": view_record(lateral_path, workspace=workspace),
            "axial": view_record(axial_path, workspace=workspace),
        },
        "notes": "",
    }


def unknown_block(*, now=None) -> Dict[str, Any]:
    """The block a backfill writes for a patch predicted before R6: both
    views ``unknown``, nothing invented."""
    def view():
        return {"backend": None, "kind": "unknown", "name": None,
                "path": None, "run_id": None, "round": None,
                "fingerprint": None, "trained_channels": None, "extra": {}}
    return {"schema": PROVENANCE_SCHEMA, "recorded_at": _iso(now),
            "views": {"lateral": view(), "axial": view()}, "notes": "",
            BACKFILLED_KEY: True}


# ---------------------------------------------------------------------------
# The manifests (read, notes, backfill)
# ---------------------------------------------------------------------------


def read_manifest(path) -> Optional[Dict[str, Any]]:
    """Any manifest as a dict, or None (missing, unreadable, not a dict).
    Unlike ``_step3_manifests.read_prediction_manifest`` it does not ask
    for ``seg_config``, so a backfilled manifest reads too.  Never raises."""
    data, _problem = _read_json(Path(_text(path) or "."))
    return data if isinstance(data, dict) else None


def read_block(manifest) -> Optional[Dict[str, Any]]:
    """A deep copy of *manifest*'s ``model_provenance`` block with every
    key kept, or None when there is none."""
    if not isinstance(manifest, dict):
        return None
    block = manifest.get(PROVENANCE_KEY)
    if not isinstance(block, dict):
        return None
    return copy.deepcopy(block)


def pick_block(reconstruction_manifest, prediction_manifest
               ) -> Optional[Dict[str, Any]]:
    """The block to show: the reconstruction manifest's copy when the patch
    is reconstructed, else the prediction manifest's."""
    return (read_block(reconstruction_manifest)
            or read_block(prediction_manifest))


def write_notes(manifest_path, notes) -> Tuple[bool, Optional[str]]:
    """Set the block's ``notes`` in the manifest at *manifest_path*, every
    other key kept, atomically.  ``(True, None)``, or ``(False, raw reason)``
    for the log -- the file is never changed then.  Works on a prediction
    manifest and on a reconstruction manifest's copy.  Never raises."""
    path = Path(_text(manifest_path) or ".")
    data, problem = _read_json(path)
    if problem is not None:
        return False, problem
    if not isinstance(data, dict):
        return False, f"not a manifest: {path}"
    block = data.get(PROVENANCE_KEY)
    if not isinstance(block, dict):
        return False, f"no {PROVENANCE_KEY} block in {path}"
    block["notes"] = "" if notes is None else str(notes)
    return _write_keeping_mtime(path, data)


def _write_keeping_mtime(path: Path, data) -> Tuple[bool, Optional[str]]:
    """``write_json_atomic`` over an EXISTING manifest, its times put back
    afterwards: a note or a backfill is not a new prediction or
    reconstruction, and both "the newest reconstruction" readers (Step 3's
    ``_newest_recon_manifest``, Step 4's ``newest_step3_manifest``) pick by
    ``st_mtime``.  A failed ``utime`` is not a failed write."""
    try:
        st = os.stat(path)
    except OSError:
        st = None
    try:
        _SM.write_json_atomic(path, data)
    except (OSError, TypeError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if st is not None:
        try:
            os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
        except OSError as exc:
            print(f"[STEP3-PROV] mtime not restored on {path}: "
                  f"{type(exc).__name__}: {exc}")
    return True, None


def backfill_manifest(path, block) -> Tuple[bool, Optional[str]]:
    """Give an old patch a block, never overwriting one.

    * no file -> a minimal prediction manifest
      ``{"manifest_version": 1, "backfilled": true, "model_provenance":
      block}``;
    * a readable manifest whose block is missing or ``null`` (a patch
      predicted between R2c and R6) -> the block is added and
      ``model_provenance_backfilled: true`` set, every other key kept;
    * a manifest that HAS a block, or a file that cannot be read -> refused,
      untouched.

    ``(True, None)`` or ``(False, raw reason)``.  Never raises."""
    target = Path(_text(path) or ".")
    if not isinstance(block, dict):
        return False, "the block is not a dict"
    try:
        exists = target.exists()
    except (OSError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not exists:
        data: Dict[str, Any] = {
            "manifest_version": _SM.PREDICTION_MANIFEST_VERSION,
            BACKFILLED_KEY: True,
            PROVENANCE_KEY: block,
        }
    else:
        data, problem = _read_json(target)
        if problem is not None:
            return False, problem
        if not isinstance(data, dict):
            return False, f"not a manifest: {target}"
        if isinstance(data.get(PROVENANCE_KEY), dict):
            return False, f"{target} already records a model"
        data[PROVENANCE_KEY] = block
        data[BACKFILLED_BLOCK_KEY] = True
        return _write_keeping_mtime(target, data)
    try:
        _SM.write_json_atomic(target, data)
    except (OSError, TypeError, ValueError) as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, None


# ---------------------------------------------------------------------------
# The chip
# ---------------------------------------------------------------------------


def _view(block, name: str) -> Optional[Dict[str, Any]]:
    if not isinstance(block, dict):
        return None
    views = block.get("views")
    if not isinstance(views, dict):
        return None
    v = views.get(name)
    return v if isinstance(v, dict) else None


def short_name(block) -> str:
    """The lateral model's name, elided to ``SHORT_NAME_MAX`` characters;
    ``""`` when the block names none."""
    lat = _view(block, "lateral")
    name = lat.get("name") if lat else None
    if not name:
        return ""
    name = str(name)
    if len(name) > PT.SHORT_NAME_MAX:
        name = name[:PT.SHORT_NAME_MAX - len(PT.ELLIPSIS)] + PT.ELLIPSIS
    return name


def made_by_text(block) -> str:
    """The row's chip: "made by {short}", or the unknown sentence."""
    short = short_name(block)
    return PT.MADE_BY.format(short=short) if short else PT.MADE_BY_UNKNOWN


def _path_key(text: str, workspace: str):
    """Step 4's ``same_model`` rule: the workspace-relative path when there
    is one, the normalised absolute path otherwise (a bare built-in name
    compares as itself)."""
    text = text.strip()
    if os.path.isabs(text):
        rel = _rel_to(text, workspace) if workspace else None
        if rel is not None:
            return ("rel", os.path.normcase(os.path.normpath(rel)))
        return ("abs", os.path.normcase(os.path.normpath(text)))
    return ("rel", os.path.normcase(os.path.normpath(text)))


def is_active_model(block, lateral_path, axial_path,
                    workspace: str = "") -> bool:
    """Whether *block* names the active model (both views), compared by
    path as ``same_model`` compares (a copied workspace compares equal).
    False with no block or a view with no path -- the caller draws the
    unknown chip without the amber border."""
    ws = str(workspace or "")
    lat_now = _text(lateral_path)
    ax_now = _text(axial_path) or lat_now
    if not lat_now:
        return False
    for name, now in (("lateral", lat_now), ("axial", ax_now)):
        v = _view(block, name)
        if name == "axial" and v is None:
            v = _view(block, "lateral")
        rec = _text(v.get("path")) if v else ""
        if not rec:
            return False
        try:
            if _path_key(rec, ws) != _path_key(now, ws):
                return False
        except (TypeError, ValueError):
            return False
    return True


# ---------------------------------------------------------------------------
# For the details window (generic: a key no code has heard of still shows)
# ---------------------------------------------------------------------------


def flatten_items(mapping, prefix: str = "") -> List[Tuple[str, Any]]:
    """``[(dotted key, value)]`` in the mapping's order, nested dicts
    flattened as ``a.b`` (an empty dict stays one item); lists are values."""
    out: List[Tuple[str, Any]] = []
    if not isinstance(mapping, dict):
        return out
    for k, v in mapping.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict) and v:
            out.extend(flatten_items(v, prefix=key + "."))
        else:
            out.append((key, v))
    return out


def display_value(value, *, empty: str) -> str:
    """One value as the window shows it."""
    if value is None or value == "" or value == [] or value == {}:
        return empty
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(display_value(v, empty=empty) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)
