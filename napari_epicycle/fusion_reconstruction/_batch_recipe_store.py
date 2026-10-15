"""Persisted store for named Batch-Refine recipes (workspace-level JSON).

Pure helpers — **no Qt, no napari**.  Backs the "Saved recipes" feature of
the Step-3 Batch Refine dialog: the user names + persists a recipe (an
ordered list of geometry-only steps) to the workspace, then reloads it into
the dialog's recipe builder later.

Storage layout
--------------
A single JSON object mapping ``name -> recipe_json`` at the workspace path
``<workspace>/artifacts/step3/batch_refine_recipes.json`` (resolved by
``fusion_reconstruction/_paths.batch_refine_recipes_path``).  ``recipe_json``
is the round-trippable list-of-dicts produced by
``packages.epicycle_annotator.batch_refine.recipe_to_json``::

    {
      "Smooth + Fill": [
        {"method": "sdf", "params": {"sigma_z": 2.0, "sigma_xy": 1.5, ...}},
        {"method": "fill_holes", "params": {"axis": "3D"}}
      ],
      ...
    }

Contracts
---------
* **Atomic writes** — tempfile in the same directory + ``os.replace`` so a
  crash / concurrent writer never leaves a half-written (corrupt) file.
* **Bad-JSON tolerant reads** — ``load_named_recipes`` returns ``{}`` for a
  missing / empty / corrupt file instead of raising, so a damaged store can
  never strand the dialog.  (Per-recipe validity is the dialog's concern at
  reload time — the store is format-agnostic about the step dicts.)
* **Name collisions overwrite** — ``save_named_recipe`` replaces an existing
  entry of the same name (the dialog warns the user before calling).
* ``value_type`` (keyword-only, default ``list``) -- the Step-3 Refine
  chain recipes (``refine_chain_recipes.json``) store a DICT per name.  The default drops every non-list entry, so
  a default-type save on that file would load nothing and write back only
  its own entry (every other recipe gone) and a default-type delete would
  be a no-op.  The Refine callers pass ``value_type=dict``; every Batch
  call is byte-identical.
* ``load_named_recipes_checked`` tells a missing file (fine) from a present
  but unreadable one (``problem``), so a caller can refuse to overwrite a
  file the owner might still recover by looking at it.
* The store does NOT mkdir the workspace root, but it DOES create the
  immediate parent dir of the recipes file if missing (the
  ``artifacts/step3/`` sub-tree), so the first save on a fresh workspace
  succeeds.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Dict, List, Union

_PathLike = Union[str, Path]


def _coerce(path: _PathLike) -> Path:
    return Path(path)


def _read(path: Path):
    """``(data, problem)`` -- ``data`` is the parsed object or ``None``."""
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, FileNotFoundError):
        return None, None
    if not raw.strip():
        return None, None
    try:
        data = json.loads(raw)
    except (ValueError, json.JSONDecodeError):
        return None, UNREADABLE
    if not isinstance(data, dict):
        return None, UNREADABLE
    return data, None


#: The one sentence for a present-but-unparseable (or non-object) file.
UNREADABLE = "The recipes file could not be read."


def load_named_recipes(path: _PathLike, *, value_type=list) -> Dict[str, List[dict]]:
    """Read the named-recipe store at ``path``.

    Returns a ``{name: recipe_json}`` dict.  A missing / empty / corrupt /
    non-object file yields ``{}`` (never raises) so a damaged store can't
    strand the dialog.  Entries whose value is not a ``value_type`` (a list,
    by default) are dropped defensively.
    """
    data, _problem = _read(_coerce(path))
    if data is None:
        return {}
    out: Dict[str, List[dict]] = {}
    for name, recipe in data.items():
        if isinstance(recipe, value_type):
            out[str(name)] = recipe
    return out


def load_named_recipes_checked(path: _PathLike, *, value_type):
    """``(store, problem)`` -- ``problem`` is ``None``, or
    :data:`UNREADABLE` for a present-but-unparseable / non-object file.

    Unlike :func:`load_named_recipes`, an entry of the wrong type is KEPT
    (as its raw value) so the caller can list it as unreadable and offer to
    delete it.  ``value_type`` is keyword-only and has no default: the caller
    says which store it is reading.
    """
    del value_type  # every entry is kept; the caller judges each value
    data, problem = _read(_coerce(path))
    if data is None:
        return {}, problem
    return {str(k): v for k, v in data.items()}, None


def _load_for_write(p: Path, value_type) -> dict:
    """The store a save / delete rewrites.

    The default (Batch, ``list``) keeps its historical behaviour exactly:
    non-list entries are dropped.  Any other type keeps EVERY entry, so an
    entry of the wrong type is neither lost by a save of another name nor
    impossible to delete by its own name (the Refine recipes rely on this).
    """
    if value_type is list:
        return load_named_recipes(p)
    return load_named_recipes_checked(p, value_type=value_type)[0]


def _atomic_write_json(path: Path, payload: dict) -> None:
    """Atomically write ``payload`` as pretty JSON to ``path``.

    tempfile in the same dir + fsync + ``os.replace``.  Creates the parent
    directory if missing.  On any failure the temp file is cleaned up and
    the error re-raised (the existing store is left untouched).
    """
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False)
    fd, tmp_path = tempfile.mkstemp(
        prefix=".batch_recipes_", suffix=".json.tmp", dir=str(parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
            f.flush()
            try:
                os.fsync(f.fileno())
            except OSError:
                pass
        os.replace(tmp_path, str(path))
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def save_named_recipe(
    path: _PathLike, name: str, recipe_json: List[dict], *, value_type=list
) -> Dict[str, List[dict]]:
    """Persist ``recipe_json`` under ``name`` in the store at ``path``.

    Reads the current store (tolerant), upserts the entry (collisions
    overwrite), and atomically rewrites the file.  Returns the updated
    ``{name: recipe_json}`` dict.

    Raises ``ValueError`` for an empty name or a non-list recipe — the
    store refuses to persist garbage.
    """
    name = (name or "").strip()
    if not name:
        raise ValueError("Recipe name must be a non-empty string.")
    if value_type is list:
        if not isinstance(recipe_json, (list, tuple)):
            raise ValueError(
                f"recipe_json must be a list of step dicts, got "
                f"{type(recipe_json).__name__}."
            )
        value = list(recipe_json)
    else:
        if not isinstance(recipe_json, value_type):
            raise ValueError(
                f"recipe_json must be a {value_type.__name__}, got "
                f"{type(recipe_json).__name__}."
            )
        value = recipe_json
    p = _coerce(path)
    store = _load_for_write(p, value_type)
    store[name] = value
    _atomic_write_json(p, store)
    return store


def delete_named_recipe(
    path: _PathLike, name: str, *, value_type=list
) -> Dict[str, List[dict]]:
    """Remove ``name`` from the store at ``path`` (no-op if absent).

    Returns the updated ``{name: recipe_json}`` dict.  A delete of the last
    entry leaves an empty ``{}`` JSON object (not a missing file).
    """
    p = _coerce(path)
    store = _load_for_write(p, value_type)
    if name in store:
        del store[name]
        _atomic_write_json(p, store)
    return store


__all__ = [
    "UNREADABLE",
    "load_named_recipes",
    "load_named_recipes_checked",
    "save_named_recipe",
    "delete_named_recipe",
]
