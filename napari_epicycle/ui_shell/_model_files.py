"""What a model FILE says about itself, read without Qt (brick R6a).

Lifted verbatim from ``segmentation/_step4_models.py`` so Step 3's model
provenance can read the same facts Step 4 does without importing a Step 4
module: the file's fingerprint ``{size, mtime_ns}``, and
the Cellpose channel codes a model was trained with, read from the training
config beside its weights.  ``_step4_models`` re-exports the three names, so
every Step 4 caller and test is unchanged (the same function objects;
``tests/segmentation/test_step4_lift_identity.py``).

Qt-free, napari-free, and imports nothing from a step package
(``tests/ui_shell/test_recon_shared_modules_imports.py``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional


def file_fingerprint(path) -> Optional[Dict[str, int]]:
    """``{"size", "mtime_ns"}`` of *path*, or None when it cannot be read."""
    try:
        st = os.stat(str(path))
    except (OSError, TypeError, ValueError):
        return None
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def _training_config_files(weights_path) -> list:
    """The training configs that may sit beside *weights_path*: Step 3's
    refined bundle (``{bucket}_training_config.json``) and Step 2's round
    model (``{weights name}_training_config.json``, e.g.
    ``Cell_lateral_round_01_training_config.json``).

    Brick B audit (M1, M2): the file named for THESE weights comes first (a
    refined bundle's ``axial`` weights read ``axial_training_config.json``,
    not the lateral one); the two bucket names and any other
    ``*_training_config.json`` in the folder ONLY when the folder holds a
    single weights file -- with both buckets side by side a missing config
    must not be filled with the other bucket's."""
    wp = Path(str(weights_path))
    folder = wp.parent
    out = [folder / f"{wp.name}_training_config.json"]
    try:
        weights = [p for p in folder.iterdir()
                   if p.is_file() and p.suffix.lower() != ".json"]
        if len(weights) <= 1:
            out += [folder / f"{b}_training_config.json"
                    for b in ("lateral", "axial")]
            out += sorted(folder.glob("*_training_config.json"))
    except OSError:
        pass
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def read_trained_channels(weights_path) -> Optional[list]:
    """The Cellpose channel codes ``[c1, c2]`` a model was TRAINED with,
    from the training config beside the weights: Step 3's
    ``{bucket}_training_config.json`` (the file Step 3's
    ``_try_sync_prediction_channels_from_bundle`` reads) or Step 2's
    ``{model}_training_config.json`` (owner live test 2026-09-29: every
    Step 2 model read as unknown).  ``None`` when nothing can be read.
    Never raises."""
    if not weights_path:
        return None
    try:
        cfgs = _training_config_files(weights_path)
    except Exception:      # noqa: BLE001
        return None
    for cfg in cfgs:
        try:
            if not cfg.is_file():
                continue
            data = json.loads(cfg.read_text(encoding="utf-8"))
            ch = (data.get("hyperparameters") or {}).get("channels")
            if isinstance(ch, list) and len(ch) == 2:
                return [int(ch[0]), int(ch[1])]
        except (OSError, ValueError, TypeError, AttributeError):
            continue
    return None
