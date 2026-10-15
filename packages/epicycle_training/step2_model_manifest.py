"""Step 2 per-bucket-round model manifest (schema 2.1) writer.

With the Phase
B.2 flat-layout shift, each Step 2 trained model bundle (post-B.2:
``artifacts/step2/runs/{run_id}/{bucket}/round_NN/model/``; pre-B.2:
``artifacts/step2/runs/{run_id}/{bucket}/models/round_NN/``) gets a
``manifest.json`` capturing:

* Identity: run_id, bucket, round_number, backend
* File pointers: weights, loss_train, loss_val, training_config
  (all Cellpose-driven filenames after the lift moves them up)
* Init provenance: scratch / inherited / cross_bucket / step3_refined
  / browse / pretrained — with mutually-exclusive groups.
* Training stats: final loss, n_pairs, epochs, batch_size,
  channels_used + snapshot

Phase B.3 (2026-05-25) schema 2.1 ADDITIVE additions:
* training.train_pair_labels (list[str]): slice labels actually trained
* training.train_pair_uris (list[dict]): per-pair URIs (basenames —
  relative to `accepted/` in the SIBLING round dir, NOT absolute).
  Resolved as ``manifest.parent.parent / "accepted" / basename``.
* training.accepted_dir_checksum (str | None): reserved slot
* training.random_seed (int | None): reserved slot
* timing.{started_at, finished_at, duration_seconds}
* environment.{cellpose_version, torch_version, python_version,
  device, platform}

Schema 2.0 → 2.1 is purely additive — pre-B.3 manifests still read
via `ACCEPTED_SCHEMA_VERSIONS`.  Permanent 2.0/2.1 coexistence is
the design intent; no migration tool required (per B.3 audit 2 Q10).

The manifest is **FLAT per bucket-round** (Step 2's per-bucket-per-
round independence — different from Step 3's joint refinement bundle
which nests lateral+axial sub-dicts in one manifest).

This module owns the data shape + a writer.  Phase 6 sub-C wired it
into the post-train hook after the Cellpose lift completes; Phase
B.3 wires the new reproducibility fields.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


SCHEMA_VERSION = "2.1"

# Schema 2.0 → 2.1 (2026-05-25) — additive, no breaking changes:
# - training.train_pair_labels (list[str])
# - training.train_pair_uris (list[{slice_label, img_path, mask_path}])
# - training.accepted_dir_checksum (str | None)
# - training.random_seed (int | None)
# - timing.started_at / finished_at / duration_seconds (str/str/float)
# - environment.cellpose_version / torch_version / python_version / device / platform (all str | None)
# Old (2.0) readers ignore unknown keys; new (2.1) readers `.get()`
# default-None for missing keys when reading 2.0 manifests.

# Accepted schema versions for read-path tolerance.  Writers always
# emit current SCHEMA_VERSION; readers tolerate any in this set.
ACCEPTED_SCHEMA_VERSIONS = frozenset({"2.0", "2.1"})

VALID_BUCKETS = ("lateral", "axial")

# Allowed init.mode enum.  Mutually-exclusive groups:
#   scratch        - no init source
#   inherited      - within-workspace cross-run fork
#   cross_bucket   - within-run cross-bucket init
#   step3_refined  - cross-step within-workspace handoff (Phase 7B)
#   browse         - arbitrary external file
#   pretrained     - Cellpose pretrained model name (e.g. "cyto3")
VALID_INIT_MODES = (
    "scratch", "inherited", "cross_bucket",
    "step3_refined", "browse", "pretrained",
)


def _validate_bucket(bucket: str) -> None:
    if bucket not in VALID_BUCKETS:
        raise ValueError(
            f"Invalid bucket {bucket!r}; expected one of {VALID_BUCKETS}"
        )


def _validate_init_mode(mode: str) -> None:
    if mode not in VALID_INIT_MODES:
        raise ValueError(
            f"Invalid init.mode {mode!r}; expected one of {VALID_INIT_MODES}"
        )


def build_step2_model_manifest(
    *,
    run_id: str,
    bucket: str,
    round_number: int,
    source_workspace: Union[str, Path],
    weights_filename: str = "weights",
    init_mode: str = "scratch",
    init_path: Optional[str] = None,
    init_combo_text: str = "",
    final_loss_train: Optional[float] = None,
    final_loss_val: Optional[float] = None,
    n_train_pairs: int = 0,
    n_val_pairs: int = 0,
    n_epochs: int = 0,
    batch_size: int = 0,
    channels_used: Optional[List[str]] = None,
    channels_snapshot: Optional[List[str]] = None,
    inherited_from_run: Optional[str] = None,
    inherited_from_run_bucket: Optional[str] = None,
    inherited_from_run_round_id: Optional[str] = None,
    inherited_from_bucket: Optional[str] = None,
    inherited_from_step3_bundle: Optional[str] = None,
    inherited_from_step3_bucket: Optional[str] = None,
    inherited_from_external_path: Optional[str] = None,
    inherited_from_external_label: Optional[str] = None,
    backend: str = "cellpose",
    created_at: Optional[str] = None,
    labels_format: str = "tiff",
    # Schema 2.1 (2026-05-25) — additive reproducibility fields.
    # All optional; build still produces a valid manifest when any are None.
    train_pair_labels: Optional[List[str]] = None,
    train_pair_uris: Optional[List[Dict[str, str]]] = None,
    accepted_dir_checksum: Optional[str] = None,
    random_seed: Optional[int] = None,
    timing_started_at: Optional[str] = None,
    timing_finished_at: Optional[str] = None,
    timing_duration_seconds: Optional[float] = None,
    env_cellpose_version: Optional[str] = None,
    env_torch_version: Optional[str] = None,
    env_python_version: Optional[str] = None,
    env_device: Optional[str] = None,
    env_platform: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a Step 2 per-bucket-round model manifest dict (schema 2.1).

    Returns a dict ready to write to
    ``artifacts/step2/runs/{run_id}/{bucket}/models/round_NN/manifest.json``.

    File pointers are RELATIVE to the manifest's parent directory —
    a reader resolves them as ``manifest.parent / weights_path``.
    This keeps the bundle portable across workspace moves.

    Loss + training_config filenames follow the Cellpose-driven naming
    (``{weights_filename}_train_loss.npy``,
    ``{weights_filename}_val_loss.npy``,
    ``{weights_filename}_training_config.json``).  The lift function
    in ``cellpose_lift.py`` moves these from Cellpose's auto-inserted
    ``models/`` subdir up to the bundle root WITHOUT renaming, so the
    manifest's path fields are accurate post-lift.

    Args:
        run_id: Slug of the parent run (e.g., "default", "nuclei_v1").
            Validated via ``_paths.validate_run_id``.
        bucket: "lateral" or "axial".
        round_number: Round index (non-negative int).
        source_workspace: Absolute path to the workspace this bundle
            was trained in (informational; readers don't depend on it
            being current).
        weights_filename: The name Cellpose was told to use as
            ``model_name``.  For Step 2: "weights".  Determines the
            filenames of all the lift output.
        init_mode: One of ``VALID_INIT_MODES``.
        init_path: Relative or absolute path to the init source weights
            (None for ``init_mode="scratch"`` or ``"pretrained"``).
        init_combo_text: Human-readable label for the init source as
            shown in the UI picker (e.g., "From lateral round 2").
        final_loss_train / final_loss_val: final epoch losses.
        n_train_pairs / n_val_pairs / n_epochs / batch_size: training
            metadata.
        channels_used / channels_snapshot: list of channel names
            used during training + snapshot for cross-workspace match.
        inherited_from_* fields: provenance for the init source.
            Mutually-exclusive GROUPS — at most ONE
            group has non-null values per manifest.  (See validation
            note below.)
        backend: Backend name; "cellpose" today, reserved field for
            future Omnipose / StarDist.
        created_at: ISO timestamp; defaults to now (UTC).
        labels_format: Reserved for future format-agnostic readers.

    Returns:
        Dict with all the fields.

    Raises:
        ValueError: if bucket / init_mode / round_number invalid, or
            if mutually-exclusive inheritance fields are set across
            groups.
    """
    _validate_bucket(bucket)
    _validate_init_mode(init_mode)
    if not isinstance(round_number, int) or round_number < 0:
        raise ValueError(
            f"round_number must be non-negative int; got {round_number!r}"
        )

    # Mutually-exclusive group check.  Each group is
    # a set of field names that must be EITHER all-null OR set-together
    # (for paired fields).  Across groups, at most one may be non-null.
    #
    # 2026-05-26 audit H2 fix: `inherited_from_run_bucket` +
    # `inherited_from_run_round_id` joined the within_workspace_cross_run
    # group as recoverable provenance for the source run's bucket +
    # round.  Pre-fix, only `inherited_from_run` (the slug) was
    # captured; bucket+round survived only in human-readable
    # `init_combo_text` which downstream readers couldn't parse.
    groups = {
        "within_workspace_cross_run": [
            inherited_from_run,
            inherited_from_run_bucket,
            inherited_from_run_round_id,
        ],
        "within_run_cross_bucket": [inherited_from_bucket],
        "cross_step": [inherited_from_step3_bundle, inherited_from_step3_bucket],
        "cross_workspace_external": [inherited_from_external_path,
                                     inherited_from_external_label],
    }
    non_null_groups = [
        name for name, fields in groups.items()
        if any(f is not None for f in fields)
    ]
    if len(non_null_groups) > 1:
        raise ValueError(
            f"inherited_from_* fields are mutually exclusive across "
            f"groups; got multiple non-null: {non_null_groups}"
        )

    # Cross-step group: both fields must be set together
    if (inherited_from_step3_bundle is None) != (inherited_from_step3_bucket is None):
        raise ValueError(
            "inherited_from_step3_bundle + inherited_from_step3_bucket "
            "must be set together as a pair (cross-step provenance)"
        )
    if (inherited_from_external_path is None) != (inherited_from_external_label is None):
        raise ValueError(
            "inherited_from_external_path + inherited_from_external_label "
            "must be set together as a pair (external provenance)"
        )

    # 2026-05-26 audit (Commit 4 H2 follow-up): within_workspace_cross_run
    # group co-population invariant.  Bucket/round_id are paired with
    # `inherited_from_run` — they describe a subset of that run's
    # trained rounds.  Either:
    #   * all three None (no within-workspace inheritance)
    #   * inherited_from_run set, bucket/round_id None (fork case —
    #     "we don't know which specific round")
    #   * inherited_from_run set + bucket + round_id (full provenance)
    # Bucket or round_id WITHOUT run is orphan provenance ("inherited
    # from bucket axial round_03 of [no run]") — reject at build time
    # to prevent silent corruption.
    if inherited_from_run is None and (
        inherited_from_run_bucket is not None
        or inherited_from_run_round_id is not None
    ):
        raise ValueError(
            "inherited_from_run_bucket / inherited_from_run_round_id "
            "require inherited_from_run to be set (orphan provenance "
            "is rejected — see schema 2.0 §within_workspace_cross_run)"
        )

    if created_at is None:
        created_at = datetime.now(timezone.utc).isoformat()

    # File pointers — relative to manifest.json (the bundle root).
    # 2026-05-26 audit C-4 fix: use Cellpose's actual on-disk val-loss
    # filename `_val_loss.npy` (per `cellpose_trainer.py:237`), NOT the
    # originally planned `_validation_loss.npy`.  Step 3's bundle writer
    # uses the same `_val_loss.npy` convention.  Pre-fix, the manifest
    # claimed a file that never existed -> any future reader hit
    # FileNotFoundError.
    weights_path = weights_filename
    loss_train_path = f"{weights_filename}_train_loss.npy"
    # Release-hardening 1b#7 (2026-09-07): only advertise the val-loss file
    # when a validation loss actually exists.  Step 2 never passes
    # `val_data`, so `CellposeTrainer.fit` writes no `_val_loss.npy` — a
    # manifest pointing at it sent every reader to a FileNotFoundError.
    loss_val_path = (
        f"{weights_filename}_val_loss.npy" if final_loss_val is not None else None
    )
    training_config_path = f"{weights_filename}_training_config.json"

    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "bucket": bucket,
        "round_number": round_number,
        "backend": backend,
        "created_at": created_at,
        "source_workspace": str(source_workspace),

        # File pointers (Cellpose-driven, post-lift)
        "weights_filename": weights_filename,
        "weights_path": weights_path,
        "loss_train_path": loss_train_path,
        "loss_val_path": loss_val_path,
        "training_config_path": training_config_path,

        # Init provenance
        "init": {
            "mode": init_mode,
            "path": init_path,
            "init_combo_text": init_combo_text,
        },
        "inherited_from_run": inherited_from_run,
        "inherited_from_run_bucket": inherited_from_run_bucket,
        "inherited_from_run_round_id": inherited_from_run_round_id,
        "inherited_from_bucket": inherited_from_bucket,
        "inherited_from_step3_bundle": inherited_from_step3_bundle,
        "inherited_from_step3_bucket": inherited_from_step3_bucket,
        "inherited_from_external_path": inherited_from_external_path,
        "inherited_from_external_label": inherited_from_external_label,

        # Training stats
        "final_loss": {
            "train": final_loss_train,
            "val": final_loss_val,
        },
        "training": {
            "n_train_pairs": n_train_pairs,
            "n_val_pairs": n_val_pairs,
            "n_epochs": n_epochs,
            "batch_size": batch_size,
            "channels_used": channels_used or [],
            # Schema 2.1 (2026-05-25): pair-identity + reproducibility.
            # Empty default keeps shape stable for 2.0 readers.
            "train_pair_labels": list(train_pair_labels) if train_pair_labels else [],
            "train_pair_uris": list(train_pair_uris) if train_pair_uris else [],
            "accepted_dir_checksum": accepted_dir_checksum,
            "random_seed": random_seed,
        },
        "channels_snapshot": channels_snapshot or [],

        # Schema 2.1 (2026-05-25): timing + environment provenance.
        # Top-level for easy grepping in cross-workspace comparisons.
        "timing": {
            "started_at": timing_started_at,
            "finished_at": timing_finished_at,
            "duration_seconds": timing_duration_seconds,
        },
        "environment": {
            "cellpose_version": env_cellpose_version,
            "torch_version": env_torch_version,
            "python_version": env_python_version,
            "device": env_device,
            "platform": env_platform,
        },

        "labels_format": labels_format,
    }

    return manifest


def write_step2_model_manifest(
    manifest_path: Union[str, Path],
    manifest: Dict[str, Any],
    *,
    overwrite: bool = True,
) -> Path:
    """Atomically write a Step 2 model manifest to disk.

    Validates the manifest dict matches schema 2.0 before writing
    (schema_version field present + correct, required fields present).

    Args:
        manifest_path: Destination ``manifest.json`` path (typically
            from ``_paths.model_manifest_path(...)``).
        manifest: The manifest dict (typically from
            :func:`build_step2_model_manifest`).
        overwrite: If False, refuses to overwrite an existing file
            (raises ``FileExistsError``).  Default True since the
            normal flow is post-train-overwrite.

    Returns:
        The path written.

    Raises:
        ValueError: if manifest doesn't conform to schema 2.0.
        FileExistsError: if ``overwrite=False`` and file exists.
    """
    manifest_path = Path(manifest_path)
    _validate_manifest_dict(manifest)

    if manifest_path.exists() and not overwrite:
        raise FileExistsError(
            f"Manifest already exists at {manifest_path} and overwrite=False"
        )

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write via temp file + rename
    tmp_path = manifest_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    tmp_path.replace(manifest_path)
    return manifest_path


def read_step2_model_manifest(manifest_path: Union[str, Path]) -> Dict[str, Any]:
    """Read + validate a Step 2 model manifest from disk.

    Raises:
        FileNotFoundError: if manifest doesn't exist.
        ValueError: if manifest is malformed JSON or fails schema validation.
    """
    manifest_path = Path(manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Manifest {manifest_path} is malformed JSON: {e}") from e
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest {manifest_path} is not a JSON object")
    _validate_manifest_dict(manifest)
    return manifest


def _validate_manifest_dict(manifest: Dict[str, Any]) -> None:
    """Lightweight schema check — catches obviously-malformed manifests
    without enforcing every field's type.  Future schema bumps can
    extend this."""
    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must be dict; got {type(manifest).__name__}")
    schema = manifest.get("schema_version")
    # Phase B.3 (2026-05-25): accept BOTH 2.0 and 2.1.  Writers always
    # emit current SCHEMA_VERSION; readers tolerate both during the
    # 2.0→2.1 migration window since the bump is purely additive.
    if schema not in ACCEPTED_SCHEMA_VERSIONS:
        raise ValueError(
            f"Manifest schema_version must be in {sorted(ACCEPTED_SCHEMA_VERSIONS)}; "
            f"got {schema!r}"
        )
    required = ("run_id", "bucket", "round_number", "backend",
                "weights_filename", "weights_path", "init")
    missing = [k for k in required if k not in manifest]
    if missing:
        raise ValueError(f"Manifest missing required fields: {missing}")
    if manifest["bucket"] not in VALID_BUCKETS:
        raise ValueError(
            f"Manifest bucket must be in {VALID_BUCKETS}; got {manifest['bucket']!r}"
        )


__all__ = [
    "SCHEMA_VERSION",
    "ACCEPTED_SCHEMA_VERSIONS",
    "VALID_BUCKETS",
    "VALID_INIT_MODES",
    "build_step2_model_manifest",
    "write_step2_model_manifest",
    "read_step2_model_manifest",
]
