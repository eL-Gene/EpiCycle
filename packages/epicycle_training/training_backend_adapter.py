"""Backend adapter for real Cellpose training and inference.

Imports CellposeTrainer and TrainingConfig from the local monorepo package.
No external sibling repo dependency required.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import tifffile

from .config import TrainingConfig
from .cellpose_trainer import CellposeTrainer


# ---------------------------------------------------------------------------
# F3.1 (2026-05-27 audit follow-up): archive-instead-of-rmtree for prior
# training state.  Pre-fix: re-training a round did `shutil.rmtree` on the
# prior weights up front.  If the user cancelled during epoch 0 (before
# new weights were written), both the prior weights AND new weights were
# gone — manifest.json then pointed at deleted files.
#
# Post-fix: prior weights move to a sibling `.pre_train_archive/<ts>_<round>/`
# dir.  On training success the archive is deleted.  On training failure
# (incl. TrainingAbortedError now that it's a BaseException) the archive is
# restored before the exception propagates.  Net effect: cancel mid-train
# preserves the prior round's weights.
# ---------------------------------------------------------------------------


def _archive_prior_training_state(
    save_path: Path,
    model_name: str,
    *,
    log_warn_cb=None,
) -> Optional[Path]:
    """Move any prior training state (Cellpose `models/` subdir + companion
    files matching `{model_name}*` at the bundle root) into a timestamped
    archive sibling directory.

    Returns the archive path on success, or None if there was nothing to
    archive (fresh round).  The caller MUST call either
    :func:`_delete_archive` on success or :func:`_restore_prior_training_state`
    on failure to keep the on-disk state coherent.

    Bundle 2 audit fix #4 (2026-05-27): if shutil.move fails (e.g.
    Windows file lock from a prior crashed training, or antivirus
    scanning), the helper falls back to rmtree.  Pre-fix this was a
    silent print() — user had no idea their prior weights got
    discarded.  Now we surface to a caller-provided log_warn_cb.  If
    log_warn_cb is None we still print() for backward compat.
    """
    import shutil
    import time

    def _warn(msg):
        # Always print for console diagnostics; ALSO surface to UI
        # log if the caller provided a callback.
        print(f"[STEP2-TRAIN] {msg}")
        if log_warn_cb is not None:
            try:
                log_warn_cb(msg)
            except Exception:
                pass

    if not save_path.is_dir():
        return None
    # Anything to archive?
    cellpose_subdir = save_path / "models"
    companions = [p for p in save_path.glob(f"{model_name}*") if p.is_file()]
    if not cellpose_subdir.exists() and not companions:
        return None

    archive_root = save_path.parent / ".pre_train_archive"
    archive_root.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%dT%H%M%S")
    archive_path = archive_root / f"{ts}_{save_path.name}_{model_name}"
    archive_path.mkdir(parents=True, exist_ok=True)

    # Archive the Cellpose subdir if present.
    if cellpose_subdir.exists():
        # 2026-09-07 (console-encoding hazard): no print inside this try —
        # a print can RAISE and this except DISCARDS the prior weights.
        _archived_models = None
        try:
            shutil.move(str(cellpose_subdir), str(archive_path / "models"))
            _archived_models = archive_path / "models"
        except Exception as e:
            # Bundle 2 audit fix #4: surface to UI.  This means the
            # prior weights got discarded due to a filesystem
            # constraint (lock, AV, permission).  User needs to know.
            _warn(
                f"WARNING: prior weights at {cellpose_subdir} could "
                f"not be archived ({type(e).__name__}: {e}) — they "
                f"will be DISCARDED to allow training to proceed.  "
                f"If this round fails, the prior round's weights are "
                f"NOT recoverable from .pre_train_archive/."
            )
            # Fall back to old behavior so we don't leave the user
            # in a worse state than pre-fix.
            shutil.rmtree(cellpose_subdir, ignore_errors=True)
        if _archived_models is not None:
            print(f"[STEP2-TRAIN] F3.1 archived Cellpose subdir -> {_archived_models}")

    # Archive companion files at bundle root.
    companion_archive = archive_path / "companions"
    if companions:
        companion_archive.mkdir(parents=True, exist_ok=True)
        for stale in companions:
            try:
                shutil.move(str(stale), str(companion_archive / stale.name))
            except Exception as e:
                _warn(
                    f"companion file {stale.name} could not be "
                    f"archived ({type(e).__name__}: {e}) — discarded."
                )
                try:
                    stale.unlink()
                except OSError:
                    pass
        print(f"[STEP2-TRAIN] F3.1 archived {len(companions)} companion(s) -> {companion_archive}")

    return archive_path


def _delete_archive(archive_path: Path) -> None:
    """Delete the archive dir.  Called on training success — new weights
    are good, prior weights can be discarded."""
    import shutil
    if archive_path is None or not archive_path.exists():
        return
    try:
        shutil.rmtree(archive_path, ignore_errors=False)
        print(f"[STEP2-TRAIN] F3.1 success: deleted archive {archive_path}")
        # Best-effort: remove the .pre_train_archive parent if empty.
        try:
            if not any(archive_path.parent.iterdir()):
                archive_path.parent.rmdir()
        except OSError:
            pass
    except Exception as e:
        print(f"[STEP2-TRAIN] F3.1 archive delete FAILED "
              f"({type(e).__name__}: {e}) — orphan archive at {archive_path}")


def _restore_prior_training_state(save_path: Path, archive_path: Path) -> None:
    """Move archived state back into save_path.  Called on training
    failure / cancellation — the new weights are gone or partial, so we
    restore the prior round's weights to maintain the cumulative chain.
    """
    import shutil
    if archive_path is None or not archive_path.exists():
        return

    # Restore Cellpose subdir.
    archived_models = archive_path / "models"
    # 2026-09-07 (console-encoding hazard): this whole function runs from
    # an `except BaseException:` handler, so a raising print here would
    # REPLACE TrainingAbortedError with UnicodeEncodeError.  Diagnostics
    # are collected and emitted after the restore + archive cleanup.
    _restore_diag = []
    if archived_models.exists():
        dst_models = save_path / "models"
        try:
            # If a partial models/ exists from the failed run, blow it
            # away first.
            if dst_models.exists():
                shutil.rmtree(dst_models, ignore_errors=True)
            shutil.move(str(archived_models), str(dst_models))
            _restore_diag.append(
                f"[STEP2-TRAIN] F3.1 restored Cellpose subdir <- {dst_models}"
            )
        except Exception as e:
            _restore_diag.append(
                f"[STEP2-TRAIN] F3.1 restore of models/ FAILED "
                f"({type(e).__name__}: {e}) - prior weights may be lost"
            )

    # Restore companions at bundle root.
    archived_companions = archive_path / "companions"
    if archived_companions.exists():
        for archived_file in archived_companions.iterdir():
            if archived_file.is_file():
                try:
                    shutil.move(str(archived_file), str(save_path / archived_file.name))
                except Exception as e:
                    _restore_diag.append(
                        f"[STEP2-TRAIN] F3.1 restore of "
                        f"{archived_file.name} FAILED "
                        f"({type(e).__name__}: {e})"
                    )
        _restore_diag.append(
            f"[STEP2-TRAIN] F3.1 restored companions <- {save_path}"
        )

    # Clean up the now-empty archive container.
    try:
        shutil.rmtree(archive_path, ignore_errors=True)
        try:
            if not any(archive_path.parent.iterdir()):
                archive_path.parent.rmdir()
        except OSError:
            pass
    except Exception:
        pass

    # Emit the collected diagnostics only once the restore + cleanup are
    # done, so a raising print cannot skip either of them (nor replace
    # the TrainingAbortedError this handler is unwinding).
    for _line in _restore_diag:
        print(_line)


def scan_orphan_archives(workspace_root: Path) -> List[Dict[str, Any]]:
    """Bundle 2 audit fix #3 (2026-05-27): scan workspace for orphan
    ``.pre_train_archive/`` directories.

    Archives accumulate when training is interrupted by a process crash
    (OOM, segfault, force-kill) BEFORE ``_restore_prior_training_state``
    can fire its except-BaseException handler.  Without this scanner,
    the user has prior weights stuck in
    ``models/round_NN/.pre_train_archive/<ts>_*/`` with no UI to recover
    them — worse than pre-fix because the prior weights LOOK gone.

    The scanner is deliberately READ-ONLY.  Returns one dict per orphan
    with enough info for the caller to decide what to do (typically:
    log to UI, optionally auto-restore if canonical models/ is empty,
    or surface to user for manual recovery).

    Save-path resolution (Bundle 2 second-pass audit fix, 2026-05-27):
    ``_archive_prior_training_state`` places archives at
    ``save_path.parent / ".pre_train_archive" / "{ts}_{save_path.name}_{model_name}"``.
    Therefore the SAVE_PATH parent of the archive is
    ``archive_root.parent`` (not ``archive_root.parent.parent``), and
    the REAL save_path is reconstructed by parsing the dir-name format
    and joining: ``archive_root.parent / save_path_name``.

    Schema:
    [
        {
            "archive_path": Path,
            "save_path": Path,        # the original save_path
            "ts": str,                # archived timestamp
            "canonical_weights_present": bool,
            "model_name_hint": str | None,  # parsed from dir name
        },
        ...
    ]
    """
    out: List[Dict[str, Any]] = []
    if workspace_root is None:
        return out
    root = Path(workspace_root)
    if not root.is_dir():
        return out

    # Bundle 2 second-pass audit (HIGH #3): wrap rglob in try/except
    # so a single permission-denied / locked subdir doesn't kill the
    # entire scan.
    def _safe_rglob(p, pattern):
        """rglob, minus anything living under a dot-prefixed directory.

        Audit fix (2026-09-07): the caller AUTO-RESTORES what this finds, and
        the workspace now has dot-prefixed holding areas — `.trash/` (a run the
        user deleted) and `.pre_step2_flat_layout_*` (a migration backup).  An
        archive found in there would be "recovered" back into a run the user
        threw away, or out of a backup.  Only the archive directory's own
        dot-name is allowed.
        """
        try:
            candidates = list(p.rglob(pattern))
        except (PermissionError, OSError) as e:
            print(
                f"[STEP2-TRAIN] F3.1 orphan-scan rglob failed at "
                f"{p}: {type(e).__name__}: {e} — continuing"
            )
            return
        for candidate in candidates:
            # rglob results are always under `p`, so relative_to cannot
            # fail here - and falling back to the ABSOLUTE parts would blank
            # the whole scan for a workspace living under, say, a .config
            # directory.
            parts = candidate.relative_to(p).parts[:-1]
            if any(str(part).startswith(".") for part in parts):
                print(
                    f"[STEP2-TRAIN] orphan-scan skipping {candidate} "
                    f"(inside a dot-prefixed directory such as .trash/)"
                )
                continue
            yield candidate

    for archive_root in _safe_rglob(root, ".pre_train_archive"):
        try:
            if not archive_root.is_dir():
                continue
        except OSError:
            continue
        # save_path_parent is the directory CONTAINING save_path —
        # because _archive_prior_training_state placed archive_root at
        # `save_path.parent / .pre_train_archive`.
        save_path_parent = archive_root.parent

        # Bundle 2 second-pass audit (HIGH #3): protect iterdir on the
        # archive directory itself too.
        try:
            archive_entries = list(archive_root.iterdir())
        except (PermissionError, OSError) as e:
            print(
                f"[STEP2-TRAIN] F3.1 orphan-scan iterdir failed at "
                f"{archive_root}: {type(e).__name__}: {e} — continuing"
            )
            continue

        for entry in archive_entries:
            try:
                if not entry.is_dir():
                    continue
            except OSError:
                continue
            # Parse the canonical format `{ts}_{save_path.name}_{model_name}`.
            # We must split this carefully: ts is one segment (no
            # underscore inside the strftime format we use), save_path.name
            # may contain underscores (e.g. "round_00"), and model_name
            # may also contain them.  Strategy: take the first segment
            # as ts, then attempt to find the save_path subdirectory by
            # walking the remaining tokens until a directory exists
            # under save_path_parent.
            ts = ""
            save_path_name = ""
            model_name_hint = None
            try:
                head, _, tail = entry.name.partition("_")
                ts = head
                # Try longest-first prefix match: progressively peel
                # tokens off the START of `tail` and check if the
                # remaining looks like a known save_path.name.
                tokens = tail.split("_")
                for split_idx in range(len(tokens) - 1, 0, -1):
                    candidate_name = "_".join(tokens[:split_idx])
                    candidate_path = save_path_parent / candidate_name
                    try:
                        if candidate_path.is_dir():
                            save_path_name = candidate_name
                            model_name_hint = "_".join(tokens[split_idx:])
                            break
                    except OSError:
                        continue
                # Fallback: if no candidate matched (the original save_path
                # might have been deleted post-archive), assume the last
                # token is the model and the rest is save_path.name.
                if not save_path_name and len(tokens) >= 2:
                    save_path_name = "_".join(tokens[:-1])
                    model_name_hint = tokens[-1]
            except Exception:
                pass

            if not save_path_name:
                # Genuinely unparseable — surface with degraded info
                # so caller can still log the path.
                save_path = save_path_parent
            else:
                save_path = save_path_parent / save_path_name

            # Heuristic for "canonical weights present": either a
            # models/ subdir exists with files, OR a *.pth/.h5/.pt
            # file lives at save_path root (post-lift layout).
            canonical_present = False
            try:
                models_sub = save_path / "models"
                if models_sub.is_dir() and any(models_sub.iterdir()):
                    canonical_present = True
                if not canonical_present and save_path.is_dir():
                    for p in save_path.iterdir():
                        if p.is_file() and p.suffix in (".pth", ".h5", ".pt"):
                            canonical_present = True
                            break
            except OSError:
                pass

            out.append({
                "archive_path": entry,
                "save_path": save_path,
                "ts": ts,
                "canonical_weights_present": canonical_present,
                "model_name_hint": model_name_hint,
            })
    return out


def restore_orphan_archive(orphan: Dict[str, Any]) -> bool:
    """Bundle 2 audit fix #3: caller-driven recovery of one orphan.
    Wraps :func:`_restore_prior_training_state` with the orphan dict
    schema returned by :func:`scan_orphan_archives`.  Returns True on
    success, False on failure.
    """
    try:
        _restore_prior_training_state(
            Path(orphan["save_path"]),
            Path(orphan["archive_path"]),
        )
        return True
    except Exception as e:
        print(
            f"[STEP2-TRAIN] F3.1 orphan restore FAILED "
            f"({type(e).__name__}: {e}) — orphan={orphan.get('archive_path')}"
        )
        return False


def _slice_label_for_sample(sample) -> str:
    """Human-readable identifier for a training pair (release-hardening #14)."""
    try:
        label = str(sample.get("slice_label", "") or "").strip()
        if label:
            return label
        return (
            f"{sample.get('stack_id', '?')}_{sample.get('view', '?')}_"
            f"{sample.get('slice_index', '?')}"
        )
    except Exception:
        return "<unknown slice>"


def _report_min_mask_drops(*, masks, samples, min_train_masks, log_warn_cb=None):
    """Warn about pairs cellpose will drop for having too few masks (#15).

    Mirrors cellpose's own `nmasks < min_train_masks` rule (cellpose/train.py)
    using the label arrays already in memory, so the user learns WHICH slices
    left the training set instead of only seeing a smaller sample count.
    Returns the number of pairs that will be dropped.
    """
    try:
        threshold = int(min_train_masks or 0)
    except Exception:
        threshold = 0
    if threshold <= 0:
        return 0

    dropped = []
    for mask, sample in zip(masks, samples):
        try:
            n_masks = int(np.unique(np.asarray(mask)).size)
            if 0 in np.unique(np.asarray(mask)):
                n_masks -= 1
        except Exception:
            continue
        if n_masks < threshold:
            dropped.append((_slice_label_for_sample(sample), n_masks))

    if not dropped:
        return 0
    detail = ", ".join(f"{lbl} ({n})" for lbl, n in dropped)
    # Release-hardening 1b#7: if EVERY pair would be dropped, cellpose is
    # left with an empty training set and fails minutes later with an
    # opaque index error.  Say so now, in the user's own vocabulary.
    if len(dropped) == len(list(samples)):
        raise ValueError(
            f"All {len(dropped)} training pair(s) have fewer than "
            f"min_train_masks={threshold} objects, so cellpose would drop "
            f"every one and training would fail with an empty dataset. "
            f"Lower 'Min masks per image' or add pairs with more objects. "
            f"Pairs: {detail}"
        )
    msg = (
        f"[TRAIN] {len(dropped)} pair(s) have fewer than "
        f"min_train_masks={threshold} objects and will be DROPPED by "
        f"cellpose: " + detail
    )
    print(msg)
    if log_warn_cb is not None:
        try:
            log_warn_cb(msg)
        except Exception:
            pass
    return len(dropped)


def build_epicycle_training_backend():
    """Returns a backend callable compatible with TrainingManager.

    Returns None if cellpose is not installed.
    """
    try:
        from cellpose import models, train  # noqa: F401 — verify cellpose available
    except ImportError:
        return None

    def _backend(
        *,
        round_id: str,
        config: Dict[str, Any],
        working_dir: str,
        training_samples: List[Dict[str, Any]],
        view_type: str,
        pretrained_model: Optional[str] = None,
        epoch_callback=None,
        save_path_override: Optional[str] = None,
        weights_filename: Optional[str] = None,
        log_warn_cb=None,
    ) -> Dict[str, Any]:
        # Phase B.3 (2026-05-25): capture timing + environment + slice
        # labels for schema 2.1 reproducibility provenance.  Done at
        # the top of training so even short-circuit returns surface what
        # we know.
        import platform as _platform
        import sys as _sys
        from datetime import datetime as _dt, timezone as _tz

        # B.3.1 CRITICAL fix (2026-05-25): pass `_tz.utc` (the singleton
        # UTC instance) NOT `_tz` (the timezone CLASS).  datetime.now()
        # requires a tzinfo INSTANCE; passing the class raises
        # `TypeError: tzinfo argument must be None or of a tzinfo
        # subclass, not type 'type'`.  Pre-fix every real Cellpose
        # training crashed at the top of `_backend()`.  Tests passed
        # vacuously because they hand-passed string timestamps to the
        # manifest builder directly, never exercising the adapter path.
        timing_started_at = _dt.now(_tz.utc).isoformat()
        try:
            import cellpose as _cp
            env_cellpose_version = getattr(_cp, "version", None) or getattr(
                _cp, "__version__", None
            )
            if env_cellpose_version is not None:
                env_cellpose_version = str(env_cellpose_version)
        except Exception:
            env_cellpose_version = None
        try:
            import torch as _torch
            env_torch_version = str(_torch.__version__)
            _use_gpu = str(config.get("device", "GPU")).upper() == "GPU"
            if _use_gpu and _torch.cuda.is_available():
                try:
                    env_device = str(_torch.cuda.get_device_name(0))
                except Exception:
                    env_device = "cuda"
            else:
                env_device = "cpu"
        except Exception:
            # B.3.1 audit 3 LOW fix (2026-05-25): when torch import fails
            # entirely, we don't actually know what device training ran on.
            # Pre-fix mis-recorded as "cpu" when device=GPU was requested
            # but torch was missing — a lie.  Post-fix: None when torch
            # is unavailable, "cpu" only when device explicitly CPU AND
            # torch present.
            env_torch_version = None
            env_device = None
        env_python_version = ".".join(map(str, _sys.version_info[:3]))
        try:
            env_platform_str = _platform.platform()
        except Exception:
            env_platform_str = None

        # Capture slice labels for schema 2.1 `train_pair_labels`.
        # B.3.1 audit 1 HIGH fix (2026-05-25): canonical slice-label
        # format is `{stack_id}_{VIEW}_z{idx:04d}` (z-prefix) per
        # `_parse_slice_label` + `_canonicalize_slice_label`.  Pre-fix
        # this builder wrote `{stack}_{VIEW}_{idx:04d}` (no z), creating
        # provenance labels that don't match the on-disk filenames.
        train_pair_labels: List[str] = []
        train_pair_uris: List[Dict[str, str]] = []
        for sample in training_samples:
            sl = sample.get("slice_label")
            if not sl:
                try:
                    sl = (
                        f"{sample['stack_id']}_{str(sample['view']).upper()}_"
                        f"z{int(sample['slice_index']):04d}"
                    )
                except Exception:
                    sl = ""
            if sl:
                train_pair_labels.append(str(sl))
            # B.3.1 audit 3 HIGH fix (2026-05-25): record URIs as
            # filename-only (relative to `accepted/`).  Pre-fix stored
            # absolute paths, breaking the schema's relative-paths-only
            # convention (cf. `weights_path`, `loss_train_path`).  A
            # workspace moved across machines would have stale absolute
            # URIs.  Post-fix: store the basename only; readers resolve
            # via `manifest.parent.parent / "accepted" / basename` (the
            # manifest lives at `round_NN/model/manifest.json`, so two
            # levels up + accepted/).  Empty string preserved when source
            # had no URI (defensive).
            _img_p = str(sample.get("img_uri", ""))
            _msk_p = str(sample.get("mask_uri", ""))
            train_pair_uris.append({
                "slice_label": str(sl),
                "img_uri": Path(_img_p).name if _img_p else "",
                "mask_uri": Path(_msk_p).name if _msk_p else "",
            })

        imgs = []
        masks = []
        for sample in training_samples:
            img = tifffile.imread(sample["img_uri"])
            mask = tifffile.imread(sample["mask_uri"])
            # Release-hardening #14 (2026-09-07): validate the pair BEFORE
            # cellpose sees it.  Externally imported pairs bypass the
            # native compaction/persist path, so a mismatched image/mask
            # used to surface as an opaque cellpose/numpy error minutes
            # into training with no indication of WHICH slice was bad.
            _label = _slice_label_for_sample(sample)
            if getattr(img, "shape", None) is None or getattr(mask, "shape", None) is None:
                raise ValueError(
                    f"Training pair {_label!r}: image or mask could not be read"
                )
            if mask.ndim != 2:
                raise ValueError(
                    f"Training pair {_label!r}: mask must be 2-D (H, W), got "
                    f"shape {tuple(mask.shape)} from {sample.get('mask_uri')}"
                )
            if tuple(img.shape[-2:]) != tuple(mask.shape):
                raise ValueError(
                    f"Training pair {_label!r}: image/mask shape mismatch — "
                    f"image {tuple(img.shape)} (last two axes "
                    f"{tuple(img.shape[-2:])}) vs mask {tuple(mask.shape)}. "
                    f"img={sample.get('img_uri')} mask={sample.get('mask_uri')}"
                )
            imgs.append(img)
            masks.append(mask)

        channel1 = int(str(config.get("channel1", "0: gray")).split(":")[0].strip())
        channel2 = int(str(config.get("channel2", "0: none")).split(":")[0].strip())

        training_config = TrainingConfig(
            n_epochs=int(config.get("epochs", 500)),
            learning_rate=float(config.get("learning_rate", 1e-2)),
            batch_size=int(config.get("batch_size", 16)),
            weight_decay=float(config.get("weight_decay", 1e-5)),
            min_train_masks=int(config.get("min_masks_per_image", 0)),
            normalize=bool(config.get("normalize", True)),
            rescale=bool(config.get("rescale", True)),
            compute_flows=bool(config.get("compute_flows", True)),
            channels=[channel1, channel2],
        )

        # Release-hardening #15 (2026-09-07): cellpose drops training
        # images with fewer than `min_train_masks` objects and reports it
        # only on its own logger.  The count is cheap to derive from the
        # masks we just read, so surface it before training starts.
        _report_min_mask_drops(
            masks=masks,
            samples=training_samples,
            min_train_masks=training_config.min_train_masks,
            log_warn_cb=log_warn_cb,
        )

        trainer = CellposeTrainer(
            config=training_config,
            use_gpu=str(config.get("device", "GPU")).upper() == "GPU",
        )

        # Phase 6c (2026-05-26): canonical save layout.
        # - If save_path_override is provided (new multi-run layout):
        #     save_path = runs/{slug}/{bucket}/models/round_NN/
        #     model_name = weights_filename (caller-supplied; controller
        #     uses f"{run_id}_{bucket}" so the filename carries identity)
        #     -> post-train, call lift_cellpose_output() to flatten
        # - Legacy fallback: workspace top-level models/{round_id}/
        #     model_name = "{name}_{view}_{round}" (legacy naming)
        #     -> NO lift (preserves pre-restructure behavior)
        archive_path = None  # F3.1: set if we archived prior weights
        if save_path_override is not None:
            save_path = Path(save_path_override)
            # Caller-supplied filename carries identity; fallback to
            # "weights" if caller didn't pass one (back-compat).
            model_name = weights_filename or "weights"
            _use_new_layout = True
            # Bundle 2 (2026-05-27 audit follow-up F3.1): ARCHIVE prior
            # weights instead of rmtree-ing them.  Pre-fix: if user
            # cancelled during epoch 0 (before new weights were written),
            # both the prior weights AND new weights were gone.  Manifest
            # then pointed at deleted files.  Post-fix: prior weights
            # move to .pre_train_archive/.  On training success → archive
            # deleted.  On training failure / cancel → archive restored.
            archive_path = _archive_prior_training_state(
                save_path, model_name, log_warn_cb=log_warn_cb
            )
        else:
            model_name = f"{config.get('model_name', 'model')}_{view_type}_{round_id}"
            save_path = Path(working_dir) / "models" / round_id
            _use_new_layout = False
        save_path.mkdir(parents=True, exist_ok=True)
        print(f"[STEP2-TRAIN] training round_id={round_id!r} view={view_type!r} "
              f"-> save_path={save_path} model_name={model_name!r} "
              f"new_layout={_use_new_layout}")

        # Determine model initialization: pretrained checkpoint > starting
        # checkpoint from config > base cellpose model > scratch.
        ckpt = pretrained_model or config.get("starting_checkpoint") or None
        # F3.1 (Bundle 2 audit): wrap trainer.fit in try/except BaseException
        # so cancel mid-train (TrainingAbortedError is BaseException-derived)
        # restores the archived prior weights instead of leaving the user
        # with neither old nor new.  Re-raise so the worker thread still
        # sees the abort and routes to _on_training_error.
        try:
            if ckpt and Path(ckpt).is_file():
                # Fine-tune from a specific checkpoint (prior round or imported).
                trainer.fit(
                    train_data=imgs,
                    train_labels=masks,
                    model_name=model_name,
                    save_path=save_path,
                    pretrained_model=str(ckpt),
                    epoch_callback=epoch_callback,
                )
            else:
                # Fine-tune from a built-in cellpose model or train from scratch.
                base = str(config.get("model_type", "scratch"))
                trainer.fit(
                    train_data=imgs,
                    train_labels=masks,
                    model_name=model_name,
                    save_path=save_path,
                    base_model=base if base != "scratch" else None,
                    epoch_callback=epoch_callback,
                )
        except BaseException:
            # Training failed OR was cancelled mid-epoch.  Restore the
            # archived prior weights so the user doesn't lose both
            # old and new.  Re-raise so the worker sees the failure
            # and routes to _on_training_error / clean abort handling.
            if archive_path is not None:
                _restore_prior_training_state(save_path, archive_path)
            raise

        losses = []
        if trainer.train_loss is not None:
            losses = [float(x) for x in trainer.train_loss.tolist()]

        # 2026-05-26 audit H-4 fix: also surface val-loss series so the
        # schema 2.0 manifest can record final_loss.val (was None).
        # Cellpose splits val internally; series may be shorter than
        # train_loss if val happens every N epochs.
        val_losses = []
        if trainer.val_loss is not None:
            try:
                val_losses = [float(x) for x in trainer.val_loss.tolist()]
            except Exception:
                val_losses = []

        checkpoint_path = str(trainer.model_path) if trainer.model_path else ""

        # Phase 6c (2026-05-26): post-train lift for the new layout.
        # Cellpose unconditionally writes the weights + loss + config
        # under {save_path}/models/ (its auto-inserted subdir).  The
        # shared lift function moves them up to {save_path}/ + deletes
        # the subdir.  This matches Step 3's pattern + lets the schema
        # 2.0 manifest paths point directly at the bundle root.
        #
        # 2026-05-26 audit C-3 + H-1 fix: only declare the model
        # "lifted" if lift_result["weights_lifted"] is True AND
        # skipped_reason is NOT "dst_exists" (the latter is a
        # false-positive where root has a STALE file blocking the
        # fresh src in models/ from moving up).  The pre-clean above
        # should prevent dst_exists in normal flow but guard anyway.
        # Controller uses this flag to gate the schema 2.0 manifest
        # write (no manifest if lift failed -> the manifest's path
        # fields would point at non-existent files).
        weights_lifted = False
        if _use_new_layout:
            try:
                from .cellpose_lift import lift_cellpose_output
                lift_result = lift_cellpose_output(
                    save_path=save_path,
                    weights_filename=model_name,
                    log_fn=print,
                )
                # Only trust the post-lift root path if the lift
                # function ACTUALLY moved the file (or confirmed it
                # was already there via "already_lifted").
                if lift_result.get("weights_lifted") and lift_result.get(
                    "skipped_reason"
                ) != "dst_exists":
                    lifted_weights = save_path / model_name
                    if lifted_weights.is_file():
                        checkpoint_path = str(lifted_weights)
                        weights_lifted = True
                        print(f"[STEP2-TRAIN] lift complete -> "
                              f"checkpoint_path={checkpoint_path}")
                    else:
                        print(f"[STEP2-TRAIN] WARN: lift reported success "
                              f"but file not at {lifted_weights} — "
                              f"treating as not-lifted; manifest write "
                              f"will be skipped.")
                else:
                    reason = lift_result.get("skipped_reason", "unknown")
                    print(f"[STEP2-TRAIN] WARN: lift did NOT lift weights "
                          f"(skipped_reason={reason!r}); checkpoint at "
                          f"pre-lift path={checkpoint_path}; manifest "
                          f"write will be skipped to avoid pointing at "
                          f"non-existent files.")
            except Exception as e:
                print(f"[STEP2-TRAIN] WARN: lift raised "
                      f"({type(e).__name__}: {e}); checkpoint at "
                      f"pre-lift trainer.model_path={checkpoint_path}; "
                      f"manifest write will be skipped.")

        # Phase B.3 (2026-05-25): finalize timing snapshot for manifest.
        # B.3.1 CRITICAL fix: pass `_tz.utc` instance, not `_tz` class.
        timing_finished_at = _dt.now(_tz.utc).isoformat()
        try:
            _start_dt = _dt.fromisoformat(timing_started_at)
            _finish_dt = _dt.fromisoformat(timing_finished_at)
            timing_duration_seconds = (_finish_dt - _start_dt).total_seconds()
        except Exception:
            timing_duration_seconds = None

        # F3.1 (Bundle 2): training + lift succeeded — delete the archive
        # of prior weights.  We're past every checkpoint where cancel
        # could have rolled us back; the new weights are the canonical
        # state for this round.
        #
        # Bundle 2 audit fix #5 (2026-05-27): only delete the archive if
        # we actually have new weights at the canonical location.  For
        # the new-layout path, that's `weights_lifted=True`.  For the
        # legacy path (no lift), trainer.model_path is the source of
        # truth — only delete if the new weights file actually exists.
        # Pre-fix: if lift_cellpose_output silently failed, the archive
        # was still deleted → user had neither prior weights (deleted)
        # nor new weights (orphaned at non-canonical path).
        if archive_path is not None:
            if _use_new_layout:
                _new_weights_present = bool(weights_lifted)
            else:
                _new_weights_present = bool(
                    checkpoint_path
                    and Path(checkpoint_path).is_file()
                )
            if _new_weights_present:
                _delete_archive(archive_path)
            else:
                print(
                    f"[STEP2-TRAIN] F3.1 archive PRESERVED at "
                    f"{archive_path} — new weights are not at the "
                    f"canonical location (use_new_layout={_use_new_layout} "
                    f"weights_lifted={weights_lifted} "
                    f"checkpoint_path={checkpoint_path!r}).  Restore "
                    f"manually if needed; orphan-scan on next workspace "
                    f"load will also surface it."
                )

        return {
            "checkpoint_path": checkpoint_path,
            "losses": losses,
            "val_losses": val_losses,
            "mode": "epicycle_training_backend",
            "view_type": view_type,
            "save_path": str(save_path),
            "new_layout": _use_new_layout,
            "weights_lifted": weights_lifted,
            # Schema 2.1 reproducibility fields
            "train_pair_labels": train_pair_labels,
            "train_pair_uris": train_pair_uris,
            "timing_started_at": timing_started_at,
            "timing_finished_at": timing_finished_at,
            "timing_duration_seconds": timing_duration_seconds,
            "env_cellpose_version": env_cellpose_version,
            "env_torch_version": env_torch_version,
            "env_python_version": env_python_version,
            "env_device": env_device,
            "env_platform": env_platform_str,
        }

    _backend.backend_mode = "epicycle_training_backend"
    return _backend


def build_epicycle_inference_backend():
    """Returns an inference backend callable compatible with TrainingManager.

    Returns None if cellpose is not installed.
    """
    try:
        from cellpose import models  # noqa: F401 — verify cellpose available
    except ImportError:
        return None

    def _backend(
        *,
        round_id: str,
        view_type: str,
        model_checkpoint_path: str,
        config: Dict[str, Any],
        working_dir: str,
        inference_samples: List[Dict[str, Any]],
        predictions_dir_override: Optional[str] = None,
    ) -> Dict[str, Any]:
        use_gpu = str(config.get("device", "GPU")).upper() == "GPU"
        trainer = CellposeTrainer(use_gpu=use_gpu)
        trainer.load(model_checkpoint_path)

        images = []
        records_meta = []
        for sample in inference_samples:
            img = tifffile.imread(sample["img_uri"])
            images.append(img)
            records_meta.append(sample)

        # 2026-05-27: forward inference-time params from config dict so
        # post-training inference matches the user's Inference Parameters
        # picks.  Cellpose's predict() defaults kick in for any param not
        # set in config.  Audit fix 2026-05-27: also forward `invert` —
        # pre-fix it was set in the cfg dict by Inference Parameters but
        # silently dropped at this layer, so the post-train inference
        # diverged from the Predict button behavior on inverted-data
        # workflows.
        preds = trainer.predict(
            images,
            channels=[
                int(str(config.get("channel1", "0: gray")).split(":")[0].strip()),
                int(str(config.get("channel2", "0: none")).split(":")[0].strip()),
            ],
            normalize=bool(config.get("normalize", True)),
            rescale=bool(config.get("rescale", True)),
            cellprob_threshold=float(config.get("cellprob_threshold", 0.0)),
            flow_threshold=float(config.get("flow_threshold", 0.4)),
            diameter=(int(config.get("diameter", 0) or 0) or None),
            n_iterations=int(config.get("n_iterations", 0) or 0),
            invert=bool(config.get("invert", False)),
        )

        # Phase 6c (2026-05-26): predictions_dir_override routes to
        # the new multi-run layout (runs/{slug}/{bucket}/rounds/round_NN/
        # predictions/) instead of legacy {working_dir}/predictions/
        # {round_id}/{view_type}/.
        if predictions_dir_override is not None:
            predictions_dir = Path(predictions_dir_override)
            print(f"[STEP2-TRAIN] inference writing to NEW layout: "
                  f"{predictions_dir}")
        else:
            predictions_dir = Path(working_dir) / "predictions" / round_id / view_type
            print(f"[STEP2-TRAIN] inference writing to LEGACY layout: "
                  f"{predictions_dir}")
        predictions_dir.mkdir(parents=True, exist_ok=True)
        records = []
        for sample, pred_mask in zip(records_meta, preds):
            stack_id = str(sample["stack_id"])
            view = str(sample["view"]).upper()
            slice_index = int(sample["slice_index"])
            pred_name = f"{stack_id}_{view}_z{slice_index:04d}_pred_mask.tif"
            pred_path = predictions_dir / pred_name
            tifffile.imwrite(str(pred_path), pred_mask.astype("int32"))
            records.append(
                {
                    "stack_id": stack_id,
                    "view": view,
                    "slice_index": slice_index,
                    "prediction_uri": str(pred_path),
                }
            )

        manifest_path = predictions_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "round_id": round_id,
                    "view_type": view_type,
                    "num_predictions": len(records),
                    "records": records,
                    "mode": "epicycle_training_backend",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return {
            "prediction_uri": str(manifest_path),
            "num_predictions": len(records),
            "mode": "epicycle_training_backend",
            "view_type": view_type,
        }

    _backend.backend_mode = "epicycle_training_backend"
    return _backend
