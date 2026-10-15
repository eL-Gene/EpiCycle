from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List

logger = logging.getLogger(__name__)

import numpy as np
import tifffile


# Slice-label regex.  Accepts BOTH formats that appear in sample_records:
#
# * Native-curation format ``{stack}_{VIEW}_{idx}`` (e.g. ``patch_2_XY_62``) —
#   written by ``_persist_curated_training_pair`` keying off the raw label.
# * Disk-derived format ``{stack}_{VIEW}_z{idx:04d}`` (e.g. ``patch_2_XY_z0055``) —
#   the canonical on-disk filename pattern.  Cherry-pick import
#   (``_import_pairs.py::list_source_pairs``) and external import write
#   sample_records keys in THIS form because they derive labels from
#   ``{label}_img.tif`` filenames.
#
# Without the optional ``z?`` prefix, training a round that included any
# imported pair raised ``Unrecognized slice label format: ..._XY_z0055``
# in ``prepare_round_dataset`` (2026-05-26 bug report).  ``slice_index``
# is parsed as ``int(...)`` either way, so the value is identical for
# the two formats — they refer to the same slice.
SLICE_LABEL_RE = re.compile(r"^(?P<stack_id>.+)_(?P<view>XY|XZ|YZ)_z?(?P<slice_index>\d+)$")


class TrainingManager:
    def __init__(
        self,
        backend_trainer: Callable[..., Dict[str, Any]] | None = None,
        backend_inferencer: Callable[..., Dict[str, Any]] | None = None,
    ):
        self.round_index = 0
        self.loss_history: List[List[float]] = []
        self.backend_trainer = backend_trainer
        self.backend_inferencer = backend_inferencer

    def training_backend_info(self) -> Dict[str, str]:
        if self.backend_trainer is None:
            return {"mode": "placeholder", "repo_path": ""}
        return {
            "mode": str(getattr(self.backend_trainer, "backend_mode", "custom_backend")),
            "repo_path": str(getattr(self.backend_trainer, "backend_repo", "")),
        }

    def inference_backend_info(self) -> Dict[str, str]:
        if self.backend_inferencer is None:
            return {"mode": "placeholder", "repo_path": ""}
        return {
            "mode": str(getattr(self.backend_inferencer, "backend_mode", "custom_backend")),
            "repo_path": str(getattr(self.backend_inferencer, "backend_repo", "")),
        }

    def _parse_slice_label(self, slice_label: str) -> Dict[str, Any]:
        match = SLICE_LABEL_RE.match(slice_label)
        if not match:
            raise ValueError(f"Unrecognized slice label format: {slice_label}")
        return {
            "stack_id": match.group("stack_id"),
            "view": match.group("view"),
            "slice_index": int(match.group("slice_index")),
        }

    def _parse_slice_label_from_filename(self, fname: str) -> Dict[str, Any]:
        """Parse stack_id, view, slice_index from a training pair filename.

        Filename format: {stack_id}_{VIEW}_z{INDEX}
        E.g.: SAMPLE01_patch_1_XY_z0088
        """
        # Try matching with z-prefix index (from _persist_curated_training_pair).
        m = re.match(r"^(?P<stack_id>.+)_(?P<view>XY|XZ|YZ)_z(?P<slice_index>\d+)$", fname)
        if m:
            return {
                "stack_id": m.group("stack_id"),
                "view": m.group("view"),
                "slice_index": int(m.group("slice_index")),
            }
        # Fallback: try without z-prefix.
        m = SLICE_LABEL_RE.match(fname)
        if m:
            return {
                "stack_id": m.group("stack_id"),
                "view": m.group("view"),
                "slice_index": int(m.group("slice_index")),
            }
        # Last resort: best-effort split.
        parts = fname.rsplit("_", 2)
        return {
            "stack_id": parts[0] if len(parts) >= 3 else fname,
            "view": parts[1] if len(parts) >= 3 else "XY",
            "slice_index": 0,
        }

    def _dedupe_by_identity(self, samples: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        seen = set()
        out: List[Dict[str, Any]] = []
        for sample in samples:
            key = (sample["stack_id"], sample["view"], int(sample["slice_index"]))
            if key in seen:
                continue
            seen.add(key)
            out.append(sample)
        return out

    def split_samples_by_view_strategy(self, samples: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        grouped: Dict[str, List[Dict[str, Any]]] = {"spatial": [], "axial": []}
        for sample in samples:
            view = str(sample.get("view", "")).upper()
            if view == "XY":
                grouped["spatial"].append(sample)
            elif view in {"XZ", "YZ"}:
                grouped["axial"].append(sample)
        return {k: v for k, v in grouped.items() if v}

    def prepare_round_dataset(
        self,
        segmentation_manager: Any,
        slice_labels: List[str],
        working_dir: str,
        round_id: str,
        dataset_id: str = "default_dataset",
        prior_pairs: List[Dict[str, Any]] | None = None,
        persisted_pairs: Dict[str, Dict[str, str]] | None = None,
        round_data_root: Path | str | None = None,
        log_warn_cb: Callable[[str], None] | None = None,
        used_slice_labels_out: List[str] | None = None,
    ) -> List[Dict[str, Any]]:
        """Prepare training dataset for a round.

        Phase A (2026-05-25) — flat layout simplification.

        Pre-Phase-A this function unconditionally created `spatial/`
        and `axial/` subdirs inside the round dir, but the conditional
        write block beneath them never fired in production (the
        `persisted_pairs` short-circuit reuses URIs from `accepted/`
        which `_persist_curated_training_pair` has already written by
        the time training launches).  Net effect: two empty subdirs
        appeared in every `runs/{run}/{bucket}/rounds/round_NN/` —
        pure cruft.

        Phase A drops the subdir creation entirely.  The fallback
        in-memory write path (for slices that have an `input_stack` +
        `mask` in `segmentation_manager` but NO `persisted_pairs`
        URI — never reached in the standard HITL flow) now writes
        directly into `round_data_root / "accepted/"` (the canonical
        curated-pair location), creating it if missing.

        Args:
            persisted_pairs: Optional mapping from slice_label to
                {"img_uri": ..., "mask_uri": ...} for pairs already on disk.
                When present, avoids re-extracting and re-writing those slices.
            round_data_root: Optional explicit override for where to
                write this round's pairs.  When provided, takes
                precedence over the legacy
                ``working_dir/training_data/round_id`` default.  Step 2
                callers pass this from ``_paths.round_dir(...)``.
            log_warn_cb: Optional ``callable(str)`` that receives every
                DROP made here (a prior-round pair whose files vanished,
                a staged slice with no pixels).  Release-hardening #5
                (2026-09-07): pre-fix those drops only went to a module
                logger, so a cumulative round could report "prior=12" and
                silently train on 8.
            used_slice_labels_out: Optional list; when supplied, every
                ``slice_labels`` entry that actually became a training
                pair is appended to it, in order.  Release-hardening #6:
                the caller commits ONLY these, so a silently-skipped
                staged slice stays staged instead of being marked
                "committed" (i.e. trained on) when it never was.  Returned
                via an out-parameter rather than the return value so the
                existing ``-> List[sample records]`` contract — and every
                caller/test of it — is unchanged.
        """
        if round_data_root is not None:
            round_dir = Path(round_data_root)
        else:
            round_dir = Path(working_dir) / "training_data" / round_id

        _persisted = persisted_pairs or {}
        sample_records: List[Dict[str, Any]] = []

        # Include prior round pairs (cumulative training).
        for pair in (prior_pairs or []):
            img_uri = str(pair.get("img_uri", ""))
            mask_uri = str(pair.get("mask_uri", ""))
            view_group = str(pair.get("view_group", "spatial"))
            if not img_uri or not mask_uri:
                continue
            if not Path(img_uri).is_file() or not Path(mask_uri).is_file():
                logger.warning("Prior pair files missing, skipping: %s / %s", img_uri, mask_uri)
                if log_warn_cb is not None:
                    try:
                        log_warn_cb(
                            f"[TRAIN] Prior-round pair files missing — "
                            f"skipping: {img_uri} / {mask_uri}"
                        )
                    except Exception:
                        pass
                continue
            view = "XY" if view_group == "spatial" else "XZ"
            fname = Path(img_uri).stem.replace("_img", "")
            # Parse stack_id, view, slice_index from filename.
            # Format: {stack_id}_{VIEW}_z{INDEX}_img.tif
            parsed = self._parse_slice_label_from_filename(fname)
            sample_records.append(
                {
                    "dataset_id": dataset_id,
                    "stack_id": parsed["stack_id"],
                    "view": parsed["view"] or view,
                    "slice_index": parsed["slice_index"],
                    "channel_map": {},
                    "img_uri": img_uri,
                    "mask_uri": mask_uri,
                }
            )

        # Lazy-init for the fallback write path (rare — only hit when
        # `persisted_pairs[label]` URI is missing on disk).
        _accepted_dir: Path | None = None

        # Write current round staged slices.
        for slice_label in slice_labels:
            parsed = self._parse_slice_label(slice_label)
            stack_id = parsed["stack_id"]
            view = parsed["view"]
            slice_index = parsed["slice_index"]

            # If pair already persisted on disk, reuse it.
            existing = _persisted.get(slice_label) or {}
            existing_img = str(existing.get("img_uri", ""))
            existing_mask = str(existing.get("mask_uri", ""))
            if existing_img and existing_mask and Path(existing_img).is_file() and Path(existing_mask).is_file():
                sample_records.append(
                    {
                        "dataset_id": dataset_id,
                        "stack_id": stack_id,
                        "view": view,
                        "slice_index": slice_index,
                        "channel_map": {},
                        "img_uri": existing_img,
                        "mask_uri": existing_mask,
                    }
                )
                if used_slice_labels_out is not None:
                    used_slice_labels_out.append(slice_label)
                continue

            input_stack = segmentation_manager.get_input_stack(slice_label)
            mask = segmentation_manager.get_masks(slice_label)

            if input_stack is None or mask is None:
                logger.warning(
                    "No image/mask available for staged slice %s — skipping",
                    slice_label,
                )
                if log_warn_cb is not None:
                    try:
                        log_warn_cb(
                            f"[TRAIN] Staged slice {slice_label!r} has no "
                            f"image/mask on disk or in memory — NOT trained "
                            f"on; it stays staged."
                        )
                    except Exception:
                        pass
                continue

            if not isinstance(input_stack, np.ndarray):
                input_stack = np.asarray(input_stack)
            if not isinstance(mask, np.ndarray):
                mask = np.asarray(mask)

            if input_stack.ndim != 3:
                raise ValueError(f"Expected input stack shape (C,H,W), got {input_stack.shape} for {slice_label}")
            if mask.ndim != 2:
                raise ValueError(f"Expected mask shape (H,W), got {mask.shape} for {slice_label}")

            # Phase A: fallback path writes directly to `accepted/`
            # (the canonical curated-pair location) instead of the
            # legacy `spatial/`/`axial/` subdirs.  Lazy-mkdir so the
            # common case (persisted_pairs URI found) never touches it.
            if _accepted_dir is None:
                _accepted_dir = round_dir / "accepted"
                _accepted_dir.mkdir(parents=True, exist_ok=True)
            fname_base = f"{stack_id}_{view}_z{slice_index:04d}"
            img_path = _accepted_dir / f"{fname_base}_img.tif"
            mask_path = _accepted_dir / f"{fname_base}_mask.tif"

            # Phase A audit follow-up: log the rare fallback path so
            # mystery files appearing in `accepted/` during training
            # are traceable.  Standard HITL flow never reaches here
            # (_persist_curated_training_pair writes on Accept and
            # populates `persisted_pairs`).
            logger.info(
                "[STEP2-TRAIN] fallback write (no persisted_pairs URI) -> %s",
                img_path,
            )
            tifffile.imwrite(str(img_path), input_stack.astype(np.float32))
            tifffile.imwrite(str(mask_path), mask.astype(np.int32))

            sample_records.append(
                {
                    "dataset_id": dataset_id,
                    "stack_id": stack_id,
                    "view": view,
                    "slice_index": slice_index,
                    "channel_map": {},
                    "img_uri": str(img_path),
                    "mask_uri": str(mask_path),
                }
            )
            if used_slice_labels_out is not None:
                used_slice_labels_out.append(slice_label)

        return self._dedupe_by_identity(sample_records)

    def run_training(
        self,
        round_id: str,
        config: Dict[str, Any],
        working_dir: str,
        training_samples: List[Dict[str, Any]],
        view_type: str,
        pretrained_model: str | None = None,
        epoch_callback: Callable | None = None,
        save_path_override: str | Path | None = None,
        weights_filename: str | None = None,
        log_warn_cb: Callable | None = None,
    ) -> Dict[str, Any]:
        """
        Executes training for one round.

        Phase B default behavior:
        - if `backend_trainer` is provided: delegate to real training backend
        - otherwise: write placeholder checkpoint metadata and synthetic loss series

        If pretrained_model is provided, it is forwarded to the backend trainer
        as the checkpoint to fine-tune from (cumulative training).

        epoch_callback: Optional callable ``(epoch, n_epochs, train_loss, test_loss)``
            forwarded to the backend trainer for real-time progress reporting.

        save_path_override (Phase 6c — 2026-05-26): if provided, the
        backend writes its checkpoint here INSTEAD of the legacy
        ``{working_dir}/models/{round_id}/`` location.  Step 2 callers
        on the new multi-run layout pass
        ``_paths.model_dir(workspace, run_id, bucket, round_idx)``.
        When provided, the backend uses canonical model_name="weights"
        + calls ``cellpose_lift.lift_cellpose_output`` post-training.
        When None, legacy behavior is preserved.
        """
        if self.backend_trainer is not None:
            # Forward log_warn_cb only if the backend accepts it
            # (Bundle 2 audit fix #4 wiring — older non-Cellpose
            # backends won't have the kwarg).
            import inspect as _inspect
            backend_sig = _inspect.signature(self.backend_trainer)
            backend_kwargs = dict(
                round_id=round_id,
                config=config,
                working_dir=working_dir,
                training_samples=training_samples,
                view_type=view_type,
                pretrained_model=pretrained_model,
                epoch_callback=epoch_callback,
                save_path_override=save_path_override,
                weights_filename=weights_filename,
            )
            if "log_warn_cb" in backend_sig.parameters:
                backend_kwargs["log_warn_cb"] = log_warn_cb
            result = self.backend_trainer(**backend_kwargs)
            losses = [float(x) for x in result.get("losses", [])]
            self.log_loss(losses)
            return result

        # Placeholder path (no backend installed) — honor override if provided
        # so tests + dummy flows write into the new layout too.
        if save_path_override is not None:
            models_dir = Path(save_path_override)
        else:
            models_dir = Path(working_dir) / "models" / round_id
        models_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        checkpoint_path = models_dir / f"{config.get('model_name', 'model')}_{view_type}_{timestamp}.ckpt.json"

        epochs = int(config.get("epochs", 200))
        points = min(max(epochs // 50, 5), 20)
        losses = [round(1.0 / (1 + i * 0.35), 6) for i in range(points)]

        payload = {
            "round_id": round_id,
            "timestamp": timestamp,
            "mode": "placeholder",
            "model_name": config.get("model_name", "model"),
            "model_type": config.get("model_type", "cyto3"),
            "view_type": view_type,
            "epochs": epochs,
            "num_samples": len(training_samples),
            "losses": losses,
        }
        checkpoint_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        self.log_loss(losses)
        return {
            "checkpoint_path": str(checkpoint_path),
            "losses": losses,
            "mode": "placeholder",
            "view_type": view_type,
        }

    def run_inference(
        self,
        *,
        round_id: str,
        view_type: str,
        model_checkpoint_path: str,
        config: Dict[str, Any],
        working_dir: str,
        inference_samples: List[Dict[str, Any]],
        predictions_dir_override: str | Path | None = None,
    ) -> Dict[str, Any]:
        """Phase 6c — 2026-05-26: predictions_dir_override routes
        per-round inference output to ``runs/{slug}/{bucket}/rounds/
        round_NN/predictions/`` (the new layout) instead of legacy
        ``{working_dir}/predictions/{round_id}/{view_type}/``.  When
        None, legacy path is used."""
        if self.backend_inferencer is not None:
            return self.backend_inferencer(
                round_id=round_id,
                view_type=view_type,
                model_checkpoint_path=model_checkpoint_path,
                config=config,
                working_dir=working_dir,
                inference_samples=inference_samples,
                predictions_dir_override=predictions_dir_override,
            )

        if predictions_dir_override is not None:
            predictions_dir = Path(predictions_dir_override)
        else:
            predictions_dir = Path(working_dir) / "predictions" / round_id / view_type
        predictions_dir.mkdir(parents=True, exist_ok=True)
        records = []

        for sample in inference_samples:
            stack_id = str(sample["stack_id"])
            view = str(sample["view"]).upper()
            slice_index = int(sample["slice_index"])
            pred_name = f"{stack_id}_{view}_z{slice_index:04d}_pred_mask.tif"
            pred_path = predictions_dir / pred_name

            # Placeholder inference path: use curated mask as pseudo-prediction artifact.
            src_mask = tifffile.imread(sample["mask_uri"])
            tifffile.imwrite(str(pred_path), np.asarray(src_mask, dtype=np.int32))

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
                    "mode": "placeholder",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "prediction_uri": str(manifest_path),
            "num_predictions": len(records),
            "mode": "placeholder",
            "view_type": view_type,
        }

    def save_model(self, model_path):
        # Reserved for explicit model export logic (not used in Phase B baseline).
        return model_path

    def log_loss(self, loss_values):
        self.loss_history.append(loss_values)
