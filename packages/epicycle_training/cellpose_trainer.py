"""
Cellpose model training wrapper.

Copied from epicycle-training/epicycle_local/trainer/cellpose_trainer.py
to consolidate into the EpiCycle monorepo.

Provides a clean API for training and fine-tuning Cellpose models
with support for:
- Training from scratch
- Fine-tuning from custom pretrained models
- Fine-tuning from built-in Cellpose models (cyto3, nuclei, etc.)
- Automatic model saving and checkpointing
- Loss curve tracking
- Validation during training
"""

import os
import json
import logging
import re
import numpy as np
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from .config import TrainingConfig, SegmentationConfig


def build_cellpose_model_kwargs(
    *,
    use_gpu: bool,
    pretrained_model: Optional[str] = None,
    base_model: Optional[str] = None,
    n_channels: int = 2,
) -> Dict[str, Any]:
    """Exact kwargs :meth:`CellposeTrainer.fit` passes to ``CellposeModel``.

    Release-hardening #10 (2026-09-07): hoisted out of ``fit`` so
    ``tests/hitl_training_annotation/test_cellpose_signature.py`` can bind
    the REAL dict against the installed cellpose signature.  A cellpose
    upgrade that renames or drops a parameter then fails in CI instead of
    minutes into the user's first real training round.
    """
    if pretrained_model:
        return {"gpu": use_gpu, "pretrained_model": pretrained_model}
    if base_model:
        return {"gpu": use_gpu, "model_type": base_model, "nchan": n_channels}
    return {"gpu": use_gpu, "model_type": None}


def build_train_seg_kwargs(
    *,
    net: Any,
    config: TrainingConfig,
    train_data: List[np.ndarray],
    train_labels: List[np.ndarray],
    save_path: Optional[str],
    model_name: Optional[str],
    val_imgs: Optional[List[np.ndarray]] = None,
    val_masks: Optional[List[np.ndarray]] = None,
) -> Dict[str, Any]:
    """Exact kwargs :meth:`CellposeTrainer.fit` passes to ``train.train_seg``.

    Release-hardening #10 — see :func:`build_cellpose_model_kwargs`.
    """
    return {
        "net": net,
        "train_data": train_data,
        "train_labels": train_labels,
        "channels": config.channels,
        "save_path": save_path,
        "n_epochs": config.n_epochs,
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "batch_size": config.batch_size,
        "normalize": config.normalize,
        "min_train_masks": config.min_train_masks,
        "model_name": model_name,
        "momentum": config.momentum,
        "rescale": config.rescale,
        "SGD": config.SGD,
        "compute_flows": config.compute_flows,
        "test_data": val_imgs,
        "test_labels": val_masks,
    }


class _EpochLogHandler(logging.Handler):
    """Intercepts cellpose.train logger to extract per-epoch metrics."""

    _PATTERN = re.compile(
        r"^(\d+),\s*train_loss=([\d.]+)(?:,\s*test_loss=([\d.]+))?.*"
    )

    def __init__(self, n_epochs: int, callback: Callable):
        super().__init__(level=logging.INFO)
        self.n_epochs = n_epochs
        self.callback = callback

    def emit(self, record):
        # Bundle 2 (2026-05-27 audit follow-up): the bare `except Exception`
        # below previously swallowed `TrainingAbortedError` (then a
        # RuntimeError subclass), preventing mid-epoch cancellation.
        # TrainingAbortedError now derives from BaseException so it
        # bypasses this clause automatically.  We narrow the except
        # anyway as defense-in-depth + clear intent.
        try:
            # Use getMessage() not format() — format() can prepend
            # logger name/level which breaks the ^(\d+) pattern.
            msg = record.getMessage()
            m = self._PATTERN.match(msg)
            if m:
                epoch = int(m.group(1))
                train_loss = float(m.group(2))
                test_loss = float(m.group(3)) if m.group(3) else None
                self.callback(epoch, self.n_epochs, train_loss, test_loss)
        except (re.error, ValueError, TypeError, AttributeError):
            # Parse / type errors are non-fatal — keep training going.
            # Note: TrainingAbortedError(BaseException) is NOT caught here
            # (intentional — it must propagate up to abort training).
            pass


# Available built-in Cellpose model types
CELLPOSE_MODEL_TYPES = [
    "cyto3",      # Latest cytoplasm model for cp3 (recommended)
    "cyto2",      # Previous cytoplasm model
    "cyto",       # Original cytoplasm model
    "nuclei",     # Nuclear segmentation
    "livecell",   # Live cell imaging
    "tissuenet",  # Tissue segmentation
    "CP",         # CellPose base
    "CPx",        # CellPose extended
]


class CellposeTrainer:
    """High-level trainer for Cellpose models.

    Wraps Cellpose's training API with a cleaner interface and
    additional features like loss tracking and model management.

    Args:
        config: TrainingConfig instance (uses defaults if None)
        use_gpu: Whether to use GPU for training

    Example:
        >>> config = TrainingConfig(n_epochs=500, learning_rate=1e-2)
        >>> trainer = CellposeTrainer(config)
        >>> model = trainer.fit(train_imgs, train_masks, val_data=(val_imgs, val_masks))
        >>> trainer.save("models/my_model")
    """

    def __init__(
        self,
        config: Optional[TrainingConfig] = None,
        use_gpu: bool = True,
    ):
        self.config = config or TrainingConfig()
        self.use_gpu = use_gpu

        self.model = None
        self.train_loss = None
        self.val_loss = None
        self.model_path = None

    def fit(
        self,
        train_data: List[np.ndarray],
        train_labels: List[np.ndarray],
        val_data: Optional[Tuple[List[np.ndarray], List[np.ndarray]]] = None,
        pretrained_model: Optional[Union[str, Path]] = None,
        base_model: Optional[str] = None,
        model_name: Optional[str] = None,
        save_path: Optional[Union[str, Path]] = None,
        epoch_callback: Optional[Callable] = None,
    ) -> "CellposeTrainer":
        """Train a Cellpose model.

        Three initialization modes:
            1. Fine-tune from custom model: provide ``pretrained_model`` path
            2. Fine-tune from built-in Cellpose model: provide ``base_model`` name
            3. Train from scratch: provide neither

        epoch_callback: Optional callable ``(epoch, n_epochs, train_loss, test_loss)``
            invoked every ~10 epochs with live training metrics (intercepted
            from cellpose's internal logger).

        Args:
            train_data: List of training images (C, H, W)
            train_labels: List of training masks (H, W)
            val_data: Optional tuple of (val_images, val_masks)
            pretrained_model: Path to custom pretrained model for fine-tuning
            base_model: Built-in Cellpose model to fine-tune from
            model_name: Name for the trained model
            save_path: Directory to save the model

        Returns:
            self (for method chaining)
        """
        from cellpose import models, train

        if pretrained_model and base_model:
            raise ValueError(
                "Cannot specify both 'pretrained_model' and 'base_model'. "
                "Use 'pretrained_model' for custom models or 'base_model' for "
                "built-in Cellpose models (cyto3, nuclei, etc.)"
            )

        if self.config.seed is not None:
            np.random.seed(self.config.seed)

        init_source = None

        if pretrained_model:
            pretrained_path = str(Path(pretrained_model).resolve())
            if not Path(pretrained_path).exists():
                raise FileNotFoundError(f"Pretrained model not found: {pretrained_path}")
            print(f"Fine-tuning from custom model: {pretrained_path}")
            self.model = models.CellposeModel(
                **build_cellpose_model_kwargs(
                    use_gpu=self.use_gpu, pretrained_model=pretrained_path
                )
            )
            init_source = {"type": "custom", "path": pretrained_path}

        elif base_model:
            if base_model not in CELLPOSE_MODEL_TYPES:
                available = ", ".join(CELLPOSE_MODEL_TYPES)
                raise ValueError(
                    f"Unknown base_model '{base_model}'. "
                    f"Available options: {available}"
                )
            print(f"Fine-tuning from Cellpose '{base_model}' model")
            self.model = models.CellposeModel(
                **build_cellpose_model_kwargs(
                    use_gpu=self.use_gpu,
                    base_model=base_model,
                    n_channels=(
                        len(self.config.channels) if self.config.channels else 2
                    ),
                )
            )
            init_source = {"type": "cellpose", "model_type": base_model}

        else:
            print("Training model from scratch")
            self.model = models.CellposeModel(
                **build_cellpose_model_kwargs(use_gpu=self.use_gpu)
            )
            init_source = {"type": "scratch"}

        val_imgs = val_data[0] if val_data else None
        val_masks = val_data[1] if val_data else None

        if save_path:
            save_path = str(Path(save_path).resolve())
            os.makedirs(save_path, exist_ok=True)

        print(f"Training model: {model_name or 'unnamed'}")
        print(f"  Epochs: {self.config.n_epochs}")
        print(f"  Learning rate: {self.config.learning_rate}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Training samples: {len(train_data)}")
        if val_imgs:
            print(f"  Validation samples: {len(val_imgs)}")

        # Install epoch logging hook to intercept cellpose's per-epoch metrics.
        _log_handler = None
        if epoch_callback is not None:
            cp_logger = logging.getLogger("cellpose.train")
            # Ensure logger level allows INFO through to our handler.
            # getEffectiveLevel() resolves NOTSET by walking the parent chain.
            if cp_logger.getEffectiveLevel() > logging.INFO:
                cp_logger.setLevel(logging.INFO)
            _log_handler = _EpochLogHandler(self.config.n_epochs, epoch_callback)
            cp_logger.addHandler(_log_handler)

        try:
            new_model_path, self.train_loss, self.val_loss = train.train_seg(
                **build_train_seg_kwargs(
                    net=self.model.net,
                    config=self.config,
                    train_data=train_data,
                    train_labels=train_labels,
                    save_path=save_path,
                    model_name=model_name,
                    val_imgs=val_imgs,
                    val_masks=val_masks,
                )
            )
        finally:
            if _log_handler is not None:
                logging.getLogger("cellpose.train").removeHandler(_log_handler)

        self.model_path = new_model_path

        # Release-hardening #7 (2026-09-07): cellpose returns an all-zeros
        # test-loss array when no validation set was supplied.  Keeping it
        # meant a zeros `{model}_val_loss.npy` on disk and
        # `final_loss.val = 0.0` in the schema 2.0 manifest — which reads
        # as a PERFECT validation score.  Validation is only "present"
        # when the caller actually passed `val_data`.
        if val_imgs is None:
            self.val_loss = None

        if save_path and model_name:
            loss_dir = Path(save_path) / "models"
            np.save(loss_dir / f"{model_name}_train_loss.npy", self.train_loss)
            if self.val_loss is not None:
                np.save(loss_dir / f"{model_name}_val_loss.npy", self.val_loss)

            training_metadata = {
                "model_name": model_name,
                "initialization": init_source,
                "hyperparameters": {
                    "n_epochs": self.config.n_epochs,
                    "learning_rate": self.config.learning_rate,
                    "batch_size": self.config.batch_size,
                    "weight_decay": self.config.weight_decay,
                    "momentum": self.config.momentum,
                    "channels": self.config.channels,
                    "normalize": self.config.normalize,
                    "rescale": self.config.rescale,
                    "min_train_masks": self.config.min_train_masks,
                    "SGD": self.config.SGD,
                    "compute_flows": self.config.compute_flows,
                    "seed": self.config.seed,
                },
                "data": {
                    "n_train_samples": len(train_data),
                    "n_val_samples": len(val_imgs) if val_imgs else 0,
                },
                "final_train_loss": float(self.train_loss[-1]) if len(self.train_loss) > 0 else None,
                "final_val_loss": float(self.val_loss[-1]) if self.val_loss is not None and len(self.val_loss) > 0 else None,
            }

            metadata_path = loss_dir / f"{model_name}_training_config.json"
            with open(metadata_path, "w") as f:
                json.dump(training_metadata, f, indent=2)
            print(f"Training config saved to: {metadata_path}")

        print(f"Training complete. Model saved to: {self.model_path}")
        return self

    def predict(
        self,
        images: List[np.ndarray],
        **kwargs,
    ) -> List[np.ndarray]:
        """Run inference on images."""
        if self.model is None:
            raise ValueError("No model loaded. Call fit() or load() first.")

        seg_config = SegmentationConfig()
        eval_kwargs = dict(
            channels=kwargs.get("channels", seg_config.channels),
            normalize=kwargs.get("normalize", seg_config.normalize),
            rescale=kwargs.get("rescale", seg_config.rescale),
            cellprob_threshold=kwargs.get("cellprob_threshold", seg_config.cellprob_threshold),
            flow_threshold=kwargs.get("flow_threshold", seg_config.flow_threshold),
            diameter=kwargs.get("diameter", seg_config.diameter),
            batch_size=kwargs.get("batch_size", seg_config.batch_size),
            compute_masks=True,
            do_3D=False,
        )
        # 2026-05-27: invert pass-through (audit fix).  Default False
        # preserves prior behavior; when set, applies the user's
        # Inference Parameters `Invert image` choice.
        if "invert" in kwargs:
            eval_kwargs["invert"] = bool(kwargs["invert"])
        # 2026-05-27: niter (dynamics iterations) — Cellpose's
        # post-prediction simulation step.  Skipped when value is None
        # or 0 (= Cellpose's auto-pick ~2× diameter) so we don't break
        # on older Cellpose versions that don't accept the arg.
        niter_val = kwargs.get("n_iterations") or kwargs.get("niter")
        try:
            niter_int = int(niter_val) if niter_val is not None else 0
        except (TypeError, ValueError):
            niter_int = 0
        if niter_int > 0:
            eval_kwargs["niter"] = niter_int
        masks, _, _ = self.model.eval(images, **eval_kwargs)
        return masks

    def load(self, model_path: Union[str, Path]) -> "CellposeTrainer":
        """Load a trained model from disk."""
        from cellpose import models

        model_path = str(Path(model_path).resolve())
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = models.CellposeModel(
            gpu=self.use_gpu,
            pretrained_model=model_path,
        )
        self.model_path = model_path
        print(f"Loaded model from: {model_path}")
        return self

    def get_loss_history(self) -> Dict[str, Any]:
        """Get training and validation loss history."""
        result = {}
        if self.train_loss is not None:
            result["train_loss"] = self.train_loss
        if self.val_loss is not None:
            result["val_loss"] = self.val_loss
        return result


def list_cellpose_models() -> List[str]:
    """List available built-in Cellpose model types."""
    return CELLPOSE_MODEL_TYPES.copy()
