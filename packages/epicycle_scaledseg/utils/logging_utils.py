"""
Logging utilities for EpiCycle ScaledSegmentation.

Per-GPU file loggers, stdout/stderr suppression, and progress monitoring.
"""

import os
import sys
import time
import logging
import contextlib
from pathlib import Path


def setup_stage_logger(
    stage_name: str,
    log_dir: str = "logs",
    level: int = logging.INFO,
) -> logging.Logger:
    """
    Create a file logger for a pipeline stage.

    Logs are written to ``<log_dir>/<stage_name>.log``.

    Parameters
    ----------
    stage_name : human-readable stage identifier (e.g. "stage2_reconstruction")
    log_dir : directory for log files
    level : logging level

    Returns
    -------
    Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"epicycle.{stage_name}")
    logger.setLevel(level)

    log_path = os.path.join(log_dir, f"{stage_name}.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(level)
    fh.setFormatter(
        logging.Formatter("%(asctime)s [%(name)s] %(levelname)s %(message)s")
    )

    # Avoid duplicate handlers on repeated calls
    if not any(
        isinstance(h, logging.FileHandler) and h.baseFilename == fh.baseFilename
        for h in logger.handlers
    ):
        logger.addHandler(fh)

    return logger


def setup_logger(gpu_id: int, log_dir: str = "logs") -> logging.Logger:
    """
    Create a file logger for a specific GPU worker.

    Logs are written to ``<log_dir>/gpu_<gpu_id>.log``.

    Parameters
    ----------
    gpu_id : GPU index
    log_dir : directory for log files

    Returns
    -------
    Configured logger instance.
    """
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger(f"gpu_{gpu_id}")
    logger.setLevel(logging.INFO)

    log_path = os.path.join(log_dir, f"gpu_{gpu_id}.log")
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s [GPU %(name)s] %(message)s"))

    if not logger.handlers:
        logger.addHandler(fh)

    return logger


@contextlib.contextmanager
def suppress_output():
    """Context manager that silences stdout and stderr (e.g. noisy library calls)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout, sys.stderr = devnull, devnull
        try:
            yield
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


def progress_monitor(counter, total: int):
    """
    Thread target that displays a shared tqdm progress bar across GPU workers.

    Parameters
    ----------
    counter : multiprocessing.Value with an integer ``value`` attribute
    total : total number of items to track
    """
    from tqdm import tqdm

    with tqdm(total=total, desc="Total Progress", dynamic_ncols=True) as pbar:
        while pbar.n < total:
            time.sleep(0.5)
            try:
                pbar.update(counter.value - pbar.n)
            except Exception:
                break
