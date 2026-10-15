"""
GPU allocation strategies for EpiCycle ScaledSegmentation.

Manages GPU assignment for multi-tile parallel processing using
pluggable allocation strategies (Strategy pattern).
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import List, Optional

logger = logging.getLogger(__name__)


class AllocationStrategy(ABC):
    """Abstract base class for GPU allocation strategies."""

    @abstractmethod
    def allocate(self, available_gpus: List[int]) -> Optional[List[int]]:
        """
        Attempt to allocate GPUs from the available pool.

        Returns
        -------
        List of GPU IDs allocated, or None if insufficient GPUs available.
        The returned GPUs are removed from available_gpus in-place.
        """

    @abstractmethod
    def max_concurrent(self, n_gpus: int) -> int:
        """Maximum tiles that can run concurrently given n_gpus."""

    @property
    @abstractmethod
    def gpus_per_tile(self) -> int:
        """Number of GPUs required per tile."""


class FixedStrategy(AllocationStrategy):
    """
    Fixed allocation: each tile gets exactly ``views_per_tile`` GPUs.

    With 3 views and 8 GPUs, only 2 tiles run in parallel (6 GPUs used, 2 idle).
    Simple but wastes GPUs when n_gpus is not a multiple of views_per_tile.
    """

    def __init__(self, views_per_tile: int = 3):
        self._views_per_tile = views_per_tile

    def allocate(self, available_gpus: List[int]) -> Optional[List[int]]:
        if len(available_gpus) < self._views_per_tile:
            return None
        allocated = available_gpus[:self._views_per_tile]
        del available_gpus[:self._views_per_tile]
        return allocated

    def max_concurrent(self, n_gpus: int) -> int:
        return n_gpus // self._views_per_tile

    @property
    def gpus_per_tile(self) -> int:
        return self._views_per_tile


class RoundRobinStrategy(AllocationStrategy):
    """
    Round-robin allocation: each tile gets 1 GPU and runs all 3 views
    sequentially on that GPU.

    Uses all available GPUs — no waste. Multiple tiles run in parallel,
    each processing XY → XZ → YZ on its assigned GPU.
    """

    def __init__(self, views_per_tile: int = 3):
        self._views_per_tile = views_per_tile

    def allocate(self, available_gpus: List[int]) -> Optional[List[int]]:
        if len(available_gpus) < 1:
            return None
        allocated = [available_gpus.pop(0)]
        return allocated

    def max_concurrent(self, n_gpus: int) -> int:
        return n_gpus

    @property
    def gpus_per_tile(self) -> int:
        return 1


class GPUAllocator:
    """
    Thread-safe GPU resource manager.

    Tracks available/busy GPUs and assigns them according to the
    chosen AllocationStrategy.

    Parameters
    ----------
    n_gpus : total number of GPUs on the node
    strategy : AllocationStrategy instance
    """

    def __init__(self, n_gpus: int, strategy: AllocationStrategy):
        self.n_gpus = n_gpus
        self.strategy = strategy
        self.available_gpus = list(range(n_gpus))
        self._lock = threading.Lock()

    def allocate_for_tile(self) -> Optional[List[int]]:
        """
        Allocate GPUs for one tile.

        Returns
        -------
        List of GPU IDs, or None if not enough GPUs are free.
        """
        with self._lock:
            result = self.strategy.allocate(self.available_gpus)
        if result is None:
            logger.debug(f"GPU allocation failed: need {self.strategy.gpus_per_tile}, have {len(self.available_gpus)}")
        else:
            logger.debug(f"Allocated GPUs {result}, remaining: {self.available_gpus}")
        return result

    def release(self, gpus: List[int]):
        """Return GPUs to the available pool."""
        with self._lock:
            self.available_gpus.extend(gpus)
            self.available_gpus.sort()
        logger.debug(f"Released GPUs {gpus}, available: {self.available_gpus}")

    def max_concurrent_tiles(self) -> int:
        """Maximum tiles that can process simultaneously."""
        return self.strategy.max_concurrent(self.n_gpus)

    @property
    def gpus_in_use(self) -> int:
        """Number of GPUs currently allocated."""
        with self._lock:
            return self.n_gpus - len(self.available_gpus)

    def __repr__(self) -> str:
        return (
            f"GPUAllocator(n_gpus={self.n_gpus}, "
            f"strategy={self.strategy.__class__.__name__}, "
            f"available={len(self.available_gpus)})"
        )
