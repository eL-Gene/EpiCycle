"""Tests for epicycle.utils.gpu module."""

import pytest
from packages.epicycle_scaledseg.utils.gpu import (
    GPUAllocator,
    FixedStrategy,
    RoundRobinStrategy,
)


class TestFixedStrategy:
    def test_allocates_3_gpus(self):
        s = FixedStrategy(views_per_tile=3)
        pool = list(range(8))
        allocated = s.allocate(pool)
        assert allocated == [0, 1, 2]
        assert pool == [3, 4, 5, 6, 7]

    def test_returns_none_when_insufficient(self):
        s = FixedStrategy(views_per_tile=3)
        pool = [0, 1]
        assert s.allocate(pool) is None
        assert pool == [0, 1]  # unchanged

    def test_max_concurrent(self):
        s = FixedStrategy(views_per_tile=3)
        assert s.max_concurrent(8) == 2
        assert s.max_concurrent(6) == 2
        assert s.max_concurrent(3) == 1

    def test_gpus_per_tile(self):
        assert FixedStrategy(3).gpus_per_tile == 3


class TestRoundRobinStrategy:
    def test_allocates_1_gpu(self):
        s = RoundRobinStrategy()
        pool = list(range(8))
        allocated = s.allocate(pool)
        assert allocated == [0]
        assert pool == list(range(1, 8))

    def test_returns_none_when_empty(self):
        s = RoundRobinStrategy()
        pool = []
        assert s.allocate(pool) is None

    def test_max_concurrent(self):
        s = RoundRobinStrategy()
        assert s.max_concurrent(8) == 8

    def test_gpus_per_tile(self):
        assert RoundRobinStrategy().gpus_per_tile == 1


class TestGPUAllocator:
    def test_allocate_and_release(self):
        alloc = GPUAllocator(4, RoundRobinStrategy())
        gpus = alloc.allocate_for_tile()
        assert gpus == [0]
        assert alloc.gpus_in_use == 1

        alloc.release(gpus)
        assert alloc.gpus_in_use == 0

    def test_exhaust_pool(self):
        alloc = GPUAllocator(2, RoundRobinStrategy())
        g1 = alloc.allocate_for_tile()
        g2 = alloc.allocate_for_tile()
        g3 = alloc.allocate_for_tile()
        assert g1 == [0]
        assert g2 == [1]
        assert g3 is None

    def test_fixed_strategy_integration(self):
        alloc = GPUAllocator(8, FixedStrategy(3))
        assert alloc.max_concurrent_tiles() == 2
        g1 = alloc.allocate_for_tile()
        g2 = alloc.allocate_for_tile()
        g3 = alloc.allocate_for_tile()  # only 2 GPUs left, need 3
        assert g1 == [0, 1, 2]
        assert g2 == [3, 4, 5]
        assert g3 is None

    def test_release_restores_order(self):
        alloc = GPUAllocator(4, RoundRobinStrategy())
        g1 = alloc.allocate_for_tile()  # [0]
        g2 = alloc.allocate_for_tile()  # [1]
        alloc.release(g2)
        alloc.release(g1)
        # After release, pool should be sorted
        g3 = alloc.allocate_for_tile()
        assert g3 == [0]

    def test_repr(self):
        alloc = GPUAllocator(8, RoundRobinStrategy())
        r = repr(alloc)
        assert "RoundRobinStrategy" in r
        assert "n_gpus=8" in r
