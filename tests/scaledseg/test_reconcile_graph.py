"""Tests for the bipartite cell<->nucleus component graph + taxonomy (epicycle.reconcile.graph).

Every cell and every nucleus must land in exactly one component, and each component must be
classified into the right taxonomy bucket by its (cell count, nucleus count)."""

import pytest

from packages.epicycle_scaledseg.reconcile.graph import (
    build_components, classify_component, component_census,
    CLEAN, MERGE, STRUNG, CELL_NO_NUCLEUS, ORPHAN_NUCLEUS, TANGLED,
)


def test_classify_all_cases():
    assert classify_component(1, 1) == CLEAN
    assert classify_component(1, 2) == MERGE
    assert classify_component(1, 5) == MERGE
    assert classify_component(2, 1) == STRUNG
    assert classify_component(1, 0) == CELL_NO_NUCLEUS
    assert classify_component(0, 1) == ORPHAN_NUCLEUS
    assert classify_component(3, 4) == TANGLED


def test_clean_pair():
    comps = build_components([(1, 10, 0.99)], cell_ids=[1], nucleus_ids=[10])
    assert len(comps) == 1
    assert comps[0]["type"] == CLEAN
    assert comps[0]["cells"] == [1] and comps[0]["nuclei"] == [10]


def test_merge_one_cell_two_nuclei():
    comps = build_components([(1, 10, 0.9), (1, 11, 0.8)], cell_ids=[1], nucleus_ids=[10, 11])
    assert len(comps) == 1 and comps[0]["type"] == MERGE
    assert comps[0]["nuclei"] == [10, 11]


def test_strung_nucleus_two_cells():
    comps = build_components([(1, 10, 0.6), (2, 10, 0.5)], cell_ids=[1, 2], nucleus_ids=[10])
    assert len(comps) == 1 and comps[0]["type"] == STRUNG
    assert comps[0]["cells"] == [1, 2]


def test_isolated_cell_and_nucleus_are_singletons():
    # a cell with no edge -> cell_no_nucleus; a nucleus with no edge -> orphan_nucleus
    comps = build_components([], cell_ids=[7], nucleus_ids=[20])
    types = sorted(c["type"] for c in comps)
    assert types == [CELL_NO_NUCLEUS, ORPHAN_NUCLEUS]


def test_tangled_component():
    # two cells, two nuclei, cross-linked into one clump
    edges = [(1, 10, 0.6), (1, 11, 0.5), (2, 11, 0.6)]
    comps = build_components(edges, cell_ids=[1, 2], nucleus_ids=[10, 11])
    assert len(comps) == 1 and comps[0]["type"] == TANGLED


def test_weak_edge_does_not_connect():
    # containment below the floor must NOT link the two -> they split into singletons
    comps = build_components([(1, 10, 0.05)], cell_ids=[1], nucleus_ids=[10], floor=0.15)
    assert len(comps) == 2
    assert sorted(c["type"] for c in comps) == [CELL_NO_NUCLEUS, ORPHAN_NUCLEUS]


def test_every_object_in_exactly_one_component():
    cell_ids = [1, 2, 3, 4]; nucleus_ids = [10, 11, 12]
    edges = [(1, 10, 0.9),              # clean
             (2, 11, 0.6), (3, 11, 0.5),  # strung nucleus 11 over cells 2,3
             (4, 12, 0.2)]              # clean-ish (above floor)
    comps = build_components(edges, cell_ids, nucleus_ids, floor=0.15)
    seen_cells = [c for comp in comps for c in comp["cells"]]
    seen_nuc = [n for comp in comps for n in comp["nuclei"]]
    assert sorted(seen_cells) == cell_ids               # every cell once
    assert sorted(seen_nuc) == nucleus_ids              # every nucleus once
    assert len(seen_cells) == len(set(seen_cells))      # no duplicates
    assert len(seen_nuc) == len(set(seen_nuc))


def test_stray_edge_id_does_not_leak_a_phantom_node():
    # an edge referencing ids not in cell_ids/nucleus_ids must be ignored, not added
    comps = build_components([(99, 88, 0.9), (1, 10, 0.9)], cell_ids=[1], nucleus_ids=[10])
    seen_cells = sorted(c for comp in comps for c in comp["cells"])
    seen_nuc = sorted(n for comp in comps for n in comp["nuclei"])
    assert seen_cells == [1] and seen_nuc == [10]      # 99/88 ignored
    assert len(comps) == 1 and comps[0]["type"] == CLEAN


def test_census_sums_to_totals():
    cell_ids = [1, 2, 3]; nucleus_ids = [10, 11, 12, 13]
    edges = [(1, 10, 0.9), (1, 11, 0.8),   # merge: cell 1 + nuclei 10,11
             (2, 12, 0.9)]                 # clean: cell 2 + nucleus 12
    #        cell 3 isolated, nucleus 13 isolated
    comps = build_components(edges, cell_ids, nucleus_ids, floor=0.15)
    cen = component_census(comps)
    assert sum(v["cells"] for v in cen.values()) == len(cell_ids)
    assert sum(v["nuclei"] for v in cen.values()) == len(nucleus_ids)
    assert cen[MERGE]["cells"] == 1 and cen[MERGE]["nuclei"] == 2
    assert cen[CELL_NO_NUCLEUS]["cells"] == 1
    assert cen[ORPHAN_NUCLEUS]["nuclei"] == 1
