"""Bipartite cell<->nucleus overlap graph -> connected components -> the full taxonomy.

The reconcile's organizing structure: every cell label and every nucleus label is a node;
an edge means they overlap (a nucleus is at least ``floor``-contained in a cell). Connected
components of that bipartite graph are the independent little puzzles — a clean 1-cell-1-
nucleus component is no puzzle and passes straight through; the rest are the work.

Classifying each component by ``(n_cells, n_nuclei)`` gives the COMPLETE taxonomy, and
because every node lands in exactly one component, **every cell and every nucleus is
accounted for** — nothing falls through:

    (1, 1)   clean             pass through
    (1, 2+)  merge             1 cell, many nuclei  -> membrane decides each split
    (2+, 1)  strung_nucleus    1 nucleus, many cells -> membrane decides the join/split
    (1, 0)   cell_no_nucleus   missed nucleus (rescue) or garbage cell
    (0, 1)   orphan_nucleus    missed cell (rescue) or garbage nucleus
    (2+, 2+) tangled           joint: membrane decides each candidate boundary

This module is pure graph bookkeeping (no images) — the per-component RESOLUTION (membrane
arbiter / rescue arbiter) lives elsewhere.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

CLEAN = "clean"
MERGE = "merge"
STRUNG = "strung_nucleus"
CELL_NO_NUCLEUS = "cell_no_nucleus"
ORPHAN_NUCLEUS = "orphan_nucleus"
TANGLED = "tangled"


def classify_component(n_cells: int, n_nuclei: int) -> str:
    """Map a component's (cell count, nucleus count) to its taxonomy label."""
    if n_cells == 1 and n_nuclei == 1:
        return CLEAN
    if n_cells == 1 and n_nuclei == 0:
        return CELL_NO_NUCLEUS
    if n_cells == 0 and n_nuclei == 1:
        return ORPHAN_NUCLEUS
    if n_cells == 1 and n_nuclei >= 2:
        return MERGE
    if n_cells >= 2 and n_nuclei == 1:
        return STRUNG
    if n_cells >= 2 and n_nuclei >= 2:
        return TANGLED
    raise ValueError(f"degenerate component ({n_cells} cells, {n_nuclei} nuclei)")


def build_components(
    edges: Iterable[Tuple[int, int, float]],
    cell_ids: Sequence[int],
    nucleus_ids: Sequence[int],
    floor: float = 0.15,
) -> List[dict]:
    """Connected components of the bipartite cell<->nucleus overlap graph.

    Args:
        edges: iterable of ``(cell_id, nucleus_id, containment)``. Only edges with
            ``containment >= floor`` connect their two nodes (weaker overlaps are incidental
            touches, not a real link).
        cell_ids, nucleus_ids: ALL real cell / nucleus ids (so isolated objects with no
            qualifying edge become their own singleton component — e.g. an orphan nucleus or a
            cell whose nucleus was missed).
        floor: containment threshold for an edge to connect.

    Returns:
        list of ``{"cells": sorted[int], "nuclei": sorted[int], "type": str}``. Every id in
        ``cell_ids`` / ``nucleus_ids`` appears in exactly one component.
    """
    parent: dict = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:                # path compression
            parent[x], x = root, parent[x]
        return root

    def union(a, b):
        parent[find(a)] = find(b)

    cset = {int(c) for c in cell_ids}
    nset = {int(n) for n in nucleus_ids}
    for c in cset:
        find(("c", c))
    for n in nset:
        find(("n", n))
    for cid, nid, cont in edges:
        # ignore edges whose endpoints aren't declared real objects (a stray edge id would
        # otherwise leak a phantom node into the census + corrupt a future auto-edit)
        if cont >= floor and int(cid) in cset and int(nid) in nset:
            union(("c", int(cid)), ("n", int(nid)))

    groups: dict = {}
    for node in list(parent.keys()):
        groups.setdefault(find(node), {"cells": [], "nuclei": []})
        groups[find(node)]["cells" if node[0] == "c" else "nuclei"].append(node[1])

    out = []
    for g in groups.values():
        g["cells"].sort()
        g["nuclei"].sort()
        g["type"] = classify_component(len(g["cells"]), len(g["nuclei"]))
        out.append(g)
    return out


def component_census(components: Sequence[dict]) -> dict:
    """Count components, cells, and nuclei by taxonomy type (every object accounted for)."""
    by_type: dict = {}
    for comp in components:
        t = comp["type"]
        rec = by_type.setdefault(t, {"components": 0, "cells": 0, "nuclei": 0})
        rec["components"] += 1
        rec["cells"] += len(comp["cells"])
        rec["nuclei"] += len(comp["nuclei"])
    return by_type


__all__ = [
    "classify_component",
    "build_components",
    "component_census",
    "CLEAN", "MERGE", "STRUNG", "CELL_NO_NUCLEUS", "ORPHAN_NUCLEUS", "TANGLED",
]
