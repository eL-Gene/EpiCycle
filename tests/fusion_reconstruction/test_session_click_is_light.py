"""A2: a Session row click is Step 4's shape -- the
highlight, the status line and the gates, nothing heavy.

Before: ``_on_patch_selected`` ``np.load``-ed every cached view of the
patch (84-190 MB, 340-690 ms on the test copy) into ``_cached_views`` --
which nothing ever read -- and reloaded the patch's QC CSV.  After: an
EXISTENCE check of the cache files only, and no QC reload (the Curation
list's click keeps it).

Bare controllers via ``__new__`` (no Qt, no napari).
"""
from __future__ import annotations

import ast
import builtins
import inspect
import io
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from napari_epicycle.fusion_reconstruction import _step3_pr_text as T
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
)

_MB = 1024 * 1024


class _SM:
    def __init__(self, status):
        self.status = status
        self.active_patch = ""

    def get_patch_status(self, name):
        return self.status


def _ctrl(tmp_path, status, *, cache=True, big=False):
    c = C.__new__(C)
    c.state_manager = _SM(status)
    c.working_dir = str(tmp_path)
    c.loaded_stack_dir = str(tmp_path / "patches")
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(exist_ok=True)
    if cache:
        size = (100 * _MB) // 6 if big else 16
        for view in ("xy", "xz", "yz"):
            for kind in ("flows", "probs"):
                with open(cache_dir / f"p1_{view}_{kind}.npy", "wb") as fh:
                    fh.truncate(size)
    c._prediction_cache_dir = lambda: cache_dir
    c._selected_patch_name = lambda: "p1.tif"
    c.said = []
    c.logged = []
    c.gated = []
    c.reloads = []
    c._pr_status = lambda text: c.said.append(text)
    c._append_log = lambda text: c.logged.append(text)
    c._refresh_pr_foot_gates = lambda **kw: c.gated.append(kw)
    c._reload_current_patch_features = lambda: c.reloads.append(1)
    return c


def _forbid_reads(monkeypatch):
    """Any file open, ``np.load`` or CSV read fails the test."""
    def _no(*a, **k):
        raise AssertionError(f"the Session click opened a file: {a!r}")
    monkeypatch.setattr(builtins, "open", _no)
    monkeypatch.setattr(io, "open", _no)
    monkeypatch.setattr(np, "load", _no)
    monkeypatch.setattr(pd, "read_csv", _no)


def test_a_click_on_a_predicted_patch_opens_no_file(tmp_path, monkeypatch):
    c = _ctrl(tmp_path, "predicted", big=True)
    _forbid_reads(monkeypatch)
    c._on_patch_selected()
    assert c.state_manager.active_patch == "p1.tif"
    # A2 fold L1: nothing is restored any more -- the words say "cached"
    assert c.said == [T.S3_STATUS_PREDICTIONS_CACHED.format(patch="p1.tif")]
    assert c.logged == [T.S3_LOG_PREDICTIONS_CACHED.format(patch="p1.tif")]
    assert not hasattr(T, "S3_STATUS_PREDICTIONS_RESTORED")
    assert c._cached_source_path == str(Path(c.loaded_stack_dir) / "p1.tif")


def test_the_click_calls_no_qc_reload(tmp_path, monkeypatch):
    ctrls = [_ctrl(tmp_path, s)
             for s in ("unprocessed", "predicted", "curating", "finalized")]
    _forbid_reads(monkeypatch)
    for c in ctrls:
        c._on_patch_selected()
        assert c.reloads == [], c.state_manager.status


def test_a_patch_with_no_cache_says_so_and_has_no_source(tmp_path,
                                                        monkeypatch):
    c = _ctrl(tmp_path, "reconstructed", cache=False)
    _forbid_reads(monkeypatch)
    c._on_patch_selected()
    assert c.said == [T.S3_STATUS_NO_CACHE.format(status="reconstructed")]
    assert c._cached_source_path == ""


def test_a_partial_cache_is_no_cache(tmp_path):
    c = _ctrl(tmp_path, "predicted")
    (tmp_path / "cache" / "p1_yz_probs.npy").unlink()
    assert c._prediction_cache_exists("p1.tif") is False
    c._on_patch_selected()
    assert c.said == [T.S3_STATUS_NO_CACHE.format(status="predicted")]


def test_an_unpredicted_patch_lets_the_gates_speak(tmp_path, monkeypatch):
    c = _ctrl(tmp_path, "unprocessed")
    _forbid_reads(monkeypatch)
    c._on_patch_selected()
    assert c.gated == [{"keep_status": False}]
    assert c.said == []
    assert c._cached_source_path == ""


def test_the_cached_views_are_gone():
    """Nothing read ``_cached_views``; the attribute and its writers go."""
    src = inspect.getsource(C)
    assert "_cached_views" not in src


def test_the_reconstruct_worker_still_restores_the_arrays():
    """The worker's reconstruct reads the arrays itself (a background
    thread) -- that restore stays."""
    src = textwrap.dedent(inspect.getsource(C._reconstruct_single_patch))
    assert "_try_restore_prediction_cache" in src


def _calls(fn):
    tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    return {n.func.attr for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}


def test_the_curation_click_still_reloads_the_features():
    """The Curation list's click switches (``_switch_to_curation_patch``)
    and the switch's async finish reloads the CSV."""
    assert "_switch_to_curation_patch" in _calls(C._on_curation_patch_clicked)
    assert "_reload_current_patch_features" in _calls(C._on_patch_loaded)
    assert "_reload_current_patch_features" not in _calls(
        C._on_patch_selected)

# (A2 fold M3: the Commit Triage resolution is held by a BEHAVIOUR test in
# test_step3_selection_rule.py -- the source-string test was vacuous)
