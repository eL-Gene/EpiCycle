r"""Tests for the Stage-1 training-pair persist perf fix
(``fix/step2-persist-tiff-perf``).

Root cause (validated by live `[SAVE-TIMING]` profiling 2026-06-02):
``_persist_curated_training_pair`` blocked the main thread ~640ms per
edit on a slow ``D:\`` drive by doing three disk ops every stroke —
(1) a ``tifffile.imread`` of the old mask just to compare, (2) rewriting
the large float32 IMAGE TIFF even though the image never changes between
mask edits, (3) writing the mask.

Stage 1 (synchronous, no threading): replace the disk-read compare with
an in-memory content signature, and skip any (image|mask) write whose
file already exists AND whose signature is unchanged since the last
persist.  This kills the read + the redundant image rewrite.

These tests cover the two pure helpers directly plus source-inspection
wiring assertions (the controller class is too heavy to instantiate in a
unit test — same approach as test_label_compaction.py).
"""

from __future__ import annotations

import inspect

import numpy as np

import tifffile

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    _array_signature,
    _atomic_imwrite,
    _AsyncPairWriter,
    _compact_label_ids,
    _pair_write_plan,
    TrainingAnnotationController,
)


def _reference_compact(mask):
    """Reference (old per-label loop) implementation of _compact_label_ids,
    used to lock the vectorized version's equivalence."""
    arr = np.asarray(mask)
    uniq = np.unique(arr)
    fg = uniq[uniq != 0]
    out = np.zeros_like(arr)
    for new_id, old_id in enumerate(fg, start=1):
        out[arr == old_id] = new_id
    return out


# ---------------------------------------------------------------------------
# _array_signature
# ---------------------------------------------------------------------------

def test_signature_identical_content_matches():
    a = np.array([[0, 1], [2, 3]], dtype=np.int32)
    b = np.array([[0, 1], [2, 3]], dtype=np.int32)
    assert _array_signature(a) == _array_signature(b)


def test_signature_differs_on_content_change():
    a = np.array([[0, 1], [2, 3]], dtype=np.int32)
    b = np.array([[0, 1], [2, 99]], dtype=np.int32)
    assert _array_signature(a) != _array_signature(b)


def test_signature_differs_on_shape_change():
    a = np.zeros((4, 4), dtype=np.int32)
    b = np.zeros((4, 5), dtype=np.int32)
    assert _array_signature(a) != _array_signature(b)


def test_signature_differs_on_dtype_change():
    a = np.zeros((4, 4), dtype=np.int32)
    b = np.zeros((4, 4), dtype=np.float32)
    assert _array_signature(a) != _array_signature(b)


def test_signature_stable_for_noncontiguous_view():
    """A transposed / sliced (non-contiguous) view must hash the same as its
    contiguous copy — the helper must not depend on memory layout."""
    base = np.arange(12, dtype=np.int32).reshape(3, 4)
    view = base.T  # non-contiguous
    assert _array_signature(view) == _array_signature(np.ascontiguousarray(view))


def test_signature_stable_across_repeated_calls():
    a = np.random.randint(0, 5, size=(8, 8)).astype(np.int32)
    assert _array_signature(a) == _array_signature(a)


# ---------------------------------------------------------------------------
# _pair_write_plan
# ---------------------------------------------------------------------------

def test_plan_writes_both_when_files_missing():
    write_img, write_mask = _pair_write_plan(
        img_exists=False, mask_exists=False, img_sig=1, mask_sig=2, prior={}
    )
    assert write_img and write_mask


def test_plan_skips_both_when_present_and_unchanged():
    prior = {"img": 1, "mask": 2}
    write_img, write_mask = _pair_write_plan(
        img_exists=True, mask_exists=True, img_sig=1, mask_sig=2, prior=prior
    )
    assert not write_img and not write_mask


def test_plan_hot_loop_writes_mask_only():
    """The curating hot-loop case: mask edited (sig changed), image invariant
    and already on disk -> write mask, skip image."""
    prior = {"img": 1, "mask": 2}
    write_img, write_mask = _pair_write_plan(
        img_exists=True, mask_exists=True, img_sig=1, mask_sig=999, prior=prior
    )
    assert not write_img
    assert write_mask


def test_plan_writes_image_when_image_sig_changes():
    prior = {"img": 1, "mask": 2}
    write_img, write_mask = _pair_write_plan(
        img_exists=True, mask_exists=True, img_sig=42, mask_sig=2, prior=prior
    )
    assert write_img
    assert not write_mask


def test_plan_first_persist_writes_both_even_if_files_exist():
    """First persist of a session (empty prior) must (re)write both even if
    files exist on disk from a previous session — we have no in-memory
    signature to trust yet."""
    write_img, write_mask = _pair_write_plan(
        img_exists=True, mask_exists=True, img_sig=1, mask_sig=2, prior={}
    )
    assert write_img and write_mask


def test_plan_missing_file_forces_write_even_if_sig_matches():
    """If the on-disk file was deleted but the signature still matches the
    prior persist, we MUST rewrite (file absence wins over sig match)."""
    prior = {"img": 1, "mask": 2}
    write_img, write_mask = _pair_write_plan(
        img_exists=False, mask_exists=True, img_sig=1, mask_sig=2, prior=prior
    )
    assert write_img       # file gone -> rewrite
    assert not write_mask  # present + unchanged -> skip


def test_plan_sequence_write_then_skip_then_rewrite():
    """The update-then-reread loop the controller actually relies on: first
    persist (empty prior) writes + records sigs; an unchanged repeat skips;
    a content change rewrites.  Exercises the prior-dict round-trip, not just
    a single isolated call."""
    prior = {}
    # First persist: nothing cached, files don't exist yet -> write both.
    wi, wm = _pair_write_plan(False, False, img_sig=10, mask_sig=20, prior=prior)
    assert wi and wm
    if wi:
        prior["img"] = 10
    if wm:
        prior["mask"] = 20
    # Unchanged repeat, files now exist -> skip both.
    wi, wm = _pair_write_plan(True, True, img_sig=10, mask_sig=20, prior=prior)
    assert not wi and not wm
    # Mask edited (sig changes), image still invariant -> rewrite mask only.
    wi, wm = _pair_write_plan(True, True, img_sig=10, mask_sig=21, prior=prior)
    assert not wi and wm


def test_plan_missing_key_writes_even_when_sig_is_zero():
    """`prior.get(key)` returns None for a missing key; that must compare
    unequal to ANY signature including the legal value 0.  Guards against a
    future `.get(key, 0)` regression that would falsely skip the first write
    of a slice whose content happens to hash to 0."""
    write_img, write_mask = _pair_write_plan(
        img_exists=True, mask_exists=True, img_sig=0, mask_sig=0, prior={}
    )
    assert write_img and write_mask


def test_signature_empty_mask_stable_and_distinct():
    """All-zero (fully-erased) mask hashes stably and differs from a
    non-empty mask — locks the erase-to-empty path the controller depends
    on (an erased slice must still force a rewrite)."""
    empty = np.zeros((16, 16), dtype=np.int32)
    nonempty = empty.copy()
    nonempty[0, 0] = 1
    assert _array_signature(empty) == _array_signature(np.zeros((16, 16), dtype=np.int32))
    assert _array_signature(empty) != _array_signature(nonempty)
    # Same content, different shape -> different signature (guards the shape term).
    assert _array_signature(empty) != _array_signature(np.zeros((16, 17), dtype=np.int32))


# ---------------------------------------------------------------------------
# Wiring (source inspection) — the controller is too heavy to instantiate.
# ---------------------------------------------------------------------------

def test_persist_no_longer_reads_mask_from_disk_to_compare():
    """The per-edit tifffile.imread() of the old mask (the ~hundreds-of-ms
    read this fix removes) must be gone from the persist path."""
    src = inspect.getsource(
        TrainingAnnotationController._persist_curated_training_pair
    )
    assert "tifffile.imread" not in src, (
        "_persist_curated_training_pair must not read the old mask back from "
        "disk to compare — use the in-memory _array_signature instead."
    )


def test_persist_uses_write_plan_and_signatures():
    src = inspect.getsource(
        TrainingAnnotationController._persist_curated_training_pair
    )
    assert "_pair_write_plan(" in src
    assert "_array_signature(arr_img)" in src
    assert "_array_signature(arr_mask)" in src


def test_persist_sig_cache_keyed_by_output_path_not_slice_label():
    """Audit fix: the per-pair signature cache must be keyed by the resolved
    on-disk target (out_dir / fname_base), NOT slice_label.  Keying by
    slice_label would let a round-advance or run-switch — which redirect the
    same slice_label to a different file — wrongly skip a write against a
    signature cached for the other round/run."""
    src = inspect.getsource(
        TrainingAnnotationController._persist_curated_training_pair
    )
    assert "_persisted_pair_sigs.setdefault(" in src
    assert "out_dir / fname_base" in src, (
        "signature cache must be keyed by the resolved output path"
    )
    # The cache must NOT be keyed by slice_label (the unsafe coarse key).
    assert "_persisted_pair_sigs.setdefault(slice_label" not in src
    assert "_persisted_pair_sigs.setdefault(\n            slice_label" not in src


def test_persist_gates_both_writes_behind_the_plan():
    """Both imwrites must sit behind the write_img / write_mask guards so an
    unchanged file is never rewritten."""
    src = inspect.getsource(
        TrainingAnnotationController._persist_curated_training_pair
    )
    assert "if write_img:" in src
    assert "if write_mask:" in src
    # Mask compaction must still happen, but only inside the write_mask guard.
    assert "_compact_label_ids(arr_mask" in src


def test_persist_still_records_uris_and_returns_them():
    """Even when both writes are skipped, the record must still carry the
    uris and the method must return them (caller relies on a truthy dict)."""
    src = inspect.getsource(
        TrainingAnnotationController._persist_curated_training_pair
    )
    assert 'record["img_uri"] = str(img_path)' in src
    assert 'record["mask_uri"] = str(mask_path)' in src
    assert 'return {"img_uri": str(img_path), "mask_uri": str(mask_path)}' in src


# ---------------------------------------------------------------------------
# Stage 2 — vectorized compaction equivalence
# ---------------------------------------------------------------------------

def test_compact_vectorized_matches_reference_loop():
    """The vectorized _compact_label_ids must produce IDENTICAL output to the
    original per-label loop across a range of label patterns."""
    rng = np.random.RandomState(0)
    cases = [
        np.zeros((8, 8), dtype=np.int32),                         # all bg
        np.array([[0, 1], [1, 0]], dtype=np.int32),               # already 1..N
        np.array([[0, 5], [17, 203]], dtype=np.int32),            # sparse hi ids
        rng.randint(0, 7, size=(16, 16)).astype(np.int32),        # dense small
        (rng.randint(0, 50, size=(12, 12)) * 1000).astype(np.int32),  # inflated
    ]
    for m in cases:
        got = _compact_label_ids(m)
        ref = _reference_compact(m)
        assert np.array_equal(got, ref), f"mismatch for case max={m.max()}"
        # Foreground partition preserved.
        assert np.array_equal(got != 0, m != 0)


def test_compact_vectorized_preserves_identity_bijection():
    """Two pixels sharing an old label must share the new label, and distinct
    old labels must map to distinct new labels."""
    m = np.array([[10, 10, 20], [30, 20, 0]], dtype=np.int32)
    out = _compact_label_ids(m)
    assert out[0, 0] == out[0, 1]                    # both old-10 share
    assert out[0, 2] == out[1, 1]                    # both old-20 share
    assert out[1, 2] == 0                            # background stays 0
    assert len(np.unique(out[out != 0])) == 3        # 3 distinct fg ids
    assert set(np.unique(out)) == {0, 1, 2, 3}       # dense 0..3 (incl bg)


# ---------------------------------------------------------------------------
# Stage 2 — atomic write + async FIFO writer (real disk, no Qt)
# ---------------------------------------------------------------------------

def test_atomic_imwrite_roundtrip_and_no_temp_leftover(tmp_path):
    target = tmp_path / "sub" / "x_mask.tif"   # parent does not exist yet
    data = np.array([[0, 1], [2, 3]], dtype=np.int32)
    _atomic_imwrite(str(target), data)
    assert target.is_file()
    assert np.array_equal(np.asarray(tifffile.imread(str(target))), data)
    # No stray temp files left in the directory.
    leftovers = [p for p in target.parent.iterdir() if p.name != target.name]
    assert leftovers == [], f"unexpected temp leftovers: {leftovers}"


def test_atomic_imwrite_overwrite_replaces_content(tmp_path):
    target = tmp_path / "m.tif"
    _atomic_imwrite(str(target), np.zeros((4, 4), dtype=np.int32))
    _atomic_imwrite(str(target), np.ones((4, 4), dtype=np.int32))
    assert np.array_equal(np.asarray(tifffile.imread(str(target))),
                          np.ones((4, 4), dtype=np.int32))


def test_async_writer_submit_then_flush_writes_file(tmp_path):
    writer = _AsyncPairWriter()
    target = tmp_path / "pair_mask.tif"
    data = np.array([[1, 2], [3, 4]], dtype=np.int32)
    writer.submit(str(target), data)
    writer.flush()   # barrier — must block until the write lands
    assert target.is_file()
    assert np.array_equal(np.asarray(tifffile.imread(str(target))), data)


def test_async_writer_fifo_last_wins_same_path(tmp_path):
    """Multiple submits to the SAME path must apply in FIFO order; the last
    submitted content wins after flush (single ordered worker guarantee)."""
    writer = _AsyncPairWriter()
    target = tmp_path / "m.tif"
    for v in range(1, 6):
        writer.submit(str(target), np.full((8, 8), v, dtype=np.int32))
    writer.flush()
    assert np.array_equal(np.asarray(tifffile.imread(str(target))),
                          np.full((8, 8), 5, dtype=np.int32))


def test_async_writer_flush_when_empty_is_noop():
    """flush() on a never-used writer must return immediately, not hang."""
    writer = _AsyncPairWriter()
    writer.flush()  # should not block / raise


def test_async_writer_failed_write_does_not_hang_flush(tmp_path):
    """A write that RAISES (e.g. disk full / bad path) must still let flush()
    return — task_done() is in a finally, so the barrier can't block forever.
    This is the load-bearing 'self-healing' guarantee: a failed async write
    must not wedge train/switch/shutdown."""
    import threading as _t
    writer = _AsyncPairWriter()
    # A directory path where a file is expected -> _atomic_imwrite's os.replace
    # (or imwrite) will raise inside the worker.
    bad_dir = tmp_path / "iam_a_directory.tif"
    bad_dir.mkdir()
    writer.submit(str(bad_dir), np.zeros((4, 4), dtype=np.int32))

    done = _t.Event()
    waiter = _t.Thread(target=lambda: (writer.flush(), done.set()))
    waiter.start()
    waiter.join(timeout=10)
    assert done.is_set(), "flush() hung after a failed async write"


def test_async_writer_submit_is_nonblocking(tmp_path, monkeypatch):
    """submit() must return to the caller WITHOUT waiting for the (slow) disk
    write — that non-blocking enqueue is the entire point of Stage 2."""
    import time
    import napari_epicycle.hitl_training_annotation.controller.training_annotation_controller as ctrl_mod

    real = ctrl_mod._atomic_imwrite

    def _slow(path, data):
        time.sleep(0.4)
        real(path, data)

    monkeypatch.setattr(ctrl_mod, "_atomic_imwrite", _slow)
    writer = _AsyncPairWriter()
    target = tmp_path / "slow.tif"
    t0 = time.perf_counter()
    writer.submit(str(target), np.zeros((4, 4), dtype=np.int32))
    enqueue_ms = (time.perf_counter() - t0) * 1000.0
    assert enqueue_ms < 100, f"submit blocked {enqueue_ms:.0f}ms (should be ~0)"
    writer.flush()  # now wait for the slow write to actually land
    assert target.is_file()


def test_async_writer_multiple_paths_all_land(tmp_path):
    writer = _AsyncPairWriter()
    paths = [tmp_path / f"m{i}.tif" for i in range(5)]
    for i, p in enumerate(paths):
        writer.submit(str(p), np.full((4, 4), i, dtype=np.int32))
    writer.flush()
    for i, p in enumerate(paths):
        assert p.is_file()
        assert int(np.asarray(tifffile.imread(str(p)))[0, 0]) == i


# ---------------------------------------------------------------------------
# Stage 2 — wiring (source inspection)
# ---------------------------------------------------------------------------

def test_persist_writes_via_async_writer_not_inline_imwrite():
    src = inspect.getsource(
        TrainingAnnotationController._persist_curated_training_pair
    )
    # async opt-in param exists; debounce hot-loop uses it.
    assert "async_write=False" in src
    # Both files routed through the _emit helper (async submit OR sync atomic).
    assert "_emit(mask_path, compact_mask)" in src
    assert "_emit(img_path," in src
    # _emit offloads to the background writer when async_write, else atomic.
    assert "_pair_writer.submit(" in src
    assert "_atomic_imwrite(str(path), data)" in src
    # The blocking *direct* tifffile.imwrite must be gone from the persist path
    # (atomicity + threading both go through _atomic_imwrite now).
    assert "tifffile.imwrite(str(mask_path)" not in src
    assert "tifffile.imwrite(str(img_path)" not in src


def test_persist_debounce_path_uses_async_write():
    """The per-stroke debounced curating-persist (the lag source) must call
    persist with async_write=True; deliberate callers must NOT."""
    flush_src = inspect.getsource(
        TrainingAnnotationController._flush_curating_persist_pending
    )
    assert "async_write=True" in flush_src


def test_hard_barriers_flush_the_async_writer():
    """Train / patch-switch / run-switch (via _flush_pending_state_timers) and
    shutdown MUST flush the async writer so files are on disk before anything
    reads them back.  The per-stroke debounce path must NOT (verified by the
    fact that _flush_curating_persist_pending itself contains no writer.flush)."""
    barrier_src = inspect.getsource(
        TrainingAnnotationController._flush_pending_state_timers
    )
    shutdown_src = inspect.getsource(TrainingAnnotationController.shutdown)
    assert "writer.flush()" in barrier_src
    assert "writer.flush()" in shutdown_src
    # The non-blocking debounce path must NOT flush (that would re-block draw).
    debounce_src = inspect.getsource(
        TrainingAnnotationController._flush_curating_persist_pending
    )
    assert "writer.flush()" not in debounce_src
    assert ".flush()" not in debounce_src
