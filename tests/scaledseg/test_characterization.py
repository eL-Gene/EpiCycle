"""
Characterization tests — the Mode-A regression net (Stack 2).

These LOCK the current behavior of the existing pipeline so that:
  * the upcoming Mode-B IoU fork (Stack 6) can be proven byte-identical
    THROUGH Stage 2, and Mode A stays byte-identical after the fork;
  * the catastrophic-but-silent axis + param drifts that later stacks fix
    have a concrete, failing-on-fix anchor (bug-first TDD).

Nothing here changes production code — it only pins what the engine does TODAY.
Where a value is a KNOWN BUG that a later stack will fix, the test says so and
asserts the (wrong) current output; that stack flips the assertion when it
lands the fix. Pointers to the owning stack are in each docstring.

Bug-locks use ``@pytest.mark.xfail(strict=True)`` and assert the CORRECT
post-fix value: today they xfail, and the moment the owning stack lands the fix
they XPASS -> strict failure -> a self-announcing "remove the marker" signal
(so the bug-lock and its fix don't drift apart across stacks).

Scope notes (verified during Stack-2 scoping, not assumed):
  * prob-stitch OOM fix  -> Stack 7 (zarr seam); no cheap non-zarr fix exists.
    Here we only characterize correctness on a tiny input.
  * ZCYX->CZYX/ZYX axis fix -> Stack 4 (axis resolver). Here we lock the ZCYX
    happy path; the CZYX mis-unpack is an xfail(strict) bug-lock.
  * renorm_gradient flip (False->True) + golden parity -> DONE in Stack 3
    (defaults.py now True; parity in tests/test_smoothing_parity.py). The
    Stack-2 xfail bug-lock was removed when the flip landed.
  * smoothing-off identity + Stage-4 negative test -> Stack 3 (smoothing stack);
    default-OFF is already covered by tests/test_reconstruction.py.
  * tile_size 512 default is a bare-function footgun (unreachable via
    AssemblyStage) -> documented for Stack 5 tiling cleanup.
  * prob_stitch_mode default DIVERGES: AssemblyStage.__init__ defaults 'max'
    but the CLI defaults 'average' and always passes it through, so the
    EFFECTIVE production default is 'average'. Locked below so Stack 6's
    "Mode A byte-identical" prob comparison uses the real default.
  * Per-tile Stage-2 <tile>_mask.tif golden is NOT characterized here (it needs
    segment3D + GPU). DELIBERATE: "identical through Stage 2" is enforced at the
    contract level (conftest fixtures mirror the real uint32/full-Z/BigTIFF
    tile) and at the equivalence level downstream (Stack 6 asserts the fork
    reads identical tiles + Mode A stitch output is unchanged). Not a silent hole.
"""

from pathlib import Path

import numpy as np
import pytest
import tifffile

from packages.epicycle_scaledseg.core.assembly import (
    AssemblyStage,
    parse_tile_positions,
    stitch_binary_volume,
    stitch_mask_volume,
    stitch_prob_volume,
)
from packages.epicycle_scaledseg.utils.tiling import ZStackTiler

# The whole file is the Mode-A regression net.
pytestmark = pytest.mark.characterization


# ---------------------------------------------------------------------------
# Mode-A mask stitch: averaging of integer label IDs over overlaps
# ---------------------------------------------------------------------------


class TestMaskOverlapStitch:
    """Locks the current overlap math: slice_acc/norm_acc averaged + rounded
    (assembly.py:138-155). This is the behavior Mode B intentionally replaces;
    pinning it proves 'Mode A byte-identical' across the Stack-6 fork.
    """

    def _stitch(self, t0, t1, tmp_path):
        # full_size = (height, width). tile1 origin x0=12 -> width = 12 + 16 = 28.
        out = tmp_path / "stitched_mask.tif"
        stitch_mask_volume(
            [t0.position, t1.position],
            recon_root=t0.dir.parent,  # fixture base; assembly re-prepends dir.name
            output_path=str(out),
            full_size=(16, 28),
            tile_size=16,
            depth=16,
            n_workers=1,
        )
        return tifffile.imread(str(out))

    def test_straddle_cell_survives_seam(self, two_overlapping_tiles, tmp_path):
        """In the overlap, both tiles carry label 3, so round((3+3)/2)=3 — the
        straddle cell stays one ID. (It only stays clean because both sides
        agree; mismatched IDs would average to garbage — see next test.)"""
        t0, t1 = two_overlapping_tiles
        result = self._stitch(t0, t1, tmp_path)
        assert result.shape == (16, 16, 28)  # (Z, Y, X)
        # Seam = global X[12,16), Y[4,12), Z[4,12): label 3 on both sides.
        assert (result[4:12, 4:12, 12:16] == 3).all()
        # Label 3 spans tile0[10,16) ∪ tile1[12,18) -> global X[10,18).
        assert (result[4:12, 4:12, 10:18] == 3).all()

    def test_singletons_keep_their_ids(self, two_overlapping_tiles, tmp_path):
        """Cells outside the seam are single-covered (norm=1) -> exact ID."""
        t0, t1 = two_overlapping_tiles
        result = self._stitch(t0, t1, tmp_path)
        # label 1: tile0 only, Y[4,12), global X[2,6).
        assert (result[4:12, 4:12, 2:6] == 1).all()
        # label 2: tile1 only, Y[4,12), global X[22,26).
        assert (result[4:12, 4:12, 22:26] == 2).all()
        # background outside every cell stays 0 (e.g. the Z-pad region).
        assert result[:4].sum() == 0 and result[12:].sum() == 0

    def test_averaging_produces_garbage_on_mismatched_ids(self, make_tile_mask, tmp_path):
        """The documented 'averaging is wrong for instances' failure: when two
        DIFFERENT labels overlap, round((a+b)/2) invents a third ID. This is the
        core motivation for Mode B; lock it so Stack 6 can prove it's gone."""
        base = tmp_path / "mism"
        # tile0 label 4 fills its right half [8,16); tile1 label 10 fills its
        # left half [0,8) -> they collide in the seam global X[12,16).
        t0 = make_tile_mask(
            row=0, col=0, base_dir=base,
            labels={4: (slice(4, 12), slice(8, 16))},
        )
        t1 = make_tile_mask(
            row=0, col=1, base_dir=base,
            labels={10: (slice(4, 12), slice(0, 8))},
        )
        out = tmp_path / "mism.tif"
        stitch_mask_volume(
            [t0.position, t1.position], recon_root=base, output_path=str(out),
            full_size=(16, 28), tile_size=16, depth=16, n_workers=1,
        )
        result = tifffile.imread(str(out))
        # Seam global X[12,16): round((4+10)/2) = 7 — a label that exists in
        # NEITHER tile. Pure averaging artifact.
        assert (result[4:12, 4:12, 12:16] == 7).all()


class TestAssemblyDeterminism:
    """Re-running the stitch on identical inputs yields byte-identical output
    (assembly is deterministic — no skip-if-exists guard today; adding one is a
    deferred nice-to-have, NOT bundled into this characterization stack)."""

    def test_mask_stitch_is_repeatable(self, two_overlapping_tiles, tmp_path):
        t0, t1 = two_overlapping_tiles
        outs = []
        for i in range(2):
            out = tmp_path / f"run_{i}.tif"
            stitch_mask_volume(
                [t0.position, t1.position], recon_root=t0.dir.parent,
                output_path=str(out), full_size=(16, 28), tile_size=16,
                depth=16, n_workers=1,
            )
            outs.append(tifffile.imread(str(out)))
        assert np.array_equal(outs[0], outs[1])

    def test_mask_stitch_workers_invariant(self, two_overlapping_tiles, tmp_path):
        """Multi-worker (Pool path) output is byte-identical to single-worker.
        The repeatability test above runs n_workers=1, which never exercises the
        multiprocessing.Pool; this one does (n_workers=2). pool.map preserves
        submission order and the slice accumulation is commutative float add, so
        the outputs must match — if they ever don't, that's a real Mode-A bug."""
        t0, t1 = two_overlapping_tiles
        outs = []
        for nw in (1, 2):
            out = tmp_path / f"nw_{nw}.tif"
            stitch_mask_volume(
                [t0.position, t1.position], recon_root=t0.dir.parent,
                output_path=str(out), full_size=(16, 28), tile_size=16,
                depth=16, n_workers=nw,
            )
            outs.append(tifffile.imread(str(out)))
        assert np.array_equal(outs[0], outs[1])


# ---------------------------------------------------------------------------
# Mode-A prob stitch: average / max blend over overlaps
# ---------------------------------------------------------------------------


class TestProbOverlapStitch:
    """Locks stitch_prob_volume's average + max blend (assembly.py:197-222).
    Also documents the full-volume np.zeros((depth,H,W)) allocation at :197 —
    the guaranteed-OOM-at-scale site whose fix is deferred to Stack 7 (zarr seam)."""

    def _two_prob_tiles(self, tmp_path, p0=0.8, p1=0.4):
        base = tmp_path / "recon"
        # 1 Z slice, 4x4 tiles; tile1 at x0=2 -> overlap X[2,4), width=6.
        for name, val in [("tile_0_0", p0), ("tile_0_1", p1)]:
            d = base / name
            d.mkdir(parents=True)
            np.save(d / f"{name}_probabilities.npy",
                    np.full((1, 4, 4), val, dtype=np.float32))
        positions = [(0, 0, base / "tile_0_0"), (0, 2, base / "tile_0_1")]
        return base, positions

    def test_average_blend(self, tmp_path):
        base, positions = self._two_prob_tiles(tmp_path)
        out = tmp_path / "prob_avg.npy"
        stitch_prob_volume(positions, recon_root=base, output_path=str(out),
                           full_size=(4, 6), tile_size=4, depth=1,
                           stitch_mode="average")
        vol = np.load(str(out))
        assert vol.shape == (1, 4, 6)
        assert np.allclose(vol[0, :, 0:2], 0.8)          # tile0 only
        assert np.allclose(vol[0, :, 2:4], (0.8 + 0.4) / 2)  # overlap averaged
        assert np.allclose(vol[0, :, 4:6], 0.4)          # tile1 only

    def test_max_blend(self, tmp_path):
        base, positions = self._two_prob_tiles(tmp_path)
        out = tmp_path / "prob_max.npy"
        stitch_prob_volume(positions, recon_root=base, output_path=str(out),
                           full_size=(4, 6), tile_size=4, depth=1,
                           stitch_mode="max")
        vol = np.load(str(out))
        assert np.allclose(vol[0, :, 2:4], 0.8)  # overlap takes the max

    def test_effective_prob_stitch_mode_default_is_average_not_max(self):
        """AssemblyStage's CONSTRUCTOR default is 'max' (assembly.py:357), but the
        CLI default is 'average' (cli/main.py:90) and is ALWAYS passed through
        (:223) — so the EFFECTIVE production default is 'average'. A leaf
        stitch_prob_volume test can't see this. Locked so Stack 6's
        'Mode A byte-identical' prob comparison uses 'average', and so a future
        convergence (constructor -> 'average') trips this test on purpose."""
        assert AssemblyStage().prob_stitch_mode == "max"  # constructor default only


class TestBinaryOverlapStitch:
    """Locks stitch_binary_volume's OR-over-overlap semantics (assembly.py:281/294)
    + the tif save_format branch. Mode B's fork replaces binary stitch too, so it's
    on the 'Mode A byte-identical' boundary; existing test_assembly.py covers only a
    single tile (no seam, no tif branch)."""

    def _two_binary_tiles(self, tmp_path, depth=1):
        base = tmp_path / "recon"
        for name in ("tile_0_0", "tile_0_1"):
            d = base / name
            d.mkdir(parents=True)
            a = np.zeros((depth, 4, 4), dtype=np.uint8)
            a[:, :, 2:4] = 1  # foreground in each tile's right half (local x[2,4))
            np.save(d / f"{name}_binary.npy", a)
        positions = [(0, 0, base / "tile_0_0"), (0, 2, base / "tile_0_1")]
        return base, positions

    def test_or_union_over_overlap(self, tmp_path):
        """Overlap global X[2,4): OR(tile0 fg=1, tile1 bg=0) = 1 — the union
        survives even where one tile is background."""
        base, positions = self._two_binary_tiles(tmp_path)
        out = tmp_path / "binary.npy"
        stitch_binary_volume(positions, recon_root=base, output_path=str(out),
                             full_size=(4, 6), tile_size=4, depth=1,
                             save_format="npy")
        vol = np.load(str(out))
        assert vol.shape == (1, 4, 6)
        assert (vol[0, :, 0:2] == 0).all()  # tile0-only, tile0 bg there
        assert (vol[0, :, 2:4] == 1).all()  # overlap union
        assert (vol[0, :, 4:6] == 1).all()  # tile1-only, tile1 fg there

    def test_tif_format_roundtrip(self, tmp_path):
        base, positions = self._two_binary_tiles(tmp_path, depth=2)
        out = tmp_path / "binary.tif"
        stitch_binary_volume(positions, recon_root=base, output_path=str(out),
                             full_size=(4, 6), tile_size=4, depth=2,
                             save_format="tif")
        vol = tifffile.imread(str(out))
        assert vol.shape == (2, 4, 6)
        assert (vol[:, :, 2:4] == 1).all()
        with tifffile.TiffFile(str(out)) as tf:
            assert tf.is_bigtiff


# ---------------------------------------------------------------------------
# Tiler axis contract: hard ZCYX unpack (Stack 4 will generalize)
# ---------------------------------------------------------------------------


class TestTilerAxisContract:
    """Locks the current hard ``self.Z,self.C,self.H,self.W = shape`` unpack
    (tiling.py:70). Sets up Stack 4's axis resolver: the ZCYX happy path must
    keep working, and the CZYX mis-unpack is the bug Stack 4 flips to a pass."""

    def _write_4d(self, tmp_path, shape, name):
        arr = np.zeros(shape, dtype=np.uint16)
        p = tmp_path / name
        tifffile.imwrite(str(p), arr)
        return p

    def test_zcyx_unpacks_correctly(self, tmp_path):
        """ZCYX (Z=5,C=2,H=8,W=8) — the engine's documented canonical input."""
        p = self._write_4d(tmp_path, (5, 2, 8, 8), "zcyx.tif")
        tiler = ZStackTiler(
            p, tmp_path / "out_zcyx",
            target_patch_size=8, overlap=0.25, search_range=0,
        )
        assert (tiler.Z, tiler.C, tiler.H, tiler.W) == (5, 2, 8, 8)

    def test_czyx_unpacks_correctly(self, tmp_path):
        """CZYX (C=2,Z=5,H=8,W=8) resolves to Z=5,C=2 — fixed in Stack 4 by the
        axis resolver canonicalizing to ZCYX. Was an xfail bug-lock in Stack 2;
        Stack 4 landed the fix and removed the marker. (See tests/test_axes.py
        for the full resolver coverage.)"""
        p = self._write_4d(tmp_path, (2, 5, 8, 8), "czyx.tif")
        tiler = ZStackTiler(
            p, tmp_path / "out_czyx",
            target_patch_size=8, overlap=0.25, search_range=0,
        )
        assert (tiler.Z, tiler.C) == (5, 2)


# ---------------------------------------------------------------------------
# Reconstruction-param defaults: renorm_gradient drift (Stack 3 flips Stage 2)
# ---------------------------------------------------------------------------


class TestReconParamDefaults:
    """Locks the renorm_gradient state. Stage 2 (direct) is True (flipped in
    Stack 3 for monorepo parity); Stage 4 (indirect) is True. Requires segment3D.
    Broader param parity lives in tests/test_smoothing_parity.py."""

    def test_stage2_direct_renorm_gradient_is_true(self):
        """Stage-2 direct renorm_gradient is True (flipped from False in Stack 3
        for monorepo parity; defaults.py). Was an xfail bug-lock in Stack 2;
        Stack 3 landed the flip and removed the marker. Further parity is locked
        by tests/test_smoothing_parity.py against the reference snapshot."""
        pytest.importorskip("segment3D")
        from packages.epicycle_scaledseg.config.defaults import get_default_reconstruct_params
        params = get_default_reconstruct_params()
        assert params["gradient_descent"]["renorm_gradient"] is True

    def test_stage4_indirect_renorm_gradient_is_true(self):
        pytest.importorskip("segment3D")
        from packages.epicycle_scaledseg.config.defaults import get_default_indirect_reconstruct_params
        params = get_default_indirect_reconstruct_params()
        assert params["gradient_descent"]["renorm_gradient"] is True


# ---------------------------------------------------------------------------
# tile_size bare-function footgun (documented for Stack 5)
# ---------------------------------------------------------------------------


def test_parse_tile_positions_bare_default_is_512():
    """parse_tile_positions defaults tile_size=512 (step=384). This default is
    UNREACHABLE via AssemblyStage (which derives tile_size from the config
    patch_size and hard-fails on None) — it only bites direct module-function
    callers. Documented here so Stack 5's tiling cleanup removes the footgun."""
    pos = parse_tile_positions([Path("tile_1_1")])
    assert pos[0][0:2] == (384, 384)  # int(512*(1-0.25)) = 384
