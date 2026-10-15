"""Unit tests for ``resolve_patch_axes`` (per-patch axis-order resolver).

The Step 3 predict + extract paths historically applied a SINGLE global
``axes`` value to every patch.  In a folder whose patches have different C/Z
orderings (some CZYX, some ZCYX) that one value is wrong for the mismatched
patches → IndexError on extract, silent mis-prediction on predict.
``resolve_patch_axes`` resolves the layout PER PATCH from the most
authoritative signal available:

    metadata → deterministic labels-shape match → size heuristic → fallback

These tests exercise each priority branch independently plus the
defensive contract (3D → "ZYX"; never raises on garbage).
"""

import numpy as np
import pytest

from napari_epicycle.fusion_reconstruction.qc_features import (
    detect_axes_order,
    resolve_patch_axes,
)


# ---------------------------------------------------------------------------
# (a) Metadata wins
# ---------------------------------------------------------------------------

def test_metadata_branch_zcyx(tmp_path):
    """An on-disk TIFF written with explicit ``axes='ZCYX'`` metadata is
    resolved as ZCYX purely from the path-metadata signal."""
    import tifffile

    # Genuinely ZCYX, thin-Z (Z=2, C=3) — the size heuristic would say CZYX,
    # so a correct ZCYX result here can ONLY have come from metadata (no
    # labels passed).
    arr = np.random.RandomState(1).randint(0, 255, (2, 3, 32, 32), dtype=np.uint8)
    p = tmp_path / "meta_zcyx.tif"
    tifffile.imwrite(str(p), arr, metadata={"axes": "ZCYX"})

    # Precondition: heuristic alone gets it wrong.
    assert detect_axes_order(arr.shape) == "CZYX"

    out = resolve_patch_axes(arr, image_path=str(p), fallback="CZYX")
    assert out == "ZCYX"


def test_metadata_branch_czyx(tmp_path):
    """Explicit ``axes='CZYX'`` metadata is honored over a wrong fallback."""
    import tifffile

    arr = np.random.RandomState(2).randint(0, 255, (3, 16, 32, 32), dtype=np.uint8)
    p = tmp_path / "meta_czyx.tif"
    tifffile.imwrite(str(p), arr, metadata={"axes": "CZYX"})

    out = resolve_patch_axes(arr, image_path=str(p), fallback="ZCYX")
    assert out == "CZYX"


def test_metadata_ignored_when_not_recognized(tmp_path):
    """Ambiguous tifffile-default axes (e.g. 'QSYX' for a plain 4D write) are
    NOT one of ZCYX/CZYX, so the metadata branch is skipped and a lower
    signal (here the heuristic) decides."""
    import tifffile

    # CZYX-ish by size: axis 0 (=2) is smallest → heuristic says CZYX.
    arr = np.random.RandomState(3).randint(0, 255, (2, 16, 32, 32), dtype=np.uint8)
    p = tmp_path / "ambiguous.tif"
    tifffile.imwrite(str(p), arr)  # no explicit axes metadata

    out = resolve_patch_axes(arr, image_path=str(p), fallback="ZCYX")
    # Falls through metadata → no labels → heuristic = CZYX.
    assert out == "CZYX"


# ---------------------------------------------------------------------------
# (b) Labels-shape match resolves a thin-Z ZCYX the heuristic gets wrong
# ---------------------------------------------------------------------------

def test_labels_match_beats_heuristic_thin_z():
    """Genuinely-ZCYX with thin Z (Z=2, C=3): the size heuristic mis-guesses
    CZYX, but the deterministic labels-shape match resolves ZCYX.  No path/
    metadata supplied — proof the labels branch is doing the work."""
    image = np.zeros((2, 3, 32, 32), dtype=np.uint8)  # (Z=2, C=3, Y, X) → ZCYX
    labels = np.zeros((2, 32, 32), dtype=np.int32)     # (Z, Y, X)

    # Precondition: heuristic alone is wrong.
    assert detect_axes_order(image.shape) == "CZYX"

    out = resolve_patch_axes(image, labels=labels, fallback="CZYX")
    assert out == "ZCYX"


def test_labels_match_czyx():
    """Channel-first 4D whose labels match arr[0] → CZYX."""
    image = np.zeros((3, 16, 32, 32), dtype=np.uint8)  # (C=3, Z=16, Y, X)
    labels = np.zeros((16, 32, 32), dtype=np.int32)     # (Z, Y, X)
    out = resolve_patch_axes(image, labels=labels, fallback="ZCYX")
    assert out == "CZYX"


def test_labels_match_zcyx_normal():
    """Z-first 4D whose labels match arr[:,0] → ZCYX."""
    image = np.zeros((16, 2, 32, 32), dtype=np.uint8)  # (Z=16, C=2, Y, X)
    labels = np.zeros((16, 32, 32), dtype=np.int32)
    out = resolve_patch_axes(image, labels=labels, fallback="CZYX")
    assert out == "ZCYX"


def test_labels_no_match_falls_through_to_heuristic():
    """If neither arr[0] nor arr[:,0] matches the labels shape (e.g. a
    downsample/shape mismatch), the labels branch is inconclusive and the
    size heuristic decides."""
    image = np.zeros((2, 16, 32, 32), dtype=np.uint8)  # heuristic → CZYX
    labels = np.zeros((99, 99, 99), dtype=np.int32)     # matches nothing
    out = resolve_patch_axes(image, labels=labels, fallback="ZCYX")
    assert out == "CZYX"  # from heuristic


# ---------------------------------------------------------------------------
# (c) Heuristic fallback when no labels / metadata
# ---------------------------------------------------------------------------

def test_heuristic_czyx_no_labels_no_path():
    image = np.zeros((2, 16, 32, 32), dtype=np.uint8)  # axis 0 smallest → CZYX
    out = resolve_patch_axes(image, fallback="ZCYX")
    assert out == "CZYX"
    assert out == detect_axes_order(image.shape)


def test_heuristic_zcyx_no_labels_no_path():
    image = np.zeros((16, 2, 32, 32), dtype=np.uint8)  # axis 1 smallest → ZCYX
    out = resolve_patch_axes(image, fallback="CZYX")
    assert out == "ZCYX"
    assert out == detect_axes_order(image.shape)


# ---------------------------------------------------------------------------
# (c2) SINGLE-CHANNEL 4D (C == 1) — the singleton channel is the smallest dim,
# so the size heuristic correctly identifies it.  These pin the C=1 case down
# explicitly because it is the most common real-world single-channel-as-4D
# form (a grayscale stack carrying a degenerate channel axis): (Z,1,Y,X) is
# ZCYX, (1,Z,Y,X) is CZYX.  Pre-existing heuristic tests only used C in {2,3}.
# ---------------------------------------------------------------------------

def test_heuristic_single_channel_z_first_is_zcyx():
    """(Z, 1, Y, X): the singleton is axis 1 → smallest → ZCYX (no labels)."""
    image = np.zeros((266, 1, 512, 512), dtype=np.uint8)
    assert detect_axes_order(image.shape) == "ZCYX"
    out = resolve_patch_axes(image, fallback="CZYX")
    assert out == "ZCYX"


def test_heuristic_single_channel_channel_first_is_czyx():
    """(1, Z, Y, X): the singleton is axis 0 → smallest → CZYX (no labels)."""
    image = np.zeros((1, 266, 512, 512), dtype=np.uint8)
    assert detect_axes_order(image.shape) == "CZYX"
    out = resolve_patch_axes(image, fallback="ZCYX")
    assert out == "CZYX"


def test_labels_match_single_channel_z_first_is_zcyx():
    """(Z, 1, Y, X) + labels (Z, Y, X): arr[:, 0].shape == labels → ZCYX.
    Deterministic match resolves it independent of the (already-correct)
    heuristic — proof the C=1 labels branch works."""
    image = np.zeros((16, 1, 32, 32), dtype=np.uint8)  # (Z, C=1, Y, X)
    labels = np.zeros((16, 32, 32), dtype=np.int32)     # (Z, Y, X)
    out = resolve_patch_axes(image, labels=labels, fallback="CZYX")
    assert out == "ZCYX"


def test_labels_match_single_channel_channel_first_is_czyx():
    """(1, Z, Y, X) + labels (Z, Y, X): arr[0].shape == labels → CZYX."""
    image = np.zeros((1, 16, 32, 32), dtype=np.uint8)  # (C=1, Z, Y, X)
    labels = np.zeros((16, 32, 32), dtype=np.int32)     # (Z, Y, X)
    out = resolve_patch_axes(image, labels=labels, fallback="ZCYX")
    assert out == "CZYX"


def test_single_channel_4d_no_recognizable_metadata_falls_to_heuristic(tmp_path):
    """tifffile writes a plain C=1 4D array with unrecognized axes ('QQYX'),
    so the metadata branch is skipped and the size heuristic (correctly)
    decides.  Confirms the real on-disk path for a 4D single-channel TIFF."""
    import tifffile

    arr = np.zeros((16, 1, 32, 32), dtype=np.uint8)  # ZCYX, C=1
    p = tmp_path / "sc_zcyx.tif"
    tifffile.imwrite(str(p), arr)  # no explicit axes metadata
    out = resolve_patch_axes(arr, image_path=str(p), fallback="CZYX")
    assert out == "ZCYX"


# ---------------------------------------------------------------------------
# (d) 3D → "ZYX"
# ---------------------------------------------------------------------------

def test_3d_returns_zyx():
    image = np.zeros((16, 32, 32), dtype=np.uint8)
    assert resolve_patch_axes(image, fallback="CZYX") == "ZYX"


def test_3d_returns_zyx_even_with_labels_and_path(tmp_path):
    import tifffile

    image = np.zeros((16, 32, 32), dtype=np.uint8)
    p = tmp_path / "vol3d.tif"
    tifffile.imwrite(str(p), image, metadata={"axes": "ZYX"})
    labels = np.zeros((16, 32, 32), dtype=np.int32)
    out = resolve_patch_axes(image, labels=labels, image_path=str(p), fallback="CZYX")
    assert out == "ZYX"


# ---------------------------------------------------------------------------
# (e) Fallback when nothing resolves
# ---------------------------------------------------------------------------

def test_fallback_used_when_heuristic_inconclusive():
    """A 4D shape the heuristic can't classify as ZCYX/CZYX (e.g. all-equal
    leading dims where it returns CZYX) — verify with a shape that forces the
    'else' branch.  Easiest: a 5D image (heuristic returns 'CZYX' via its
    len!=4 fallback)... instead use a shape the heuristic itself rejects."""
    # detect_axes_order returns "CZYX" for any non-4D; to truly hit the
    # fallback branch we need detect_axes_order to NOT return ZCYX/CZYX,
    # which only happens for len(shape)==3 → "ZYX".  A 3D array short-circuits
    # to "ZYX" earlier, so we exercise the fallback via a 4D shape where the
    # heuristic + labels are absent but fallback is explicit and valid.
    # Force heuristic to be skipped by passing an object whose .shape raises.
    class _Bad:
        ndim = 4

        @property
        def shape(self):
            raise RuntimeError("boom")

    out = resolve_patch_axes(_Bad(), fallback="CZYX")
    assert out == "CZYX"


def test_fallback_normalized_when_garbage():
    """A fallback that isn't ZCYX/CZYX is normalized to the EpiCycle default
    layout, CZYX."""
    class _Bad:
        ndim = 4

        @property
        def shape(self):
            raise RuntimeError("boom")

    out = resolve_patch_axes(_Bad(), fallback="WHATEVER")
    assert out == "CZYX"


def test_fallback_default_is_czyx():
    """No signal at all → CZYX (EpiCycle's default for its own extracted
    stacks; ZCYX is the anticipated alternative users may bring)."""
    class _Bad:
        ndim = 4

        @property
        def shape(self):
            raise RuntimeError("boom")

    # No fallback kwarg → default "CZYX".
    assert resolve_patch_axes(_Bad()) == "CZYX"


# ---------------------------------------------------------------------------
# (f) Never raises on garbage input
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "garbage",
    [
        None,
        "not an array",
        123,
        object(),
        np.zeros((5,), dtype=np.uint8),       # 1D
        np.zeros((4, 4), dtype=np.uint8),     # 2D
        np.zeros((2, 3, 4, 5, 6), dtype=np.uint8),  # 5D
    ],
)
def test_never_raises(garbage):
    out = resolve_patch_axes(garbage, fallback="ZCYX")
    assert out in ("ZYX", "ZCYX", "CZYX")


def test_never_raises_bad_path(tmp_path):
    image = np.zeros((2, 16, 32, 32), dtype=np.uint8)
    # Non-existent path → metadata branch must swallow the error and fall
    # through to the heuristic.
    out = resolve_patch_axes(
        image, image_path=str(tmp_path / "does_not_exist.tif"), fallback="ZCYX"
    )
    assert out == "CZYX"  # heuristic


def test_never_raises_bad_labels():
    """A labels object whose .shape blows up must not propagate."""
    image = np.zeros((2, 16, 32, 32), dtype=np.uint8)

    class _BadLabels:
        @property
        def shape(self):
            raise RuntimeError("boom")

    out = resolve_patch_axes(image, labels=_BadLabels(), fallback="ZCYX")
    assert out == "CZYX"  # falls through to heuristic


# ---------------------------------------------------------------------------
# Predict-path wiring: _predict_single_patch resolves per-patch and passes
# the resolved axes to segment_views (segment_views itself is mocked — no GPU).
# ---------------------------------------------------------------------------

class _PredictHarness:
    """Bare FusionReconstructionController to drive _predict_single_patch."""

    @staticmethod
    def make():
        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )
        c = FusionReconstructionController.__new__(FusionReconstructionController)
        c._persist_calls = []
        # Stub out the cache persist (no disk / no GUI).
        # R2c: the persist also takes the patch's prediction manifest
        c._persist_prediction_cache = lambda views, patch_name="", **_kw: (
            c._persist_calls.append(patch_name)
        )
        return c


def _patch_segment_views(monkeypatch):
    """Replace packages.epicycle_reconstruction.logic.segment_views with a
    capture stub so _predict_single_patch never touches Cellpose/GPU.
    Returns the captured-kwargs list (populated on call)."""
    import packages.epicycle_reconstruction.logic as logic_mod

    captured = []

    def fake_segment_views(image, lateral, axial, seg_config, axes="ZCYX", progress_cb=None):
        captured.append({"shape": tuple(image.shape), "axes": axes})
        return {"xy": (None, None, None), "xz": (None, None, None), "yz": (None, None, None)}

    monkeypatch.setattr(logic_mod, "segment_views", fake_segment_views)
    return captured


def test_predict_resolves_czyx_over_wrong_global(tmp_path, monkeypatch):
    """A CZYX patch on disk must be predicted as CZYX even when the global
    combo says ZCYX — _predict_single_patch resolves per patch."""
    import tifffile

    captured = _patch_segment_views(monkeypatch)
    arr = np.random.RandomState(5).randint(0, 255, (3, 16, 32, 32), dtype=np.uint8)
    p = tmp_path / "czyx_patch.tif"
    tifffile.imwrite(str(p), arr, metadata={"axes": "CZYX"})

    c = _PredictHarness.make()
    out = c._predict_single_patch(
        patch_name="czyx_patch.tif",
        stack_path=str(p),
        lateral_path="lat",
        axial_path="ax",
        seg_config=None,
        axes="ZCYX",  # WRONG global combo
    )
    assert captured, "segment_views was not called"
    assert captured[0]["axes"] == "CZYX"  # resolved per-patch, not the global
    assert out["patch_name"] == "czyx_patch.tif"
    assert c._persist_calls == ["czyx_patch.tif"]


def test_predict_resolves_zcyx_via_metadata(tmp_path, monkeypatch):
    import tifffile

    captured = _patch_segment_views(monkeypatch)
    # thin-Z ZCYX (heuristic would say CZYX) — metadata must win.
    arr = np.random.RandomState(6).randint(0, 255, (2, 3, 32, 32), dtype=np.uint8)
    p = tmp_path / "zcyx_patch.tif"
    tifffile.imwrite(str(p), arr, metadata={"axes": "ZCYX"})

    c = _PredictHarness.make()
    c._predict_single_patch(
        patch_name="zcyx_patch.tif", stack_path=str(p),
        lateral_path="lat", axial_path="ax", seg_config=None, axes="CZYX",
    )
    assert captured[0]["axes"] == "ZCYX"


def test_predict_3d_passes_zyx(tmp_path, monkeypatch):
    import tifffile

    captured = _patch_segment_views(monkeypatch)
    arr = np.random.RandomState(7).randint(0, 255, (16, 32, 32), dtype=np.uint8)
    p = tmp_path / "vol3d.tif"
    tifffile.imwrite(str(p), arr)

    c = _PredictHarness.make()
    c._predict_single_patch(
        patch_name="vol3d.tif", stack_path=str(p),
        lateral_path="lat", axial_path="ax", seg_config=None, axes="CZYX",
    )
    # 3D → resolver returns "ZYX"; segment_views treats 3D as single-channel
    # regardless, so passing "ZYX" is safe (it ignores axes for ndim==3).
    assert captured[0]["axes"] == "ZYX"


def test_predict_single_channel_4d_zcyx(tmp_path, monkeypatch):
    """A 4D single-channel (Z, 1, Y, X) patch resolves to ZCYX (singleton is
    axis 1 → smallest) and is handed to segment_views as ZCYX, even when the
    global fallback says CZYX."""
    import tifffile

    captured = _patch_segment_views(monkeypatch)
    arr = np.random.RandomState(8).randint(0, 255, (16, 1, 32, 32), dtype=np.uint8)
    p = tmp_path / "sc_zcyx.tif"
    tifffile.imwrite(str(p), arr)  # plain write → unrecognized axes → heuristic

    c = _PredictHarness.make()
    c._predict_single_patch(
        patch_name="sc_zcyx.tif", stack_path=str(p),
        lateral_path="lat", axial_path="ax", seg_config=None, axes="CZYX",
    )
    assert captured[0]["axes"] == "ZCYX"
    assert captured[0]["shape"] == (16, 1, 32, 32)


def test_predict_single_channel_4d_czyx(tmp_path, monkeypatch):
    """A 4D single-channel (1, Z, Y, X) patch resolves to CZYX (singleton is
    axis 0 → smallest) and is handed to segment_views as CZYX."""
    import tifffile

    captured = _patch_segment_views(monkeypatch)
    arr = np.random.RandomState(9).randint(0, 255, (1, 16, 32, 32), dtype=np.uint8)
    p = tmp_path / "sc_czyx.tif"
    tifffile.imwrite(str(p), arr)

    c = _PredictHarness.make()
    c._predict_single_patch(
        patch_name="sc_czyx.tif", stack_path=str(p),
        lateral_path="lat", axial_path="ax", seg_config=None, axes="ZCYX",
    )
    assert captured[0]["axes"] == "CZYX"
    assert captured[0]["shape"] == (1, 16, 32, 32)


def test_predict_single_patch_uses_resolver_in_source():
    """Source-level guard: _predict_single_patch must call resolve_patch_axes
    and pass the resolved value (not the raw global ``axes``) to segment_views."""
    import inspect
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    src = inspect.getsource(FusionReconstructionController._predict_single_patch)
    assert "resolve_patch_axes" in src
    # The resolved variable must be what's handed to segment_views.
    assert "axes=resolved_axes" in src
