"""The header reader and ImageJ stacks that carry no ``spacing`` key.

ImageJ's own TIFF writer puts ``spacing=`` into its image description only for
a stack (more than one image) whose voxel depth is NOT 1.0, and its reader
keeps the default depth of 1.0 when the key is absent.  So a multi-slice
stack that ImageJ (or Fiji) saved, with a physical unit, an XY pixel size and
no ``spacing``, has a voxel depth of exactly 1 unit: Z is 1 unit, read from
the file, not assumed from XY.

tifffile writes the same ``ImageJ=`` marker with its own fixed version
(``1.11a``) and no such rule -- a scripted stack whose depth nobody knows --
so a file with that version keeps the old rule (Z filled from XY, flagged),
as do a one-slice image, a non-ImageJ TIFF and a stack with no physical unit.

The files under ``fixtures/imagej_depth/`` are 3-slice ZYX stacks (0.2 um
pixels, unit "um"): three with a real ImageJ version (no ``spacing`` key,
``spacing=1.0``, ``spacing=2.0``) and one with tifffile's version and no key.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import tifffile

from packages.epicycle_largedata.axes import (
    read_voxel_spacing, read_voxel_spacing_detail,
)

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "imagej_depth")
REAL = "1.54f"          # a version string ImageJ / Fiji writes


def _fx(name):
    return os.path.join(FIX, name)


def _description(path):
    with tifffile.TiffFile(path) as tf:
        return tf.pages[0].description


def _stack(path, shape=(3, 8, 8), version=REAL, **md):
    meta = {"axes": "ZYX" if len(shape) == 3 else "ZCYX", "unit": "um"}
    if version is not None:
        meta["ImageJ"] = version
    meta.update(md)
    res = meta.pop("res", (1 / 0.2, 1 / 0.2))
    tifffile.imwrite(str(path), np.zeros(shape, np.uint8), imagej=True,
                     resolution=res, metadata=meta)
    return str(path)


# --- the fixtures are what they claim to be --------------------------------

def test_the_absent_fixture_is_an_imagej_stack_without_spacing():
    d = _description(_fx("zyx_depth_absent.tif"))
    assert d.startswith(f"ImageJ={REAL}\n")
    assert "slices=3" in d and "unit=um" in d
    assert "spacing" not in d


@pytest.mark.parametrize("name,value", [("zyx_depth_1.tif", "1.0"),
                                        ("zyx_depth_2.tif", "2.0")])
def test_the_spacing_fixtures_carry_their_key(name, value):
    d = _description(_fx(name))
    assert d.startswith(f"ImageJ={REAL}\n")
    assert f"spacing={value}" in d and "slices=3" in d


def test_the_tifffile_fixture_carries_tifffiles_version():
    d = _description(_fx("zyx_depth_absent_tifffile.tif"))
    assert d.startswith("ImageJ=1.11a\n") and "spacing" not in d


def test_tifffile_still_writes_the_fixtures_the_same_way(tmp_path):
    """The committed bytes match what tifffile writes today for the same call,
    so the fixtures are not a stale hand-made header."""
    p = _stack(tmp_path / "again.tif")
    assert _description(p) == _description(_fx("zyx_depth_absent.tif"))
    q = _stack(tmp_path / "again2.tif", version=None)
    assert _description(q) == _description(
        _fx("zyx_depth_absent_tifffile.tif"))


# --- the rule ----------------------------------------------------------------

def test_a_stack_imagej_saved_without_spacing_reads_a_depth_of_one_unit():
    d = read_voxel_spacing_detail(_fx("zyx_depth_absent.tif"))
    assert d["zyx"] == pytest.approx((1.0, 0.2, 0.2))
    assert d["z_from_xy"] is False
    assert d["z_imagej_default"] is True


def test_the_absent_and_the_explicit_one_read_the_same():
    assert read_voxel_spacing(_fx("zyx_depth_absent.tif")) == pytest.approx(
        read_voxel_spacing(_fx("zyx_depth_1.tif")))
    d = read_voxel_spacing_detail(_fx("zyx_depth_1.tif"))
    assert d["z_from_xy"] is False and d["z_imagej_default"] is False


def test_an_explicit_spacing_is_read_as_written():
    d = read_voxel_spacing_detail(_fx("zyx_depth_2.tif"))
    assert d["zyx"] == pytest.approx((2.0, 0.2, 0.2))
    assert d["z_from_xy"] is False


def test_tifffiles_own_stack_keeps_the_xy_fill():
    """tifffile's version marker: the depth is unknown, Z stays assumed."""
    d = read_voxel_spacing_detail(_fx("zyx_depth_absent_tifffile.tif"))
    assert d["zyx"] == pytest.approx((0.2, 0.2, 0.2))
    assert d["z_from_xy"] is True
    assert d["z_imagej_default"] is False


def test_the_depth_of_one_unit_follows_the_unit(tmp_path):
    p = _stack(tmp_path / "nm.tif", unit="nm", res=(1 / 200.0, 1 / 200.0))
    d = read_voxel_spacing_detail(p)
    assert d["zyx"] == pytest.approx((0.001, 0.2, 0.2))
    assert d["z_from_xy"] is False


def test_a_one_slice_image_keeps_the_xy_fill(tmp_path):
    p = str(tmp_path / "one.tif")
    tifffile.imwrite(p, np.zeros((8, 8), np.uint8), imagej=True,
                     resolution=(1 / 0.2, 1 / 0.2),
                     metadata={"unit": "um", "ImageJ": REAL})
    d = read_voxel_spacing_detail(p)
    assert d["zyx"] == pytest.approx((0.2, 0.2, 0.2))
    assert d["z_from_xy"] is True


def test_channels_without_slices_keep_the_xy_fill(tmp_path):
    """A CYX image has one slice: its depth means nothing, Z stays assumed."""
    p = _stack(tmp_path / "cyx.tif", shape=(2, 8, 8), axes="CYX")
    assert read_voxel_spacing_detail(p)["z_from_xy"] is True


def test_a_zcyx_stack_without_spacing_reads_one_unit(tmp_path):
    p = _stack(tmp_path / "zcyx.tif", shape=(3, 2, 8, 8))
    d = read_voxel_spacing_detail(p)
    assert d["zyx"] == pytest.approx((1.0, 0.2, 0.2))
    assert d["z_from_xy"] is False


def test_a_non_imagej_xy_only_tiff_keeps_the_xy_fill(tmp_path):
    p = str(tmp_path / "plain.tif")
    tifffile.imwrite(p, np.zeros((4, 16, 16), np.uint16),
                     resolution=(1000, 1000), resolutionunit="CENTIMETER")
    d = read_voxel_spacing_detail(p)
    assert d["zyx"] == pytest.approx((10.0, 10.0, 10.0))
    assert d["z_from_xy"] is True


def test_a_stack_with_no_physical_unit_gives_no_depth(tmp_path):
    """No unit (or inch) means no physical size at all: nothing is invented."""
    p = str(tmp_path / "nounit.tif")
    tifffile.imwrite(p, np.zeros((3, 8, 8), np.uint8), imagej=True,
                     metadata={"axes": "ZYX", "ImageJ": REAL})
    assert read_voxel_spacing(p) is None
    q = _stack(tmp_path / "inch.tif", unit="inch", res=(72, 72))
    assert read_voxel_spacing(q) is None


def test_a_tifffile_mm_stack_with_the_default_resolution_stays_assumed(
        tmp_path):
    """tifffile writes a default resolution of 1 pixel per unit: 1 mm XY.
    With tifffile's version the Z is NOT read as 1 unit (it stays the XY
    fill, flagged); such a size is never recorded as detected (an XY over
    50 um is treated as print resolution)."""
    p = str(tmp_path / "mm.tif")
    tifffile.imwrite(p, np.zeros((3, 8, 8), np.uint8), imagej=True,
                     metadata={"axes": "ZYX", "unit": "mm"})
    d = read_voxel_spacing_detail(p)
    assert d["zyx"] == pytest.approx((1000.0, 1000.0, 1000.0))
    assert d["z_from_xy"] is True
