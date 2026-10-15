"""Tests for the large-data load routing.

Pure and Qt-free: `plan_load_route` is a function of header numbers only, which
is why the routing decision was put in `logic.py` rather than in the controller.

The MEMMAP branch is the one worth testing hard.  The routing table has THREE view
states, and an implementation that collapses them to EAGER/LAZY_ZARR **falsely
refuses uncompressed files** — which is the format large microscopy stacks
usually ship in, and the one case that needs no conversion at all.
"""
from __future__ import annotations

import pytest

from napari_epicycle.data_preparation.logic import (
    EAGER_RAM_FRACTION,
    format_bytes,
    plan_load_route,
)

GB = 1024 ** 3
MB = 1024 ** 2


class TestEagerBranch:
    """`view = EAGER if decompressed <= 0.6 · free_ram`."""

    def test_small_stack_is_eager(self):
        route = plan_load_route(2 * GB, 32 * GB, n_channels=2)
        assert route["view"] == "EAGER"
        assert route["eager"] is True
        assert route["needs_conversion"] is False

    def test_exactly_at_the_budget_is_eager(self):
        available = 32 * GB
        route = plan_load_route(int(EAGER_RAM_FRACTION * available), available)
        assert route["view"] == "EAGER"

    def test_one_byte_over_the_budget_is_not(self):
        available = 32 * GB
        route = plan_load_route(int(EAGER_RAM_FRACTION * available) + 1,
                                available, compression="ADOBE_DEFLATE")
        assert route["view"] == "LAZY_ZARR"
        assert route["eager"] is False


class TestMemmapBranch:
    """`view = MEMMAP elif compression == raw` — the branch a two-way split
    silently drops, turning a loadable file into a refused one."""

    @pytest.mark.parametrize("tag", ["NONE", "none", "COMPRESSION.NONE", "1"])
    def test_uncompressed_over_budget_is_memmappable(self, tag):
        route = plan_load_route(220 * GB, 32 * GB, n_channels=2,
                                compression=tag)
        assert route["view"] == "MEMMAP"
        assert route["needs_conversion"] is False, (
            "an uncompressed file can be read zero-copy — no re-save needed"
        )

    def test_compressed_over_budget_needs_conversion(self):
        route = plan_load_route(220 * GB, 32 * GB, n_channels=2,
                                compression="ADOBE_DEFLATE")
        assert route["view"] == "LAZY_ZARR"
        assert route["needs_conversion"] is True

    def test_memmap_is_still_not_eager(self):
        """`eager` means "safe to hand to `imread`".  MEMMAP needs `memmap()`,
        so it must not be treated as an ordinary load."""
        route = plan_load_route(220 * GB, 32 * GB, compression="NONE")
        assert route["eager"] is False

    def test_unknown_compression_is_treated_as_compressed(self):
        """Fail toward the safer branch: assume a re-save is needed rather than
        promising a zero-copy read we cannot deliver."""
        route = plan_load_route(220 * GB, 32 * GB, compression=None)
        assert route["view"] == "LAZY_ZARR"


class TestProcessBudget:
    """`process = IN_MEMORY if per_channel_float_peak <= 0.6 · free_ram`."""

    def test_peak_divides_by_channel_count(self):
        """"÷ n_channels since ops run one channel at a time"."""
        one = plan_load_route(8 * GB, 64 * GB, n_channels=1)
        four = plan_load_route(8 * GB, 64 * GB, n_channels=4)
        assert one["proc_peak"] == 4 * four["proc_peak"]

    def test_peak_is_four_times_the_per_channel_size(self):
        route = plan_load_route(8 * GB, 64 * GB, n_channels=2)
        assert route["proc_peak"] == pytest.approx(4 * (8 * GB) / 2)

    def test_out_of_core_when_the_peak_does_not_fit(self):
        route = plan_load_route(220 * GB, 32 * GB, n_channels=2,
                                compression="NONE")
        assert route["process"] == "OUT_OF_CORE"

    def test_in_memory_when_it_does(self):
        route = plan_load_route(1 * GB, 64 * GB, n_channels=2)
        assert route["process"] == "IN_MEMORY"

    def test_worked_example_does_not_follow_its_own_rule(self):
        """The copy says a 220 GB stack "needs ~1.3 TB", but its stated rule —
        4× the per-channel size — gives 440 GB for 2 channels and 880 GB for 1.

        Recorded as a test so the discrepancy is visible rather than quietly
        resolved in someone's favour: the RULE is what is implemented.
        """
        two_ch = plan_load_route(220 * GB, 32 * GB, n_channels=2,
                                 compression="NONE")["proc_peak"]
        one_ch = plan_load_route(220 * GB, 32 * GB, n_channels=1,
                                 compression="NONE")["proc_peak"]
        assert two_ch == pytest.approx(440 * GB)
        assert one_ch == pytest.approx(880 * GB)
        assert one_ch < 1.3 * 1024 ** 4, (
            "neither reading reaches the spec's illustrative 1.3 TB"
        )


class TestUnknownInputs:
    """A missing header number must not block the user."""

    @pytest.mark.parametrize("nbytes,available", [
        (None, 32 * GB), (0, 32 * GB), (2 * GB, None), (2 * GB, 0),
    ])
    def test_unknown_does_not_refuse(self, nbytes, available):
        route = plan_load_route(nbytes, available)
        assert route["view"] == "UNKNOWN"
        assert route["eager"] is True, (
            "an unmeasurable file must load, not be refused on a guess"
        )
        assert route["needs_conversion"] is False


class TestFormatBytes:
    @pytest.mark.parametrize("value,expected", [
        (512, "512 B"),
        (2 * 1024, "2 KB"),
        (293 * MB, "293.0 MB"),
        (220 * GB, "220.0 GB"),
        (3 * 1024 ** 4, "3.0 TB"),
    ])
    def test_units(self, value, expected):
        assert format_bytes(value) == expected

    @pytest.mark.parametrize("bad", [None, "", "abc"])
    def test_unreadable_input(self, bad):
        assert format_bytes(bad) == "—"


class TestHeaderPassCarriesWhatRoutingNeeds:
    """The header pass reads shape, dtype, axes AND compression.  Without
    the last one the MEMMAP branch cannot exist."""

    def test_scan_returns_the_routing_keys(self, tmp_path):
        import numpy as np
        import tifffile

        path = tmp_path / "stack.tif"
        tifffile.imwrite(str(path), np.zeros((4, 8, 8), dtype=np.uint16))

        from napari_epicycle.data_preparation.logic import scan_tiff_folder

        entry = scan_tiff_folder(str(tmp_path))[0]
        for key in ("path", "filename", "stem", "shape", "dtype", "nbytes",
                    "axes", "compression"):
            assert key in entry, f"{key} missing from the header pass"

    def test_nbytes_is_decompressed_not_file_size(self, tmp_path):
        """It is is explicit: file size under-reports 2–10× under DEFLATE, so the
        decision must be made on shape × itemsize."""
        import numpy as np
        import tifffile

        from napari_epicycle.data_preparation.logic import scan_tiff_folder

        shape = (16, 64, 64)
        path = tmp_path / "zeros.tif"
        # All-zero data compresses to almost nothing, so a file-size-based
        # check would read this as tiny.
        tifffile.imwrite(str(path), np.zeros(shape, dtype=np.uint16),
                         compression="deflate")
        entry = scan_tiff_folder(str(tmp_path))[0]

        expected = int(np.prod(shape)) * 2
        assert entry["nbytes"] == expected
        assert entry["nbytes"] > path.stat().st_size, (
            "decompressed size must exceed the compressed file on zeros"
        )
