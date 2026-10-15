"""The Add page's words.

* the description equals the accepted wording, kept as a fixture beside this
  test (``fixtures/add_about_description.md``);
* every Add control has a sentence in ``ADD_HELP_LINES`` and carries it as
  its tooltip (the named tooltip-only exceptions aside);
* each micron / voxel claim is checked NUMERICALLY on a synthetic object.
"""

from __future__ import annotations

import html
import pathlib
import re

import numpy as np
import pytest

from packages.epicycle_annotator import add_preview as AP
from packages.epicycle_annotator.reshape import region as R

from .add_widget_fixture import qapp, widget  # noqa: F401

SPEC = pathlib.Path(__file__).parent / "fixtures" / "add_about_description.md"

#: every control the Add page's help names
CONTROLS = (
    "add_preset_combo", "add_start_btn", "add_size_s_btn", "add_size_m_btn",
    "add_size_l_btn", "add_numbers_btn", "add_recenter_btn",
    "add_marker_source_combo", "add_nuclei_channel_combo",
    "add_nuclei_threshold_sb", "add_nuclei_auto_btn",
    "add_place_markers_btn", "add_click_undo_btn", "add_click_clear_btn",
    "add_click_list",
    "add_wall_channel_combo", "add_prep_zcorrect_cb", "add_prep_bgsub_cb",
    "add_prep_lift_cb", "add_prep_denoise_cb", "add_wall_kind_combo",
    "add_channel_combo", "add_clahe_cb", "add_threshold_sb", "add_otsu_btn",
    "add_preview_btn", "add_stop_btn", "add_all_btn", "add_none_btn",
    "add_show_walls_cb", "add_contour_cb", "add_click_radius_sb", "add_core_erode_sb",
    "add_dark_outside_cb", "add_dark_clearance_sb", "add_dark_factor_sb",
    "add_separate_cb", "add_min_distance_sb", "add_clahe_clip_sb",
    "add_clear_border_cb", "add_edge_margin_sb", "add_min_size_sb",
    "add_auto_smooth_cb", "add_smooth_sigma_sb", "add_batch_refine_btn",
    "add_refine_piece_btn", "add_again_btn", "add_undo_all_btn",
    "add_accept_refine_btn",
)


def _spec_paragraphs():
    text = SPEC.read_text(encoding="utf-8")
    sec = text.split("### 4.10", 1)[1].split("\n### ", 1)[0]
    return [ln[2:].strip() for ln in sec.splitlines() if ln.startswith("> ")]


def _plain(s):
    s = re.sub(r"\*\*|<b>|</b>", "", s)
    return html.unescape(s).strip()


def test_the_description_fixture_exists():
    assert SPEC.is_file() and SPEC.stat().st_size > 0


def test_the_description_is_the_accepted_wording_verbatim():
    paras = _spec_paragraphs()
    assert len(paras) == 2
    assert _plain(paras[0]) == _plain(AP.ADD_ABOUT_LINE)
    assert _plain(paras[1]) == _plain(AP.ADD_ABOUT_BODY)


def test_every_control_has_a_sentence_and_carries_it(widget):
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    assert set(CONTROLS) == set(ADD_HELP_LINES)
    for attr in CONTROLS:
        w = getattr(widget, attr)
        text = ADD_HELP_LINES[attr]
        assert text.strip().endswith((".", "?")), attr
        assert w.toolTip() == text, attr


def test_no_sentence_names_an_engine_token():
    from napari_epicycle.annotator.advanced_features_widget import (
        ADD_HELP_LINES,
    )

    tokens = ("nuclei_masks", "paint", "faces+dark", "watershed", "ridge",
              "labels_curated")
    for attr, text in ADD_HELP_LINES.items():
        for t in tokens:
            assert t not in text, (attr, t)


# --------------------------------------------------------------------------- #
# the numbers behind the words
# --------------------------------------------------------------------------- #
SP = (0.517, 0.517, 0.517)


def test_a_two_micron_ball_is_about_four_microns_across():
    shape = (31, 31, 31)
    ball = R.rasterise_points([(15, 15, 15)], shape, radius_um=2.0,
                              spacing=SP) > 0
    z, y, x = np.nonzero(ball)
    d = np.sqrt(((z - 15) * SP[0]) ** 2 + ((y - 15) * SP[1]) ** 2
                + ((x - 15) * SP[2]) ** 2)
    assert d.max() <= 2.0
    for axis in (z, y, x):
        span = (axis.max() - axis.min()) * 0.517
        assert span >= 2 * (2.0 - 0.517)
    assert "about 4 µm across" in AP.help_clicks(2.0)


def test_a_one_micron_erosion_removes_the_expected_shell():
    lab = np.zeros((40, 40, 40), np.int32)
    lab[5:35, 5:35, 5:35] = 3
    cores, n = R.eroded_cores(lab, erode_um=1.0, spacing=SP)
    assert n == 0
    # the outer voxel sits 0.517 um inside (<= 1 um: it goes); the next one
    # sits 1.034 um inside (> 1 um: it stays) -- a one-voxel shell here
    assert cores[6:34, 6:34, 6:34].all()
    assert not cores[5].any() and not cores[:, 5].any()
    assert not cores[34].any()


def test_the_dark_clearance_reaches_two_microns():
    from packages.epicycle_annotator.reshape.arbiter import physical_ball

    ball = physical_ball(2.0, SP)
    c = (np.array(ball.shape) - 1) // 2
    off = np.argwhere(ball) - c
    reach = np.sqrt(((off * 0.517) ** 2).sum(axis=1))
    assert reach.max() <= 2.0 + 1e-9
    assert reach.max() >= 2.0 - 0.517


def test_the_cells_limit_refuses_four_million_and_one():
    from packages.epicycle_annotator.reshape import ReshapeSpec

    spec = ReshapeSpec(target="region", seeds="paint", walls="membrane",
                       params={"wall_channel": "membrane", "ridge": False,
                               "outside": "faces"})
    roi = (slice(0, 1), slice(0, 1), slice(0, 4_000_001))
    labels = np.zeros((1, 1, 4_000_001), np.uint8)
    out = R.reshape_empty_region(labels, {"membrane": labels}, spec, roi=roi,
                                 seed_points=[(0, 0, 5)], spacing=SP)
    assert out.status == "refused"
    assert out.messages[-1] == AP.ADD_ENGINE_TOO_BIG


@pytest.mark.parametrize("depth_um,kept", [(0.9, False), (1.1, True)])
def test_a_nuclei_sliver_under_one_micron_is_skipped(depth_um, kept):
    """A piece whose deepest voxel is ``depth_um`` from a kept voxel."""
    sx = depth_um / 3.0                       # three voxels deep along x
    spacing = (5.0, 5.0, sx)
    existing = np.zeros((3, 3, 20), np.int32)
    existing[:, :, :5] = 9
    seeds = np.zeros_like(existing)
    seeds[1, 1, 5:8] = 1                      # its far end is 3 voxels in
    out, n = R.drop_slivers(seeds, existing, min_depth_um=AP_SLIVER,
                            spacing=spacing)
    assert bool(out.any()) is kept
    assert n == (0 if kept else 1)


AP_SLIVER = R.NUCLEI_SLIVER_UM


def test_the_voxel_sentences_name_a_button_that_exists():
    """Live-shot fix F1 (now ADD_NOTE_DEFAULT_SPACING) sends the user to
    "QC Features -> Settings..." in Step 3 -- that section, that button and
    its "Spacing" row must really be there."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parents[2]
    ui = (root / "napari_epicycle" / "fusion_reconstruction"
          / "ui.py").read_text(encoding="utf-8")
    assert '"QC Features"' in ui
    assert 'self.qc_settings_btn = QPushButton("Settings...")' in ui
    assert 'self.extract_features_btn = QPushButton("Extract")' in ui
    dlg = (root / "napari_epicycle" / "fusion_reconstruction"
           / "qc_extraction_dialog.py").read_text(encoding="utf-8")
    assert 'addRow("Spacing:"' in dlg
    for const in (AP.ADD_NOTE_DEFAULT_SPACING,):
        assert "QC Features → Settings…" in const
        assert "Spacing" in const
