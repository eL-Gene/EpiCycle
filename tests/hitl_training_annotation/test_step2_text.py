"""Step 2's words (brick L): display_round is the disk's numbering (D16),
the status table, the sentences, and the voxel wording of Step 3."""
import pytest

from napari_epicycle.hitl_training_annotation.ui import _step2_text as T


class TestDisplayRound:
    @pytest.mark.parametrize("given,shown", [
        ("round_00", 0), ("round_03", 3), ("lateral_2", 2),
        ("axial_0", 0), ("lateral_1/axial", 1), (4, 4), (0, 0),
        ("Cell_lateral_round_07", 7),
    ])
    def test_the_disk_number_is_unchanged(self, given, shown):
        assert T.display_round(given) == shown

    @pytest.mark.parametrize("given", ["-", "", None, True, -1, "lateral"])
    def test_a_tag_that_names_no_round_is_none(self, given):
        assert T.display_round(given) is None

    def test_the_chip_text_is_the_zero_padded_tag(self):
        assert T.round_tag_text("round_01") == "round_01"
        assert T.round_tag_text(0) == "round_00"
        assert T.round_tag_text("-") == ""


class TestStatusTable:
    def test_every_stored_status_has_a_word_a_colour_and_an_icon(self):
        for s in T.STATUS_ORDER:
            assert s in T.STATUS_WORDS and s in T.STATUS_COLOURS
            assert s in T.STATUS_ICONS

    def test_the_words_are_the_decided_ones(self):
        assert T.STATUS_WORDS["staged"] == "finalized"
        assert T.STATUS_WORDS["committed"] == "trained"
        assert T.STATUS_COLOURS["predicted"] == "#d4a843"      # D12: amber
        assert T.STATUS_COLOURS["committed"] == "#8c9eff"
        assert "rejected" not in T.STATUS_WORDS                # D8: retired

    def test_the_order_runs_predicted_to_trained(self):
        assert T.STATUS_ORDER[0] == "predicted"
        assert T.STATUS_ORDER[-1] == "committed"


class TestSentences:
    def test_the_placeholders_fill(self):
        assert "3 patches" in T.SESSION_PATCH_FOLDER_QUALIFIER.format(n=3)
        s = T.SESSION_PATCHES_SUMMARY.format(n=15, t=4, f=2, p=8, s=1)
        assert s.startswith("15 patches") and s.endswith("1 not started")
        assert "Round 4 (lateral)" in T.SESSION_LOCK_LINE.format(
            n=4, bucket="lateral")
        assert T.ROW_CURATED.format(k=2, z=124, bucket="lateral") == \
            "2 of 124 lateral slices curated"
        assert T.RUN_SUMMARY_MODEL.format(run="Cell", n=2) == \
            "Cell · round 2"

    def test_the_curation_desc_says_up_and_down_load(self):
        assert T.CURATION_PATCHES_DESC == \
            "Click a patch to load it. Up / Down load too."

    def test_the_empty_states_are_two_lines_not_zeros(self):
        assert len(T.SESSION_PATCHES_EMPTY_LINES) == 2
        assert len(T.RUN_SUMMARY_EMPTY_LINES) == 2
        assert "default" in T.RUN_SUMMARY_EMPTY_LINES[1]

    def test_the_dims_text(self):
        assert T.dims_text((124, 240, 240)) == "124 × 240 × 240"
        assert T.dims_text((2, 124, 240, 240)) == "124 × 240 × 240"
        assert T.dims_text(None) == ""
        assert T.dims_text("ab") == ""


class TestVoxelWording:
    def test_it_matches_step_3s_row_text(self):
        from napari_epicycle.fusion_reconstruction import _step3_pr_text as S3

        for spacing in (None, {}, {"zyx": [1, 0.52, 0.52], "source": "step1"},
                        {"zyx": [0.5, 0.5, 1], "source": "file"},
                        {"zyx": [2, 2, 2], "source": "user"},
                        {"zyx": ["a", 1, 1], "source": "user"}):
            assert T.voxel_text(spacing) == S3.spacing_row_text(spacing)

    def test_none_is_the_amber_word(self):
        assert T.voxel_text(None) == T.ROW_VOXEL_NONE == "voxels"
