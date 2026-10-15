"""Brick L's row builders and tally, over fake
``sample_records``; no widget is built."""
import pytest

from napari_epicycle.hitl_training_annotation.ui import _step2_lists as L
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T
from napari_epicycle.hitl_training_annotation._step2_thumbs import PatchMeta


def _rec(patch, bucket, status, n, round_tag="-"):
    sid = f"{patch}_{'XY' if bucket == 'lateral' else 'XZ'}_{n}"
    return sid, {"sample_id": sid, "patch_id": patch, "bucket": bucket,
                 "status": status, "round_tag": round_tag, "slice_index": n}


def _records(*rows):
    return dict(_rec(*r) for r in rows)


RECS = _records(
    ("p1", "lateral", "predicted", 1), ("p1", "lateral", "curating", 2),
    ("p1", "lateral", "committed", 3, "round_01"),
    ("p1", "lateral", "committed", 4, "round_02"),
    ("p1", "axial", "accepted", 5),
    ("p2", "lateral", "staged", 1),
    ("p3", "axial", "predicted", 1),
    ("p4", "lateral", "rejected", 1),
)


class TestTally:
    def test_it_counts_both_buckets(self):
        t = L.tally_records(RECS)
        assert t["p1"]["lateral"]["predicted"] == 1
        assert t["p1"]["lateral"]["committed"] == 2
        assert t["p1"]["axial"]["accepted"] == 1

    def test_rejected_is_not_counted(self):
        assert "p4" not in L.tally_records(RECS)

    def test_the_patch_is_derived_from_the_sample_id_when_absent(self):
        t = L.tally_records({"WholeScale_patch_1_XY_31": {
            "bucket": "lateral", "status": "predicted"}})
        assert "WholeScale_patch_1" in t

    def test_junk_records_are_skipped(self):
        assert L.tally_records({"a": None, "b": {"bucket": "x",
                                                 "status": "predicted"}}) == {}
        assert L.tally_records(None) == {}

    def test_the_round_chip_comes_from_committed_records_only(self):
        t = L.tally_records(RECS)
        assert L.latest_round(t, "p1") == 2
        assert L.latest_round(t, "p2") is None       # staged, not trained
        assert L.latest_round(t, "nope") is None

    def test_a_committed_record_with_no_round_tag_has_no_chip(self):
        t = L.tally_records(_records(("p", "lateral", "committed", 1, "-")))
        assert L.latest_round(t, "p") is None

    def test_classes_are_exclusive_and_sum_to_the_patch_count(self):
        t = L.tally_records(RECS)
        names = ["p1", "p2", "p3", "p5"]
        kinds = [L.classify_patch(t, n) for n in names]
        assert kinds == ["trained", "finalized", "in_progress", "not_started"]
        assert L.session_summary(t, names) == (
            "4 patches · 1 in a trained round · 1 finalized · "
            "1 in progress · 1 not started")
        assert L.session_summary(t, []) == ""


class TestStrip:
    def test_fractions_sum_to_at_most_one(self):
        c = {"predicted": 3, "curating": 1, "accepted": 2, "staged": 3,
             "committed": 0}
        segs = L.strip_segments(c, 153)
        assert sum(f for f, _c in segs) == pytest.approx(1.0)
        assert segs[-1][1] == L.STRIP_REST

    def test_more_records_than_slices_is_scaled_to_fit(self):
        segs = L.strip_segments({"predicted": 10}, 4)
        assert sum(f for f, _c in segs) <= 1.0 + 1e-9

    def test_no_records_is_all_grey(self):
        assert L.strip_segments({}, 10) == ((1.0, L.STRIP_REST),)

    def test_the_total_is_z_for_lateral_and_y_plus_x_for_axial(self):
        assert L.slice_total((124, 240, 200), "lateral", 5) == 124
        assert L.slice_total((124, 240, 200), "axial", 5) == 440
        assert L.slice_total(None, "lateral", 5) == 5
        assert L.slice_total(None, "lateral", 0) == 1


META = PatchMeta("p1.tiff", "CZYX", (2, 124, 240, 240), 2, "")


class TestSessionRow:
    def test_the_lines(self):
        t = L.tally_records(RECS)
        rec = L.session_row_record(
            "p1.tiff", tally=t, shape=(124, 240, 240), meta=META,
            spacing={"zyx": [0.52, 0.52, 0.52], "source": "step1"})
        assert rec["line2"] == ("CZYX · 124 × 240 × 240 "
                                "· 2 ch · 3 of 124 lateral "
                                "slices curated")
        assert rec["line3"] == \
            "0.52 × 0.52 × 0.52 µm (from Step 1)"
        assert rec["line3_warn"] is False
        assert rec["word"] == ""                    # D2: no status word
        assert "grey_word" not in rec

    def test_no_voxel_size_is_the_amber_word_with_a_tip(self):
        rec = L.session_row_record("p1.tiff", tally={}, shape=None, meta=None)
        assert rec["line3"] == "voxels" and rec["line3_warn"] is True
        assert rec["row_tip"] == T.ROW_VOXEL_TIP_NONE

    def test_chips_lat_ax_and_the_round_chip(self):
        t = L.tally_records(RECS)
        rec = L.session_row_record("p1.tiff", tally=t, shape=(124, 240, 240),
                                   meta=META)
        chips = dict(rec["chips_inline"])
        assert "lat 4" in chips and "ax 1" in chips
        assert chips["round_02"] == "round"          # the chip, disk numbering

    def test_a_patch_with_no_records_says_so_and_has_no_strip(self):
        rec = L.session_row_record("p5.tiff", tally=L.tally_records(RECS),
                                   shape=(124, 240, 240), meta=META)
        assert rec["grey_word"] == T.ROW_NO_RECORDS
        assert "strip" not in rec and "status_counts" not in rec

    def test_records_only_in_the_other_bucket_dim_the_slots(self):
        t = L.tally_records(RECS)
        rec = L.session_row_record("p3.tiff", tally=t, shape=(124, 240, 240),
                                   meta=META, bucket="lateral")
        assert rec["dim_slots"] is True
        assert "grey_word" not in rec
        assert "strip" in rec

    def test_counts_are_all_five_slots_in_order(self):
        t = L.tally_records(RECS)
        rec = L.session_row_record("p1.tiff", tally=t, shape=(124, 240, 240),
                                   meta=META)
        words = [w for _i, _c, _n, w in rec["status_counts"]]
        assert words == ["predicted", "curating", "accepted", "finalized",
                         "trained"]
        assert [n for _i, _c, n, _w in rec["status_counts"]] == [1, 1, 0, 0, 2]

    def test_the_axial_view_counts_the_axial_bucket(self):
        t = L.tally_records(RECS)
        rec = L.session_row_record("p1.tiff", tally=t, shape=(124, 240, 200),
                                   meta=META, bucket="axial")
        assert "1 of 440 axial slices curated" in rec["line2"]


class TestCuratedCountAndRoundChip:
    def test_predicted_only_slices_are_not_curated(self):
        t = L.tally_records(_records(
            *[("p", "lateral", "predicted", i) for i in range(9)]))
        rec = L.session_row_record("p.tiff", tally=t, shape=(153, 9, 9),
                                   meta=META)
        assert "0 of 153 lateral slices curated" in rec["line2"]
        assert dict(rec["chips_inline"])["lat 9"] == ""   # the chip still counts records

    def test_curating_and_beyond_count(self):
        t = L.tally_records(_records(
            ("p", "lateral", "predicted", 1), ("p", "lateral", "curating", 2),
            ("p", "lateral", "accepted", 3), ("p", "lateral", "staged", 4),
            ("p", "lateral", "committed", 5, "round_00")))
        rec = L.session_row_record("p.tiff", tally=t, shape=(50, 9, 9),
                                   meta=META)
        assert "4 of 50 lateral slices curated" in rec["line2"]

    def test_the_round_chip_follows_the_view_bucket(self):
        t = L.tally_records(_records(
            ("p", "lateral", "predicted", 1),
            ("p", "axial", "committed", 2, "round_00")))
        lat = L.session_row_record("p.tiff", tally=t, shape=(50, 9, 9),
                                   meta=META, bucket="lateral")
        texts = [c[0] for c in lat["chips_inline"]]
        assert "round_00" not in texts          # not the active bucket's
        assert "ax round_00" in texts
        ax = L.session_row_record("p.tiff", tally=t, shape=(50, 9, 9),
                                  meta=META, bucket="axial")
        texts = [c[0] for c in ax["chips_inline"]]
        assert "round_00" in texts and "ax round_00" not in texts

    def test_both_buckets_trained_show_both_chips(self):
        t = L.tally_records(_records(
            ("p", "lateral", "committed", 1, "round_01"),
            ("p", "axial", "committed", 2, "round_00")))
        rec = L.session_row_record("p.tiff", tally=t, shape=(50, 9, 9),
                                   meta=META, bucket="lateral")
        texts = [c[0] for c in rec["chips_inline"]]
        assert "round_01" in texts and "ax round_00" in texts

    def test_latest_round_can_name_a_bucket(self):
        t = L.tally_records(_records(
            ("p", "axial", "committed", 2, "round_00")))
        assert L.latest_round(t, "p", "lateral") is None
        assert L.latest_round(t, "p", "axial") == 0
        assert L.latest_round(t, "p") == 0


class TestRunHead:
    def test_desc_and_pairs(self):
        assert L.run_head_tail("whole cell", 83) == "whole cell · 83 pairs"

    def test_pairs_only_desc_only_and_none(self):
        assert L.run_head_tail("", 83) == "83 pairs"
        assert L.run_head_tail("whole cell", None) == "whole cell"
        assert L.run_head_tail("", None) == ""
        assert L.run_head_tail("", "junk") == ""


class TestCurationRow:
    def test_a_narrow_row_with_no_icon_and_a_plain_second_line(self):
        t = L.tally_records(RECS)
        rec = L.curation_patch_record("p1.tiff", tally=t,
                                      shape=(124, 240, 240), meta=META,
                                      loaded=True)
        assert rec["no_icon"] is True and rec["loaded"] is True
        assert rec["line2_plain"] == \
            "CZYX · 124 × 240 × 240 · 2 ch"
        assert rec["strip"]
        assert "chips_inline" not in rec            # the narrow row has none

    def test_the_voxel_size_is_the_hover(self):
        rec = L.curation_patch_record(
            "p1.tiff", tally={}, spacing={"zyx": [1, 1, 1], "source": "file"})
        assert "from the file" in rec["row_tip"]
        rec = L.curation_patch_record("p1.tiff", tally={})
        assert rec["row_tip"] == T.ROW_VOXEL_TIP_NONE

    def test_not_started_is_grey_and_other_bucket_dims(self):
        t = L.tally_records(RECS)
        assert L.curation_patch_record(
            "p5.tiff", tally=t)["grey_word"] == "not started"
        rec = L.curation_patch_record("p3.tiff", tally=t, bucket="lateral")
        assert rec.get("dim_slots") is True and "grey_word" not in rec

    def test_counts_follow_the_view(self):
        t = L.tally_records(RECS)
        lat = L.curation_patch_record("p1.tiff", tally=t, bucket="lateral")
        ax = L.curation_patch_record("p1.tiff", tally=t, bucket="axial")
        assert [n for *_x, n, _w in lat["status_counts"]] == [1, 1, 0, 0, 2]
        assert [n for *_x, n, _w in ax["status_counts"]] == [0, 0, 1, 0, 0]


class TestRunSummaryRounds:
    def test_next_is_the_disk_index_and_last_is_the_one_before(self):
        r = L.run_summary_rounds({"lateral": 3, "axial": 2}, "Cell")
        assert r["lateral"] == {"next": "3", "last": "2",
                                "model": "Cell · round 2"}
        assert r["axial"]["next"] == "2" and r["axial"]["last"] == "1"

    def test_round_zero_has_no_last_round_or_model(self):
        r = L.run_summary_rounds({"lateral": 0}, "Cell")
        assert r["lateral"]["last"] == r["lateral"]["model"] == "—"
        assert r["lateral"]["next"] == "0"
        assert r["axial"]["next"] == "0"

    def test_garbage_is_zero_never_parsed_from_a_string(self):
        assert L.run_summary_rounds(None, "")["lateral"]["next"] == "0"
        assert L.run_summary_rounds({"lateral": "Lat 3 | Axi 2"},
                                    "x")["lateral"]["next"] == "0"

    def test_chip_and_run_summary_agree_on_the_number(self):
        # the last trained round of the tally is the "Last round" of the card
        t = L.tally_records(_records(
            ("p", "lateral", "committed", 1, "round_02")))
        chip = L.latest_round(t, "p")
        last = L.run_summary_rounds({"lateral": 3}, "c")["lateral"]["last"]
        assert str(chip) == last
