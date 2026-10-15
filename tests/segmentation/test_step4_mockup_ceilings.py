"""The Step 4 revamp's mockup-side ceiling check .

Qt-free.  The numbers in ``data/step4_mockup_ceilings_v7.json`` were measured on the
accepted v7 mockup with headless Chrome (the generator is
``EpiCycle-drivers/step4_mockup_gen_v5/measure_v7_ceilings.py``: one single-dock page
per state, a small JS that reports column widths, scroll heights, the pinned feet, the
lock lines and any sideways overflow).  They are measured independently of the Qt build
so that a ceiling is never measured from what it constrains; the Qt tests
(``test_step4_ceilings.py``, brick B1a on) import ``MOCKUP_CEILINGS`` from here.

Never raise a number here without the owner.
"""
from __future__ import annotations

import math
import json
from pathlib import Path

import pytest

DATA_PATH = Path(__file__).with_name("data") / "step4_mockup_ceilings_v7.json"

STATES = ("A", "B", "B2", "B3", "C", "D", "D2", "D2win", "D3", "E")
PINNED_STATES = ("B2", "B3", "C", "D", "D2", "D3")  # ② and ③: two pinned columns
SCROLL_STATES = ("A", "B", "E")                        # ① and Diagnostics: one scroll area

# The spec's three stated mockup numbers and the tolerance they are checked to.
SPEC_SEG_RIGHT_COL_W = 324
SPEC_MT_RIGHT_D = 489
SPEC_MT_RIGHT_D3 = 464
TOLERANCE = 0.05


def _load() -> dict:
    return json.loads(DATA_PATH.read_text(encoding="utf-8"))


_DATA = _load() if DATA_PATH.exists() else {"meta": {}, "states": {}, "ceilings": {}}

#: The frozen ceilings (named constants).  ``None`` = the mockup has no
#: state to measure it from (e.g. ③ with Advanced open); the Qt brick must ask the owner.
MOCKUP_CEILINGS: dict = dict(_DATA.get("ceilings", {}))
#: Per-state raw measurements, for the Qt tests that want a specific mockup number.
MOCKUP_STATES: dict = dict(_DATA.get("states", {}))
HAND_ENTERED: bool = bool(_DATA.get("meta", {}).get("hand_entered", False))


def _near(value, target) -> bool:
    return value is not None and abs(value - target) <= TOLERANCE * target


def test_the_data_file_is_present_and_names_its_source():
    assert DATA_PATH.exists(), f"missing {DATA_PATH}; run measure_v7_ceilings.py --write-repo"
    meta = _load()["meta"]
    assert meta["mockup"].endswith("STEP4_REVAMP_MOCKUP_v7_2026-09-28.html")
    assert "hand_entered" in meta
    if not meta["hand_entered"]:
        assert meta["chrome_version"] and meta["chrome_version"] != "unknown"
        assert len(meta["mockup_sha1"]) == 40


def test_every_state_was_measured():
    if HAND_ENTERED:
        pytest.skip("hand-entered: only the ceilings are known")
    assert set(STATES) <= set(MOCKUP_STATES), sorted(set(STATES) - set(MOCKUP_STATES))


def test_nothing_overflows_sideways_in_any_state():
    if HAND_ENTERED:
        pytest.skip("hand-entered")
    bad = {s: MOCKUP_STATES[s]["sideways_overflow"] for s in STATES if MOCKUP_STATES[s]["sideways_overflow"]}
    assert not bad, bad
    for s in SCROLL_STATES:
        sec = MOCKUP_STATES[s]["sec_scroll"]
        assert sec is not None, s
    for s in PINNED_STATES:
        r = MOCKUP_STATES[s]["right"]
        assert r["content_w"] <= r["client_w"], s


def test_the_docks_are_the_real_dock_size():
    if HAND_ENTERED:
        pytest.skip("hand-entered")
    for s in STATES:
        if s == "D2win":
            continue
        st = MOCKUP_STATES[s]
        assert (st["dock_w"], st["dock_h"]) == (580, 970), s  # 578 x 968 + the 1 px border


def test_measure_and_tidy_right_column_does_not_scroll_at_D_and_D3():
    if HAND_ENTERED:
        pytest.skip("hand-entered")
    for s in ("D", "D3"):
        r = MOCKUP_STATES[s]["right"]
        assert r["scrolls"] is False, s
        assert r["content_h"] <= r["client_h"], s


def test_the_spec_stated_numbers_match_the_measurement():
    if HAND_ENTERED:
        pytest.skip("hand-entered")
    for s in ("B2", "B3", "C", "D", "D3"):
        assert _near(MOCKUP_STATES[s]["col_r_w"], SPEC_SEG_RIGHT_COL_W), (s, MOCKUP_STATES[s]["col_r_w"])
    assert _near(MOCKUP_STATES["D"]["right"]["scroll_h"], SPEC_MT_RIGHT_D)
    assert _near(MOCKUP_STATES["D3"]["right"]["scroll_h"], SPEC_MT_RIGHT_D3)


def test_the_ceilings_are_the_spec_named_constants():
    names = {
        "STEP4_MT_RIGHT_D_PX", "STEP4_MT_RIGHT_D3_PX", "STEP4_MT_RIGHT_D_ADV_PX", "STEP4_MT_FOOT_PX",
        "STEP4_SEG_RIGHT_B2_PX", "STEP4_SEG_RIGHT_B3_PX", "STEP4_SEG_FOOT_PX", "STEP4_SEG_LEFT_FIXED_PX",
        "STEP4_LOCK_LINE_PX", "STEP4_SEG_LIST_MIN_ROWS", "STEP4_MT_LIST_MIN_ROWS", "STEP4_SESSION_B_PX",
        "STEP4_SEG_RIGHT_COL_W", "STEP4_LEFT_COL_W", "STEP4_EXPORT_W",
    }
    assert names <= set(MOCKUP_CEILINGS), sorted(names - set(MOCKUP_CEILINGS))
    # the spec's own values
    assert MOCKUP_CEILINGS["STEP4_MT_RIGHT_D_PX"] == 514
    assert MOCKUP_CEILINGS["STEP4_MT_RIGHT_D3_PX"] == 487
    assert MOCKUP_CEILINGS["STEP4_SEG_LIST_MIN_ROWS"] == 3
    assert MOCKUP_CEILINGS["STEP4_MT_LIST_MIN_ROWS"] == 3
    assert MOCKUP_CEILINGS["STEP4_SEG_RIGHT_COL_W"] == 322
    assert MOCKUP_CEILINGS["STEP4_LEFT_COL_W"] == 218
    assert MOCKUP_CEILINGS["STEP4_EXPORT_W"] == 480


def test_every_height_ceiling_is_at_least_what_the_mockup_measured():
    """A ceiling below its own mockup number would fail the mockup itself."""
    if HAND_ENTERED:
        pytest.skip("hand-entered")
    s, c = MOCKUP_STATES, MOCKUP_CEILINGS
    assert c["STEP4_MT_RIGHT_D_PX"] >= s["D"]["right"]["scroll_h"]
    assert c["STEP4_MT_RIGHT_D3_PX"] >= s["D3"]["right"]["scroll_h"]
    assert c["STEP4_MT_FOOT_PX"] >= max(s["D"]["foot_h"], s["D3"]["foot_h"])
    # F4: B2 is held to the column's CONTENT, not its (taller) viewport
    assert c["STEP4_SEG_RIGHT_B2_PX"] >= max(s["B2"]["right"]["content_h"], s["C"]["right"]["content_h"])
    assert c["STEP4_SEG_RIGHT_B2_PX"] == math.ceil(
        max(s["B2"]["right"]["content_h"], s["C"]["right"]["content_h"]) * 1.05)
    assert c["STEP4_SEG_RIGHT_B3_PX"] >= s["B3"]["right"]["scroll_h"]
    assert c["STEP4_SEG_FOOT_PX"] >= max(s[x]["foot_h"] for x in ("B2", "B3", "C"))
    assert c["STEP4_SEG_LEFT_FIXED_PX"] >= max(s[x]["left_fixed_h"] for x in ("B2", "B3", "C"))
    assert c["STEP4_SESSION_B_PX"] >= s["B"]["sec_scroll"]["content_h"]
    locks = [l["h"] for x in STATES for l in s[x]["lock_lines"]]
    assert locks and c["STEP4_LOCK_LINE_PX"] >= max(locks)
    for x in ("B2", "B3", "C", "D", "D3"):
        assert s[x]["left_rows_visible"] >= c["STEP4_SEG_LIST_MIN_ROWS"], x
    # ② / ③ left column content = 218 in the mockup
    for x in PINNED_STATES:
        assert s[x]["col_l_w"] == c["STEP4_LEFT_COL_W"], x


def test_the_export_window_is_480_wide():
    if HAND_ENTERED:
        pytest.skip("hand-entered")
    assert MOCKUP_STATES["D2win"]["dock_w"] == 482  # 480 + the 1 px border
    assert MOCKUP_STATES["D2win"]["sideways_overflow"] == []
