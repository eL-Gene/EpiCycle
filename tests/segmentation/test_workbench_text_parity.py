"""Every Workbench sentence is the accepted wording, word for word.

Reads the sentence table kept as a fixture beside this test
(``fixtures/workbench_sentences.json``) and holds
``napari_epicycle/segmentation/_workbench_text.py`` to it.  Step 4's own
departures, which were added on purpose, are listed in
``STEP4_OWN`` and must exist too.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

FIXTURE = Path(__file__).parent / "fixtures" / "workbench_sentences.json"

#: Sentences added beyond the accepted table.
STEP4_OWN = ("TOTALS_STEP4", "WB_LOCKED", "STOPPING", "LOG_MAP_ABANDONED",
             "PCA_TABLE_CAPTION")
#: Spec additions from the B3a audits: not in the mockup, not in the spec
#: text; recorded as departures the owner sees.
SPEC_ADDITIONS = ("PCA_EXCLUDED", "CATEGORY_HEADER_SHUT",
                  "CATEGORY_HEADER_TIP",
                  # a map that raises / has too few cells
                  "MAP_FAILED", "MAP_TOO_FEW_CELLS",
                  # busy wording
                  "BUSY_WARMUP", "BUSY_FEW", "MAP_NEEDS_FEATURES",
                  # singulars the mockup table leaves out
                  "MARKED_ONE", "UNMARKED_ONE", "UNDONE_ONE",
                  "LASSO_DONE_ONE",
                  # B2 (LOW-10): the Workbench outside Step 4, the dock
                  # title, the detail numbers line, and the Features-tab
                  # words that were literals (MED-1)
                  "WB_STEP_INACTIVE", "WB_WINDOW_TITLE", "WB_DETAIL_STATS",
                  "WB_NO_DESCRIPTION", "WB_SKEW_TIP", "WB_SKEW_TIP_NONE",
                  "WB_NORM_TIP_NONE", "WB_NORM_TIP_NORMAL",
                  "WB_NORM_TIP_NON", "WB_NORM_TEST_SHAPIRO",
                  "WB_NORM_TEST_DAGOSTINO", "COMPUTE_PCA_BTN",
                  "COMPUTE_RESIDUAL_PCA_BTN",
                  # v2.12 audit LOW-2: a Workbench rebuilt after napari's X
                  "WB_REBUILT_FROM_SAVED")
#: Mockup sentences Step 4 rewords on purpose (recorded departures):
#: Step 4's one word "unmarked".
STEP4_REWORDED = {"UNMARKED": "Returned {m} cells to unmarked."}


def mockup_sentences() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))["data"]


def test_the_fixture_exists():
    assert FIXTURE.is_file() and FIXTURE.stat().st_size > 0


def test_the_mockup_table_is_read():
    s = mockup_sentences()
    assert len(s) >= 39
    assert "SEL_NONE" in s and "MAP_AXES" in s


@pytest.mark.parametrize("key", sorted(mockup_sentences()))
def test_each_sentence_is_verbatim(key):
    from napari_epicycle.segmentation import _workbench_text as T

    assert hasattr(T, key), f"{key} missing from _workbench_text.py"
    if key in STEP4_REWORDED:
        assert getattr(T, key) == STEP4_REWORDED[key]
        return
    assert getattr(T, key) == mockup_sentences()[key]


@pytest.mark.parametrize("key", STEP4_OWN + SPEC_ADDITIONS)
def test_step4_own_sentences_exist(key):
    from napari_epicycle.segmentation import _workbench_text as T

    value = getattr(T, key)
    assert isinstance(value, str) and value


def test_spec_wording_of_step4_own():
    from napari_epicycle.segmentation import _workbench_text as T

    assert T.TOTALS_STEP4 == "All cells: good {g} · bad {b} · unmarked {u}"
    assert T.WB_LOCKED == ("A Step 4 job is running ({job}). Marking and maps "
                           "wait until it ends.")
    assert T.STOPPING == "Stopping {map}… a new map can start when it has."
    assert T.LOG_MAP_ABANDONED == "{map} was stopped because the cells changed."
    assert T.PCA_TABLE_CAPTION.startswith("Every ticked feature against every")


def test_split_title():
    from napari_epicycle.segmentation import _workbench_text as T

    assert T.split_title(T.UMAP_EMPTY) == (
        "No UMAP yet.",
        "Press Compute UMAP on the Map tab. It takes about 45 s for 3,000 cells.")
    assert T.split_title("plain") == ("plain", "")


@pytest.mark.parametrize("skew,kurt,bell", [
    (0.0, 0.0, True), (0.49, 0.99, True), (0.5, 0.0, False),
    (0.0, 1.0, False), (-0.3, -0.5, True), (float("nan"), 0.0, False),
    (None, 0.0, False),
])
def test_the_bell_rule(skew, kurt, bell):
    from napari_epicycle.segmentation._workbench_text import is_bell

    assert is_bell(skew, kurt) is bell
