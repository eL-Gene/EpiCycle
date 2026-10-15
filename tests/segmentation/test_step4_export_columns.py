"""F14 (B2's floor): the merged export's ``meta_*``
columns -- the model, and the units and spacing the numbers were MEASURED
in (``measured_spacing``, never ``spacing``, grill 1 G1-2).  Qt-free."""
from __future__ import annotations

import pandas as pd

from napari_epicycle.segmentation import _step4_rows as R


def _df(*names):
    return pd.DataFrame({"stack_name": list(names), "volume": [1.0] * len(
        names)})


def test_units_come_from_measure_time():
    """The record's spacing changed AFTER the Measure: the export reports
    the measure-time spacing."""
    recs = {"a.tif": {"spacing": {"zyx": [9.0, 9.0, 9.0], "source": "user"},
                      "measured_spacing": {"zyx": [2.0, 0.5, 0.5],
                                           "source": "step1"}}}
    df = R.add_meta_columns(_df("a.tif"), recs)
    row = df.iloc[0]
    assert row["meta_units"] == "um"
    assert (row["meta_spacing_z"], row["meta_spacing_y"],
            row["meta_spacing_x"]) == (2.0, 0.5, 0.5)


def test_measured_in_voxels_leaves_the_spacing_empty():
    recs = {"a.tif": {"spacing": {"zyx": [2.0, 0.5, 0.5]},
                      "measured_spacing": None}}
    row = R.add_meta_columns(_df("a.tif"), recs).iloc[0]
    assert row["meta_units"] == "voxels"
    assert pd.isna(row["meta_spacing_z"])


def test_a_pre_revamp_measure_is_unknown():
    row = R.add_meta_columns(_df("a.tif"), {"a.tif": {}}).iloc[0]
    assert row["meta_units"] == "unknown"


def test_mixed_units_per_stack():
    """Each row keeps its own stack's units."""
    recs = {"a.tif": {"measured_spacing": {"zyx": [1, 1, 1]}},
            "b.tif": {"measured_spacing": None}}
    df = R.add_meta_columns(_df("a.tif", "b.tif"), recs)
    assert list(df["meta_units"]) == ["um", "voxels"]


def test_the_columns_are_all_meta_prefixed_and_the_data_is_untouched():
    df = R.add_meta_columns(_df("a.tif"), {"a.tif": {}})
    added = [c for c in df.columns if c not in ("stack_name", "volume")]
    assert added == list(R.META_COLUMNS)
    assert all(c.startswith("meta_") for c in added)
    assert list(df["volume"]) == [1.0]


def test_the_model_label_comes_from_made_with():
    snap = {"lateral": "L", "axial": "A", "lateral_label": "Cell · round_01",
            "axial_label": "Cell · round_01", "kind": "step2_run"}
    row = R.add_meta_columns(_df("a.tif"), {"a.tif": {"made_with": snap}}
                             ).iloc[0]
    assert row["meta_model"]
    empty = R.add_meta_columns(_df("a.tif"), {"a.tif": {"made_with": None}}
                               ).iloc[0]
    assert empty["meta_model"] == ""
