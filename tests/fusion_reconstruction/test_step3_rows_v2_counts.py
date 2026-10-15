"""A3: the tier-count module generalised with a bucket rule --
Step 4's good / bad / unmarked from the saved tier column -- and the worker
shared by both steps (one real QThread).  Run one file at a time with
``-o addopts=""``.
"""
from __future__ import annotations

import os
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pd = pytest.importorskip("pandas")

from napari_epicycle.fusion_reconstruction import _step3_tier_counts as TC  # noqa: E402

BUCKETS = ("good", "bad", "unmarked")


def _bucket_of(tier):
    if tier == "good":
        return "good"
    if tier in ("bad", "garbage"):
        return "bad"
    return "unmarked"


def _csv(tmp_path, tiers, status=None, col="qc_tier"):
    df = pd.DataFrame({"label": range(len(tiers)), col: tiers})
    if status is not None:
        df["label_status"] = status
    p = tmp_path / "f.csv"
    df.to_csv(p, index=False)
    return p


def test_the_default_counts_are_step_3s_five_tiers(tmp_path):
    p = _csv(tmp_path, ["good", "review", "bad", "garbage", "curated"])
    assert TC.tier_counts_from_csv(p) == {
        "good": 1, "review": 1, "bad": 1, "garbage": 1, "curated": 1}


def test_buckets_use_the_given_rule_and_order(tmp_path):
    p = _csv(tmp_path, ["good", "good", "bad", "garbage", "review", None,
                        "curated"])
    got = TC.tier_counts_from_csv(p, order=BUCKETS, bucket_of=_bucket_of)
    assert got == {"good": 2, "bad": 2, "unmarked": 3}


def test_buckets_skip_inactive_rows_and_need_the_column(tmp_path):
    p = _csv(tmp_path, ["good", "bad", "good"],
             status=["active", "deleted", "superseded"])
    assert TC.tier_counts_from_csv(p, order=BUCKETS,
                                   bucket_of=_bucket_of) == {
        "good": 1, "bad": 0, "unmarked": 0}
    p2 = tmp_path / "n.csv"
    pd.DataFrame({"label": [1]}).to_csv(p2, index=False)
    # (no tier column: Step 3 gets None; a bucket rule counts unmarked)
    assert TC.tier_counts_from_csv(p2) is None


def test_items_follow_the_order_given():
    counts = {"good": 2, "bad": 0, "unmarked": 5}
    colours = {"good": (1, 2, 3), "bad": (4, 5, 6), "unmarked": (7, 8, 9)}
    items = TC.tier_items(counts, colours, order=BUCKETS)
    assert items == (("good", 2, (1, 2, 3)), ("bad", 0, (4, 5, 6)),
                     ("unmarked", 5, (7, 8, 9)))
    assert TC.tier_items({"good": 0, "bad": 0, "unmarked": 0}, colours,
                         order=BUCKETS) is None


def test_the_batch_carries_the_rule(tmp_path):
    p = _csv(tmp_path, ["good", "bad"])
    out = TC.count_batch([("a", str(p))], None, order=BUCKETS,
                         bucket_of=_bucket_of)
    assert out == {"a": {"good": 1, "bad": 1, "unmarked": 0}}


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_a_real_qthread_counts_with_the_rule(qapp, tmp_path):
    p = _csv(tmp_path, ["good", "garbage", None])
    got, fin = [], []

    class Sink:
        def counted(self, stamp, result):
            got.append((stamp, result))

        def finished(self):
            fin.append(1)

    sink = Sink()
    state = TC.start_tier_counts([("a", str(p))], ("s",), sink.counted,
                                 on_finished=sink.finished,
                                 order=BUCKETS, bucket_of=_bucket_of)
    t0 = time.monotonic()
    while time.monotonic() - t0 < 20 and not (fin and not state.running()):
        qapp.processEvents()
        time.sleep(0.01)
    assert got == [(("s",), {"a": {"good": 1, "bad": 1, "unmarked": 1}})]
    assert state.thread is None or state.thread.quit_called
    TC.stop_tier_counts(state)


def test_a_csv_without_a_tier_column_counts_every_active_row_unmarked(tmp_path):
    """L3: a fresh Measure may write no qc_tier column."""
    p = tmp_path / "n.csv"
    pd.DataFrame({"label": [1, 2, 3],
                  "label_status": ["active", "active", "deleted"]}
                 ).to_csv(p, index=False)
    got = TC.tier_counts_from_csv(p, order=BUCKETS, bucket_of=_bucket_of)
    assert got == {"good": 0, "bad": 0, "unmarked": 2}
    assert TC.tier_counts_from_csv(p) is None      # Step 3's path as it was
