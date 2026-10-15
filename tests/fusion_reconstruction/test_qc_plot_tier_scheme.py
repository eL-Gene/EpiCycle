"""The Workbench plot's tier scheme.

``STEP3_SCHEME`` keeps Step 3's five tiers (curated · good · review ·
bad · garbage) plus unassigned and a ghosts legend entry, with the ghosts
on by default and drawn behind; ``STEP4_SCHEME`` is Step 4's three
buckets, unchanged (the Workbench golden holds it); the classic layout
ignores both keywords (held against the classic tier-legend golden).
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest

from tests.fusion_reconstruction import _qc_plot_snap as S

FIVE = ("curated", "good", "review", "bad", "garbage")


def _step3_plot(df=None):
    S.app()
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        STEP3_PLOT_TEXTS,
    )
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        STEP3_SCHEME,
        QCPlotWidget,
    )
    from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE

    pw = QCPlotWidget(layout="workbench", texts=STEP3_PLOT_TEXTS,
                      palette=STEP3_PLOT_PALETTE, tier_scheme=STEP3_SCHEME)
    pw.resize(562, 600)
    pw.set_features(S.five_tier_df() if df is None else df)
    return pw


@pytest.fixture
def pw():
    w = _step3_plot()
    yield w
    w.deleteLater()


def _links(html: str) -> list:
    return re.findall(r'href="cat://qc_tier/([^"]+)"', html)


def _active(df):
    return df[df["label_status"] == "active"]


# --- the pure scheme ---------------------------------------------------------


def test_step4_scheme_is_the_old_rule():
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        STEP4_SCHEME,
        WB_TIER_BUCKETS,
        wb_tier_bucket,
        wb_tier_buckets,
    )

    assert STEP4_SCHEME.buckets == WB_TIER_BUCKETS == ("good", "bad",
                                                       "unmarked")
    cases = {"good": "good", "bad": "bad", "garbage": "bad",
             "review": "unmarked", "curated": "unmarked", "": "unmarked",
             None: "unmarked", float("nan"): "unmarked",
             "unmarked": "unmarked"}
    for tier, bucket in cases.items():
        assert wb_tier_bucket(tier) == bucket == STEP4_SCHEME.bucket_of(tier)
    df = S.five_tier_df()
    assert list(wb_tier_buckets(df)) == [wb_tier_bucket(t)
                                         for t in df["qc_tier"]]
    assert list(wb_tier_buckets(df.drop(columns="qc_tier"))) == (
        ["unmarked"] * len(df))
    assert len(wb_tier_buckets(None)) == 0
    assert not STEP4_SCHEME.ghost_entry
    assert not STEP4_SCHEME.ghosts_default_on
    assert not STEP4_SCHEME.ghosts_behind


def test_step3_scheme_has_five_tiers_and_unassigned():
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        _TIER_COLORS,
        STEP3_SCHEME,
    )

    assert STEP3_SCHEME.buckets == FIVE + ("unassigned",)
    assert STEP3_SCHEME.unassigned == "unassigned"
    for t in FIVE:
        assert STEP3_SCHEME.bucket_of(t) == t
        assert STEP3_SCHEME.rgb[t] == _TIER_COLORS[t]
    assert STEP3_SCHEME.rgb["unassigned"] == (140, 150, 170)
    for t in ("", None, float("nan"), "unmarked", "whatever"):
        assert STEP3_SCHEME.bucket_of(t) == "unassigned"
    assert STEP3_SCHEME.ghost_entry and STEP3_SCHEME.ghosts_default_on
    assert STEP3_SCHEME.ghosts_behind
    assert STEP3_SCHEME.unassigned_word == "unassigned"
    assert list(STEP3_SCHEME.colours()) == list(STEP3_SCHEME.buckets)


def test_a_scheme_is_validated():
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import TierScheme

    base = dict(name="x", buckets=("a", "b"), rgb={"a": (1, 2, 3),
                                                   "b": (4, 5, 6)},
                unassigned_word="b", ghost_entry=False,
                ghosts_default_on=False, ghosts_behind=False)
    TierScheme(**base)
    with pytest.raises(ValueError):
        TierScheme(**{**base, "rgb": {"a": (1, 2, 3)}})
    with pytest.raises(ValueError):
        TierScheme(**{**base, "buckets": ()})
    with pytest.raises(ValueError):
        TierScheme(**{**base, "aliases": {"z": "nope"}})


# --- the Step 3 plot -----------------------------------------------------------


def test_the_legend_lists_five_tiers_unassigned_and_ghosts(pw):
    assert pw.set_colour_by("QC tier")
    html = pw._legend_label.text()
    assert _links(html) == list(FIVE) + ["unassigned"]
    df = _active(S.five_tier_df())
    for t in FIVE:
        n = int((df["qc_tier"] == t).sum())
        assert (f'cat://qc_tier/{t}"' in html
                and re.search(rf"{t}</a>&nbsp;<span[^>]*>{n}</span>", html)), t
    n_un = int((df["qc_tier"] == "").sum())
    assert re.search(rf"unassigned</a>&nbsp;<span[^>]*>{n_un}</span>", html)
    # the ghosts entry: plain text, grey ×, the count, never a link
    assert "×&nbsp;ghosts</span>&nbsp;" in html
    assert re.search(r"ghosts</span>&nbsp;<span[^>]*>5</span>$", html)
    assert "cat://" not in html.split("×")[1]


def test_ghosts_start_on_and_are_drawn_behind(pw):
    assert pw._history_cb.isChecked()
    assert pw._history_cb.text() == "Ghosts"
    items = S.scatter_items(pw)
    assert len(items) == 2
    active, ghosts = sorted(items, key=lambda r: -r["z"])
    assert active["z"] == 0.0 and active["n"] == 55
    assert ghosts["z"] < active["z"]
    assert ghosts["n"] == 5
    assert ghosts["size"] == 7.0
    assert ghosts["symbol"] == "x"
    # P2 audit MED-5: a pen-less "x" draws NOTHING in pyqtgraph (the "x"
    # is all outline), so the ghosts were invisible in every live shot --
    # a faint 1 px pen and a brush at alpha 110.
    assert ghosts["brush"] == [140, 140, 160, 110]
    assert ghosts["pen"][:4] == [140, 140, 160, 120]
    assert ghosts["pen"][4] == 1.0
    assert ghosts["pen"][5] != 0          # not Qt.NoPen


def test_ghosts_off_hides_them_and_their_entry(pw):
    pw.set_colour_by("QC tier")
    pw._history_cb.setChecked(False)
    assert len(S.scatter_items(pw)) == 1
    assert "×" not in pw._legend_label.text()
    pw._history_cb.setChecked(True)
    assert "×&nbsp;ghosts" in pw._legend_label.text()


def test_the_ghosts_entry_rides_every_categorical_legend(pw):
    for name in ("Cluster", "Border cell"):
        pw.set_colour_by(name)
        assert "×&nbsp;ghosts" in pw._legend_label.text(), name


def test_no_unassigned_entry_when_every_label_has_a_tier():
    df = S.five_tier_df()
    df["qc_tier"] = [FIVE[i % 5] for i in range(len(df))]
    w = _step3_plot(df)
    try:
        w.set_colour_by("QC tier")
        assert _links(w._legend_label.text()) == list(FIVE)
    finally:
        w.deleteLater()


def test_a_legend_pick_selects_its_active_labels(pw):
    df = S.five_tier_df()
    act = _active(df)
    for t in FIVE:
        pw._toggle_category_selection("qc_tier", t)
        assert pw.selected_label_ids() == sorted(
            act.loc[act["qc_tier"] == t, "label"].tolist()), t
        pw._clear_selection()
    pw._toggle_category_selection("qc_tier", "unassigned")
    assert pw.selected_label_ids() == sorted(
        act.loc[act["qc_tier"] == "", "label"].tolist())


def test_the_viewer_takes_the_five_tier_colours(pw):
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        _TIER_COLORS,
    )

    pw.set_colour_by("QC tier")
    ids, rgba = pw.viewer_colours()
    act = _active(S.five_tier_df())
    assert ids.tolist() == act["label"].tolist()
    for lid, tier, c in zip(ids, act["qc_tier"], rgba):
        r, g, b = _TIER_COLORS.get(tier, (140, 150, 170))
        np.testing.assert_allclose(c, [r / 255, g / 255, b / 255, 0.85],
                                   atol=1e-6, err_msg=str(lid))


def test_a_kept_pick_survives_a_plot_switch(pw):
    pw._toggle_category_selection("qc_tier", "review")
    before = pw.selected_label_ids()
    pw.plot_type_combo.setCurrentText("PCA")
    assert pw.selected_label_ids() == before


# --- Step 4 and classic unchanged -------------------------------------------


def test_step4_explicit_scheme_equals_the_default():
    S.app()
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        STEP4_SCHEME,
        QCPlotWidget,
    )
    from napari_epicycle.segmentation.analysis_widget import (
        WORKBENCH_PLOT_TEXTS,
    )

    a = QCPlotWidget(layout="workbench", texts=WORKBENCH_PLOT_TEXTS)
    b = QCPlotWidget(layout="workbench", texts=WORKBENCH_PLOT_TEXTS,
                     tier_scheme=STEP4_SCHEME)
    try:
        for w in (a, b):
            w.set_features(S.five_tier_df())
            w.set_colour_by("QC tier")
        assert a._legend_label.text() == b._legend_label.text()
        assert "×" not in a._legend_label.text()
        assert not a._history_cb.isChecked()
        assert a._history_cb.text() == "Removed"
        assert S.tree(a) == S.tree(b)
    finally:
        a.deleteLater()
        b.deleteLater()


def test_a_wrong_scheme_type_is_refused_by_the_workbench_only():
    S.app()
    from napari_epicycle.fusion_reconstruction._qc_plot_text import (
        STEP3_PLOT_TEXTS,
    )
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        QCPlotWidget,
    )

    with pytest.raises(TypeError):
        QCPlotWidget(layout="workbench", texts=STEP3_PLOT_TEXTS,
                     tier_scheme="step3")
    w = QCPlotWidget(tier_scheme="step3")      # classic: never read
    w.deleteLater()


def test_classic_ignores_both_keywords():
    """Against the classic tier-legend golden, captured before P1."""
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        STEP3_SCHEME,
    )
    from napari_epicycle.ui_shell.theme import STEP3_PLOT_PALETTE
    from tests.fusion_reconstruction.test_qc_plot_classic_tier_legend_golden import (  # noqa: E501
        GOLDEN,
        snapshot,
    )

    expected = json.loads(GOLDEN.read_text(encoding="utf-8"))
    for kwargs in ({"tier_scheme": STEP3_SCHEME},
                   {"tier_scheme": STEP3_SCHEME, "palette": STEP3_PLOT_PALETTE,
                    "layout": "classic"}):
        got = json.loads(json.dumps(snapshot(**kwargs)))
        assert got == expected, kwargs


def test_step3_legend_words_are_neutral_and_the_dot_is_the_tier_colour(pw):
    """Decision 14: tier colours appear as dots, never as text."""
    from napari_epicycle.fusion_reconstruction.qc_plot_widget import (
        _TIER_COLORS,
        STEP3_SCHEME,
        STEP4_SCHEME,
    )

    assert STEP4_SCHEME.colour_words is True
    assert STEP3_SCHEME.colour_words is False
    pw.set_colour_by("QC tier")
    html = pw._legend_label.text()
    anchors = re.findall(r'<a href="cat://qc_tier/([^"]+)" style="([^"]*)">'
                         r'(.*?)</a>', html)
    assert [a[0] for a in anchors] == list(FIVE) + ["unassigned"]
    for tier, style, inner in anchors:
        assert style.startswith("color:#cccccc;"), (tier, style)
        r, g, b = _TIER_COLORS.get(tier, (140, 150, 170))
        assert inner.startswith(
            f'<span style="color:rgb({r},{g},{b});">●</span>&nbsp;'), inner
        assert inner.endswith(tier)
    # a cluster legend follows the same rule
    pw.set_colour_by("Cluster")
    assert 'href="cat://cluster_id/0" style="color:#cccccc;' in (
        pw._legend_label.text())


def test_the_ghosts_entry_only_where_ghosts_are_drawn(pw):
    pw.set_colour_by("QC tier")
    for ptype, drawn in (("Scatter", True), ("PCA", True), ("Violin", False),
                         ("UMAP", False), ("t-SNE", False)):
        pw.plot_type_combo.setCurrentText(ptype)
        assert ("×&nbsp;ghosts" in pw._legend_label.text()) is drawn, ptype


def test_the_ghosts_entry_on_a_drawn_map():
    df = S.five_tier_df()
    rng = np.random.RandomState(0)
    df["UMAP_1"], df["UMAP_2"] = rng.rand(len(df)), rng.rand(len(df))
    w = _step3_plot(df)
    try:
        w.set_colour_by("QC tier")
        w.plot_type_combo.setCurrentText("UMAP")
        assert any(r["n"] == 5 and r["z"] < 0 for r in S.scatter_items(w))
        assert "×&nbsp;ghosts" in w._legend_label.text()
    finally:
        w.deleteLater()


def test_step3_cluster_legend_reads_c_ids(pw):
    """GROUP_WORD "C{id}" (Step 3's cluster names); Step 4's "G{n}"
    ignores ``id`` and stays G1, G2... (the Workbench golden holds it)."""
    pw.set_colour_by("Cluster")
    html = pw._legend_label.text()
    words = re.findall(r"&nbsp;([^<&]+)</a>", html)
    assert words == ["noise", "C0", "C1", "C2", "C3"], words


# --- missing values, both schemes -------------------------------------------


@pytest.mark.parametrize("scheme_name", ["STEP4_SCHEME", "STEP3_SCHEME"])
def test_missing_tiers_fall_into_the_catch_all(scheme_name):
    import pandas as pd

    from napari_epicycle.fusion_reconstruction import qc_plot_widget as M

    scheme = getattr(M, scheme_name)
    for missing in (None, float("nan"), np.nan, pd.NA, ""):
        assert scheme.bucket_of(missing) == scheme.unassigned, repr(missing)
    df = pd.DataFrame({"label": [1, 2, 3, 4, 5],
                       "qc_tier": pd.array(["good", None, np.nan, pd.NA, ""],
                                           dtype=object)})
    got = list(scheme.buckets_of(df))
    assert got == ["good"] + [scheme.unassigned] * 4
    # a pandas "string" column turns every missing value into pd.NA
    df2 = df.astype({"qc_tier": "string"})
    assert list(scheme.buckets_of(df2)) == got
    if scheme is M.STEP4_SCHEME:
        assert got == [M.wb_tier_bucket(t) for t in df["qc_tier"]]
        assert list(M.wb_tier_buckets(df)) == got


def test_a_ghost_cannot_be_clicked(pw):
    from tests.fusion_reconstruction.test_qc_plot_p2_api import click_at

    df = S.five_tier_df()
    ghosts = df[df["label_status"] != "active"]
    for _, row in ghosts.iterrows():
        click_at(pw, float(row["area"]), float(row["solidity"]))
        picked = pw.selected_label_ids()
        assert len(picked) == 1 and picked[0] not in set(ghosts["label"]), (
            int(row["label"]), picked)
