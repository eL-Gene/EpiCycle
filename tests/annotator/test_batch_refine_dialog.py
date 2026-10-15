"""Tests for ``BatchRefineDialog`` — the floating Batch Refine view.

Uses a plain ``QApplication`` (``QT_QPA_PLATFORM=offscreen``; pytest-qt
is NOT required) to construct the real QWidget with stub callbacks and exercise the view contract:
- recipe building (method + params → ``RefineStep`` list, in order),
- target spec reflecting radio + tier-checkbox selection,
- ``count_provider`` called live on target change,
- ``set_state`` visibility toggling (idle / running / staged),
- invalid params → ``recipe()`` returns None + status set.

NOTE (project history): some Qt tests crash on PROCESS EXIT (teardown)
even when every test PASSES.  If the run shows all tests passing but a
nonzero exit / STATUS_STACK_BUFFER_OVERRUN, that is the known teardown
issue — not a real failure.
"""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


@pytest.fixture(scope="module")
def qapp():
    """A plain QApplication (pytest-qt is not installed, nor required)."""
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def widgets(qapp):
    """Collects the dialogs a test builds; closes + releases them after it
    (what pytest-qt's ``qtbot.addWidget`` used to do)."""
    made = []
    yield made
    for w in made:
        try:
            if hasattr(w, "mark_clean_close"):
                w.mark_clean_close()
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass  # already deleted by Qt
    qapp.processEvents()

from packages.epicycle_annotator.batch_refine import RefineStep
from napari_epicycle.annotator.batch_refine_dialog import BatchRefineDialog


# -------- builders --------------------------------------------------


def _make_dialog(widgets, *, count_provider=None, available_tiers=None, calls=None):
    """Construct a dialog with stub callbacks.  Returns (dialog, calls)."""
    if calls is None:
        calls = {
            "preview": 0,
            "run": 0,
            "cancel": 0,
            "accept": 0,
            "reject": 0,
            "close": 0,
        }
    if count_provider is None:
        # Default: "all" → 142, "selected" → 7, "tier" → 1 per checked tier.
        def count_provider(spec):
            if spec["mode"] == "all":
                return 142
            if spec["mode"] == "selected":
                return 7
            return 10 * len(spec["tiers"])

    dlg = BatchRefineDialog(
        count_provider=count_provider,
        all_count=142,
        selected_count=7,
        available_tiers=available_tiers
        if available_tiers is not None
        else ["good", "review", "bad", "garbage", "curated"],
        on_preview_one=lambda: calls.__setitem__("preview", calls["preview"] + 1),
        on_run=lambda: calls.__setitem__("run", calls["run"] + 1),
        on_cancel=lambda: calls.__setitem__("cancel", calls["cancel"] + 1),
        on_accept=lambda: calls.__setitem__("accept", calls["accept"] + 1),
        on_reject=lambda: calls.__setitem__("reject", calls["reject"] + 1),
        on_close=lambda: calls.__setitem__("close", calls["close"] + 1),
    )
    widgets.append(dlg)
    return dlg, calls


# -------- refresh_count (live "Selected" count) ---------------------


def test_refresh_count_only_requeries_in_selected_idle(widgets):
    n = {"calls": 0}

    def cp(spec):
        n["calls"] += 1
        return 5 if spec["mode"] == "selected" else 99

    dlg, _ = _make_dialog(widgets, count_provider=cp)
    base = n["calls"]

    # Default mode is "all" → refresh_count must NOT re-query.
    dlg.refresh_count()
    assert n["calls"] == base

    # Switching to "Selected" triggers one count via _on_target_changed.
    dlg._radio_selected.setChecked(True)
    after_switch = n["calls"]
    assert after_switch > base

    # refresh_count now re-queries (selected + idle).
    dlg.refresh_count()
    assert n["calls"] == after_switch + 1

    # Not idle → no-op.
    dlg.set_state("running")
    frozen = n["calls"]
    dlg.refresh_count()
    assert n["calls"] == frozen


# -------- construction ---------------------------------------------


def test_construct_seeds_one_sdf_row(widgets):
    dlg, _ = _make_dialog(widgets)
    assert len(dlg._rows) == 1
    recipe = dlg.recipe()
    assert recipe is not None
    assert len(recipe) == 1
    assert recipe[0].method == "sdf"
    # Batch SDF defaults are 1.5 / 1.5 / topo-on.
    assert recipe[0].params["sigma_z"] == pytest.approx(1.5)
    assert recipe[0].params["sigma_xy"] == pytest.approx(1.5)
    assert recipe[0].params["preserve_topology"] is True


def test_default_target_is_all(widgets):
    dlg, _ = _make_dialog(widgets)
    spec = dlg.target_spec()
    assert spec["mode"] == "all"
    assert spec["tiers"] == []


# -------- recipe building ------------------------------------------


def test_add_two_steps_recipe_in_order(widgets):
    dlg, _ = _make_dialog(widgets)
    # Default row is SDF.  Change it to dilate_erode, then add a fill row.
    dlg._rows[0]._method_combo.setCurrentIndex(1)  # Dilate/Erode
    dlg._rows[0]._de_radius.setValue(-3)
    dlg._rows[0]._de_axis.setCurrentText("XY")

    dlg._on_add_step_clicked()
    assert len(dlg._rows) == 2
    dlg._rows[1]._method_combo.setCurrentIndex(2)  # Fill Holes
    dlg._rows[1]._fill_axis.setCurrentText("XZ")

    recipe = dlg.recipe()
    assert recipe is not None
    assert [s.method for s in recipe] == ["dilate_erode", "fill_holes"]
    assert recipe[0].params == {"radius": -3, "axis": "XY"}
    assert recipe[1].params == {"axis": "XZ"}
    assert all(isinstance(s, RefineStep) for s in recipe)


def test_preset_replaces_recipe_with_sdf_then_fill(widgets):
    dlg, _ = _make_dialog(widgets)
    dlg._on_add_step_clicked()
    dlg._on_add_step_clicked()
    assert len(dlg._rows) == 3
    dlg._on_preset_clicked()
    recipe = dlg.recipe()
    assert recipe is not None
    assert [s.method for s in recipe] == ["sdf", "fill_holes"]


def test_remove_step(widgets):
    dlg, _ = _make_dialog(widgets)
    dlg._on_add_step_clicked()
    assert len(dlg._rows) == 2
    dlg._on_row_remove(dlg._rows[0])
    assert len(dlg._rows) == 1


def test_reorder_moves_step_up(widgets):
    dlg, _ = _make_dialog(widgets)
    dlg._rows[0]._method_combo.setCurrentIndex(0)  # sdf
    dlg._on_add_step_clicked()
    dlg._rows[1]._method_combo.setCurrentIndex(2)  # fill_holes
    # Move the fill row up → it should now be first.
    dlg._on_row_move_up(dlg._rows[1])
    recipe = dlg.recipe()
    assert [s.method for s in recipe] == ["fill_holes", "sdf"]


# -------- target spec + count_provider -----------------------------


def test_target_spec_reflects_radio_selection(widgets):
    dlg, _ = _make_dialog(widgets)
    dlg._radio_selected.setChecked(True)
    assert dlg.target_spec()["mode"] == "selected"
    dlg._radio_plot.setChecked(True)
    assert dlg.target_spec()["mode"] == "plot"
    dlg._radio_tier.setChecked(True)
    assert dlg.target_spec()["mode"] == "tier"


def test_refresh_count_also_requeries_in_plot_idle(widgets):
    n = {"calls": 0}

    def cp(spec):
        n["calls"] += 1
        return 3 if spec["mode"] == "plot" else 99

    dlg, _ = _make_dialog(widgets, count_provider=cp)
    dlg._radio_plot.setChecked(True)
    before = n["calls"]
    dlg.refresh_count()
    assert n["calls"] == before + 1
    assert "3" in dlg._count_lbl.text()


def test_selected_radio_text_follows_set_target_counts(widgets):
    """It used to be frozen at construction ("Selected (7)" for ever)."""
    dlg, _ = _make_dialog(widgets)
    assert dlg._radio_selected.text() == "Selected (7)"
    dlg.set_target_counts(selected=2, plot=5)
    assert dlg._radio_selected.text() == "Selected (2)"
    assert dlg._radio_plot.text() == "Plot selection (5)"


def test_target_spec_reflects_tier_checkboxes(widgets):
    dlg, _ = _make_dialog(widgets)
    dlg._radio_tier.setChecked(True)
    dlg._tier_checks["review"].setChecked(True)
    dlg._tier_checks["bad"].setChecked(True)
    spec = dlg.target_spec()
    assert spec["mode"] == "tier"
    assert set(spec["tiers"]) == {"review", "bad"}


def test_count_provider_called_on_target_change(widgets):
    seen = []

    def cp(spec):
        seen.append(dict(spec))
        return 5

    dlg, _ = _make_dialog(widgets, count_provider=cp)
    seen.clear()  # ignore the construction-time call
    dlg._radio_selected.setChecked(True)
    assert any(s["mode"] == "selected" for s in seen)
    assert "5 objects" in dlg._count_lbl.text()


def test_run_disabled_when_count_zero(widgets):
    dlg, _ = _make_dialog(widgets, count_provider=lambda spec: 0)
    dlg.set_state("idle")
    assert dlg._run_btn.isEnabled() is False


def test_run_enabled_when_count_positive(widgets):
    dlg, _ = _make_dialog(widgets, count_provider=lambda spec: 9)
    dlg.set_state("idle")
    assert dlg._run_btn.isEnabled() is True


# -------- state transitions / visibility ---------------------------


# NOTE: we assert via ``isHidden()`` (the per-widget explicit-visibility
# flag) rather than ``isVisible()``.  ``isVisible()`` is False for every
# child until the top-level window is actually shown on screen, which we
# don't do in headless tests.  ``isHidden()`` reflects only whether
# ``setVisible(False)`` was called on the widget itself — exactly the
# state ``set_state`` toggles.


def test_state_idle_shows_preview_run_hides_rest(widgets):
    dlg, _ = _make_dialog(widgets)
    dlg.set_state("idle")
    assert not dlg._preview_btn.isHidden()
    assert not dlg._run_btn.isHidden()
    assert dlg._cancel_btn.isHidden()
    assert dlg._progress.isHidden()
    assert dlg._accept_btn.isHidden()
    assert dlg._reject_btn.isHidden()
    assert dlg._target_section.isEnabled()
    assert dlg._recipe_section.isEnabled()


def test_state_running_shows_progress_cancel_locks_sections(widgets):
    dlg, _ = _make_dialog(widgets)
    dlg.set_state("running")
    assert not dlg._cancel_btn.isHidden()
    assert not dlg._progress.isHidden()
    assert dlg._preview_btn.isHidden()
    assert dlg._run_btn.isHidden()
    assert dlg._accept_btn.isHidden()
    assert not dlg._target_section.isEnabled()
    assert not dlg._recipe_section.isEnabled()


def test_state_staged_shows_accept_reject_summary(widgets):
    dlg, _ = _make_dialog(widgets)
    dlg.set_state("staged")
    assert not dlg._accept_btn.isHidden()
    assert not dlg._reject_btn.isHidden()
    assert not dlg._summary_lbl.isHidden()
    assert dlg._preview_btn.isHidden()
    assert dlg._run_btn.isHidden()
    assert dlg._cancel_btn.isHidden()
    assert dlg._progress.isHidden()
    assert not dlg._target_section.isEnabled()


def test_invalid_state_raises(widgets):
    dlg, _ = _make_dialog(widgets)
    with pytest.raises(ValueError):
        dlg.set_state("bogus")


# -------- invalid recipe -------------------------------------------


def test_invalid_param_returns_none_and_sets_status(widgets):
    dlg, _ = _make_dialog(widgets)
    # Force an out-of-range sigma by bypassing the spinbox clamp: build a
    # row whose to_step() yields invalid params.  We patch the row's
    # to_step to emit an invalid step (simulates a future widget that
    # could produce an out-of-range value).
    bad = dlg._rows[0]
    bad.to_step = lambda: RefineStep(method="sdf", params={"sigma_z": 99.0})
    result = dlg.recipe()
    assert result is None
    # A fixed sentence on screen (UX audit M5); the raw reason is logged.
    from packages.epicycle_annotator.batch_refine import BATCH_STEP_INVALID

    assert dlg._status_lbl.text() == BATCH_STEP_INVALID.format(k=1)


# -------- callbacks ------------------------------------------------


def test_buttons_fire_injected_callbacks(widgets):
    dlg, calls = _make_dialog(widgets)
    dlg.set_state("idle")
    dlg._preview_btn.click()
    assert calls["preview"] == 1
    # Preview optimistically disables Run; the annotator returns to idle
    # if it was a no-op.  Reset state, then exercise Run.
    dlg.set_state("idle")
    dlg._run_btn.click()
    assert calls["run"] == 1

    dlg.set_state("running")
    dlg._cancel_btn.click()
    assert calls["cancel"] == 1

    dlg.set_state("staged")
    dlg._accept_btn.click()
    dlg.set_state("staged")
    dlg._reject_btn.click()
    assert calls["accept"] == 1
    assert calls["reject"] == 1


def test_mark_clean_close_suppresses_on_close(widgets):
    dlg, calls = _make_dialog(widgets)
    dlg.mark_clean_close()
    dlg.close()
    assert calls["close"] == 0


def test_x_close_fires_on_close(widgets):
    dlg, calls = _make_dialog(widgets)
    dlg.close()  # not marked clean → treated as X/teardown
    assert calls["close"] == 1


# -------- progress / summary setters -------------------------------


def test_set_progress_and_summary(widgets):
    dlg, _ = _make_dialog(widgets)
    dlg.set_state("running")
    dlg.set_progress(37, 142)
    assert dlg._progress.maximum() == 142
    assert dlg._progress.value() == 37
    dlg.set_state("staged")
    dlg.set_preview_summary(142, 11800)
    txt = dlg._summary_lbl.text()
    assert "142" in txt
    assert "11,800" in txt
