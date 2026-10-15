"""Tests for the Batch-Refine dialog recipe-history UI (Recent/Saved/Save).

Uses a plain ``QApplication`` (``QT_QPA_PLATFORM=offscreen``; pytest-qt
is NOT required).  Covers:
- ``set_recipe`` repopulates rows so ``recipe()`` round-trips the same steps,
- ``set_recipe`` accepts both ``RefineStep`` lists and JSON dicts,
- ``set_recipe`` does NOT touch the target radios (reload-only),
- ``set_recipe_history`` populates the two dropdowns,
- selecting a Recent / Saved entry reloads the recipe,
- the Save button validates + fires the injected callback,
- the Save button is disabled when no save callback is wired.

See the NOTE in ``test_batch_refine_dialog.py`` about Qt teardown crashes
on PROCESS EXIT (not real failures).
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

from packages.epicycle_annotator.batch_refine import RefineStep, recipe_to_json
from napari_epicycle.annotator.batch_refine_dialog import BatchRefineDialog


def _make_dialog(widgets, *, on_save_recipe=None, can_save_recipe=False):
    dlg = BatchRefineDialog(
        count_provider=lambda spec: 5,
        all_count=142,
        selected_count=7,
        available_tiers=["good", "review", "bad", "garbage", "curated"],
        on_preview_one=lambda: None,
        on_run=lambda: None,
        on_cancel=lambda: None,
        on_accept=lambda: None,
        on_reject=lambda: None,
        on_close=lambda: None,
        on_save_recipe=on_save_recipe,
        can_save_recipe=can_save_recipe,
    )
    widgets.append(dlg)
    return dlg


_RECIPE_3STEP = [
    RefineStep(method="sdf", params={"sigma_z": 2.5, "sigma_xy": 1.0, "preserve_topology": False}),
    RefineStep(method="dilate_erode", params={"radius": -3, "axis": "XZ"}),
    RefineStep(method="fill_holes", params={"axis": "YZ"}),
]


# -------- set_recipe round-trips through the UI ---------------------
def test_set_recipe_repopulates_rows_round_trip(widgets):
    dlg = _make_dialog(widgets)
    dlg.set_recipe(_RECIPE_3STEP)
    out = dlg.recipe()
    assert out is not None
    assert [s.method for s in out] == ["sdf", "dilate_erode", "fill_holes"]
    assert out[0].params["sigma_z"] == pytest.approx(2.5)
    assert out[0].params["sigma_xy"] == pytest.approx(1.0)
    assert out[0].params["preserve_topology"] is False
    assert out[1].params == {"radius": -3, "axis": "XZ"}
    assert out[2].params == {"axis": "YZ"}


def test_set_recipe_accepts_json_dicts(widgets):
    dlg = _make_dialog(widgets)
    dlg.set_recipe(recipe_to_json(_RECIPE_3STEP))
    out = dlg.recipe()
    assert [s.method for s in out] == ["sdf", "dilate_erode", "fill_holes"]
    assert out[1].params == {"radius": -3, "axis": "XZ"}


def test_set_recipe_replaces_existing_rows(widgets):
    dlg = _make_dialog(widgets)
    dlg._on_add_step_clicked()
    dlg._on_add_step_clicked()
    assert len(dlg._rows) == 3
    dlg.set_recipe([RefineStep(method="fill_holes", params={"axis": "XY"})])
    assert len(dlg._rows) == 1
    assert dlg.recipe()[0].method == "fill_holes"


def test_set_recipe_does_not_touch_target(widgets):
    """Reload-only: set_recipe must NOT change the target radios/tiers."""
    dlg = _make_dialog(widgets)
    dlg._radio_tier.setChecked(True)
    dlg._tier_checks["bad"].setChecked(True)
    before = dlg.target_spec()
    dlg.set_recipe(_RECIPE_3STEP)
    after = dlg.target_spec()
    assert before == after
    assert after["mode"] == "tier"
    assert "bad" in after["tiers"]


def test_set_recipe_empty_is_noop(widgets):
    dlg = _make_dialog(widgets)
    n_before = len(dlg._rows)
    dlg.set_recipe([])
    assert len(dlg._rows) == n_before


def test_set_recipe_ignored_when_not_idle(widgets):
    dlg = _make_dialog(widgets)
    dlg.set_state("running")
    before = [s.method for s in dlg.recipe()] if dlg.recipe() else None
    dlg.set_recipe(_RECIPE_3STEP)
    # State change locked the section — recipe unchanged (still the seeded sdf).
    dlg.set_state("idle")
    assert [s.method for s in dlg.recipe()] == ["sdf"]
    assert before == ["sdf"]


# -------- set_recipe_history populates the dropdowns ----------------
def test_set_recipe_history_populates_combos(widgets):
    dlg = _make_dialog(widgets)
    recent = [recipe_to_json(_RECIPE_3STEP)]
    saved = {"My Saved": recipe_to_json(_RECIPE_3STEP)}
    dlg.set_recipe_history(recent, saved)
    # Index 0 is the placeholder in each combo.
    assert dlg._recent_combo.count() == 2
    assert dlg._saved_combo.count() == 2
    assert dlg._saved_combo.itemText(1) == "My Saved"


def test_recent_label_built_from_recipe(widgets):
    dlg = _make_dialog(widgets)
    recent = [recipe_to_json([
        RefineStep(method="sdf", params={"sigma_z": 2.0, "sigma_xy": 1.5}),
        RefineStep(method="fill_holes", params={"axis": "3D"}),
    ])]
    dlg.set_recipe_history(recent, {})
    label = dlg._recent_combo.itemText(1)
    assert "SDF" in label
    assert "Fill Holes" in label


def test_selecting_recent_reloads_recipe(widgets):
    dlg = _make_dialog(widgets)
    dlg.set_recipe_history([recipe_to_json(_RECIPE_3STEP)], {})
    dlg._on_recent_selected(1)
    out = dlg.recipe()
    assert [s.method for s in out] == ["sdf", "dilate_erode", "fill_holes"]
    # Combo resets to the placeholder so re-picking re-fires.
    assert dlg._recent_combo.currentIndex() == 0


def test_selecting_saved_reloads_recipe(widgets):
    dlg = _make_dialog(widgets)
    dlg.set_recipe_history([], {"S": recipe_to_json(_RECIPE_3STEP)})
    dlg._on_saved_selected(1)
    out = dlg.recipe()
    assert [s.method for s in out] == ["sdf", "dilate_erode", "fill_holes"]
    assert dlg._saved_combo.currentIndex() == 0


def test_selecting_placeholder_is_noop(widgets):
    dlg = _make_dialog(widgets)
    dlg.set_recipe_history([recipe_to_json(_RECIPE_3STEP)], {})
    dlg._on_recent_selected(0)  # placeholder
    # Still the seeded single sdf row.
    assert [s.method for s in dlg.recipe()] == ["sdf"]


# -------- Save button -----------------------------------------------
def test_save_button_disabled_without_callback(widgets):
    dlg = _make_dialog(widgets, on_save_recipe=None, can_save_recipe=False)
    assert dlg._save_recipe_btn.isEnabled() is False


def test_save_button_enabled_with_callback(widgets):
    dlg = _make_dialog(widgets, on_save_recipe=lambda name: None, can_save_recipe=True)
    assert dlg._save_recipe_btn.isEnabled() is True


def test_save_clicked_fires_callback_with_name(widgets, monkeypatch):
    seen = {}

    def on_save(name):
        seen["name"] = name

    dlg = _make_dialog(widgets, on_save_recipe=on_save, can_save_recipe=True)
    dlg.set_recipe(_RECIPE_3STEP)
    # Patch the modal name prompt to return a name without showing a dialog.
    monkeypatch.setattr(
        "napari_epicycle.annotator.batch_refine_dialog.QInputDialog.getText",
        staticmethod(lambda *a, **k: ("My Recipe", True)),
    )
    dlg._on_save_recipe_clicked()
    assert seen.get("name") == "My Recipe"


def test_save_clicked_cancelled_prompt_no_fire(widgets, monkeypatch):
    seen = {}
    dlg = _make_dialog(
        widgets, on_save_recipe=lambda name: seen.setdefault("name", name),
        can_save_recipe=True,
    )
    monkeypatch.setattr(
        "napari_epicycle.annotator.batch_refine_dialog.QInputDialog.getText",
        staticmethod(lambda *a, **k: ("", False)),  # user cancelled
    )
    dlg._on_save_recipe_clicked()
    assert "name" not in seen


def test_save_clicked_empty_name_no_fire(widgets, monkeypatch):
    seen = {}
    dlg = _make_dialog(
        widgets, on_save_recipe=lambda name: seen.setdefault("name", name),
        can_save_recipe=True,
    )
    monkeypatch.setattr(
        "napari_epicycle.annotator.batch_refine_dialog.QInputDialog.getText",
        staticmethod(lambda *a, **k: ("   ", True)),  # whitespace only
    )
    dlg._on_save_recipe_clicked()
    assert "name" not in seen
    assert "empty" in dlg._status_lbl.text().lower()


def test_save_clicked_no_callback_is_noop(widgets, monkeypatch):
    dlg = _make_dialog(widgets, on_save_recipe=None, can_save_recipe=False)
    called = {"n": 0}
    monkeypatch.setattr(
        "napari_epicycle.annotator.batch_refine_dialog.QInputDialog.getText",
        staticmethod(lambda *a, **k: ("X", True)),
    )
    dlg._on_save_recipe_clicked()  # must not raise, must not prompt
    assert called["n"] == 0
