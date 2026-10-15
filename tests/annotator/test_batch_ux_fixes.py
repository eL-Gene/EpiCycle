"""The UX-audit fixes of Batch Refine commits 4-5 (M1-M7, L1-L3, L6-L8) and
the wiring audit's M1 / M2.

Behavioural: every refusal is READ back off the real dialog's status line
after the annotator is done with it (``set_state("idle")`` clears that
line, so a refusal must set the state first and the sentence second); the
staged state names what to do; counts read "1 object" / "1,204 objects";
the plot's third reason; the plain-open pre-check; one plot read per
refresh; the running marquee before the snapshot copy; and a golden table
of every reworded sentence, literally.

Plain ``QApplication`` (pytest-qt is not installed).
"""
import os
import types

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from packages.epicycle_annotator import batch_refine as br  # noqa: E402
from tests.annotator.test_batch_chain_dialog import (  # noqa: E402
    make_ann,
    make_dialog,
    pump,
)


@pytest.fixture(scope="module")
def qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


@pytest.fixture
def widgets(qapp):
    made = []
    yield made
    for w in made:
        try:
            if hasattr(w, "mark_clean_close"):
                w.mark_clean_close()
            w.close()
            w.deleteLater()
        except RuntimeError:
            pass
    qapp.processEvents()


# ================================================= the four refusals (M4)


class TestRefusalsStayOnScreen:
    def test_no_targets(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        # every object vanishes after the open: the cache still counts 5
        ann.sync_manager.shared_labels["lbl"][:] = 0
        ann._on_batch_run()
        assert dlg._status_lbl.text() == br.BATCH_NO_TARGETS
        # the count line and Run agree with the refusal (wiring audit M1)
        assert dlg._count_lbl.text().startswith(
            br.BATCH_COUNT.format(k="0 objects"))
        assert not dlg._run_btn.isEnabled()

    def test_no_volume(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        ann.sync_manager = types.SimpleNamespace(shared_labels={})
        ann._run_batch_refine([1, 2])
        assert dlg._status_lbl.text() == br.BATCH_NO_VOLUME

    def test_add_step(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        for row in list(dlg._rows):
            dlg._remove_row(row)
        ann._run_batch_refine([1, 2])
        assert dlg._status_lbl.text() == br.BATCH_ADD_STEP

    def test_preview_with_no_object(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        ann._selected = 0
        ann._on_batch_preview_one()
        assert dlg._status_lbl.text() == br.BATCH_PREVIEW_NO_OBJECT


# ======================================================== staged (M1)


def test_the_staged_status_names_what_to_do(monkeypatch, widgets, qapp):
    ann = make_ann(monkeypatch, widgets)
    ann._open_batch_refine_dialog()
    dlg = ann._batch_refine_dialog
    ann._on_batch_run()
    assert pump(qapp, lambda: not ann._batch_running)
    assert ann._batch_staged
    assert dlg._status_lbl.text() == br.BATCH_STAGED
    assert "Writing" not in dlg._status_lbl.text()


# ================================================== the running marquee (L7)


def test_running_and_the_marquee_come_before_the_copy(monkeypatch, widgets,
                                                     qapp):
    ann = make_ann(monkeypatch, widgets)
    ann._open_batch_refine_dialog()
    dlg = ann._batch_refine_dialog
    seen = {}

    class _Vol(np.ndarray):
        def copy(self, *a, **k):
            # the FIRST copy is the annotator's snapshot (the worker's own
            # working copy comes later, on its thread)
            seen.setdefault("state", dlg._state)
            seen.setdefault("marquee", dlg._progress.maximum() == 0)
            seen.setdefault("status", dlg._status_lbl.text())
            return np.ndarray.copy(self, *a, **k)

    vol = ann.sync_manager.shared_labels["lbl"].view(_Vol)
    ann.sync_manager.shared_labels["lbl"] = vol
    ann._on_batch_run()
    assert pump(qapp, lambda: not ann._batch_running)
    assert seen["state"] == "running" and seen["marquee"]
    assert seen["status"] == br.BATCH_STAGE_PREPARING.format(n="5 objects")


# ============================================================ counts (M2, L1)


class TestCounts:
    def test_objects_word(self):
        assert br.objects_word(1) == "1 object"
        assert br.objects_word(0) == "0 objects"
        assert br.objects_word(1204) == "1,204 objects"

    def test_one_object(self, widgets):
        dlg = make_dialog(widgets, count=1)
        assert dlg._count_lbl.text() == "→ Will process 1 object"
        assert dlg._run_btn.text() == "Run on 1 object"
        dlg.set_preview_summary(1, 1204)
        assert dlg._summary_lbl.text() == \
            "✓ Previewed 1 object · 1,204 voxels changed"

    def test_thousands(self, widgets):
        dlg = make_dialog(widgets, count=1204)
        dlg.set_target_counts(all=1204, selected=2500, plot=10000)
        assert dlg._radio_all.text() == "All active (1,204)"
        assert dlg._radio_selected.text() == "Selected (2,500)"
        assert dlg._radio_plot.text() == "Plot selection (10,000)"
        assert dlg._count_lbl.text() == "→ Will process 1,204 objects"
        assert br.BATCH_STAGE_REFINING.format(done=1000, total=2000) == \
            "Refining object 1,000 of 2,000"

    def test_stage_lines_say_object(self, monkeypatch):
        from tests.annotator.test_batch_progress_states import _ann

        ann = _ann(monkeypatch, {"t": 0.0})
        ann._batch_n_targets = 1
        ann._on_batch_worker_stage("preparing", 0, 0)
        assert ann._batch_refine_dialog.status[-1] == \
            "Preparing — copying the labels and finding 1 object…"
        ann._on_batch_worker_stage("writing", 0, 1)
        assert ann._batch_refine_dialog.status[-1] == \
            "Writing 1 object into the volume…"


# ============================================================ the plot (M3, M6)


class TestPlotReasons:
    def test_no_plot_is_short_and_wraps(self, widgets, qapp):
        dlg = make_dialog(widgets, count=0)
        dlg._radio_plot.setChecked(True)
        dlg.set_target_counts(plot=0, plot_available=False)
        assert dlg._count_lbl.text() == br.BATCH_PLOT_NONE
        assert dlg._count_lbl.wordWrap()
        dlg.show()
        qapp.processEvents()
        with_reason = dlg.minimumSizeHint().width()
        dlg._radio_all.setChecked(True)
        qapp.processEvents()
        without = dlg.minimumSizeHint().width()
        # the reason no longer widens the dialog (it made it 948 px)
        assert with_reason <= without, (with_reason, without)

    def test_another_patch(self, widgets):
        dlg = make_dialog(widgets, count=0)
        dlg._radio_plot.setChecked(True)
        dlg.set_target_counts(plot=0, plot_available=True,
                              plot_other_patch=True)
        assert dlg._count_lbl.text() == br.BATCH_PLOT_OTHER_PATCH
        dlg.set_target_counts(plot_other_patch=False)
        assert dlg._count_lbl.text() == br.BATCH_PLOT_EMPTY

    def test_the_annotator_turns_the_marker_into_the_reason(self, monkeypatch,
                                                           widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._get_plot_selection_cb = lambda: br.PLOT_OTHER_PATCH
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        dlg._radio_plot.setChecked(True)
        ann._do_refresh_batch_count()
        assert dlg._count_lbl.text().startswith(br.BATCH_PLOT_OTHER_PATCH)
        assert ann._resolve_batch_targets({"mode": "plot"}) == []


# ============================================================ M7


class TestPlainOpenStartsOnThePlot:
    def test_plot_points_highlighted(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._get_plot_selection_cb = lambda: [2, 3]
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        assert dlg._radio_plot.isChecked()
        assert dlg.target_spec()["mode"] == "plot"

    def test_nothing_highlighted(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._get_plot_selection_cb = lambda: []
        ann._open_batch_refine_dialog()
        assert ann._batch_refine_dialog._radio_all.isChecked()

    def test_an_explicit_open_keeps_selected(self, monkeypatch, widgets):
        ann = make_ann(monkeypatch, widgets)
        ann._get_plot_selection_cb = lambda: [2, 3]
        ann._open_batch_refine_dialog(explicit_targets=[1],
                                      default_target="selected")
        assert ann._batch_refine_dialog._radio_selected.isChecked()


# ============================================================ L6


class TestOneReadPerRefresh:
    def test_the_plot_is_read_once_and_printed_once(self, monkeypatch,
                                                    widgets, capsys):
        ann = make_ann(monkeypatch, widgets)
        calls = []
        ann._get_plot_selection_cb = lambda: calls.append(1) or [2, 3]
        ann._open_batch_refine_dialog()
        dlg = ann._batch_refine_dialog
        dlg._radio_plot.setChecked(True)
        calls.clear()
        capsys.readouterr()
        ann._do_refresh_batch_count()
        assert calls == [1]
        out = capsys.readouterr().out
        assert out.count("[BATCH] resolve plot") == 1

    def test_a_radio_switch_counts_once(self, widgets):
        n = {"calls": 0}

        def cp(spec):
            n["calls"] += 1
            return 3

        dlg = make_dialog(widgets, count_provider=cp)
        before = n["calls"]
        dlg._radio_selected.setChecked(True)     # All off, Selected on
        assert n["calls"] == before + 1


# ============================================================ M5


class TestFixedSentences:
    def test_an_invalid_step(self, widgets, monkeypatch, capsys):
        dlg = make_dialog(widgets)

        def bad(self):
            raise ValueError("raw-secret-reason")

        monkeypatch.setattr(br.RefineStep, "validate", bad)
        assert dlg.recipe() is None
        assert dlg._status_lbl.text() == br.BATCH_STEP_INVALID.format(k=1)
        assert "raw-secret-reason" in capsys.readouterr().out
        assert "raw-secret" not in dlg._status_lbl.text()

    def test_a_failed_save(self, widgets, monkeypatch, capsys):
        import napari_epicycle.annotator.batch_refine_dialog as mod

        def boom(name):
            raise OSError("C:\\\\very\\\\long\\\\raw\\\\path")

        dlg = make_dialog(widgets, on_save_recipe=boom, can_save_recipe=True)
        monkeypatch.setattr(mod.QInputDialog, "getText",
                            staticmethod(lambda *a, **k: ("mine", True)))
        dlg._on_save_recipe_clicked()
        assert dlg._status_lbl.text() == br.BATCH_SAVE_FAILED
        assert "long" in capsys.readouterr().out

    def test_the_annotator_save_failure(self, monkeypatch, widgets, capsys):
        ann = make_ann(monkeypatch, widgets)

        def boom(name, rj):
            raise PermissionError("WinError 5 raw")

        ann._save_batch_recipe_cb = boom
        ann._open_batch_refine_dialog()
        ann._save_batch_recipe("x")
        dlg = ann._batch_refine_dialog
        assert dlg._status_lbl.text() == br.BATCH_SAVE_FAILED
        assert "WinError" in capsys.readouterr().out

    def test_keep_tier_tooltip_says_objects(self, widgets):
        dlg = make_dialog(widgets)
        tip = dlg._keep_tier_cb.toolTip()
        assert tip == br.BATCH_KEEP_TIER_TIP
        assert "labels" not in tip


# ============================================================ L8: golden


GOLDEN = {
    "BATCH_TARGET_ALL": "All active ({n:,})",
    "BATCH_TARGET_SELECTED": "Selected ({n:,})",
    "BATCH_TARGET_PLOT": "Plot selection ({n:,})",
    "BATCH_TARGET_ADDED": "Added this session ({n:,})",
    "BATCH_PLOT_EMPTY": "→ Lasso or click points in the QC plot first",
    "BATCH_PLOT_NONE":
        "→ No QC plot for this patch yet — extract QC features first.",
    "BATCH_PLOT_OTHER_PATCH":
        "→ The QC plot shows another patch — open this patch's plot first.",
    "BATCH_COUNT": "→ Will process {k}",
    "BATCH_WARNING_BANNER":
        "⚠ Batch refine modifies many objects at once. Preview first. After "
        "Save Curation it's permanent.",
    "BATCH_NO_VOLUME": "No label volume loaded.",
    "BATCH_NO_TARGETS": "No objects in the target set.",
    "BATCH_ADD_STEP": "Add at least one step.",
    "BATCH_STAGE_PREPARING":
        "Preparing — copying the labels and finding {n}…",
    "BATCH_STAGE_REFINING": "Refining object {done:,} of {total:,}",
    "BATCH_STAGE_WRITING": "Writing {n} into the volume…",
    "BATCH_CANCELLING": "Cancelling…",
    "BATCH_CANCELLED": "Batch cancelled — no changes applied.",
    "BATCH_PATCH_CHANGED": "Patch changed — batch discarded.",
    "BATCH_LAUNCH_FAILED":
        "The batch could not start — nothing was changed. The console has "
        "the details.",
    "BATCH_FAILED":
        "The batch stopped with an error — nothing was changed. The console "
        "has the details.",
    "BATCH_SUMMARY": "✓ Previewed {n} · {v:,} voxels changed",
    "BATCH_SUMMARY_EMPTIED":
        "{n} unchanged — a step would have removed all of their voxels, so "
        "none of their steps were kept: {ids}",
    "BATCH_SUMMARY_ABSENT": "{n} no longer in the volume: {ids}",
    "BATCH_SUMMARY_STOPPED":
        "{n} stopped part-way: step {k} ({method}) could not run on them, so "
        "they keep the steps before it (details in the console): {ids}",
    "BATCH_SUMMARY_STOPPED_FIRST":
        "{n} left unchanged: step 1 ({method}) could not run on them "
        "(details in the console): {ids}",
    "BATCH_SUMMARY_STOPPED_ENCLOSED":
        "{n} already touching their neighbours everywhere, so Settle had "
        "nothing to move; their earlier steps were kept: {ids}",
    "SETTLE_ENCLOSED_REASON":
        "There is no background left around this object",
    "BATCH_STAGED": "Review the result, then Accept or Reject.",
    "BATCH_RECIPE_TITLE": "RECIPE  (top → bottom, per object)",
    "BATCH_SAVED_STEPS_LABEL": "Saved steps",
    "BATCH_RUN_BTN": "Run on {n}",
    "BATCH_PREVIEW_BTN": "Preview on object {id}",
    "BATCH_PREVIEW_NO_OBJECT":
        "Select one object in the viewer or the navigator to preview on.",
    "BATCH_STEP_INVALID":
        "Step {k} has a setting out of range — check its numbers.",
    "BATCH_LOAD_FAILED":
        "Could not load that recipe — it is empty or unreadable.",
    "BATCH_SAVE_NO_WORKSPACE":
        "Recipes can only be saved inside a workspace.",
    "BATCH_SAVE_INVALID": "Fix the steps before saving.",
    "BATCH_SAVE_NO_NAME": "A recipe name cannot be empty.",
    "BATCH_SAVE_OVERWRITING": "Overwriting the saved recipe “{name}”.",
    "BATCH_SAVE_FAILED":
        "Could not save the recipe — the console has the details.",
    "BATCH_SAVED": "Saved the recipe “{name}”.",
    "BATCH_PRESET_READY": "Smooth is ready — click Run to preview it.",
    "BATCH_STILL_STOPPING":
        "The last batch is still stopping — try again in a moment.",
    "BATCH_BUSY":
        "Step 3 is busy with another job — try again when it finishes.",
    "BATCH_NOT_CLEAN": "Accept or Discard the change on screen first.",
    "NAV_BATCH_REFINE_TIP":
        "Run steps, or a saved Refine recipe, on all, selected, "
        "plot-selected or tier-filtered objects at once.",
    "ADD_BATCH_BUSY":
        "Batch Refine is working on these cells — Accept or Reject it first.",
    # the audit of commits 6-8: every sentence the dialog shows, literally
    "BATCH_ALREADY_OPEN":
        "Batch Refine is already open — Accept, Reject or Cancel it there "
        "first.",
    "BATCH_ORPHANS_RUNNING":
        "{n} from the stopped batch are still finishing — Run is available "
        "again in a few seconds.",
    "BATCH_CANCELLING_SETTLE":
        "Cancelling… a Settle step cannot stop halfway; the objects already "
        "started finish and are thrown away.",
    "BATCH_ROWS_OFF":
        "The steps are the Refine recipe's. Pick “none” above to build steps "
        "here.",
    "BATCH_CHAIN_LABEL": "Refine-tab recipe",
    "BATCH_CHAIN_NONE": "— none: build steps below —",
    "BATCH_CHAIN_TIP":
        "Run a recipe saved on the Refine tab (its Recipes menu) on every "
        "target object. The steps below are then the recipe's.",
    "BATCH_CHAIN_UNREADABLE":
        "{name} — cannot be read (details in the console)",
    "BATCH_CHANNEL_LABEL": "Channel",
    "BATCH_CHANNEL_TIP":
        "The image channel the recipe's wall or brightness steps read. A "
        "recipe stores no channel.",
    "BATCH_PREVIEW_BTN_NONE": "Preview on one object",
    "BATCH_PREVIEW_TIP":
        "Run the recipe on this one object first — the object selected in "
        "the viewer or the navigator.",
    "BATCH_REASON_RUN_ONLY": "Run: ",
    "BATCH_REASON_PREVIEW_ONLY": "Preview: ",
    "BATCH_NOTICE_TITLE": "Batch Refine",
    "BATCH_KEEP_TIER_LABEL": "Keep original QC tier (don't mark curated)",
    "BATCH_KEEP_TIER_TIP":
        "When checked, refined objects stay in their current QC tier "
        "(good / review / …) instead of being reclassified to 'curated'.  "
        "Their measurements are still flagged stale — run Recalculate QC "
        "to refresh.",
    "BATCH_RECENT_TIP":
        "Reload a recipe used earlier this session (steps + params only).",
    "BATCH_SAVED_TIP":
        "Reload a recipe saved to this workspace (steps + params only).",
    "BATCH_SAVE_PROMPT_TITLE": "Save recipe",
    "BATCH_SAVE_PROMPT_LABEL": "Recipe name:",
}

#: The batch refusals that live beside `step_blocker` (refine_recipe.py).
GOLDEN_RR = {
    "BATCH_NEEDS_CHANNEL":
        "Step {k} ({word}) reads an image channel — pick one under Channel.",
    "BATCH_NEEDS_PROB":
        "Step {k} ({word}) needs a probability map, and this patch has none.",
    "BATCH_SPACING_KNOWN":
        "The recipe was saved with a voxel size; this patch has none — set "
        "it in QC Settings, or save the recipe again on the Refine tab.",
    "BATCH_SPACING_UNKNOWN":
        "The recipe was saved without a voxel size; this patch has one — "
        "save the recipe again on the Refine tab.",
    "BATCH_NO_DIFFUSION":
        "Step {k} (Settle) uses Diffused edge, which needs too much memory "
        "to run on many objects at once. Save a copy of the recipe with "
        "Sharp edge or Soft edge on the Refine tab.",
    "BATCH_CONTOUR_3D":
        "Step {k} (Active contour) works in 3D, about {lo}–{hi} s per object "
        "— at most {cap} objects in one batch. Pick fewer, or save the "
        "recipe with Orientation XY.",
    "BATCH_PROB_SHAPE":
        "The probability map does not match this patch's size (it may be "
        "from before a downsample), so steps that read it cannot run here.",
    "BATCH_CHANNEL_SHAPE":
        "The channel does not match this patch's size, so steps that read "
        "it cannot run here. Pick another channel.",
    "BATCH_TOO_LARGE":
        "The largest object here would need about {gb} GB ({w} at a time) — "
        "more than a batch may use ({budget} GB). Pick smaller objects, or "
        "save the recipe with a smaller Edge may move.",
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_sentence(name):
    assert getattr(br, name) == GOLDEN[name]


@pytest.mark.parametrize("name", sorted(GOLDEN_RR))
def test_golden_refusal(name):
    from packages.epicycle_annotator import refine_recipe as rr

    assert getattr(rr, name) == GOLDEN_RR[name]


def test_the_diffusion_refusal_uses_the_pages_words():
    from napari_epicycle.annotator.advanced_features_widget import (
        REFINE_SOLVER_LABELS,
    )
    from packages.epicycle_annotator import refine_recipe as rr

    text = rr.BATCH_NO_DIFFUSION
    assert "label diffusion" not in text and "Watershed" not in text
    sharp, soft, diffused = REFINE_SOLVER_LABELS
    assert sharp.split(" (")[0] in text and soft in text
    assert diffused.split(" (")[0] in text


def test_the_3d_cap_sentence_follows_the_constant():
    from packages.epicycle_annotator import refine_recipe as rr

    assert rr.CONTOUR_3D_SECONDS == (15, 25)
    lo, hi = rr.CONTOUR_3D_SECONDS
    text = rr.BATCH_CONTOUR_3D.format(k=2, lo=lo, hi=hi,
                                      cap=rr.BATCH_CONTOUR_3D_MAX)
    assert "about 15–25 s per object" in text


def test_the_worker_cap_is_the_measured_knee():
    from packages.epicycle_annotator import batch_chain as bc

    assert bc.CHAIN_MAX_WORKERS == 2


def test_save_with_no_steps_says_add_a_step(widgets):
    dlg = make_dialog(widgets, can_save_recipe=True,
                      on_save_recipe=lambda n: None)
    for row in list(dlg._rows):
        dlg._remove_row(row)
    dlg._on_save_recipe_clicked()
    assert dlg._status_lbl.text() == br.BATCH_ADD_STEP
    assert not hasattr(br, "BATCH_SAVE_NO_STEPS")


def test_the_setters_extras_are_keyword_only(widgets):
    import inspect

    dlg = make_dialog(widgets)
    for name, extras in (("set_preview_object", ["obj_id"]),
                         ("set_orphans_running", ["n"]),
                         ("set_chain_recipes", ["problem", "no_workspace"])):
        params = inspect.signature(getattr(dlg, name)).parameters
        for p in extras:
            assert params[p].kind is inspect.Parameter.KEYWORD_ONLY, (name, p)
            assert params[p].default is None, (name, p)
