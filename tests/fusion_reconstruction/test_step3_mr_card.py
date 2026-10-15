"""Model Refinement look, Brick B2 -- the result card and its
empty sentences.  The REAL launch / finish handlers run through
``_refine_launch_harness``; ``ui.set_refine_card`` / ``show_refine_status``
are recorders.  The validation shown is the LAST SAMPLED value, never
``final_loss.val`` (usually 0.0)."""
from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace

from napari_epicycle.fusion_reconstruction import _paths
from napari_epicycle.fusion_reconstruction import _step3_mr_text as M
from tests.fusion_reconstruction._refine_launch_harness import (
    make_launch_controller, run_timers,
)


def _controller(tmp_path, monkeypatch, **kw):
    cards, status = [], []
    label = SimpleNamespace(text="", setText=lambda t: setattr(
        label, "text", t))
    ui = SimpleNamespace(
        set_refine_card=lambda title="", **k: cards.append((title, k)),
        show_refine_status=lambda text, kind="neutral": status.append(
            (text, kind)),
        refinement_output_label=label,
    )
    for p in ("patchA", "patchB"):
        (_paths.refinement_pairs_root(tmp_path) / p).mkdir(parents=True,
                                                          exist_ok=True)
    ctrl, h = make_launch_controller(tmp_path, monkeypatch, ui=ui, **kw)
    ctrl._append_log = lambda _m: None
    return ctrl, h, cards, status, label


def _finish(ctrl, worker, *, n_train, losses, val_losses, final=0.1):
    weights = Path(worker.save_path) / f"{worker.bucket}_model"
    weights.parent.mkdir(parents=True, exist_ok=True)
    weights.touch()
    ctrl._on_refinement_finished({
        "model_path": str(weights), "model_name": worker.bucket,
        "bundle_name": worker.bundle_name, "bucket": worker.bucket,
        "final_train_loss": final, "final_val_loss": 0.0,
        "n_train": n_train, "n_val": 5 if val_losses else 0,
        "train_losses": losses, "val_losses": val_losses,
    })


def test_one_pass_card(tmp_path, monkeypatch):
    ctrl, h, cards, status, label = _controller(
        tmp_path, monkeypatch, mode="Fine-tune Lateral", name="wound_v1")
    ctrl._launch_refinement()
    ctrl._refinement_run_started = time.monotonic() - 125
    _finish(ctrl, h.workers[0], n_train=40, losses=[0.3, 0.2],
            val_losses=[(5, 0.3), (10, 0.25)], final=0.1234)
    assert len(cards) == 1
    title, k = cards[0]
    assert title == "wound_v1"
    assert k["trained_on"] == M.MR_CARD_TRAINED_VALUE.format(
        t="40", n="40", m="0", k=2, patches="patches")
    assert k["lateral_loss"] == "train 0.12 · validation 0.25"
    assert "0.0000" not in k["lateral_loss"]          # never final_loss.val
    assert k["axial_loss"] == ""
    assert k["started_from"] == M.MR_CARD_INIT_SCRATCH
    assert k["took"] == "2m 05s"
    assert k["new_run_hint"] == M.MR_NEW_RUN_HINT.format(name="wound_v1")
    assert label.text.endswith("wound_v1") or "wound_v1" in label.text
    assert status[-1][1] == "info"
    assert status[-1][0].startswith("Complete: wound_v1 · lateral model "
                                    "(lateral 0.12)")


def test_no_validation_set_shows_the_train_loss_only(tmp_path, monkeypatch):
    ctrl, h, cards, status, _ = _controller(
        tmp_path, monkeypatch, mode="Fine-tune Axial", name="ax")
    ctrl._launch_refinement()
    _finish(ctrl, h.workers[0], n_train=9, losses=[0.3], val_losses=[],
            final=0.5)
    k = cards[0][1]
    assert k["axial_loss"] == "train 0.50"
    assert k["lateral_loss"] == ""
    assert k["trained_on"].startswith("9 pairs (0 lateral, 9 axial)")


def test_both_fills_the_card_once_after_the_second_pass(
        tmp_path, monkeypatch):
    ctrl, h, cards, status, _ = _controller(
        tmp_path, monkeypatch, mode="Fine-tune Both", name="both_v1")
    ctrl._launch_refinement()
    _finish(ctrl, h.workers[0], n_train=30, losses=[0.3],
            val_losses=[(5, 0.4)], final=0.2)
    assert cards == [], "no card between the passes"
    run_timers(h)
    assert len(h.workers) == 2
    _finish(ctrl, h.workers[1], n_train=20, losses=[0.4],
            val_losses=[(5, 0.6), (10, 0.5)], final=0.3)
    assert len(cards) == 1
    title, k = cards[0]
    assert title == "both_v1"
    assert k["trained_on"].startswith("50 pairs (30 lateral, 20 axial) from 2")
    assert k["lateral_loss"] == "train 0.20 · validation 0.40"
    assert k["axial_loss"] == "train 0.30 · validation 0.50"
    assert k["started_from"] == M.MR_CARD_STARTED_BOTH.format(
        lat="scratch", ax="scratch")
    assert status[-1] == (M.MR_STATUS_DONE.format(
        name="both_v1", which="both models", l="0.20", a="0.30",
        time=status[-1][0].rsplit(" ", 1)[-1]), "info")


def test_the_empty_card_sentences(tmp_path, monkeypatch):
    ctrl, h, cards, status, _ = _controller(tmp_path, monkeypatch)
    ctrl._refresh_refine_card()
    assert cards[-1][0] == M.MR_OUT_EMPTY
    # two bundles in the run: the newest created_at is named
    root = _paths.refined_models_root(tmp_path)
    for name, stamp in (("refined_01", "2026-10-01T10:00:00Z"),
                        ("refined_02", "2026-10-09T08:30:00Z")):
        (root / name).mkdir(parents=True)
        (root / name / "manifest.json").write_text(
            json.dumps({"created_at": stamp}), encoding="utf-8")
    ctrl._refresh_refine_card()
    assert cards[-1][0] == M.MR_OUT_EMPTY_SESSION.format(
        name="refined_02", when="2026-10-09 08:30")
    # a run switch / workspace reset forgets the card
    ctrl._refinement_last_card = {"title": "x"}
    ctrl._refinement_last_card = None
    ctrl._refresh_refine_card()
    assert cards[-1][0].startswith("Nothing trained in this session yet.")
