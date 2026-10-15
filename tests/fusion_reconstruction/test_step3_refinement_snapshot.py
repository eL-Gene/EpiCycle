"""Model Refinement Brick A, I4: the launch snapshot.

The REAL launch path runs (``_refine_launch_harness``); only the worker,
thread and timer are faked.  A widget edit between the lateral and the
axial pass must not reach the axial pass, and a bucket that would start
from scratch is refused at launch for every bucket the Mode trains.
"""

from __future__ import annotations

from types import SimpleNamespace

from tests.fusion_reconstruction._refine_launch_harness import (
    _Combo,
    make_launch_controller,
)


class _Label:
    def __init__(self):
        self._t = ""

    def setStyleSheet(self, _s):
        pass

    def setText(self, t):
        self._t = t

    def text(self):
        return self._t


def _record_worker_args(monkeypatch):
    """Wrap the harness fake worker so a test can read what it was given."""
    from napari_epicycle.fusion_reconstruction import controller as cmod

    inner = cmod._RefinementTrainingWorker
    seen = []

    class _Rec(inner):
        def __init__(self, *args, **kwargs):
            seen.append((args, kwargs))
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(cmod, "_RefinementTrainingWorker", _Rec)
    return seen


def _wire(ctrl):
    logs = []
    ctrl._append_log = logs.append
    return logs


def test_axial_pass_ignores_a_widget_edit_between_passes(tmp_path, monkeypatch):
    ctrl, h = make_launch_controller(tmp_path, monkeypatch)
    _wire(ctrl)
    seen = _record_worker_args(monkeypatch)
    state = {"epochs": 2, "train_frac": 0.5}
    ctrl._build_training_config = lambda: {"n_epochs": state["epochs"]}
    ctrl._read_split_config = lambda: dict(
        train_frac=state["train_frac"], spatial_split=False, chunk_size=1,
    )
    inits = {"lateral": ("scratch", None), "axial": ("scratch", None)}
    ctrl._resolve_init_mode_for_bucket = lambda b: inits[b]

    ctrl._launch_refinement()               # lateral pass starts
    # The user edits the widgets while the lateral pass trains.
    state["epochs"] = 999
    state["train_frac"] = 0.9
    inits["axial"] = ("custom", "C:/somewhere/other_model")
    ctrl.ui.refine_init_combo = _Combo("Import model")
    ctrl._launch_next_refinement()          # axial pass

    assert [w.bucket for w in h.workers] == ["lateral", "axial"]
    lat_args, _ = seen[0]
    ax_args, ax_kwargs = seen[1]
    assert lat_args[0]["n_epochs"] == 2
    assert ax_args[0]["n_epochs"] == 2, "the axial pass saw a widget edit"
    assert len(ax_args[1]) == 2 and len(ax_args[3]) == 2, "split changed"
    assert ax_args[5] == "scratch" and ax_args[6] is None, "init changed"
    assert ax_kwargs["init_combo_text"] == "From scratch"


def test_snapshot_holds_everything_the_passes_read(tmp_path, monkeypatch):
    ctrl, _h = make_launch_controller(tmp_path, monkeypatch, name="mine")
    _wire(ctrl)
    ctrl._launch_refinement()
    snap = ctrl._refinement_launch_snapshot
    assert snap["mode"] == "Fine-tune Both"
    assert snap["name"] == "mine"
    assert set(snap["inits"]) == {"lateral", "axial"}
    assert snap["config"] == {"n_epochs": 2}
    assert snap["split"]["train_frac"] == 0.5
    assert snap["init_combo_text"] == "From scratch"
    assert isinstance(snap["patches"], list)
    assert ctrl._refinement_run_started is not None


def test_snapshot_cleared_when_the_run_ends(tmp_path, monkeypatch):
    ctrl, _h = make_launch_controller(tmp_path, monkeypatch)
    _wire(ctrl)
    ctrl._launch_refinement()
    assert ctrl._refinement_launch_snapshot is not None
    ctrl._restore_launch_button()
    assert getattr(ctrl, "_refinement_launch_snapshot", None) is None


def test_both_refuses_when_only_the_axial_bucket_has_no_source(
    tmp_path, monkeypatch,
):
    status = _Label()
    ui = SimpleNamespace(
        refinement_status_label=status,
        show_refine_status=lambda text, kind="neutral": status.setText(text),
        refine_init_combo=_Combo("Import model"),
    )
    ctrl, h = make_launch_controller(tmp_path, monkeypatch, ui=ui)
    logs = _wire(ctrl)
    ctrl._resolve_init_mode_for_bucket = lambda b: (
        ("custom", "C:/m/lat") if b == "lateral" else ("scratch", None)
    )

    ctrl._launch_refinement()

    sentence = (
        "No source for axial: it would start from scratch \u2014 "
        "pick a Source or choose From scratch."
    )
    assert status.text() == sentence
    assert any(sentence in m for m in logs)
    assert h.workers == [], "a half run must never start"
    assert ctrl._refinement_queue == []
    assert not getattr(ctrl, "_refinement_running", False)
    assert getattr(ctrl, "_refinement_bundle_name", None) is None
    assert getattr(ctrl, "_refinement_launch_snapshot", None) is None


def test_single_bucket_mode_only_checks_its_own_bucket(tmp_path, monkeypatch):
    status = _Label()
    ui = SimpleNamespace(
        refinement_status_label=status,
        refine_init_combo=_Combo("Import model"),
    )
    ctrl, h = make_launch_controller(
        tmp_path, monkeypatch, mode="Fine-tune Lateral", ui=ui,
    )
    _wire(ctrl)
    asked = []

    def _resolve(b):
        asked.append(b)
        return ("custom", "C:/m/lat") if b == "lateral" else ("scratch", None)

    ctrl._resolve_init_mode_for_bucket = _resolve
    ctrl._launch_refinement()
    assert asked == ["lateral"]
    assert [w.bucket for w in h.workers] == ["lateral"]
    assert "No source" not in status.text()


def test_from_scratch_choice_is_not_a_refusal(tmp_path, monkeypatch):
    ctrl, h = make_launch_controller(tmp_path, monkeypatch)
    _wire(ctrl)
    ctrl._launch_refinement()
    assert [w.bucket for w in h.workers] == ["lateral"]
