"""Qt-free tests for the ``experimental.async_`` slicing policy helpers in
``packages.epicycle_largedata.progressive``.

Policy (per-step slicing):

* :func:`ensure_async_slicing` — Step 1 turns async slicing ON for 2D browsing
  (measured: 36 -> 18 ms of UI-thread time per Z step on the (32,128,128)
  store).  No loader guard is needed: ON is what a progressive loader wants too.
* :func:`ensure_sync_slicing` — annotator launch / Step 1 exit turn it OFF,
  UNLESS a Step-1 progressive loader is live on the viewer (progressive 3D
  needs async).

The settings object and the loader probe are both monkeypatched, so no napari
settings file is touched.
"""
from __future__ import annotations

import types

import pytest

from packages.epicycle_largedata import progressive as prog


class _Settings:
    def __init__(self, async_):
        self.experimental = types.SimpleNamespace(async_=async_)


@pytest.fixture
def settings(monkeypatch):
    st = _Settings(True)
    monkeypatch.setattr(prog, "_get_settings", lambda: st)
    return st


def test_flips_off_when_no_loader_live(settings, monkeypatch, capsys):
    monkeypatch.setattr(prog, "_live_progressive_loaders", lambda v: [])
    assert prog.ensure_sync_slicing(object(), reason="test") is True
    assert settings.experimental.async_ is False
    assert "[SLICING] async slicing turned OFF (test)" in capsys.readouterr().out


def test_left_on_while_progressive_loader_live(settings, monkeypatch, capsys):
    monkeypatch.setattr(prog, "_live_progressive_loaders", lambda v: [object()])
    assert prog.ensure_sync_slicing(object(), reason="test") is False
    assert settings.experimental.async_ is True
    assert "left ON" in capsys.readouterr().out


def test_noop_when_already_sync(settings, monkeypatch, capsys):
    settings.experimental.async_ = False
    calls = []
    monkeypatch.setattr(prog, "_live_progressive_loaders", lambda v: calls.append(v) or [])
    assert prog.ensure_sync_slicing(object(), reason="test") is False
    assert calls == []  # loader probe not even consulted
    assert capsys.readouterr().out == ""


def test_no_viewer_treated_as_no_loader(settings, monkeypatch):
    monkeypatch.setattr(prog, "_live_progressive_loaders", lambda v: [object()])
    assert prog.ensure_sync_slicing(None, reason="test") is True
    assert settings.experimental.async_ is False


def test_settings_failure_is_swallowed(monkeypatch):
    def boom():
        raise RuntimeError("no napari")

    monkeypatch.setattr(prog, "_get_settings", boom)
    assert prog.ensure_sync_slicing(object()) is False


def test_live_loader_probe_survives_bad_viewer():
    assert prog._live_progressive_loaders(object()) == []


# ---------------------------------------------------------------------------
# Step 1 exit restores synchronous slicing
# ---------------------------------------------------------------------------

class _ExitViewer:
    """The slice of the viewer `on_step_exit` touches."""

    def __init__(self):
        self.layers = []
        self.scale_bar = types.SimpleNamespace(visible=True)


class _ExitCtl:
    """A controller stand-in bound to the REAL `on_step_exit`, with every
    teardown collaborator stubbed to record its call order."""

    def __init__(self, calls):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C)

        self.viewer = _ExitViewer()
        self.widget = types.SimpleNamespace()
        self._calls = calls
        self._left_step1 = False
        self._extract_running = False
        self.box_tool = None
        self._capture_dialogs = []
        for name in ("_set_review_busy", "_flush_source_state_save",
                     "_teardown_preview", "_clear_patch_locations",
                     "_clear_processed_store_layers",
                     "_detach_intensity_worker"):
            setattr(self, name, self._recorder(name))
        self._remove_source_layers_from_viewer = self._recorder(
            "_remove_source_layers_from_viewer")
        # The REAL helper (B5 lifted it out of `on_step_exit` so the
        # workspace-change teardown could share one implementation).  It only
        # walks `_capture_dialogs`, which this fake has, so binding it keeps
        # the extraction under test rather than stubbing it away.
        self._close_capture_dialogs = types.MethodType(
            C._close_capture_dialogs, self)
        self.on_step_exit = types.MethodType(C.on_step_exit, self)

    def _recorder(self, name):
        def _rec(*_a, **_k):
            self._calls.append(name)
        return _rec

    def _log(self, *_a, **_k):
        pass


def test_step_exit_restores_sync_slicing_last(monkeypatch):
    """Step 1's progressive engine persists `experimental.async_` ON; leaving
    the step must hand a synchronous-slicing viewer to Steps 2-5 (the engine's
    own settings snapshot otherwise makes the poison self-perpetuating)."""
    calls = []
    seen = {}

    def fake_ensure(viewer, *, reason=""):
        calls.append("ensure_sync_slicing")
        seen["viewer"] = viewer
        seen["reason"] = reason
        return True

    monkeypatch.setattr(prog, "ensure_sync_slicing", fake_ensure)

    ctl = _ExitCtl(calls)
    ctl.on_step_exit()

    assert "ensure_sync_slicing" in calls, "Step 1 exit never restored sync slicing"
    assert seen["viewer"] is ctl.viewer
    assert "Step 1 exit" in seen["reason"]
    # AFTER the layer teardown: a still-attached progressive loader would
    # (correctly) make the helper decline.
    assert calls.index("ensure_sync_slicing") > calls.index(
        "_remove_source_layers_from_viewer")
    assert calls[-1] == "ensure_sync_slicing"


def test_step_exit_survives_a_raising_sync_slicing_helper(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(prog, "ensure_sync_slicing", boom)
    ctl = _ExitCtl([])
    ctl.on_step_exit()          # must not raise
    assert ctl._left_step1 is True


def test_step_exit_clears_the_processed_store_layers(monkeypatch):
    """The streaming ⟨processed⟩ progressive layers were the ONE Step-1 layer
    class with no step-exit teardown: they stranded on the shared canvas (the
    pinned Step-3 all-Image-layers abort) AND kept a progressive loader live,
    which makes `ensure_sync_slicing` correctly DECLINE — so the async_ poison
    survived into Steps 2-5.  Clear them before the guard runs."""
    calls = []
    monkeypatch.setattr(
        prog, "ensure_sync_slicing",
        lambda *_a, **_k: calls.append("ensure_sync_slicing") or True)

    ctl = _ExitCtl(calls)
    ctl.on_step_exit()

    assert "_clear_processed_store_layers" in calls
    assert calls.index("_clear_processed_store_layers") < calls.index(
        "ensure_sync_slicing")


def test_step_exit_survives_a_raising_processed_store_clear(monkeypatch):
    monkeypatch.setattr(prog, "ensure_sync_slicing", lambda *_a, **_k: True)
    ctl = _ExitCtl([])

    def boom(*_a, **_k):
        raise RuntimeError("layer already gone")

    ctl._clear_processed_store_layers = boom
    ctl.on_step_exit()          # must not raise
    assert ctl._left_step1 is True


# ===========================================================================
# ensure_async_slicing (Brick 4d part 1)
# ===========================================================================

def test_async_flips_on_and_prints(settings, capsys):
    settings.experimental.async_ = False
    assert prog.ensure_async_slicing(object(), reason="step 1") is True
    assert settings.experimental.async_ is True
    assert "[SLICING] async slicing turned ON (step 1)" in capsys.readouterr().out


def test_async_noop_when_already_on(settings, capsys):
    assert settings.experimental.async_ is True          # fixture default
    assert prog.ensure_async_slicing(object(), reason="step 1") is False
    assert settings.experimental.async_ is True
    assert capsys.readouterr().out == ""                 # silent when nothing to do


def test_async_needs_no_viewer_and_no_loader_guard(settings, monkeypatch):
    """Turning async ON is safe with a progressive loader live -- ON is what a
    loader wants -- so the ON helper must NOT consult the loader probe at all."""
    settings.experimental.async_ = False
    calls = []
    monkeypatch.setattr(prog, "_live_progressive_loaders",
                        lambda v: calls.append(v) or [object()])
    assert prog.ensure_async_slicing(None, reason="step 1") is True
    assert settings.experimental.async_ is True
    assert calls == []


def test_async_settings_read_failure_is_swallowed(monkeypatch):
    def boom():
        raise RuntimeError("no napari")

    monkeypatch.setattr(prog, "_get_settings", boom)
    assert prog.ensure_async_slicing(object(), reason="test") is False


def test_async_settings_write_failure_is_swallowed(monkeypatch, capsys):
    class _ReadOnly:
        def __init__(self):
            self.experimental = self

        @property
        def async_(self):
            return False

        @async_.setter
        def async_(self, value):
            raise RuntimeError("read-only settings")

    monkeypatch.setattr(prog, "_get_settings", _ReadOnly)
    assert prog.ensure_async_slicing(object(), reason="test") is False
    assert "turned ON" not in capsys.readouterr().out   # no lie about the flip


def test_async_default_reason_is_usable(settings, capsys):
    settings.experimental.async_ = False
    assert prog.ensure_async_slicing(object()) is True
    out = capsys.readouterr().out
    assert "[SLICING] async slicing turned ON (" in out


def test_round_trip_on_then_off(settings, monkeypatch, capsys):
    """The two helpers are inverses over the same persisted setting."""
    monkeypatch.setattr(prog, "_live_progressive_loaders", lambda v: [])
    settings.experimental.async_ = False

    assert prog.ensure_async_slicing(object(), reason="step 1 enter") is True
    assert settings.experimental.async_ is True
    assert prog.ensure_sync_slicing(object(), reason="step 1 exit") is True
    assert settings.experimental.async_ is False
    # and back again, still idempotent at each end
    assert prog.ensure_async_slicing(object(), reason="re-enter") is True
    assert prog.ensure_async_slicing(object(), reason="re-enter") is False

    out = capsys.readouterr().out
    assert "turned ON (step 1 enter)" in out
    assert "turned OFF (step 1 exit)" in out


def test_async_is_exported():
    assert "ensure_async_slicing" in prog.__all__
    assert "ensure_sync_slicing" in prog.__all__


# ===========================================================================
# Step ENTER -- the inverse policy (Brick 4d part 2)
# ===========================================================================
class _EnterCtl:
    """A controller stand-in bound to the REAL `on_step_enter`, with every
    collaborator stubbed to record its call order."""

    def __init__(self, calls):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C)

        self.viewer = _ExitViewer()
        self.widget = types.SimpleNamespace(
            apply_progressive_default_open=None)
        self._calls = calls
        self._left_step1 = True
        for name in ("_note_source_unloaded_on_reentry",
                     "_apply_live_scale_bar",
                     "_maybe_restore_step1_sources",
                     "_maybe_offer_step1_migration"):
            setattr(self, name, self._recorder(name))
        self.on_step_enter = types.MethodType(C.on_step_enter, self)

    def _recorder(self, name):
        def _rec(*_a, **_k):
            self._calls.append(name)
        return _rec

    def _log(self, *_a, **_k):
        pass


def test_step_enter_turns_async_slicing_on_first(monkeypatch):
    """Step 1 browses 2D slices of a chunked store; async slicing takes that
    fetch off the UI thread (36 -> 18 ms per Z step, measured).  It must run
    BEFORE anything else on enter, so whatever is added while the step is
    active slices under the new policy."""
    calls = []
    seen = {}

    def fake_ensure(viewer, *, reason=""):
        seen["viewer"] = viewer
        seen["reason"] = reason
        calls.append("ensure_async_slicing")
        return True

    monkeypatch.setattr(prog, "ensure_async_slicing", fake_ensure)

    ctl = _EnterCtl(calls)
    ctl.on_step_enter()

    assert "ensure_async_slicing" in calls, "Step 1 enter never asked for async"
    assert seen["viewer"] is ctl.viewer
    assert "Step 1 enter" in seen["reason"]
    assert calls[0] == "ensure_async_slicing"
    assert calls.index("ensure_async_slicing") < calls.index(
        "_note_source_unloaded_on_reentry")
    assert ctl._left_step1 is False        # the rest of enter still ran


def test_step_enter_survives_a_raising_async_slicing_helper(monkeypatch):
    def boom(*_a, **_k):
        raise RuntimeError("settings unavailable")

    monkeypatch.setattr(prog, "ensure_async_slicing", boom)

    calls = []
    ctl = _EnterCtl(calls)
    ctl.on_step_enter()                    # must not raise

    # A settings backend that cannot be written is a perf regression, never a
    # broken step entry: everything after it still ran.
    assert "_note_source_unloaded_on_reentry" in calls
    assert "_maybe_restore_step1_sources" in calls
    assert ctl._left_step1 is False


def test_the_two_helpers_are_a_matched_pair():
    """Enter turns it ON, exit turns it OFF -- neither call was dropped."""
    import inspect

    from napari_epicycle.data_preparation.controller import (
        DataPreparationController as C)

    def _calls(fn):
        """The source with comment lines dropped -- the two helpers name each
        other in prose, and this test is about the CALLS."""
        return "\n".join(ln for ln in inspect.getsource(fn).splitlines()
                          if not ln.lstrip().startswith("#"))

    enter, exit_ = _calls(C.on_step_enter), _calls(C.on_step_exit)
    assert "ensure_async_slicing(self.viewer" in enter
    assert "ensure_sync_slicing(self.viewer" in exit_
    # ...and neither leaked into the other.
    assert "ensure_sync_slicing(" not in enter
    assert "ensure_async_slicing(" not in exit_


def test_the_very_first_step_activation_fires_on_step_enter():
    """The initial activation is NOT a special path: the shell selects the
    first step through the same `_select_step`, whose enter hook is
    unconditional -- so Step 1 gets async slicing on a cold start too, with no
    second call site to keep in sync.

    Pinned as a test because the alternative (a separate first-activation
    path) would need its own `ensure_async_slicing` call, and a future
    refactor that introduced one must fail here rather than silently leave a
    cold start on synchronous slicing."""
    import inspect

    from napari_epicycle.ui_shell import workflow_shell as ws_mod

    build = inspect.getsource(ws_mod.EpiCycleWorkflowShell)
    # The first step is activated through `_select_step`, not by poking the
    # stack index directly.
    assert "self._select_step(first_step_id)" in build

    select = inspect.getsource(ws_mod.EpiCycleWorkflowShell._select_step)
    # ...and `_select_step` calls the enter hook for whatever it selects,
    # with no "skip on first activation" branch.
    assert "on_step_enter" in select
    assert "new_ctrl.on_step_enter()" in select
    enter_at = select.index("new_ctrl.on_step_enter()")
    # The only guard before it is the has-a-controller check, not a
    # previous-step one (the exit hook above IS previous-step guarded).
    guard = select[:enter_at]
    assert 'hasattr(new_ctrl, "on_step_enter")' in guard
    assert "if previous_step" not in guard.rsplit(
        "# --- Step enter hook ---", 1)[-1]
