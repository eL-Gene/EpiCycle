"""Brick 5b - Step 1 says "Load first" instead of degrading, once the source
is loaded but OFF the canvas.

Brick 5 made step-exit tear the source layers down and step-enter leave them
down.  Two operations read the LIVE layers rather than the retained channel
refs, and silently degraded instead of saying so:

* **Extract** - `_live_channel_names` filters every ticked name against
  `viewer.layers`, returns [], the lazy crop is skipped and the EAGER fallback
  runs a whole-volume read only to die on "No valid 3D image layers selected."
* **(3) Box mode** - `_display_layers_present()` is False, so `start()` runs
  `_build_overview`: a strided read of the whole lazy store on the main thread,
  with no indicator.

Both now decline with the same plain line.  Qt-free: `__new__` controllers.
"""
from __future__ import annotations

import types

import pytest

from napari_epicycle.data_preparation import controller as ctlmod

C = ctlmod.DataPreparationController


def _ctl(*, channels=(("ch0", []),), on_canvas=False, processed=False):
    c = C.__new__(C)
    c._source_display_channels = list(channels) if channels else None
    c._source_layers_on_canvas = on_canvas
    c._is_processed_view_showing = lambda: processed
    c.status = []
    c.logged = []
    c._log = lambda tag, msg: c.logged.append((tag, msg))
    c._set_export_status = lambda m, success=True: c.status.append((m, success))
    return c


# --------------------------------------------------------------------------- #
# the predicate
# --------------------------------------------------------------------------- #

def test_loaded_but_off_canvas_is_off_canvas():
    assert _ctl()._source_is_off_canvas() is True


def test_nothing_loaded_is_not_off_canvas():
    """With no source at all the EXISTING messages are the honest ones."""
    assert _ctl(channels=None)._source_is_off_canvas() is False


def test_layers_on_canvas_is_not_off_canvas():
    assert _ctl(on_canvas=True)._source_is_off_canvas() is False


def test_the_processed_view_is_not_off_canvas():
    """The clean raw/processed swap removes the raw layers on PURPOSE, and
    extract-from-processed is a supported flow - blocking it would be a
    regression, not a fix."""
    assert _ctl(processed=True)._source_is_off_canvas() is False


def test_a_raising_processed_probe_does_not_crash_the_predicate():
    c = _ctl()

    def _boom():
        raise RuntimeError("layers gone")

    c._is_processed_view_showing = _boom
    assert c._source_is_off_canvas() is True


# --------------------------------------------------------------------------- #
# extract
# --------------------------------------------------------------------------- #

def _extract_ctl(**kw):
    c = _ctl(**kw)
    c._extract_running = c._extract_launching = False
    c._trim_launching = c._step1_busy = False
    c._migration_offer_timer = None
    c.reads = []
    c._selected_channel_names = lambda: c.reads.append("selected") or ["ch0"]
    c._resolve_extract_source = (
        lambda names: c.reads.append("resolve") or (None, None, None, None))
    return c


def test_extract_after_re_entry_asks_for_a_load_and_reads_nothing():
    c = _extract_ctl()
    c.extract_patches()
    assert c.status == [
        ("Click Load to view the source before extracting.", False)]
    assert c.reads == [], "no eager whole-volume read may be attempted"
    assert c._extract_launching is False, "the launch flag must not stick"


def test_extract_is_not_blocked_while_the_processed_view_is_showing():
    c = _extract_ctl(processed=True)
    with pytest.raises(Exception):
        c.extract_patches()               # proceeds past the gate, then stubs out
    assert c.reads[:1] == ["selected"]
    assert c.status == [] or c.status[0][0] != (
        "Click Load to view the source before extracting.")


def test_extract_with_nothing_loaded_still_uses_the_old_message_path():
    c = _extract_ctl(channels=None)
    with pytest.raises(Exception):
        c.extract_patches()
    assert c.reads[:1] == ["selected"]


# --------------------------------------------------------------------------- #
# (3) box mode
# --------------------------------------------------------------------------- #

class _BoxTool:
    def __init__(self):
        self.started = 0
        self.added = 0
        self.synced = []

    def start(self):
        self.started += 1
        return True

    def is_active(self):
        return self.started > 0

    def add_box(self):
        self.added += 1

    def stop(self, keep_visible=False):
        pass

    def _sync_mode_button(self, on):
        self.synced.append(on)

    def _set_ndisplay(self, n):
        pass


def _box_ctl(**kw):
    c = _ctl(**kw)
    c.box_tool = _BoxTool()
    c.viewer = types.SimpleNamespace(layers=[])
    c._arbitrate_for_box_mode = lambda: c.logged.append(("ARB", ""))
    return c


def test_box_mode_after_re_entry_declines_and_never_builds_an_overview():
    c = _box_ctl()
    c.on_box_mode_toggled(True)
    assert c.box_tool.started == 0, "start() would run a whole-store overview"
    assert c.box_tool.synced == [False], "the toggle snaps back off"
    assert c.status == [
        ("Click Load to view the source before extracting.", False)]
    assert not any(t == "ARB" for t, _ in c.logged), "no arbitration either"


def test_add_box_after_re_entry_declines():
    c = _box_ctl()
    c.add_box()
    assert c.box_tool.started == 0 and c.box_tool.added == 0
    assert c.status


def test_click_to_place_after_re_entry_declines():
    c = _box_ctl()
    c.on_place_toggled(True)
    assert c.box_tool.started == 0
    assert c.status


def test_box_mode_still_works_when_the_source_is_on_canvas():
    c = _box_ctl(on_canvas=True)
    c.on_box_mode_toggled(True)
    assert c.box_tool.started == 1
    assert c.status == []


def test_turning_box_mode_OFF_is_never_gated():
    """Leaving a mode must never be blocked - that would strand the user in it."""
    c = _box_ctl()
    c.box_tool.started = 1
    c.on_box_mode_toggled(False)
    assert c.status == []
