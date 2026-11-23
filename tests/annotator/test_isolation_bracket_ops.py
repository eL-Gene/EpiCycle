"""The isolation bracket: what Reshape and Refine Further leave isolated.

Start of an edit records what is isolated and takes it away; Accept, Cancel
and Discard put it back (the new pieces join what was isolated before).  The
rule itself is pure and lives in ``_isolation_bracket``; these tests pin the
widget side: the three helpers, the Reshape Accept / Cancel paths, the
teardown paths, the removed Refine Further flag, and the Reshape tab's
"pieces kept in view" knob.  Qt-free fakes; the offscreen panel only for the
spin box.
"""
from __future__ import annotations

import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from napari_epicycle.annotator._isolation_bracket import DEFAULT_PIECES_CAP
from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
from napari_epicycle.annotator.refinement_state import StagedOp

from .test_reshape_fragment_refine_everywhere import (
    FakeAnnotator,
    FakeLayer,
    _guided_fake,
    bind,
    solved,  # noqa: F401  (module fixture the guided fake needs)
)

_K = Curation3DAnnotator

_HELPERS = (
    "_iso_owner_alive", "_iso_bracket_open", "_iso_take", "_iso_drop",
    "_iso_begin", "_iso_suspend", "_iso_end", "_iso_live_ids",
    "_reshape_pieces_cap", "_layer_by_name", "_iso_release_authority",
    "_iso_release_if_idle", "_iso_pieces_by_size",
)


class _Layers(list):
    def __getitem__(self, key):
        if isinstance(key, str):
            for lyr in self:
                if getattr(lyr, "name", None) == key:
                    return lyr
            raise KeyError(key)
        return super().__getitem__(key)


class _Nav:
    """Navigator stand-in: the Isolate box and the auto-advance box."""

    def __init__(self, ticked=False, auto=False, next_ok=True):
        self.next_ok = next_ok
        self.on_next = None
        self.log = []
        self.box = ticked
        self.set_calls = []
        self.advanced = 0
        self._isolate_cb = SimpleNamespace(
            blockSignals=lambda b: None,
            setChecked=self._set, isChecked=lambda: self.box)
        self._auto_advance_cb = SimpleNamespace(isChecked=lambda: auto)

    def _set(self, v):
        self.box = bool(v)

    def is_isolate_checked(self):
        return self.box

    def set_isolate_checked(self, v):
        self.set_calls.append(bool(v))
        self.box = bool(v)

    def clear_selection_silently(self):
        self.log.append("clear")

    def select_rows_silently(self, ids):
        self.selected = set(int(i) for i in ids)
        self.log.append("select")

    def selected_label_ids(self):
        return set(getattr(self, "selected", set()))

    def select_next_triage(self):
        self.log.append("next")
        self.advanced += 1
        if self.on_next is not None:
            self.on_next()
        return self.next_ok


def _bind_helpers(ann):
    for n in _HELPERS:
        setattr(ann, n, types.MethodType(getattr(_K, n), ann))
    ann._ISO_OWNER_KINDS = _K._ISO_OWNER_KINDS
    return ann


def _live(ann, mode, ids):
    """Put the live isolation state of *ann* into *mode* showing *ids*."""
    ids = tuple(ids)
    if mode == "multi":
        ann._multi_label_proxy_set = frozenset(ids)
        ann._isolation_label_set = frozenset(ids)
        ann._isolation_proxy_active = True
        ann._isolation_active = True
        ann._pre_isolation_colormap = "orig-cmap"
    elif mode == "single":
        ann._isolation_proxy_active = True
        ann._isolation_proxy_label_id = ids[0]
        ann._isolation_active = True
    elif mode == "single2d":
        ann._isolation_active = True


def _env(mode="off", ids=(1, 2, 3), box=True, staged=(StagedOp.RESHAPE,),
         cap=None):
    """A bare widget with the bracket helpers bound and every seam recorded."""
    main = SimpleNamespace(name="labels", visible=True, selected_label=ids[0],
                           colormap="cmap-now")
    ann = SimpleNamespace(
        viewer=SimpleNamespace(layers=_Layers([main])),
        _active_labels_name="labels",
        _isolation_proxy_layer=None, _isolation_proxy_active=False,
        _isolation_proxy_label_id=None, _multi_label_proxy_layers={},
        _multi_label_proxy_set=None, _isolation_label_set=None,
        _isolation_active=False, _pre_isolation_colormap=None,
        _colormap_authority=None, _iso_bracket=None,
        _split_label_id=ids[0], _interp_label_id=ids[0],
        _navigator_widget=_Nav(ticked=box),
        calls=[], main=main, oi_kwargs=[],
    )
    _bind_helpers(ann)
    ann.staged_kinds = lambda: frozenset(ann._staged)
    ann._staged = set(staged)
    ann._get_advanced_features_widget = lambda: (
        None if cap is None else SimpleNamespace(
            get_split_params=lambda: {"pieces_cap": cap}))

    def _remove(restore_visibility=True):
        ann.calls.append(("REMOVE", restore_visibility))
        ann._isolation_proxy_active = False
        ann._isolation_proxy_label_id = None
        ann._multi_label_proxy_set = None
        ann._multi_label_proxy_layers = {}

    ann._remove_isolation_proxy = _remove
    ann._propagate_show_selected_label = (
        lambda v, **k: ann.calls.append(("SHOWSEL", v)))
    ann._propagate_colormap_to_subviewers = (
        lambda cm: ann.calls.append(("SUBCMAP", cm)))
    def _oi(ids, **k):
        ann.calls.append(("OI", frozenset(ids)))
        ann.oi_kwargs.append(k)

    ann._on_isolation_changed = _oi
    _live(ann, mode, ids)
    return ann


def _ois(ann):
    return [c[1] for c in ann.calls if c[0] == "OI"]


# --------------------------------------------------------------------------- #
# begin / suspend / end
# --------------------------------------------------------------------------- #
class TestBeginSuspendEnd:
    def test_single_round_trip(self):
        ann = _env("single", ids=(5,), box=True)
        snap = ann._iso_begin("reshape")
        assert (snap.mode, snap.ids, snap.checkbox, snap.main_visible) == (
            "single", frozenset({5}), True, True)
        ann._iso_suspend()
        assert not ann._isolation_active and not ann._isolation_proxy_active
        ann._iso_end("reshape")
        assert _ois(ann) == [frozenset({5})]
        assert ann._iso_bracket is None
        assert ann._navigator_widget.set_calls == [True]

    def test_multi_round_trip_restores_the_whole_set(self):
        ann = _env("multi", ids=(1, 2, 3))
        ann._iso_begin("reshape")
        assert ann._iso_bracket.snapshot.mode == "multi"
        ann._iso_suspend()
        assert ann._multi_label_proxy_set is None
        assert ann._isolation_label_set is None
        ann._iso_end("reshape")
        assert _ois(ann) == [frozenset({1, 2, 3})]

    def test_end_selects_the_rows_of_what_is_isolated(self):
        ann = _env("multi", ids=(1, 2, 3))
        ann._iso_begin("reshape")
        ann._iso_suspend()
        ann._iso_end("reshape", created=(7,), consumed=())
        isolated = _ois(ann)[-1]
        assert ann._navigator_widget.selected_label_ids() == set(isolated)
        assert {1, 2, 3} <= set(isolated)

    def test_the_rows_are_selected_before_the_isolation_is_applied(self):
        ann = _env("multi", ids=(1, 2, 3))
        nav = ann._navigator_widget
        nav.select_rows_silently = lambda ids: ann.calls.append(("SELECT",))
        ann._iso_begin("reshape")
        ann._iso_suspend()
        ann.calls.clear()
        ann._iso_end("reshape")
        names = [c[0] for c in ann.calls]
        assert names.index("SELECT") < names.index("OI")

    def test_the_2d_multi_set_has_no_proxy_but_is_still_multi(self):
        ann = _env("off", ids=(1, 2))
        ann._isolation_label_set = frozenset({1, 2})
        ann._isolation_active = True
        snap = ann._iso_begin("reshape")
        assert snap.mode == "multi" and snap.ids == frozenset({1, 2})

    def test_off_makes_no_isolation_call(self):
        ann = _env("off", box=False)
        snap = ann._iso_begin("reshape")
        assert snap.mode == "off" and snap.ids == frozenset()
        ann._iso_suspend()
        ann._iso_end("reshape")
        assert _ois(ann) == []
        assert ann._navigator_widget.set_calls == [False]

    def test_begin_reads_main_visibility_and_the_box(self):
        ann = _env("single", ids=(5,), box=False)
        ann.main.visible = False
        snap = ann._iso_begin("reshape")
        assert snap.main_visible is False and snap.checkbox is False

    def test_suspend_puts_the_main_colormap_back_and_keeps_ownership(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority, ColormapOwner)

        auth = ColormapAuthority()
        auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        auth.set_restore_point("authority-point")
        ann = _env("multi", ids=(1, 2))
        ann._colormap_authority = auth
        ann._iso_begin("reshape")
        ann._iso_suspend()
        # The widget's own saved colormap goes back, not the authority's.
        assert ann.main.colormap == "orig-cmap"
        assert ("SUBCMAP", "orig-cmap") in ann.calls
        assert ann._pre_isolation_colormap is None
        # The authority's restore point is popped, its owner is unchanged.
        assert auth.restore() is None
        assert auth.current_owner == ColormapOwner.ANNOTATOR_ISOLATION
        assert ("REMOVE", False) in ann.calls

    def test_suspend_restores_from_the_widget_when_the_authority_has_none(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority)

        ann = _env("multi", ids=(1, 2))
        ann._colormap_authority = ColormapAuthority()
        ann._iso_begin("reshape")
        ann._iso_suspend()
        assert ann.main.colormap == "orig-cmap"

    def test_suspend_turns_the_2d_single_shader_flag_off(self):
        ann = _env("single2d", ids=(5,))
        ann.main.show_selected_label = True
        ann._iso_begin("reshape")
        ann._iso_suspend()
        assert ann.main.show_selected_label is False

    def test_end_in_mode_off_releases_the_ownership(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority, ColormapOwner)

        auth = ColormapAuthority()
        auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        ann = _env("off", box=False)
        ann._colormap_authority = auth
        ann._iso_begin("reshape")
        ann._iso_suspend()
        assert auth.current_owner == ColormapOwner.ANNOTATOR_ISOLATION
        ann._iso_end("reshape")
        assert auth.current_owner is None

    def test_end_clearing_a_forced_isolation_is_a_forget(self):
        ann = _env("off", ids=(3,), box=False, staged=(StagedOp.INTERPOLATE,))
        ann._iso_begin("interp")
        _live(ann, "single", (3,))
        ann._iso_end("interp")
        assert ann.calls[-1] == ("OI", frozenset())
        assert ann.oi_kwargs == [{"reason": "forget"}]

    def test_end_with_every_isolated_cell_gone_isolates_nothing(self):
        ann = _env("single", ids=(7,), box=True)
        ann._iso_begin("reshape", target=7)
        ann._iso_suspend()
        ann._iso_end("reshape", (7,), ())
        assert _ois(ann) == []
        assert ann._navigator_widget.set_calls == [False]

    def test_take_with_another_kind_leaves_the_bracket_alone(self):
        ann = _env("single", ids=(5,))
        ann._iso_begin("reshape")
        assert ann._iso_take("interp") is None
        assert ann._iso_bracket is not None
        assert ann._iso_take("reshape").kind == "reshape"
        assert ann._iso_bracket is None

    def test_a_raising_navigator_is_printed_not_swallowed(self, capsys):
        ann = _env("single", ids=(5,))

        def _boom():
            raise RuntimeError("deleted")

        ann._navigator_widget.is_isolate_checked = _boom
        snap = ann._iso_begin("reshape")
        assert snap.checkbox is False
        ann._navigator_widget.set_isolate_checked = _boom
        ann._iso_end("reshape")
        out = capsys.readouterr().out
        assert "begin: navigator read raised" in out
        assert "end: navigator set raised" in out

    def test_suspend_leaves_the_main_layer_hidden_state_alone(self):
        ann = _env("single", ids=(5,))
        ann.main.visible = False
        ann._iso_begin("reshape")
        ann._iso_suspend()
        assert ann.main.visible is False

    def test_owner_kind_only(self):
        ann = _env("single", ids=(5,))
        ann._iso_begin("reshape")
        ann._iso_end("interp")
        assert ann._iso_bracket is not None
        assert _ois(ann) == []

    def test_end_with_no_bracket_is_a_no_op(self):
        ann = _env("single", ids=(5,))
        ann._iso_end("reshape")
        assert ann.calls == [] and ann._navigator_widget.set_calls == []

    def test_restore_false_just_drops(self):
        ann = _env("multi", ids=(1, 2))
        ann._iso_begin("reshape")
        ann._iso_suspend()
        ann.calls.clear()
        ann._iso_end("reshape", restore=False)
        assert ann._iso_bracket is None
        assert ann.calls == [] and ann._navigator_widget.set_calls == []

    def test_a_stale_bracket_is_replaced(self):
        ann = _env("multi", ids=(1, 2), staged=())
        ann._iso_begin("reshape")
        ann._isolation_label_set = None
        ann._multi_label_proxy_set = None
        ann._isolation_active = False
        ann._isolation_proxy_active = False
        ann._staged = {StagedOp.INTERPOLATE}
        ann._iso_begin("interp")
        assert ann._iso_bracket.kind == "interp"
        assert ann._iso_bracket.snapshot.mode == "off"

    def test_a_live_owner_keeps_its_bracket(self):
        ann = _env("multi", ids=(1, 2))
        ann._iso_begin("reshape")
        ann._staged = {StagedOp.RESHAPE, StagedOp.REFINE}
        snap = ann._iso_begin("interp")
        assert ann._iso_bracket.kind == "reshape"
        assert snap is ann._iso_bracket.snapshot

    def test_target_comes_from_the_edit(self):
        ann = _env("single", ids=(5,))
        ann._split_label_id = 9
        ann._iso_begin("reshape")
        assert ann._iso_bracket.target == 9

    def test_end_clears_what_the_edit_isolated_when_nothing_was_before(self):
        # Refine Further forces {label}; mode "off" before -> clear it.
        ann = _env("off", ids=(3,), box=False, staged=(StagedOp.INTERPOLATE,))
        ann._iso_begin("interp")
        _live(ann, "single", (3,))          # the forced isolation
        ann._iso_end("interp")
        assert _ois(ann) == [frozenset()]
        assert ann._navigator_widget.set_calls == [False]

    def test_end_survives_a_raising_isolation_call(self):
        ann = _env("single", ids=(5,))
        ann._iso_begin("reshape")
        ann._on_isolation_changed = lambda *a, **k: 1 / 0
        ann._iso_end("reshape")
        assert ann._iso_bracket is None
        assert ann._navigator_widget.set_calls == [True]

    def test_diagnostics_are_cp1252(self, capsys):
        ann = _env("multi", ids=(1, 2))
        ann._iso_begin("reshape")
        ann._iso_suspend()
        ann._iso_end("reshape")
        out = capsys.readouterr().out
        assert out.count("[ISO-BRACKET]") >= 3
        out.encode("cp1252")


# --------------------------------------------------------------------------- #
# the pieces cap
# --------------------------------------------------------------------------- #
class TestPiecesCap:
    def test_end_reads_the_cap_at_accept_time(self):
        ann = _env("single", ids=(7,), cap=2)
        ann._iso_begin("reshape", target=7)
        ann._iso_suspend()
        ann._iso_end("reshape", (7,), (11, 12, 13))
        assert _ois(ann) == [frozenset({11})]       # 3 > cap 2: first only

    def test_at_or_below_the_cap_shows_every_piece(self):
        ann = _env("single", ids=(7,), cap=3)
        ann._iso_begin("reshape", target=7)
        ann._iso_suspend()
        ann._iso_end("reshape", (7,), (11, 12, 13))
        assert _ois(ann) == [frozenset({11, 12, 13})]

    def test_superseded_original_leaves_and_survivor_stays(self):
        ann = _env("multi", ids=(7, 50), cap=10)
        ann._iso_begin("reshape", target=7)
        ann._iso_suspend()
        ann._iso_end("reshape", (7,), (11, 12))
        assert _ois(ann) == [frozenset({50, 11, 12})]

    def test_a_surviving_original_stays_isolated(self):
        ann = _env("multi", ids=(7, 50), cap=10)
        ann._iso_begin("reshape", target=7)
        ann._iso_suspend()
        ann._iso_end("reshape", (), (11,))
        assert _ois(ann) == [frozenset({7, 50, 11})]

    def test_fallback_is_the_default(self):
        ann = _env("off")
        assert ann._reshape_pieces_cap() == DEFAULT_PIECES_CAP == 10

    def test_unreadable_setting_falls_back(self):
        ann = _env("off")
        ann._get_advanced_features_widget = lambda: SimpleNamespace(
            get_split_params=lambda: 1 / 0)
        assert ann._reshape_pieces_cap() == DEFAULT_PIECES_CAP

    def test_the_setting_is_read(self):
        ann = _env("off", cap=4)
        assert ann._reshape_pieces_cap() == 4


# --------------------------------------------------------------------------- #
# Reshape Accept / Cancel on the real methods
# --------------------------------------------------------------------------- #
_ISO_NAMES = _HELPERS + ("_iso_forget",)


def _arm(fake, mode, ids, *, box=True, auto=False, cap=10, next_ok=True,
         takes_over=True):
    """Give a Reshape fake the real bracket helpers, begin the bracket the
    way Start does, and leave it open."""
    for n in _HELPERS:
        if n == "_layer_by_name":
            continue
        setattr(fake, n, types.MethodType(getattr(_K, n), fake))
    fake._ISO_OWNER_KINDS = _K._ISO_OWNER_KINDS
    fake.staged_kinds = lambda: frozenset({StagedOp.RESHAPE})
    fake._iso_bracket = None
    fake._navigator_widget = _Nav(ticked=box, auto=auto, next_ok=next_ok)
    if takes_over:
        # The next row's own isolation: one cell, shown by the navigator.
        fake._navigator_widget.on_next = (
            lambda: _live(fake, "single", (900,)))
    fake._is_auto_advance_enabled = lambda: auto
    fake._adv._params["pieces_cap"] = cap
    fake._multi_label_proxy_set = None
    fake._isolation_label_set = None
    fake._isolation_proxy_active = False
    fake._isolation_proxy_label_id = None
    fake._isolation_active = False
    fake._pre_isolation_colormap = None
    fake._colormap_authority = None
    fake._propagate_show_selected_label = lambda *a, **k: None
    fake._propagate_colormap_to_subviewers = lambda cm: None
    fake._remove_isolation_proxy = lambda restore_visibility=True: None
    _live(fake, mode, ids)
    fake._split_isolate_was_on = fake._iso_begin("reshape").mode != "off"
    fake._iso_suspend()
    fake.isolation_calls.clear()
    return fake


class TestReshapeAcceptPutsTheIsolationBack:
    def test_guided_accept_adds_the_pieces_and_drops_the_original(self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "multi", (7, 50))
        fake._enter_fragment_refine_mode()
        fake.isolation_calls.clear()
        fake._accept_split()
        assert fake.isolation_calls == [{50} | set(final.values())]
        assert fake._iso_bracket is None
        assert fake._navigator_widget.set_calls[-1] is True

    def test_an_unticked_piece_keeps_the_original_isolated(self, solved):
        fake, res, final = _guided_fake(solved)
        keep, _drop = sorted(final.values())
        fake._adv.set_checked_split_pieces([keep])
        fake._render_reshape_preview()
        _arm(fake, "multi", (7, 50))
        fake._enter_fragment_refine_mode()
        fake.isolation_calls.clear()
        fake._accept_split()
        assert fake.isolation_calls == [{7, 50, keep}]

    def test_above_the_cap_only_the_biggest_piece_stays(self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "single", (7,), cap=1)
        fake._enter_fragment_refine_mode()
        fake.isolation_calls.clear()
        fake._accept_split()
        vol = fake.sync_manager.shared_labels["curation_3d_x"]
        ids = fake.reshape_cb_calls[0][1]
        biggest = max(ids, key=lambda i: (int((vol == i).sum()), -i))
        assert fake.isolation_calls == [{biggest}]

    def test_nothing_isolated_before_means_nothing_isolated_after(self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "off", (7,), box=False)
        fake._enter_fragment_refine_mode()
        fake.isolation_calls.clear()
        fake._accept_split()
        assert fake.isolation_calls == []
        assert fake._navigator_widget.set_calls[-1] is False

    def test_accept_with_the_box_off_leaves_the_box_off(self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "multi", (7, 50), box=False)
        fake._enter_fragment_refine_mode()
        fake._accept_split()
        assert fake._navigator_widget.box is False

    def test_the_normal_branch_with_the_box_off_leaves_the_box_off(self):
        fake = _legacy_fake()
        _arm(fake, "multi", (7, 50), box=False)
        fake._accept_split()
        assert fake._navigator_widget.box is False

    def test_auto_advance_drops_without_restoring(self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "multi", (7, 50), auto=True)
        fake._enter_fragment_refine_mode()
        fake.isolation_calls.clear()
        fake._accept_split()
        assert fake.isolation_calls == []
        assert fake._iso_bracket is None
        assert fake._navigator_widget.advanced == 1

    def test_manual_accept_consumes_the_original(self):
        fake = _legacy_fake()
        _arm(fake, "multi", (7, 50))
        fake._enter_fragment_refine_mode()
        fake.isolation_calls.clear()
        fake._accept_split()
        assert fake.split_cb_calls and not fake.reshape_cb_calls
        assert fake.isolation_calls == [{50, 101, 102}]

    def test_the_normal_branch_ends_the_bracket_too(self):
        fake = _legacy_fake()
        _arm(fake, "multi", (7, 50))
        fake._accept_split()
        assert fake.split_cb_calls[0][1] == [101, 102]
        assert fake.isolation_calls == [{50, 101, 102}]
        assert fake._iso_bracket is None

    def test_the_normal_branch_with_auto_advance_does_not_restore(self):
        fake = _legacy_fake()
        _arm(fake, "multi", (7, 50), auto=True)
        fake._accept_split()
        assert fake.isolation_calls == []
        assert fake._iso_bracket is None
        # Start cleared the live flags: the skip reads the snapshot.
        assert fake.synced[-1][2].get("skip_3d_refresh") is True

    def test_the_normal_branch_with_no_next_row_restores_exactly_once(self):
        fake = _legacy_fake()
        _arm(fake, "multi", (7, 50), auto=True, next_ok=False)
        fake._accept_split()
        assert fake._navigator_widget.advanced == 1
        assert fake.isolation_calls == [{50, 101, 102}]
        assert fake._iso_bracket is None

    def test_the_fragment_branch_with_no_next_row_restores_exactly_once(
            self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "multi", (7, 50), auto=True, next_ok=False)
        fake._enter_fragment_refine_mode()
        fake.isolation_calls.clear()
        fake._accept_split()
        assert fake._navigator_widget.advanced == 1
        assert fake.isolation_calls == [{50} | set(final.values())]
        assert fake._iso_bracket is None

    def test_the_fragment_branch_with_a_next_row_makes_no_call(self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "multi", (7, 50), auto=True, next_ok=True)
        fake._enter_fragment_refine_mode()
        fake.isolation_calls.clear()
        fake._accept_split()
        assert fake.isolation_calls == []
        assert fake._iso_bracket is None

    def test_a_next_row_that_isolates_nothing_is_restored_not_dropped(self):
        fake = _legacy_fake()
        _arm(fake, "multi", (7, 50, 60), auto=True, takes_over=False)
        fake._accept_split()
        assert fake._navigator_widget.advanced == 1
        assert fake.isolation_calls == [{50, 60, 101, 102}]
        assert fake._iso_bracket is None

    def test_the_fragment_branch_restores_when_the_next_row_isolates_nothing(
            self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "multi", (7, 50, 60), auto=True, takes_over=False)
        record = fake._on_isolation_changed

        def _forced(ids, **k):
            # Refine Further force-isolates the one fragment: that isolation
            # is live, and stays live when the advance isolates nothing.
            record(ids, **k)
            ids = tuple(ids)
            if len(ids) == 1:
                _live(fake, "single", ids)

        fake._on_isolation_changed = _forced
        fake._enter_fragment_refine_mode()
        assert fake._isolation_active      # the forced isolation is live
        fake.isolation_calls.clear()
        fake._accept_split()
        assert fake.isolation_calls == [{50, 60} | set(final.values())]

    def test_the_selection_is_cleared_before_the_advance(self):
        fake = _legacy_fake()
        _arm(fake, "multi", (7, 50, 60), auto=True)
        fake._accept_split()
        assert fake._navigator_widget.log == ["clear", "next"]

    def test_the_fragment_branch_clears_the_selection_before_the_advance(
            self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "multi", (7, 50, 60), auto=True)
        fake._enter_fragment_refine_mode()
        fake._accept_split()
        assert fake._navigator_widget.log == ["clear", "next"]

    def test_the_bracket_survives_the_reset_it_runs_after(self):
        fake = _legacy_fake()
        _arm(fake, "single", (7,))
        seen = []
        fake._on_label_split_cb = lambda o, n, **kw: seen.append(
            fake._iso_bracket)
        fake._accept_split()
        assert seen == [None]               # taken before the callback ran
        assert fake.isolation_calls == [{101, 102}]


class TestReshapeCancelRestoresTheExactSnapshot:
    @pytest.mark.parametrize("branch", ["normal", "fragment"])
    def test_multi_comes_back_as_it_was(self, branch, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "multi", (7, 50), box=True)
        if branch == "fragment":
            fake._enter_fragment_refine_mode()
        fake.isolation_calls.clear()
        fake._reject_split()
        assert fake.isolation_calls == [{7, 50}]
        assert fake._navigator_widget.set_calls[-1] is True
        assert fake._iso_bracket is None

    @pytest.mark.parametrize("branch", ["normal", "fragment"])
    def test_a_2d_multi_isolation_keeps_its_cache_after_cancel(
            self, branch, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "off", (7,), box=True)
        fake._isolation_label_set = frozenset({7, 50})
        fake._iso_bracket = None
        fake._split_isolate_was_on = fake._iso_begin("reshape").mode != "off"
        fake._iso_suspend()
        assert fake._isolation_label_set is None

        def _oi(ids, **k):
            fake.isolation_calls.append(set(ids))
            fake._isolation_label_set = frozenset(ids)

        fake._on_isolation_changed = _oi
        if branch == "fragment":
            fake._enter_fragment_refine_mode()
        fake._reject_split()
        assert fake._isolation_label_set == frozenset({7, 50})

    def test_the_box_is_restored_exactly(self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "single", (7,), box=False)
        fake._reject_split()
        assert fake._navigator_widget.set_calls[-1] is False

    def test_off_makes_no_isolation_call(self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "off", (7,), box=False)
        fake._reject_split()
        assert fake.isolation_calls == []

    def test_the_old_forced_single_isolation_calls_are_gone(self):
        import inspect
        src = inspect.getsource(_K._reject_split)
        assert "_on_isolation_changed({label_id})" not in src

    def test_a_missing_volume_drops_without_restoring(self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "multi", (7, 50))
        fake.sync_manager.shared_labels.clear()
        fake._reject_split()
        assert fake.isolation_calls == []
        assert fake._iso_bracket is None

    def test_accept_found_no_result_restores_the_snapshot(self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "multi", (7, 50))
        fake._write_reshape_result = lambda *a, **k: None
        fake._restore_reshape_target = lambda why="": None
        fake._teardown_reshape_for_switch = types.MethodType(
            _K._teardown_reshape_for_switch, fake)
        for n, v in (("_abandon_reshape_worker", lambda: None),
                     ("_abandon_seed_find_worker", lambda: None),
                     ("_remove_all_temp_layers", lambda: None),
                     ("_guide_thread_cleanup", lambda: None),
                     ("_remove_guide_layer", lambda: None),
                     ("_teardown_refine_for_switch", lambda: None)):
            setattr(fake, n, v)
        from napari.layers import Labels

        main = Labels(np.zeros((2, 2, 2), np.int32))
        main.visible = False
        main.editable = False
        fake._layer_by_name = lambda layers, name: main
        fake._split_main_visible = True
        fake._accept_split()
        assert fake.isolation_calls == [{7, 50}]
        assert fake._iso_bracket is None
        assert main.visible is True and main.editable is True

    def test_the_teardown_alone_drops_the_bracket_and_gives_main_back(
            self, solved):
        from napari.layers import Labels

        fake, res, final = _guided_fake(solved)
        _arm(fake, "multi", (7, 50))
        main = Labels(np.zeros((2, 2, 2), np.int32))
        main.visible = False
        main.editable = False
        fake._layer_by_name = lambda layers, name: main
        fake._split_main_visible = True
        fake._teardown_reshape_for_switch = types.MethodType(
            _K._teardown_reshape_for_switch, fake)
        for n, v in (("_abandon_reshape_worker", lambda: None),
                     ("_abandon_seed_find_worker", lambda: None),
                     ("_remove_all_temp_layers", lambda: None),
                     ("_guide_thread_cleanup", lambda: None),
                     ("_remove_guide_layer", lambda: None),
                     ("_teardown_refine_for_switch", lambda: None)):
            setattr(fake, n, v)
        fake._restore_reshape_target = lambda why="": None
        fake._teardown_reshape_for_switch()
        assert fake.isolation_calls == []
        assert fake._iso_bracket is None
        assert main.visible is True and main.editable is True


class TestRealStart:
    """The real ``_start_split`` over a real (Qt-free) Labels layer."""

    def _fake(self, mode, ids, box):
        from napari.layers import Labels

        shared = np.zeros((4, 4, 4), np.int32)
        shared[1:3, 1:3, 1:3] = ids[0]
        main = Labels(shared.copy(), name="labels")
        main.selected_label = ids[0]
        layers = _Layers([main])
        layers.selection = SimpleNamespace(active=None)
        f = SimpleNamespace(
            viewer=SimpleNamespace(layers=layers),
            _active_labels_name="labels", main=main, shared=shared,
            _split_running=False, _smooth_running=False,
            _batch_staged=False, _batch_running=False,
            _interp_snapshot=None, _reshape_active=False,
            _split_roi=None, _split_snapshot=None, _split_label_id=None,
            _split_seed_ids=[], _split_painted_marker_ids=[],
            _split_isolate_was_on=False, _split_main_visible=True,
            _refine_layer=None, _refine_baseline=None,
            _detected_peaks=None, _auto_detect_active=False,
            _multi_label_proxy_set=None, _isolation_label_set=None,
            _isolation_active=False, _isolation_proxy_active=False,
            _isolation_proxy_label_id=None, _isolation_proxy_layer=None,
            _multi_label_proxy_layers={}, _pre_isolation_colormap=None,
            _colormap_authority=None, _iso_bracket=None,
            _navigator_widget=_Nav(ticked=box), calls=[],
        )
        for n in _HELPERS:
            if n != "_layer_by_name":
                setattr(f, n, types.MethodType(getattr(_K, n), f))
        f._start_split = types.MethodType(_K._start_split, f)
        f._ISO_OWNER_KINDS = _K._ISO_OWNER_KINDS
        f.staged_kinds = lambda: frozenset()
        f._get_advanced_features_widget = lambda: None
        f._ensure_shared_data_aligned = lambda name: shared
        f._get_selected_label = lambda: ids[0]
        f._reshape_search_padding = lambda: 0
        f._snapshot_label_roi = lambda data, label, _precomputed_mask=None, \
            margin=0: ((slice(0, 4),) * 3, data[(slice(0, 4),) * 3].copy())
        f._refresh_merge_button_state = lambda: None
        f._refresh_label_action_buttons_state = lambda: None
        f._clear_draw_snapshot = lambda: None
        f._layer_by_name = lambda layers, name: main
        f._create_refine_layer = lambda data, label, _precomputed_mask=None: \
            SimpleNamespace(data=np.where(data == label, label, 0),
                            selected_label=0, mode="pan_zoom")
        f._sync_all_views_after_operation = lambda **k: None
        f._next_label_id_after_accept = lambda: 300
        f._auto_enable_edt_for_split = lambda: None
        f._update_split_ui = lambda **k: None
        f._marker_finder_name = lambda: "Find"
        def _remove(restore_visibility=True):
            f._isolation_proxy_active = False
            f._isolation_proxy_label_id = None
            f._multi_label_proxy_set = None

        f._remove_isolation_proxy = _remove
        f._propagate_show_selected_label = lambda *a, **k: None
        f._propagate_colormap_to_subviewers = lambda cm: None
        f._on_isolation_changed = lambda ids, **k: f.calls.append(set(ids))
        _live(f, mode, ids)
        f._pre_isolation_colormap = None    # a real Labels has no "orig-cmap"
        return f

    def test_start_opens_a_bracket_and_takes_the_isolation_away(self):
        f = self._fake("multi", (7, 50), box=True)
        f._start_split()
        assert f._iso_bracket is not None
        assert f._iso_bracket.kind == "reshape"
        assert f._iso_bracket.snapshot.mode == "multi"
        assert f._iso_bracket.snapshot.ids == frozenset({7, 50})
        assert f._iso_bracket.snapshot.checkbox is True
        assert f._iso_bracket.target == 7
        assert f._split_isolate_was_on is True
        assert not f._isolation_active
        assert not f._isolation_proxy_active
        assert f._multi_label_proxy_set is None
        assert f._isolation_label_set is None
        assert f.main.visible is False and f.main.editable is False

    def test_start_with_nothing_isolated_records_off(self):
        f = self._fake("off", (7,), box=False)
        f._start_split()
        assert f._iso_bracket.snapshot.mode == "off"
        assert f._split_isolate_was_on is False


def _legacy_fake():
    """A hand-painted split with two pieces, ready for the normal Accept."""
    shape = (6, 8, 8)
    labels = np.zeros(shape, np.int32)
    labels[1:5, 1:7, 1:7] = 7
    roi = (slice(0, 6), slice(0, 8), slice(0, 8))
    snapshot = labels[roi].copy()
    vol = labels.copy()
    vol[vol == 7] = 0
    refine_data = np.zeros(shape, np.int32)
    refine_data[1:5, 1:4, 1:7] = 101
    refine_data[1:5, 4:6, 1:7] = 102
    fake = FakeAnnotator(vol, roi, 7, refine=FakeLayer(refine_data, "Refine"),
                         snapshot=snapshot)
    fake._reshape_active = False
    fake._reshape_result = None
    fake._split_previewed = True
    fake._split_seed_ids = [101, 102]
    bind(fake, "_enter_fragment_refine_mode", "_accept_split", "_reject_split")
    return fake


# --------------------------------------------------------------------------- #
# teardown
# --------------------------------------------------------------------------- #
def _teardown_env(kind, mode="multi"):
    ann = _env(mode, ids=(1, 2), staged=(StagedOp.RESHAPE,))
    ann._iso_begin(kind)
    ann.calls.clear()
    return ann


class TestTeardownDropsTheBracket:
    def test_close_event_drops_after_discard_and_before_the_discard_runs(self):
        ann = _teardown_env("reshape")
        order = []
        ann._has_uncommitted_work = lambda **k: (True, "work")
        ann._confirm_discard_dialog = lambda reason: True
        ann._discard_uncommitted_work = lambda: order.append(
            ("discard", ann._iso_bracket))
        ann._close_cleanup = lambda: order.append(("cleanup",))
        ann._close_subviewers = lambda: None
        ann._navigator_widget = None
        event = SimpleNamespace(ignore=lambda: order.append(("ignore",)))
        try:
            _K.closeEvent(ann, event)
        except Exception:
            pass                    # the Qt tail of closeEvent is not under test
        assert order[:2] == [("discard", None), ("cleanup",)]
        assert _ois(ann) == []

    def test_go_back_keeps_the_bracket(self):
        ann = _teardown_env("reshape")
        ann._has_uncommitted_work = lambda **k: (True, "work")
        ann._confirm_discard_dialog = lambda reason: False
        ignored = []
        _K.closeEvent(ann, SimpleNamespace(ignore=lambda: ignored.append(1)))
        assert ignored == [1] and ann._iso_bracket is not None

    def test_close_event_without_work_drops_it_before_cleanup(self):
        ann = _teardown_env("interp")
        seen = []
        ann._has_uncommitted_work = lambda **k: (False, "")
        ann._close_cleanup = lambda: seen.append(ann._iso_bracket)
        ann._close_subviewers = lambda: None
        try:
            _K.closeEvent(ann, SimpleNamespace(ignore=lambda: None))
        except Exception:
            pass
        assert seen == [None]

    def test_close_cleanup_drops_without_relying_on_close_event(self):
        ann = _teardown_env("reshape")
        seen = []
        ann._reject_split = lambda: seen.append(ann._iso_bracket)
        ann._split_running = True
        ann._smooth_running = False
        ann._add_running = False
        ann._interp_snapshot = None
        with pytest.raises(Exception):
            _K._close_cleanup(ann)      # the fake is too bare to finish
        assert seen == [None] or ann._iso_bracket is None
        assert _ois(ann) == []

    def test_load_selected_layers_drops_an_interp_bracket(self):
        ann = _teardown_env("interp")
        ann._smooth_running = False
        ann._split_running = False
        ann._add_running = False
        ann._teardown_refine_for_switch = lambda: None
        with pytest.raises(Exception):
            _K._load_selected_layers(ann)
        assert ann._iso_bracket is None
        assert _ois(ann) == []


class TestPiecesCapKeepsTheBiggest:
    """Ids with no voxels left must not use up the cap."""

    @staticmethod
    def _stats(fake, counts):
        fake._live_piece_stats = lambda ids, roi: [
            {"id": i, "voxels": counts[i], "centroid_zyx": (0, 0, 0)}
            for i in ids if counts.get(i, 0) > 0]

    def test_the_normal_branch_keeps_the_biggest_piece(self):
        fake = _legacy_fake()
        _arm(fake, "single", (7,), cap=1)
        fake._split_seed_ids = [101, 102]
        self._stats(fake, {101: 0, 102: 12763})
        fake._accept_split()
        assert fake.isolation_calls == [{102}]

    def test_the_fragment_branch_keeps_the_biggest_piece(self, solved):
        fake, res, final = _guided_fake(solved)
        _arm(fake, "single", (7,), cap=1)
        fake._enter_fragment_refine_mode()
        ids = sorted(final.values())
        counts = {i: 0 for i in ids}
        counts[ids[-1]] = 500
        self._stats(fake, counts)
        fake.isolation_calls.clear()
        fake._accept_split()
        assert fake.isolation_calls == [{ids[-1]}]

    def test_an_unreadable_volume_falls_back_to_the_given_order(self):
        fake = _legacy_fake()
        _arm(fake, "single", (7,), cap=1)

        def boom(ids, roi):
            raise RuntimeError("no volume")
        fake._live_piece_stats = boom
        fake._accept_split()
        assert fake.isolation_calls == [{101}]


class TestPiecesBySizeHelper:
    @staticmethod
    def _ann(rows):
        calls = []

        def measure(ids, roi):
            calls.append(list(ids))
            return rows

        ann = SimpleNamespace(_live_piece_stats=measure)
        return ann, calls

    def test_every_id_empty_gives_an_empty_list(self, capsys):
        ann, _ = self._ann([])
        ann.sync_manager = SimpleNamespace(
            shared_labels={"labels": object()})
        ann._active_labels_name = "labels"
        assert _K._iso_pieces_by_size(ann, [4, 5], (slice(0, 1),)) == []
        assert "every created id is empty" in capsys.readouterr().out

    def test_given_stats_are_not_measured_again(self):
        ann, calls = self._ann([])
        rows = [{"id": 4, "voxels": 3}, {"id": 5, "voxels": 9}]
        got = _K._iso_pieces_by_size(ann, [4, 5], None, stats=rows)
        assert got == [5, 4]
        assert calls == []


class TestDropReleasesTheColormapOwnership:
    @staticmethod
    def _owned(mode="multi"):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapAuthority, ColormapOwner)

        auth = ColormapAuthority()
        auth.claim(ColormapOwner.ANNOTATOR_ISOLATION)
        ann = _env(mode, ids=(1, 2), staged=(StagedOp.RESHAPE,))
        ann._colormap_authority = auth
        ann._iso_begin("reshape")
        ann._iso_suspend()
        assert auth.current_owner == ColormapOwner.ANNOTATOR_ISOLATION
        return ann, auth

    def test_a_drop_with_nothing_isolated_gives_the_colormap_back(self):
        ann, auth = self._owned()
        ann._iso_drop("test")
        assert auth.current_owner is None

    def test_a_drop_while_a_next_row_is_isolated_keeps_it(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapOwner)

        ann, auth = self._owned()
        _live(ann, "single", (9,))
        ann._iso_drop("auto-advance")
        assert auth.current_owner == ColormapOwner.ANNOTATOR_ISOLATION

    def test_closing_mid_reshape_gives_the_colormap_back(self):
        ann, auth = self._owned()
        ann._reject_split = lambda: None
        ann._split_running = True
        ann._smooth_running = False
        ann._add_running = False
        ann._interp_snapshot = None
        for name in ("_disconnect_viewer_events", "_disconnect_label_handlers",
                     "_restore_main_viewer_arrow_keys", "_reset_add_state",
                     "_abandon_seed_find_worker",
                     "_teardown_reshape_for_switch"):
            setattr(ann, name, lambda *a, **k: None)
        ann._on_isolation_cleared_cb = None
        ann._navigator_dock = None
        _K._close_cleanup(ann)
        assert auth.current_owner is None

    def test_closing_mid_split_shows_main_again(self):
        ann, _auth = self._owned()
        ann.main.visible = False
        ann._split_main_visible = False     # recorded while the proxy hid it
        ann._reject_split = lambda: None
        ann._split_running = True
        ann._smooth_running = False
        ann._add_running = False
        ann._interp_snapshot = None
        for name in ("_disconnect_viewer_events", "_disconnect_label_handlers",
                     "_restore_main_viewer_arrow_keys", "_reset_add_state",
                     "_abandon_seed_find_worker",
                     "_teardown_reshape_for_switch"):
            setattr(ann, name, lambda *a, **k: None)
        ann._on_isolation_cleared_cb = None
        ann._navigator_dock = None
        _K._close_cleanup(ann)
        assert ann.main.visible is True

    def test_ending_without_a_restore_gives_it_back_too(self):
        ann, auth = self._owned()
        ann._iso_end("reshape", restore=False)
        assert auth.current_owner is None


# --------------------------------------------------------------------------- #
# the removed Refine Further flag
# --------------------------------------------------------------------------- #
def test_the_old_refine_further_restore_is_gone():
    assert not hasattr(Curation3DAnnotator, "_end_interp_refine_isolation")
    src = Path(
        __import__("napari_epicycle.annotator.curation_3d_widget",
                   fromlist=["x"]).__file__
    ).read_text(encoding="utf-8")
    assert "_interp_refine_iso_was_on" not in src
    assert "_end_interp_refine_isolation" not in src


# --------------------------------------------------------------------------- #
# the Reshape tab's knob
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def _qapp():
    from qtpy.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def test_the_spin_box_and_its_round_trip(_qapp):
    from napari_epicycle.annotator.advanced_features_widget import (
        AdvancedFeaturesWidget,
    )

    w = AdvancedFeaturesWidget(mode="3d_curation")
    try:
        sb = w.split_pieces_cap_sb
        assert (sb.minimum(), sb.maximum(), sb.value()) == (1, 50, 10)
        assert w.get_split_params()["pieces_cap"] == 10
        sb.setValue(4)
        assert w.get_split_params()["pieces_cap"] == 4
        ann = SimpleNamespace(_get_advanced_features_widget=lambda: w)
        assert _K._reshape_pieces_cap(ann) == 4
    finally:
        w.deleteLater()


class TestRestoreAfterAdvanceIn2D:
    """In 2D nothing lives in the 3D proxies: the mark must still tell one
    isolated cell from another."""

    def _run(self, change_selection):
        from napari_epicycle.annotator.curation_3d_widget import (
            _iso_mark, _restore_after_advance)
        ann = _env("single2d", ids=(3,))
        before = _iso_mark(ann)
        if change_selection:
            ann.main.selected_label = 9
        return _restore_after_advance(ann, True, before)

    def test_a_different_isolated_cell_is_a_drop(self):
        assert self._run(True) is False

    def test_the_same_isolated_cell_restores(self):
        assert self._run(False) is True

    def test_a_different_2d_multi_set_is_a_drop(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            _iso_mark, _restore_after_advance)
        ann = _env("single2d", ids=(3,))
        ann._isolation_label_set = frozenset({3, 4})
        before = _iso_mark(ann)
        ann._isolation_label_set = frozenset({5, 6})
        assert _restore_after_advance(ann, True, before) is False


# --------------------------------------------------------------------------- #
# the teardown runs inside one batch of layer removals
# --------------------------------------------------------------------------- #
class _BatchRecorder:
    def __init__(self):
        self.events = []

    def __call__(self, *qt_viewers):
        rec = self

        class _Cm:
            def __enter__(self_):
                rec.events.append("enter")

            def __exit__(self_, *exc):
                rec.events.append("exit")
                return False
        return _Cm()


def test_reshape_accept_teardown_is_one_batch(monkeypatch, solved):
    from napari_epicycle.annotator import curation_3d_widget as mod
    from .test_reshape_fragment_refine_everywhere import _guided_fake
    rec = _BatchRecorder()
    monkeypatch.setattr(mod, "batched_layer_removals", rec)
    fake, _res, _final = _guided_fake(solved)
    order = []
    fake._remove_edt_overlay = lambda: order.append(("edt", list(rec.events)))
    fake._remove_refine_layer = lambda: order.append(("refine", list(rec.events)))
    fake._accept_split()
    assert rec.events[:1] == ["enter"] and rec.events.count("enter") >= 1
    assert [o[0] for o in order] == ["edt", "refine"]
    assert all(o[1] == ["enter"] for o in order)
    assert rec.events[-1] == "exit"


class _LoggedLayers(_Layers):
    """A layer list that records each removal into a shared event list."""

    def __init__(self, items, events):
        super().__init__(items)
        self._events = events

    def remove(self, layer):
        self._events.append("remove")
        super().remove(layer)


def test_multi_proxy_teardown_enters_a_batch(monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as mod
    rec = _BatchRecorder()
    monkeypatch.setattr(mod, "batched_layer_removals", rec)
    ann = _env("multi", ids=(1, 2))
    ann.viewer.layers = _LoggedLayers([ann.main], rec.events)
    ann._isolation_proxy_layer = None
    ann._remove_isolation_proxy = types.MethodType(
        _K._remove_isolation_proxy, ann)
    proxy = SimpleNamespace(name="p1", visible=True)
    ann._multi_label_proxy_layers = {1: proxy}
    ann.viewer.layers.append(proxy)
    ann._remove_isolation_proxy(restore_visibility=False)
    assert "remove" in rec.events
    first, last = rec.events.index("enter"), len(rec.events) - 1 - rec.events[::-1].index("exit")
    assert all(first < k < last for k, e in enumerate(rec.events) if e == "remove")


def _temp_env(events):
    main = SimpleNamespace(name="labels", visible=True)
    temp = SimpleNamespace(name="markers", visible=True)
    ann = SimpleNamespace(
        viewer=SimpleNamespace(
            layers=_LoggedLayers([main, temp], events)),
        _active_labels_name="labels", _temp_guide_layers={"markers": temp},
        _reshape_guide_layer=None, sync_manager=None)
    ann.viewer.layers.selection = SimpleNamespace(active=main)
    ann._layer_by_name = lambda layers, name: main
    ann._remove_temp_layer = types.MethodType(_K._remove_temp_layer, ann)
    ann._remove_all_temp_layers = types.MethodType(
        _K._remove_all_temp_layers, ann)
    return ann


def _assert_removed_inside(events):
    assert "remove" in events
    assert events[0] == "enter" and events[-1] == "exit"
    assert events.index("remove") > events.index("enter")


def test_remove_temp_layer_is_inside_a_batch(monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as mod
    rec = _BatchRecorder()
    monkeypatch.setattr(mod, "batched_layer_removals", rec)
    ann = _temp_env(rec.events)
    ann._remove_temp_layer("markers")
    _assert_removed_inside(rec.events)


def test_remove_all_temp_layers_is_one_outer_batch(monkeypatch):
    from napari_epicycle.annotator import curation_3d_widget as mod
    rec = _BatchRecorder()
    monkeypatch.setattr(mod, "batched_layer_removals", rec)
    ann = _temp_env(rec.events)
    ann._remove_all_temp_layers()
    _assert_removed_inside(rec.events)
    assert rec.events[0] == "enter" and rec.events[1] == "enter"
    assert rec.events[-1] == "exit" and rec.events[-2] == "exit"
