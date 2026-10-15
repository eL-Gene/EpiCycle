"""Regression tests for the QC-plot lasso ↔ 3D-annotator isolation sync.

Background: reproduced live on real data (identical on napari 0.6.2 and 0.8.0):

1. ``_set_selected_label_all`` hides the main labels layer for perf;
   ``subviewers.py::sync_layer_visibility`` cascades that hide to the
   XY/XZ/YZ label layers, and when a proxy is active no re-show is
   scheduled → the subviewers stay dark.
2. The perf re-show ``QTimer`` scheduled by the plain-3D branch fires AFTER
   ``_update_isolation_proxy`` hid the main layer → main visible under the
   proxy.
3. ``_apply_viewer_isolation_adapter`` → ``_isolate_labels_in_viewer``
   silently returns while the annotator owns isolation, so a plot lasso
   updates the plot + navigator but never the viewer.

Design constraint (verified live, audit A / H1): the main-layer hide must
stay UNBLOCKED — napari's vispy node only learns about a hide through
``events.visible``.  So the cascade DOES fire; the fix restores the subviewer
label layers right after it in the proxy case, and makes the deferred re-show
token-guarded.

All tests are Qt-free (real napari Labels layers for ``isinstance`` fidelity,
SimpleNamespace annotator, patched ``QTimer.singleShot``).
"""
from __future__ import annotations

import types
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

class _FakeLayerList(list):
    def __contains__(self, item):
        if isinstance(item, str):
            return any(getattr(l, "name", None) == item for l in self)
        return super().__contains__(item)

    def __getitem__(self, key):
        if isinstance(key, str):
            for l in self:
                if getattr(l, "name", None) == key:
                    return l
            raise KeyError(key)
        return super().__getitem__(key)

    def remove(self, layer):
        list.remove(self, layer)


def _make_env(*, ndisplay=3, label_id=5, shape=(20, 20, 20)):
    """Real napari Labels main layer + three real (tiny) subviewer Labels
    layers behind a fake sync manager, with the REAL visibility cascade
    semantics of ``subviewers.py::sync_layer_visibility`` wired on the main
    layer's ``events.visible``.  Returns (ann, main_layer, subs, cascade_log)."""
    from napari.layers import Labels as _Labels
    from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator

    data = np.zeros(shape, dtype=np.int32)
    data[5:10, 5:10, 5:10] = label_id
    data[12:15, 12:15, 12:15] = 99
    main = _Labels(data, name="labels")
    main.selected_label = 99

    subs = {}
    for suffix in ("XY", "XZ", "YZ"):
        subs[suffix] = _Labels(np.zeros((4, 4, 4), dtype=np.int32), name=f"labels_{suffix}")

    sync = types.SimpleNamespace()
    sync._base_label_name = (
        lambda name: name.replace("_XY", "").replace("_XZ", "").replace("_YZ", "")
    )
    sync.viewer_model_xy = types.SimpleNamespace(layers=[subs["XY"]])
    sync.viewer_model_xz = types.SimpleNamespace(layers=[subs["XZ"]])
    sync.viewer_model_yz = types.SimpleNamespace(layers=[subs["YZ"]])

    viewer = types.SimpleNamespace(
        layers=_FakeLayerList([main]),
        dims=types.SimpleNamespace(ndisplay=ndisplay),
    )
    ann = types.SimpleNamespace()
    ann.viewer = viewer
    ann.sync_manager = sync
    ann._active_labels_name = "labels"
    ann._isolation_proxy_layer = None
    ann._isolation_proxy_active = False
    ann._isolation_proxy_label_id = None
    ann._isolation_active = False
    for name in (
        "_layer_by_name",
        "_set_selected_label_all",
        "_update_isolation_proxy",
        "_remove_isolation_proxy",
        "_schedule_main_perf_reshow",
        "_cancel_main_perf_reshow",
        "_visible_subviewer_label_layers",
    ):
        setattr(ann, name, types.MethodType(getattr(Curation3DAnnotator, name), ann))

    # The cascade, verbatim in spirit: subviewer label layers mirror the
    # main layer's visibility on every events.visible.
    cascade_log: list = []

    def _cascade(_event):
        cascade_log.append(main.visible)
        for lyr in subs.values():
            lyr.visible = bool(main.visible)

    main.events.visible.connect(_cascade)
    return ann, main, subs, cascade_log


def _all_visible(subs):
    return all(l.visible for l in subs.values())


@pytest.fixture
def captured_timers(monkeypatch):
    """Replace ``QTimer.singleShot`` with a capture list so the deferred
    re-show can be fired by the test at the moment of its choosing."""
    from qtpy.QtCore import QTimer

    calls: list = []
    monkeypatch.setattr(QTimer, "singleShot", staticmethod(lambda ms, cb: calls.append(cb)))
    return calls


# ---------------------------------------------------------------------------
# 1: subviewers must not stay dark
# ---------------------------------------------------------------------------

class TestSelectedLabelPerfHide:
    def test_hide_stays_unblocked_so_vispy_and_cascade_see_it(self, captured_timers):
        """Audit A / H1: the hide must be a real, event-emitting hide."""
        ann, main, subs, cascade = _make_env(ndisplay=3)

        ann._set_selected_label_all(5)

        assert main.selected_label == 5
        assert main.visible is False
        assert cascade == [False], "the cascade (and napari's vispy node) must see the hide"
        assert not _all_visible(subs), "subviewers follow the hide (pre-existing cascade)"
        assert len(captured_timers) == 1

        captured_timers[0]()  # the event loop tick → deferred re-show

        assert main.visible is True
        assert cascade == [False, True]
        assert _all_visible(subs), "the re-show cascade switches them back on"

    def test_proxy_active_keeps_main_hidden_but_restores_subviewers(self, captured_timers):
        """THE bug: with a proxy active no re-show follows, so the cascade
        used to leave the XY/XZ/YZ label layers off for good."""
        ann, main, subs, cascade = _make_env(ndisplay=3)
        ann._update_isolation_proxy(5, main)          # proxy mode: main hidden by design
        assert ann._isolation_proxy_active is True
        assert main.visible is False
        for lyr in subs.values():                     # what _propagate_show_selected_label does
            lyr.visible = True
        # simulate the controller's re-show race having left main visible
        main.visible = True
        cascade.clear()

        ann._set_selected_label_all(99)               # the lasso's first-id select

        assert main.selected_label == 99
        assert main.visible is False, "proxy mode: main stays hidden"
        assert cascade == [False], "the hide itself is still a real hide"
        assert _all_visible(subs), (
            "subviewer label layers must be switched back on immediately — "
            "nothing else will (this is 'selected cells not rendered in the "
            "multiview annotator')"
        )
        assert captured_timers == [], "no re-show may be scheduled under a proxy"

    def test_proxy_active_does_not_resurrect_a_user_hidden_subviewer(self, captured_timers):
        ann, main, subs, cascade = _make_env(ndisplay=3)
        ann._update_isolation_proxy(5, main)
        main.visible = True
        subs["XZ"].visible = False                    # user hid it with the 's' key
        subs["XY"].visible = subs["YZ"].visible = True

        ann._set_selected_label_all(99)

        assert subs["XY"].visible is True and subs["YZ"].visible is True
        assert subs["XZ"].visible is False, "only layers that WERE visible are restored"

    def test_proxy_active_and_main_already_hidden_is_a_noop_for_visibility(self, captured_timers):
        ann, main, subs, cascade = _make_env(ndisplay=3)
        ann._update_isolation_proxy(5, main)
        for lyr in subs.values():
            lyr.visible = True
        cascade.clear()

        ann._set_selected_label_all(99)

        assert main.visible is False
        assert cascade == [], "no visibility event when main was already hidden"
        assert _all_visible(subs)
        assert captured_timers == []

    def test_2d_path_unchanged(self, captured_timers):
        ann, main, subs, cascade = _make_env(ndisplay=2)
        ann._set_selected_label_all(5)
        assert main.selected_label == 5
        assert main.visible is True
        assert cascade == []
        assert captured_timers == []


# ---------------------------------------------------------------------------
# 2: a stale perf re-show must not fire after a proxy took over
# ---------------------------------------------------------------------------

class TestStalePerfReshowCancelled:
    def test_proxy_creation_cancels_pending_reshow(self, captured_timers):
        ann, main, subs, cascade = _make_env(ndisplay=3)

        ann._set_selected_label_all(5)               # plain-3D: hide + schedule re-show
        assert len(captured_timers) == 1
        assert main.visible is False

        ann._update_isolation_proxy(5, main)         # proxy takes over in the same tick
        assert ann._isolation_proxy_active is True

        captured_timers[0]()                         # the stale timer fires now

        assert main.visible is False, (
            "a perf re-show scheduled BEFORE the proxy was created must not "
            "un-hide the main layer underneath the proxy"
        )

    def test_reshow_still_fires_when_no_proxy_took_over(self, captured_timers):
        ann, main, subs, cascade = _make_env(ndisplay=3)
        ann._set_selected_label_all(5)
        captured_timers[0]()
        assert main.visible is True
        assert _all_visible(subs)

    def test_second_click_before_the_tick_reshows_once(self, captured_timers):
        ann, main, subs, cascade = _make_env(ndisplay=3)
        ann._set_selected_label_all(5)
        ann._set_selected_label_all(99)              # second nav click before the tick
        assert main.selected_label == 99
        assert len(captured_timers) == 1
        captured_timers[0]()
        assert main.visible is True

    def test_explicit_cancel_blocks_pending_reshow(self, captured_timers):
        """``_cancel_main_perf_reshow`` is what ``_close_cleanup`` calls."""
        ann, main, subs, cascade = _make_env(ndisplay=3)
        ann._set_selected_label_all(5)
        ann._cancel_main_perf_reshow()
        captured_timers[0]()
        assert main.visible is False

    def test_reshow_after_proxy_removed_works(self, captured_timers):
        """The token is per-schedule: once the proxy is gone, a NEW click
        schedules a fresh re-show that does fire."""
        ann, main, subs, cascade = _make_env(ndisplay=3)
        ann._update_isolation_proxy(5, main)
        ann._remove_isolation_proxy()                # default restores main visibility
        assert main.visible is True
        ann._set_selected_label_all(99)
        assert len(captured_timers) == 1
        captured_timers[0]()
        assert main.visible is True


# ---------------------------------------------------------------------------
# 3: controller routes a plot lasso through the annotator when it owns
# ---------------------------------------------------------------------------

def _make_controller(*, owner=None):
    from napari_epicycle.fusion_reconstruction.colormap_authority import (
        ColormapAuthority,
    )
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )

    ctrl = MagicMock()
    ctrl._colormap_authority = ColormapAuthority()
    if owner is not None:
        assert ctrl._colormap_authority.claim(owner)
    ctrl._isolated_label_ids = set()
    ctrl._isolated_df_indices = []
    # MagicMock attributes are truthy — pin the guards the router reads.
    ctrl._patch_switch_loading = False
    ctrl._patch_load_in_flight = False
    ctrl._qc_view_patch_name = "p.tif"
    for name in ("_apply_viewer_isolation_adapter", "_route_isolation_to_annotator"):
        setattr(ctrl, name, types.MethodType(
            getattr(FusionReconstructionController, name), ctrl))
    return ctrl


def _sel(ids, mode="isolate", source="plot", patch_key="p.tif"):
    from napari_epicycle.fusion_reconstruction.curation_selection_sync import (
        SelectionRef,
    )
    return SelectionRef(
        scope="single_patch", patch_key=patch_key, label_ids=list(ids),
        primary_label_id=(ids[0] if ids else None), df_indices=[],
        source=source, mode=mode,
    )


class TestRoutingGuards:
    """Audit-B follow-ups: the routed path must carry the same guards as
    the navigator adapter, never re-apply a navigator-sourced request, and
    never leave the controller's own isolation bookkeeping stale."""

    def _owned(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapOwner,
        )
        ctrl = _make_controller(owner=ColormapOwner.ANNOTATOR_ISOLATION)
        ann = _fake_ann()
        ctrl._get_curation_annotator_widget.return_value = ann
        return ctrl, ann

    def test_navigator_sourced_isolate_is_not_rerouted(self):
        """The annotator already applied it (ctrl-click / Render Selected);
        re-routing would double-apply and defeat the ctrl-click perf gate."""
        ctrl, ann = self._owned()
        ctrl._apply_viewer_isolation_adapter(_sel([1, 2], source="navigator"))
        ann._on_isolation_changed.assert_not_called()
        ctrl._isolate_labels_in_viewer.assert_called_once()  # old path (claim fails → no-op)

    def test_dropped_during_patch_switch(self):
        ctrl, ann = self._owned()
        ctrl._patch_load_in_flight = True
        ctrl._apply_viewer_isolation_adapter(_sel([1, 2]))
        ann._on_isolation_changed.assert_not_called()
        ctrl._isolate_labels_in_viewer.assert_not_called()

    def test_dropped_on_patch_mismatch(self):
        ctrl, ann = self._owned()
        ctrl._apply_viewer_isolation_adapter(_sel([1, 2], patch_key="other.tif"))
        ann._on_isolation_changed.assert_not_called()
        ctrl._isolate_labels_in_viewer.assert_not_called()

    def test_dock_gone_releases_stale_ownership_then_controller_path_runs(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapOwner,
        )
        ctrl = _make_controller(owner=ColormapOwner.ANNOTATOR_ISOLATION)
        ctrl._get_curation_annotator_widget.return_value = None
        ctrl._apply_viewer_isolation_adapter(_sel([1, 2]))
        assert ctrl._colormap_authority.current_owner is None, (
            "stale annotator ownership must be released so the controller "
            "path can actually claim (was a silent no-op)"
        )
        ctrl._isolate_labels_in_viewer.assert_called_once()

    def test_raising_op_predicate_refuses(self):
        ctrl, ann = self._owned()
        ann._refinement_op_active.side_effect = RuntimeError("boom")
        ctrl._apply_viewer_isolation_adapter(_sel([1, 2]))
        ann._on_isolation_changed.assert_not_called()
        ctrl._set_plot_status.assert_called()

    def test_refused_clear_still_resets_controller_isolation_state(self):
        ctrl, ann = self._owned()
        ctrl._isolated_label_ids = {5, 6}          # left over from before the handover
        ann._refinement_op_active.return_value = True
        ctrl._apply_viewer_isolation_adapter(_sel([], mode="focus"))
        assert ctrl._isolated_label_ids == set()
        ann._on_isolation_changed.assert_not_called()

    def test_routed_request_resets_controller_isolation_state(self):
        ctrl, ann = self._owned()
        ctrl._isolated_label_ids = {5, 6}
        ctrl._apply_viewer_isolation_adapter(_sel([1, 2]))
        assert ctrl._isolated_label_ids == set()
        ann._on_isolation_changed.assert_called_once()

    def test_handler_exception_is_handled_not_fallen_back(self):
        ctrl, ann = self._owned()
        ann._on_isolation_changed.side_effect = RuntimeError("mid-apply")
        ctrl._apply_viewer_isolation_adapter(_sel([1, 2]))
        ctrl._isolate_labels_in_viewer.assert_not_called()
        ctrl._set_plot_status.assert_called()
        assert ann._isolation_bridge_dispatched is False


def _fake_ann(*, op_active=False):
    ann = MagicMock()
    ann._refinement_op_active.return_value = op_active
    return ann


class TestPlotLassoWhileAnnotatorOwns:
    def test_lasso_is_routed_to_annotator_isolation(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapOwner,
        )
        ctrl = _make_controller(owner=ColormapOwner.ANNOTATOR_ISOLATION)
        ann = _fake_ann()
        ctrl._get_curation_annotator_widget.return_value = ann

        ctrl._apply_viewer_isolation_adapter(_sel([1, 2, 3]))

        ann._on_isolation_changed.assert_called_once()
        (ids,), _ = ann._on_isolation_changed.call_args
        assert set(ids) == {1, 2, 3}
        ctrl._isolate_labels_in_viewer.assert_not_called()

    def test_clear_is_routed_to_annotator_isolation(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapOwner,
        )
        ctrl = _make_controller(owner=ColormapOwner.ANNOTATOR_ISOLATION)
        ann = _fake_ann()
        ctrl._get_curation_annotator_widget.return_value = ann

        ctrl._apply_viewer_isolation_adapter(_sel([], mode="focus"))

        ann._on_isolation_changed.assert_called_once()
        (ids,), _ = ann._on_isolation_changed.call_args
        assert set(ids) == set()
        ctrl._clear_label_isolation.assert_not_called()

    def test_controller_path_unchanged_when_it_owns(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapOwner,
        )
        ctrl = _make_controller(owner=ColormapOwner.CONTROLLER_QC)
        ann = _fake_ann()
        ctrl._get_curation_annotator_widget.return_value = ann

        ctrl._apply_viewer_isolation_adapter(_sel([1, 2, 3]))

        ctrl._isolate_labels_in_viewer.assert_called_once()
        ann._on_isolation_changed.assert_not_called()

    def test_focus_mode_never_routes(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapOwner,
        )
        ctrl = _make_controller(owner=ColormapOwner.ANNOTATOR_ISOLATION)
        ann = _fake_ann()
        ctrl._get_curation_annotator_widget.return_value = ann

        ctrl._apply_viewer_isolation_adapter(_sel([7], mode="focus"))

        ann._on_isolation_changed.assert_not_called()
        ctrl._isolate_labels_in_viewer.assert_not_called()

    def test_refused_while_refinement_op_is_staged(self):
        """Same gate as the navigator's three entry points (FM3/FM7): a lasso
        during a split / smooth / interp preview must not touch isolation —
        neither through the annotator nor through the controller path."""
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapOwner,
        )
        ctrl = _make_controller(owner=ColormapOwner.ANNOTATOR_ISOLATION)
        ann = _fake_ann(op_active=True)
        ctrl._get_curation_annotator_widget.return_value = ann

        ctrl._apply_viewer_isolation_adapter(_sel([1, 2, 3]))

        ann._on_isolation_changed.assert_not_called()
        ctrl._isolate_labels_in_viewer.assert_not_called()
        ctrl._set_plot_status.assert_called()  # user gets told why

    def test_bridge_echo_flag_is_reset_after_routing(self):
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapOwner,
        )
        ctrl = _make_controller(owner=ColormapOwner.ANNOTATOR_ISOLATION)
        ann = _fake_ann()
        ctrl._get_curation_annotator_widget.return_value = ann

        ctrl._apply_viewer_isolation_adapter(_sel([], mode="focus"))  # clear

        ann._on_isolation_changed.assert_called_once()
        assert ann._isolation_bridge_dispatched is False

    def test_falls_back_to_controller_when_annotator_gone(self):
        """Authority says annotator owns but the dock is closed (stale
        ownership) — the old behaviour (silent no-op) is NOT acceptable;
        the controller must do its own isolation."""
        from napari_epicycle.fusion_reconstruction.colormap_authority import (
            ColormapOwner,
        )
        ctrl = _make_controller(owner=ColormapOwner.ANNOTATOR_ISOLATION)
        ctrl._get_curation_annotator_widget.return_value = None

        ctrl._apply_viewer_isolation_adapter(_sel([1, 2]))

        ctrl._isolate_labels_in_viewer.assert_called_once()


# ---------------------------------------------------------------------------
# 3: every re-show must run the napari>=0.7 render heal AFTER the layer is
#    visible again (a colormap node rebuilt against the hidden-layer
#    placeholder slice is only repairable once the real slice is back)
# ---------------------------------------------------------------------------

class TestRenderHealAfterReshow:
    def _attach_recorder(self, ann, main):
        calls = []

        def _heal(layer, *, tag=""):
            calls.append((layer, tag, bool(getattr(main, "visible", None))))
            return 0

        ann._heal_label_render = _heal
        return calls

    def test_perf_reshow_heals_after_main_is_visible(self, captured_timers):
        ann, main, subs, cascade = _make_env(ndisplay=3)
        calls = self._attach_recorder(ann, main)
        ann._set_selected_label_all(5)
        assert calls == [], "no heal while the layer is still hidden"
        captured_timers[0]()
        assert main.visible is True
        assert calls and calls[-1][0] is main and calls[-1][1] == "perf-reshow"
        assert calls[-1][2] is True, "heal must run AFTER visible=True"

    def test_proxy_subviewer_restore_heals(self, captured_timers):
        ann, main, subs, cascade = _make_env(ndisplay=3)
        ann._update_isolation_proxy(5, main)
        main.visible = True
        for lyr in subs.values():
            lyr.visible = True
        calls = self._attach_recorder(ann, main)
        ann._set_selected_label_all(99)
        assert any(tag == "proxy-sub-restore" for _l, tag, _v in calls)
        assert _all_visible(subs)

    def test_missing_heal_method_is_tolerated(self, captured_timers):
        ann, main, subs, cascade = _make_env(ndisplay=3)
        assert not hasattr(ann, "_heal_label_render")
        ann._set_selected_label_all(5)
        captured_timers[0]()                          # must not raise
        assert main.visible is True
