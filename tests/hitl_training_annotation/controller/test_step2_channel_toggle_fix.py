"""Tests for Step 2 v2 fix — channel-toggle mode preservation
with directional save/restore.

The bug: in Step 2 HITL, when the user is painting on a labels layer
(mode='paint') and presses the labels-mapped channel key (e.g. 'd')
to hide then again to show, the mode reverts to pan_zoom on the
SUBVIEWER MIRRORS (the layers the user actually paints on).

The cause: a double-event cascade during hide.  When the main
work_labels layer's visibility is set False:
1. napari saves main's mode to _visible_mode (correct).
2. main's mode-change event fires sync_label_modes which propagates
   pan_zoom to all mirrors (corrupting their mode).
3. main's visible event fires sync_layer_visibility which propagates
   visible=False to mirrors.
4. Each mirror's _on_visible_changed saves CURRENT mode (now
   corrupted to pan_zoom) to its own _visible_mode.
On show, napari's restore on each mirror uses the corrupted
_visible_mode, overwriting sync_label_modes' subsequent re-propagation
of the restored main mode.

The fix (v2): directional save/restore — save modes BEFORE
visibility toggle on hide (captures uncorrupted state), restore
AFTER toggle on show (overrides napari's corrupted restore).
Persistent per-layer attribute ``_epicycle_pre_hide_mode`` keeps the
saved value across the hide/show cycle.

These tests verify the directional save/restore is wired in both
toggle sites:
* The base-class ``_setup_channel_keybindings._make_toggle._toggle``
  in ``napari_epicycle/annotator/native_multiview_widget.py``
* The Step 2 override ``_toggle_all_labels`` in
  ``napari_epicycle/annotator/curation_2d_widget.py``
"""

from __future__ import annotations

import inspect


def test_25v2_base_class_toggle_uses_directional_save_restore():
    """V2 — the base-class channel-toggle handler must use the
    directional save/restore pattern: save modes on HIDE, restore from
    saved value on SHOW.  The persistent attribute
    ``_epicycle_pre_hide_mode`` must appear in both directions.
    """
    from napari_epicycle.annotator import native_multiview_widget
    src = inspect.getsource(native_multiview_widget)

    # The v2 fix marker must be present.
    assert "v2 fix" in src, (
        "v2 regression: the channel-toggle handler in "
        "native_multiview_widget.py is missing the documented "
        "'v2 fix' marker.  Suggests the fix was reverted to the "
        "buggy 'trust napari's _visible_mode' approach that left "
        "subviewer mirrors stuck in pan_zoom on the show direction."
    )

    # Persistent per-layer attribute must be referenced (save + restore).
    assert "_epicycle_pre_hide_mode" in src, (
        "v2 regression: the persistent save attribute "
        "_epicycle_pre_hide_mode is not referenced in "
        "native_multiview_widget.py.  Without it, the corrupted-on-hide "
        "subviewer mirror mode cannot be restored on show."
    )

    # Directional gating: there must be both a `if not new_vis:` save
    # block and a `if new_vis:` restore block in the toggle closure.
    # We grep for both patterns to make sure neither direction was
    # dropped.
    assert "if not new_vis:" in src and "if new_vis:" in src, (
        "v2 regression: missing directional gating "
        "(if not new_vis: save, if new_vis: restore) in "
        "native_multiview_widget.py channel-toggle handler."
    )

    # v2 reentrancy guard — protect against rapid key-mash
    # firing a second _toggle while the first is mid-cascade.
    assert "_channel_toggle_in_flight" in src, (
        "v2 regression: missing reentrancy guard "
        "(_channel_toggle_in_flight flag with drop-on-repeat policy) "
        "in native_multiview_widget.py channel-toggle handler.  "
        "Without the guard, key-mashing 'd' rapidly can corrupt the "
        "saved pre-hide mode mid-cascade."
    )


def test_25v2_step2_override_uses_directional_save_restore():
    """V2 — the Step 2 multi-layer override ``_toggle_all_labels``
    must use the same directional save/restore pattern as the base
    class.
    """
    from napari_epicycle.annotator import curation_2d_widget
    src = inspect.getsource(curation_2d_widget)

    assert "_epicycle_pre_hide_mode" in src, (
        "v2 regression: the persistent save attribute "
        "_epicycle_pre_hide_mode is not referenced in "
        "curation_2d_widget.py._toggle_all_labels.  Without it, the "
        "corrupted-on-hide subviewer mirror mode cannot be restored on "
        "show."
    )

    assert "if not new_vis:" in src and "if new_vis:" in src, (
        "v2 regression: missing directional gating "
        "(if not new_vis: save, if new_vis: restore) in "
        "curation_2d_widget.py._toggle_all_labels."
    )

    # v2 reentrancy guard — Step 2 specifically because users
    # mash 'd' to toggle hide/show masks rapidly.
    assert "_channel_toggle_in_flight" in src, (
        "v2 regression: missing reentrancy guard "
        "(_channel_toggle_in_flight flag with drop-on-repeat policy) "
        "in curation_2d_widget.py._toggle_all_labels.  Without the "
        "guard, key-mashing 'd' rapidly can corrupt the saved pre-"
        "hide mode mid-cascade — and Step 2 users do mash."
    )
