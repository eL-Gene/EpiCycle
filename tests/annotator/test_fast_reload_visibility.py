"""Subviewer FAST-reload visibility propagation.

Bug repro (user-reported 2026-05-03): on patch switch, the FAST reload
path renames + data-swaps existing subviewer layers in place but did
NOT propagate the new (main-viewer) layer's ``visible`` attribute.
If the user toggled the image off with `D` while working on patch A,
switching to patch B left the subviewer image layers ``visible=False``
even though the new main-viewer image was ``visible=True``.

The fix: ``try_fast_reload`` now copies ``proxy.visible`` to each
subviewer layer alongside the existing data/name/style propagation,
keeping FAST and SLOW reload paths symmetric.

Tests use real napari Image/Labels layers (no viewer required) so
``isinstance(l, Image)`` checks in the production code work.
"""

from __future__ import annotations

import numpy as np
from unittest.mock import MagicMock

from napari.layers import Image, Labels


def _make_subviewer_manager_with_layers(image_data, labels_data, *, image_visible, labels_visible):
    """Build a SubviewerManager with three viewer-model fakes pre-populated
    with one Image + one Labels layer each, matching the given shapes/visibility.
    """
    from packages.epicycle_annotator.subviewers import SubviewerManager
    from packages.epicycle_annotator.orientation import transpose_data

    def _build_vm(orient):
        img = Image(transpose_data(image_data, orient), name=f"old_image_{orient}")
        lbl = Labels(transpose_data(labels_data, orient), name=f"old_labels_{orient}")
        img.visible = image_visible
        lbl.visible = labels_visible
        vm = MagicMock()
        vm.layers = [img, lbl]
        return vm, img, lbl

    vm_xy, img_xy, lbl_xy = _build_vm("XY")
    vm_xz, img_xz, lbl_xz = _build_vm("XZ")
    vm_yz, img_yz, lbl_yz = _build_vm("YZ")

    mgr = SubviewerManager(
        vm_xy, vm_xz, vm_yz,
        MagicMock(), MagicMock(), MagicMock(),
    )
    return mgr, [img_xy, img_xz, img_yz], [lbl_xy, lbl_xz, lbl_yz]


class TestFastReloadVisibility:
    """try_fast_reload must propagate visible state from the new
    (main-viewer) proxy layer to the subviewer layers."""

    def test_visible_proxy_makes_hidden_subviewer_visible(self):
        """The exact bug: subviewer image was hidden from user toggling
        `D` on previous patch, new patch's main-viewer image is visible,
        FAST reload should restore subviewer image visibility."""
        old_img = np.zeros((4, 8, 8), dtype=np.float32)
        new_img = np.ones((4, 8, 8), dtype=np.float32)
        labels_data = np.zeros((4, 8, 8), dtype=np.uint32)

        mgr, sub_imgs, sub_lbls = _make_subviewer_manager_with_layers(
            old_img, labels_data,
            image_visible=False,   # ← user had toggled image off
            labels_visible=True,
        )

        # Build new proxies (the new patch's main-viewer layers).
        new_image_proxy = Image(new_img, name="new_image")
        new_image_proxy.visible = True
        new_labels_proxy = Labels(labels_data, name="new_labels")
        new_labels_proxy.visible = True

        ok, _ = mgr.try_fast_reload([new_image_proxy], [new_labels_proxy])
        assert ok, "Fast reload should succeed when shapes match"

        for layer in sub_imgs:
            assert layer.visible is True, (
                f"Subviewer image {layer.name} should be visible after fast "
                f"reload (proxy.visible=True)"
            )
        for layer in sub_lbls:
            assert layer.visible is True

    def test_hidden_proxy_makes_visible_subviewer_hidden(self):
        """Symmetric case: if the new main-viewer layer is hidden, the
        subviewer layers should follow.  Defends against the inverse
        regression where we always force-True."""
        labels_data = np.zeros((4, 8, 8), dtype=np.uint32)

        mgr, sub_imgs, sub_lbls = _make_subviewer_manager_with_layers(
            np.zeros((4, 8, 8), dtype=np.float32), labels_data,
            image_visible=True,
            labels_visible=True,
        )

        new_image_proxy = Image(np.ones((4, 8, 8), dtype=np.float32), name="new_image")
        new_image_proxy.visible = False
        new_labels_proxy = Labels(labels_data, name="new_labels")
        new_labels_proxy.visible = False

        ok, _ = mgr.try_fast_reload([new_image_proxy], [new_labels_proxy])
        assert ok

        for layer in sub_imgs:
            assert layer.visible is False
        for layer in sub_lbls:
            assert layer.visible is False

    def test_data_and_name_still_propagated(self):
        """Sanity check: the visibility fix did not break the existing
        rename + data-swap behavior."""
        old_data = np.zeros((4, 8, 8), dtype=np.float32)
        new_data = np.ones((4, 8, 8), dtype=np.float32)
        labels_data = np.zeros((4, 8, 8), dtype=np.uint32)

        mgr, sub_imgs, _ = _make_subviewer_manager_with_layers(
            old_data, labels_data,
            image_visible=True,
            labels_visible=True,
        )

        new_image_proxy = Image(new_data, name="new_image")
        new_image_proxy.visible = True
        new_labels_proxy = Labels(labels_data, name="new_labels")
        new_labels_proxy.visible = True

        mgr.try_fast_reload([new_image_proxy], [new_labels_proxy])

        for orient, layer in zip(("XY", "XZ", "YZ"), sub_imgs):
            assert layer.name == f"new_image_{orient}"
            np.testing.assert_array_equal(layer.data, np.ones_like(layer.data))
