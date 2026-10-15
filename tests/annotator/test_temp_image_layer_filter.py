"""An UNMANAGED Image layer is not a channel.

``guide_walls`` -- the wall picture "Show what the tool sees" puts up --
is an ``Image`` layer the user did not create.  Two list builders in
``native_multiview_widget`` walk ``viewer.layers`` and collect every
``Image`` they find:

* ``_refresh_layer_options`` fills the annotator's image combo;
* ``_resolve_source_image_layers`` picks the channel set the three
  subviewer mirrors show, and its result becomes
  ``_active_image_layer_names`` -- which is what the a/s/d/f keys are
  mapped onto, positionally.

So without a filter, a wall picture present during a subviewer reload is
promoted to a real channel and SHIFTS the keymap under the user's
fingers (the annotator takes every Image layer).  Six
lines below the first site, the ``Labels`` branch has always skipped
``_epicycle_temporary``; this is the same two lines on the ``Image`` one.

Neither function reads or writes ``visible``, ``colormap``, ``contour``,
``opacity``, ``shared_labels``, any proxy or any sync handler -- what
changes is whether a layer the user did not create is OFFERED to them as
if they had.

**Both classes, because Step 3 never runs the one this file used to
bind.**  ``Curation3DAnnotator`` OVERRIDES
``_resolve_source_image_layers``: it calls ``super()`` (filtered) and,
when one layer or none comes back, falls through to its OWN loop over
``viewer.layers``.  On a single-channel patch -- the common case -- that
is the loop that answers, so a filter present only on the parent leaves
``guide_walls`` a channel exactly where the bug bites.  Every test here
therefore runs against ``NativeMultiviewAnnotator`` AND
``Curation3DAnnotator``.

Behavioural: production methods bound to lightweight harnesses, asserting
observable state.  Pattern borrowed from
``test_temporary_layer_filter_real.py``.
"""

from __future__ import annotations

import types
from unittest.mock import MagicMock

import pytest


def _image(name, ndim=3, temporary=False, shape=None):
    """A fake that passes ``isinstance(layer, Image)``."""
    from napari.layers import Image

    layer = MagicMock(spec=Image)
    layer.name = name
    layer.data = MagicMock()
    layer.data.ndim = ndim
    layer.data.shape = shape or (8, 16, 16)
    if temporary:
        layer._epicycle_temporary = True
    else:
        # MagicMock(spec=Image) would happily invent the attribute, and
        # a truthy Mock reads as "temporary" -- so it has to be deleted,
        # not merely left unset.
        del layer._epicycle_temporary
    return layer


def _labels(name, ndim=3, temporary=False):
    from napari.layers import Labels

    layer = MagicMock(spec=Labels)
    layer.name = name
    layer.data = MagicMock()
    layer.data.ndim = ndim
    if temporary:
        layer._epicycle_temporary = True
    else:
        del layer._epicycle_temporary
    return layer


def _parent():
    from napari_epicycle.annotator.native_multiview_widget import (
        NativeMultiviewAnnotator,
    )

    return NativeMultiviewAnnotator


def _subclass():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


@pytest.fixture(params=("parent", "curation3d"))
def cls(request):
    """The annotator class under test -- the base one and the one Step 3
    actually instantiates."""
    return _parent() if request.param == "parent" else _subclass()


def _bind(cls, name, harness):
    return types.MethodType(getattr(cls, name), harness)


def _instance(cls, **attrs):
    """An UNINITIALISED instance of *cls*, for the methods that ``super()``.

    ``Curation3DAnnotator._resolve_source_image_layers`` opens with the
    zero-argument ``super()``, which type-checks ``isinstance(self,
    Curation3DAnnotator)`` -- a ``MagicMock`` fails that, so the subclass
    half of this file cannot be driven with one.  ``cls.__new__(cls)``
    gives a real instance whose ``__init__`` never ran: no Qt object is
    created, and every attribute the method reads is one set below.
    """
    harness = cls.__new__(cls)
    for key, value in attrs.items():
        setattr(harness, key, value)
    return harness


class _Viewer:
    def __init__(self, layers):
        self.layers = list(layers)


# --------------------------------------------------------------------------- #
# 1. the image combo
# --------------------------------------------------------------------------- #
class TestTheImageComboSkipsATemporaryImage:
    def _run(self, cls, layers):
        harness = MagicMock()
        harness.viewer.layers = layers
        harness.image_combo = MagicMock()
        harness.image_combo.currentText.return_value = ""
        harness.labels_combo = MagicMock()
        harness.labels_combo.currentText.return_value = ""
        _bind(cls, "_refresh_layer_options", harness)()
        return list(harness.image_combo.addItems.call_args_list[0][0][0])

    def test_the_wall_picture_is_not_offered_as_a_channel(self, cls):
        added = self._run(cls, [
            _image("membrane"),
            _image("guide_walls", temporary=True),
            _labels("curation_3d_p01"),
        ])
        assert "membrane" in added
        assert "guide_walls" not in added, (
            "an unmanaged Image layer must not be offered as a channel")

    def test_every_other_image_layer_is_kept(self, cls):
        """The filter is a skip, not a rewrite: nothing else moves."""
        added = self._run(cls, [
            _image("membrane"),
            _image("nuclei"),
            _image("rgb", ndim=4, shape=(8, 16, 16, 3)),
            _image("guide_walls", temporary=True),
        ])
        assert added == ["membrane", "nuclei", "rgb"]

    def test_a_layer_that_says_False_is_still_a_channel(self, cls):
        layer = _image("membrane")
        layer._epicycle_temporary = False
        assert "membrane" in self._run(cls, [layer])

    def test_the_labels_branch_is_unchanged(self, cls):
        harness = MagicMock()
        harness.viewer.layers = [
            _image("membrane"),
            _image("guide_walls", temporary=True),
            _labels("curation_3d_p01"),
            _labels("Refine", temporary=True),
        ]
        harness.image_combo = MagicMock()
        harness.image_combo.currentText.return_value = ""
        harness.labels_combo = MagicMock()
        harness.labels_combo.currentText.return_value = ""
        _bind(cls, "_refresh_layer_options", harness)()
        added = list(harness.labels_combo.addItems.call_args_list[0][0][0])
        assert added == ["curation_3d_p01"]


# --------------------------------------------------------------------------- #
# 2. the subviewer source channels
# --------------------------------------------------------------------------- #
class TestTheSourceChannelsSkipATemporaryImage:
    def _resolve(self, cls, layers, image_layer, labels_name=""):
        harness = _instance(
            cls,
            viewer=_Viewer(layers),
            _infer_plane_from_image_name=lambda _n: "",
            _active_selected_view="XY",
            _labels_patch_prefix=lambda _n: "",
        )
        return _bind(cls, "_resolve_source_image_layers", harness)(
            image_layer, labels_name)

    def test_the_wall_picture_is_never_collected(self, cls):
        membrane = _image("membrane")
        nuclei = _image("nuclei")
        walls = _image("guide_walls", temporary=True)
        out = self._resolve(cls, [membrane, walls, nuclei], membrane)
        assert [lyr.name for lyr in out] == ["membrane", "nuclei"]

    def test_a_lone_real_channel_still_answers_itself(self, cls):
        """THE single-channel case -- and the one the subclass answers
        by ITSELF.

        ``Curation3DAnnotator`` takes the parent's (filtered) list only
        when it holds MORE THAN ONE layer; one layer means "the patch
        naming did not match", and it re-walks ``viewer.layers`` in a
        loop of its own.  A filter on the parent alone is therefore
        invisible here, which is the whole of the bug: a curation patch
        with one channel is the ordinary case.
        """
        membrane = _image("membrane")
        walls = _image("guide_walls", temporary=True)
        out = self._resolve(cls, [membrane, walls], membrane)
        assert [lyr.name for lyr in out] == ["membrane"]

    def test_the_fallback_is_still_reachable(self, cls):
        """Nothing but a temporary layer in the viewer: the caller's own
        layer comes back, never an empty channel set."""
        walls = _image("guide_walls", temporary=True)
        membrane = _image("membrane")
        out = self._resolve(cls, [walls], membrane)
        assert out == [membrane]

    def test_a_4D_rgb_channel_survives_the_skip(self, cls):
        """The subclass's own loop has an RGB branch of its own, and the
        two added lines must not cost it."""
        rgb = _image("rgb", ndim=4, shape=(8, 16, 16, 3))
        walls = _image("guide_walls", temporary=True)
        out = self._resolve(cls, [rgb, walls], rgb)
        assert [lyr.name for lyr in out] == ["rgb"]


# --------------------------------------------------------------------------- #
# 3. the a/s/d/f keymap
# --------------------------------------------------------------------------- #
class TestTheKeymapDoesNotShift:
    def _keymap(self, cls, image_names, labels_names):
        """Bind the real ``_setup_channel_keybindings`` and read back
        which layer NAME each of a/s/d/f ended up toggling."""
        asked: list = []

        harness = MagicMock()
        harness._active_image_layer_names = list(image_names)
        harness._active_labels_names = list(labels_names)
        harness._CHANNEL_KEYS = cls._CHANNEL_KEYS
        harness._channel_toggle_in_flight = False

        def _targets(name):
            asked.append(name)
            return []

        harness._get_toggle_targets_for_name = _targets

        bound_keys = {}

        class _V:
            def bind_key(self, key, fn, overwrite=False):
                bound_keys[key] = fn

        viewer = _V()
        harness.viewer = viewer
        harness.viewer_model_xy = viewer
        harness.viewer_model_xz = viewer
        harness.viewer_model_yz = viewer
        harness._teardown_channel_keybindings = lambda: None

        _bind(cls, "_setup_channel_keybindings", harness)()

        out = {}
        for key, fn in bound_keys.items():
            asked.clear()
            fn(None)
            out[key] = asked[0] if asked else None
        return out

    def test_the_keys_land_on_the_same_layers_with_and_without_it(self, cls):
        """The filter's whole point.  ``_active_image_layer_names`` is
        built from ``_resolve_source_image_layers``, so a collected
        ``guide_walls`` would take a key and push every channel after it
        one letter along."""
        clean = self._keymap(cls, ["membrane", "nuclei"],
                             ["curation_3d_p01"])
        assert clean == {"a": "membrane", "s": "nuclei",
                         "d": "curation_3d_p01"}

        # what the UNFILTERED builder would have produced, for contrast
        shifted = self._keymap(
            cls, ["membrane", "guide_walls", "nuclei"],
            ["curation_3d_p01"])
        assert shifted["s"] == "guide_walls", (
            "this is the bug the filter prevents -- if it ever stops "
            "being reachable, delete this contrast, not the filter")
        assert shifted != clean

    def test_the_resolved_sources_are_what_the_keymap_is_built_from(
            self, cls):
        """The join between part 2 and part 3, run rather than asserted
        about: resolve the channels with the temporary layer present and
        feed THAT into the keymap."""
        membrane = _image("membrane")
        nuclei = _image("nuclei")
        walls = _image("guide_walls", temporary=True)

        harness = _instance(
            cls,
            viewer=_Viewer([membrane, walls, nuclei]),
            _infer_plane_from_image_name=lambda _n: "",
            _active_selected_view="XY",
            _labels_patch_prefix=lambda _n: "",
        )
        resolved = _bind(cls, "_resolve_source_image_layers", harness)(
            membrane, "")

        keymap = self._keymap(cls, [lyr.name for lyr in resolved],
                              ["curation_3d_p01"])
        assert keymap == {"a": "membrane", "s": "nuclei",
                          "d": "curation_3d_p01"}
        assert "f" not in keymap

    def test_the_single_channel_patch_keeps_its_keymap(self, cls):
        """The case the subclass answers with its OWN loop.

        One real channel and the wall picture: ``a`` must still be the
        channel and ``s`` the labels layer.  Unfiltered, the subclass
        collects both Image layers and ``s`` becomes ``guide_walls``.
        """
        membrane = _image("membrane")
        walls = _image("guide_walls", temporary=True)
        harness = _instance(
            cls,
            viewer=_Viewer([membrane, walls]),
            _infer_plane_from_image_name=lambda _n: "",
            _active_selected_view="XY",
            _labels_patch_prefix=lambda _n: "",
        )
        resolved = _bind(cls, "_resolve_source_image_layers", harness)(
            membrane, "")
        keymap = self._keymap(cls, [lyr.name for lyr in resolved],
                              ["curation_3d_p01"])
        assert keymap == {"a": "membrane", "s": "curation_3d_p01"}
