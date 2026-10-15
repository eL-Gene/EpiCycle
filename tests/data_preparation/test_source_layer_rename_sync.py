"""Step 1 controller — a source-layer rename keeps the name-keyed caches in sync.

Root cause (pre-fix, silent mis-render): the progressive<->native 2D/3D swap
identifies its source layers BY NAME from two load-time caches —
``_loaded_source_layers`` (a list of name strings) and
``_source_display_channels`` (a list of ``(name, levels)`` tuples).  Renaming a
source layer in napari fired ``_on_layer_renamed``, which only re-enumerated the
channel picker / recipe columns; it did NOT touch those two caches.  So after a
rename the caches held the OLD name, and the next 2D/3D toggle:
``_capture_layer_appearance(old)`` missed (contrast/colormap lost ->
over-saturation), ``_remove_layer_by_name(old)`` no-op'd (the renamed layer was
ORPHANED), and a fresh layer was re-added under the OLD name (DUPLICATE ->
the picker / recipe columns double-count).  The two sibling readers
(``_apply_3d_detail_level`` + the step-exit removal) desync the same way.

The fix keeps BOTH caches correct on rename via an object->last-known-name map
(a ``weakref.WeakKeyDictionary``), identifying the renamed layer by OBJECT
IDENTITY (its old name is already gone by the time the event fires).  These are
pure-state tests of that resync — the full render is GUI-dependent.
"""
import types
import weakref


class _FakeLayer:
    """Minimal napari-layer stand-in: a rebindable ``.name`` and (being a plain
    object) weak-referenceable, so it can key a WeakKeyDictionary."""

    def __init__(self, name):
        self.name = name


class _RenameEvent:
    """napari ``layer.events.name`` event — ``.source`` is the renamed layer."""

    def __init__(self, source):
        self.source = source


class _RenameCtl:
    """Bare stand-in binding the real rename-sync method(s) off the controller so
    the object->name cache-resync runs without a live napari viewer."""

    def __init__(self):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController as C)
        self._loaded_source_layers = []
        self._source_display_channels = None
        self._source_layer_name_map = weakref.WeakKeyDictionary()
        # `_on_layer_renamed` exists pre-fix — bind it always so failing-first is
        # a clean assertion failure, not an AttributeError.
        self._on_layer_renamed = types.MethodType(C._on_layer_renamed, self)
        # The map-refresh helper is added by the fix; bind it only if present so
        # the module still imports pre-fix.
        tracker = getattr(C, "_track_source_layer_objects", None)
        if tracker is not None:
            self._track_source_layer_objects = types.MethodType(tracker, self)

    def _on_layers_changed(self, *_a, **_k):
        pass

    def _log(self, *_a, **_k):
        pass


def _levels():
    # A "multiscale" channel arrives as a LIST of pyramid-level arrays; the
    # resync only cares that this exact object is PRESERVED across the rename, so
    # a plain string-list stand-in suffices (avoids numpy `==` ambiguity).
    return ["L0", "L1"]


class TestSourceLayerRenameSync:
    def _loaded(self, name="wound.tif"):
        ctl = _RenameCtl()
        layer = _FakeLayer(name)
        lv = _levels()
        ctl._loaded_source_layers = [name]
        ctl._source_display_channels = [(name, lv)]
        ctl._source_layer_name_map[layer] = name
        return ctl, layer, lv

    def test_rename_updates_both_caches(self):
        """FAILING-FIRST: renaming a tracked source layer updates BOTH name
        caches + the object->name map (pre-fix they stay 'wound.tif')."""
        ctl, layer, lv = self._loaded("wound.tif")
        layer.name = "membrane"                       # napari already renamed it
        ctl._on_layer_renamed(_RenameEvent(layer))

        assert ctl._loaded_source_layers == ["membrane"]
        assert len(ctl._source_display_channels) == 1
        name0, levels0 = ctl._source_display_channels[0]
        assert name0 == "membrane"
        assert levels0 is lv                          # same levels object kept
        assert ctl._source_layer_name_map[layer] == "membrane"

    def test_rename_untracked_layer_leaves_caches_unchanged(self):
        """A rename of a layer NOT in the map (preview / box / foreign) must not
        touch the source caches."""
        ctl, _tracked, lv = self._loaded("wound.tif")
        foreign = _FakeLayer("some_preview")          # never seeded into the map
        foreign.name = "renamed_preview"
        ctl._on_layer_renamed(_RenameEvent(foreign))

        assert ctl._loaded_source_layers == ["wound.tif"]
        assert len(ctl._source_display_channels) == 1
        name0, levels0 = ctl._source_display_channels[0]
        assert name0 == "wound.tif"
        assert levels0 is lv

    def test_rename_follows_new_object_after_swap_readd(self):
        """The 2D/3D swap creates NEW layer objects and refreshes the map at its
        re-add site; a rename of the post-swap object must still resync."""
        ctl, _old, lv = self._loaded("wound.tif")
        # Simulate the swap re-add: a fresh layer object under the same name,
        # with the map refreshed onto it (via the real helper if present).
        swapped = _FakeLayer("wound.tif")
        if hasattr(ctl, "_track_source_layer_objects"):
            ctl._track_source_layer_objects([swapped])
        else:                                          # pre-fix fallback
            ctl._source_layer_name_map = weakref.WeakKeyDictionary()
            ctl._source_layer_name_map[swapped] = "wound.tif"

        swapped.name = "cadherin"
        ctl._on_layer_renamed(_RenameEvent(swapped))

        assert ctl._loaded_source_layers == ["cadherin"]
        assert len(ctl._source_display_channels) == 1
        name0, levels0 = ctl._source_display_channels[0]
        assert name0 == "cadherin"
        assert levels0 is lv
        assert ctl._source_layer_name_map[swapped] == "cadherin"

    def test_no_op_when_name_unchanged(self):
        """A `name` event that fires without an actual change (old == new) is a
        no-op on the caches (defensive: some viewers re-emit)."""
        ctl, layer, lv = self._loaded("wound.tif")
        ctl._on_layer_renamed(_RenameEvent(layer))     # name still 'wound.tif'
        assert ctl._loaded_source_layers == ["wound.tif"]
        name0, _levels0 = ctl._source_display_channels[0]
        assert name0 == "wound.tif"

    def test_rename_leaves_sibling_channels_untouched(self):
        """Multi-channel: renaming one channel updates ONLY its entry; the other
        channel's name + its levels OBJECT are untouched (the break-after-first
        invariant + the positional resync staying in step — audit F3)."""
        ctl = _RenameCtl()
        la, lb = _FakeLayer("chA"), _FakeLayer("chB")
        lva, lvb = _levels(), _levels()
        ctl._loaded_source_layers = ["chA", "chB"]
        ctl._source_display_channels = [("chA", lva), ("chB", lvb)]
        ctl._source_layer_name_map[la] = "chA"
        ctl._source_layer_name_map[lb] = "chB"

        la.name = "cadherin"
        ctl._on_layer_renamed(_RenameEvent(la))

        assert ctl._loaded_source_layers == ["cadherin", "chB"]
        name0, levels0 = ctl._source_display_channels[0]
        name1, levels1 = ctl._source_display_channels[1]
        assert name0 == "cadherin" and levels0 is lva   # own entry updated, levels kept
        assert name1 == "chB" and levels1 is lvb         # sibling untouched
        assert ctl._source_layer_name_map[la] == "cadherin"
        assert ctl._source_layer_name_map[lb] == "chB"   # sibling map entry untouched
