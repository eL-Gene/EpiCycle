"""A partial ``set_curation_callbacks`` must not null the other callbacks.

The regression this file exists for (found pre-merge, 2026-09-19, origin
`a24d01fc`): ``_post_switch_annotator_sync`` re-wires the annotator after
every patch switch, and it passes THREE kwargs --
``on_reshape_lock`` / ``on_reshape_committed`` / ``on_reshape_context``.
``set_curation_callbacks`` assigned seven of its callbacks
UNCONDITIONALLY from their ``None`` defaults, so that call silently wiped
``on_label_smoothed``, ``on_label_split``, ``on_interpolation_accepted``,
``on_recalculate``, ``on_new_label_painted``, ``on_batch_refined``,
``on_batch_refine_lock`` and the two batch-recipe providers.

What that cost the user: after any patch switch, a Smooth / Split /
Interpolate / Batch-Refine Accept called nobody.  The volume changed in
memory, but ``_commit_curation_action`` never ran, so no undo entry, no
``_curation_dirty``, no Save prompt -- and the next patch load discarded
the edit without a word.  Silent data loss, reachable by the most
ordinary action in Step 3.

The rule, stated once: **a keyword that was not passed means "leave it
alone", never "set it to None".**  Every one of these assignments is
guarded now, so a partial call is additive by construction.
"""

from __future__ import annotations

import pytest

pytest.importorskip("qtpy")


def _annotator_cls():
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator,
    )

    return Curation3DAnnotator


#: Every attribute `set_curation_callbacks` can write, and the keyword
#: that writes it.  If a callback is added to the method without being
#: added here, `test_every_callback_the_method_sets_is_covered` fails.
CALLBACKS = {
    "on_interpolation_accepted": "_on_interpolation_accepted_cb",
    "on_label_split": "_on_label_split_cb",
    "on_label_smoothed": "_on_label_smoothed_cb",
    "on_recalculate": "_on_recalculate_cb",
    "on_new_label_painted": "_on_new_label_painted_cb",
    "on_batch_refined": "_on_batch_refined_cb",
    "on_batch_refine_lock": "_batch_refine_lock_cb",
    "on_add_mode_lock": "_add_mode_lock_cb",
    "on_add_committed": "_on_add_committed_cb",
    "on_add_undo_session": "_on_add_undo_session_cb",
    "on_save_batch_recipe": "_save_batch_recipe_cb",
    "get_saved_batch_recipes": "_get_saved_batch_recipes_cb",
    "on_reshape_lock": "_reshape_lock_cb",
    "on_refine_lock": "_refine_lock_cb",
    "on_reshape_committed": "_on_reshape_committed_cb",
    "on_reshape_context": "_reshape_context_cb",
    # Refine recipes
    "on_save_refine_recipe": "_save_refine_recipe_cb",
    "get_saved_refine_recipes": "_get_saved_refine_recipes_cb",
    "on_delete_refine_recipe": "_delete_refine_recipe_cb",
    # Batch "Plot selection (N)"
    "get_plot_selection": "_get_plot_selection_cb",
    # The QC df's highest id, ghosts included
    "get_label_id_floor": "_get_label_id_floor_cb",
}

#: What `_post_switch_annotator_sync` actually passes.  The FOUR that must
#: not cost the others: Brick B1 added `on_refine_lock`, and a lock callback
#: that is wired at launch but not re-wired after a patch switch is a lock
#: that silently stops working on the second patch.
POST_SWITCH_KWARGS = (
    "on_reshape_lock", "on_refine_lock", "on_reshape_committed",
    "on_reshape_context",
)


class _Bare:
    """The method under test, bound to a bare object.

    `set_curation_callbacks` is pure assignment -- no Qt, no napari, no
    layers -- so it runs on anything.  Binding it rather than building a
    real annotator keeps this test about the contract and nothing else.
    """

    def __init__(self):
        cls = _annotator_cls()
        self.set_curation_callbacks = (
            cls.set_curation_callbacks.__get__(self, _Bare)
        )


def _full_set():
    """One distinct sentinel per keyword, so a swap is visible too."""
    return {kw: (lambda *a, _k=kw, **k: _k) for kw in CALLBACKS}


class TestAPartialCallIsAdditive:
    def test_the_post_switch_kwargs_leave_the_others_intact(self):
        obj = _Bare()
        full = _full_set()
        obj.set_curation_callbacks(**full)
        before = {attr: getattr(obj, attr) for attr in CALLBACKS.values()}

        # exactly what the patch switch does
        obj.set_curation_callbacks(
            **{kw: full[kw] for kw in POST_SWITCH_KWARGS})

        lost = [
            CALLBACKS[kw] for kw in CALLBACKS
            if getattr(obj, CALLBACKS[kw]) is not before[CALLBACKS[kw]]
        ]
        assert not lost, (
            "a partial set_curation_callbacks nulled: {}".format(sorted(lost))
        )

    @pytest.mark.parametrize("kw", sorted(CALLBACKS))
    def test_setting_any_ONE_of_them_keeps_the_other_fourteen(self, kw):
        obj = _Bare()
        full = _full_set()
        obj.set_curation_callbacks(**full)
        obj.set_curation_callbacks(**{kw: full[kw]})
        for other, attr in CALLBACKS.items():
            assert getattr(obj, attr) is full[other], (
                "setting {!r} cleared {!r}".format(kw, other)
            )

    def test_a_call_with_NO_kwargs_changes_nothing(self):
        obj = _Bare()
        full = _full_set()
        obj.set_curation_callbacks(**full)
        obj.set_curation_callbacks()
        for kw, attr in CALLBACKS.items():
            assert getattr(obj, attr) is full[kw], kw

    def test_a_kwarg_that_IS_passed_still_replaces_the_old_one(self):
        """Guarding must not make the method write-once."""
        obj = _Bare()
        obj.set_curation_callbacks(**_full_set())
        second = _full_set()
        obj.set_curation_callbacks(on_label_smoothed=second[
            "on_label_smoothed"])
        assert obj._on_label_smoothed_cb is second["on_label_smoothed"]

    def test_the_first_full_call_still_wires_everything(self):
        """The launcher's call: nothing was set before, everything after."""
        obj = _Bare()
        full = _full_set()
        obj.set_curation_callbacks(**full)
        for kw, attr in CALLBACKS.items():
            assert getattr(obj, attr) is full[kw], kw


class TestTheTableIsCompleteAndTrue:
    def test_every_keyword_in_the_signature_is_in_the_table(self):
        """A callback added to the method without a row here would not be
        covered by the guard test above -- which is how the seven got
        through in the first place."""
        import inspect

        sig = inspect.signature(_annotator_cls().set_curation_callbacks)
        kwargs = {
            name for name, p in sig.parameters.items()
            if p.kind is inspect.Parameter.KEYWORD_ONLY
        }
        assert kwargs == set(CALLBACKS), (
            "signature and table disagree: only-in-signature={} "
            "only-in-table={}".format(
                sorted(kwargs - set(CALLBACKS)),
                sorted(set(CALLBACKS) - kwargs))
        )

    def test_every_keyword_defaults_to_None(self):
        """The guard reads `is not None`, so a non-None default would
        make that keyword un-skippable."""
        import inspect

        sig = inspect.signature(_annotator_cls().set_curation_callbacks)
        for name in CALLBACKS:
            assert sig.parameters[name].default is None, name

    def test_the_post_switch_call_site_still_passes_these_four(self):
        """If the controller's partial call ever grows to the full set the
        guard is still right, but this list would be a lie."""
        import inspect

        from napari_epicycle.fusion_reconstruction.controller import (
            FusionReconstructionController,
        )

        src = inspect.getsource(
            FusionReconstructionController._post_switch_annotator_sync)
        assert "set_curation_callbacks(" in src
        for kw in POST_SWITCH_KWARGS:
            assert kw + "=" in src, kw
