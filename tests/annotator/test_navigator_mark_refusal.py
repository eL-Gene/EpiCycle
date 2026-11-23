"""A refused tier mark does not auto-advance; the Mark buttons grey while the annotator is load-locked."""
from __future__ import annotations

from types import SimpleNamespace

from napari_epicycle.annotator.curation_3d_widget import Curation3DAnnotator
from napari_epicycle.annotator.curation_label_navigator import CurationLabelNavigator


class _Check:
    def isChecked(self):
        return True


def _nav(cb):
    nav = SimpleNamespace(
        _on_mark_tier_cb=cb,
        _auto_advance_cb=_Check(),
        advanced=0,
    )
    nav.current_label_id = lambda: 7
    nav._advance_to_next_visible = lambda: setattr(nav, "advanced", nav.advanced + 1)
    return nav


def test_refused_mark_does_not_advance():
    nav = _nav(lambda lid, tier: False)
    CurationLabelNavigator._on_mark_tier_clicked(nav, "good")
    assert nav.advanced == 0


def test_accepted_mark_still_advances():
    for result in (True, None):
        nav = _nav(lambda lid, tier, r=result: r)
        CurationLabelNavigator._on_mark_tier_clicked(nav, "good")
        assert nav.advanced == 1


class _Btn:
    def __init__(self):
        self.enabled = True
        self.tip = ""

    def setEnabled(self, v):
        self.enabled = bool(v)

    def setToolTip(self, t):
        self.tip = t


def test_mark_buttons_follow_the_flag():
    names = ("good", "review", "bad", "garbage", "curated")
    nav = SimpleNamespace(**{f"_mark_{n}_btn": _Btn() for n in names})
    CurationLabelNavigator.set_mark_buttons_enabled(nav, False)
    assert all(not getattr(nav, f"_mark_{n}_btn").enabled for n in names)
    CurationLabelNavigator.set_mark_buttons_enabled(nav, True)
    assert all(getattr(nav, f"_mark_{n}_btn").enabled for n in names)


def test_load_lock_greys_and_restores_the_mark_buttons():
    calls = []

    class Nav:
        def set_label_action_buttons_enabled(self, v): pass
        def set_merge_enabled(self, v): pass
        def set_mark_buttons_enabled(self, v): calls.append(bool(v))

    fake = SimpleNamespace(
        _navigator_widget=Nav(),
        _lock_subviewers_for_preview=lambda: None,
        _is_op_active_for_navigator=lambda: False,
        _restore_ready_to_draw_state=lambda: None,
    )
    Curation3DAnnotator.set_load_locked(fake, True)
    Curation3DAnnotator.set_load_locked(fake, False)
    assert calls == [False, True]


def test_annotator_forwards_the_refusal():
    fake = SimpleNamespace(_on_mark_tier_cb=lambda lid, tier: False)
    assert Curation3DAnnotator._on_mark_tier_from_navigator(fake, 3, "bad") is False
    fake = SimpleNamespace(_on_mark_tier_cb=lambda lid, tier: None)
    assert Curation3DAnnotator._on_mark_tier_from_navigator(fake, 3, "bad") is None
