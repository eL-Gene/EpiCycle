"""Step 1 `adopt_workspace` — the shared-workspace hand-over.

Adopt puts the shared workspace into Step 1's path box and does the quiet
part of Browse — and nothing else.  In particular it PRE-SEEDS the
layout-migration one-shot so the migration modal is not thrown at a user who
merely walked back into Step 1; a real Browse to the same folder clears that
seed, so Browse still offers the migration exactly as before.

Qt-free: the controller methods are exercised on a bare instance with a stub
widget, the same style as the rest of `tests/data_preparation/`.
"""
from __future__ import annotations

import os

import pytest

from napari_epicycle.data_preparation.controller import (
    DataPreparationController,
)


class _Line:
    """A line-edit whose `setText` NOTIFIES, like the real one.

    The notification is routed at the publish handler on purpose: if adopt
    ever reached the publish path (directly, or because someone wires the
    publish into the `textChanged` chain), these tests would see it.  Only
    the `_suppress_publish` flag stops it.
    """

    def __init__(self, text=""):
        self._text = text
        self.sets = []
        self.on_changed = None

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value
        self.sets.append(value)
        if self.on_changed is not None:
            self.on_changed(value)


class _Widget:
    def __init__(self):
        self.output_file_path = _Line()


class _Timer:
    def __init__(self):
        self.stopped = 0

    def stop(self):
        self.stopped += 1


def _controller():
    ctrl = DataPreparationController.__new__(DataPreparationController)
    ctrl.widget = _Widget()
    # The real box notifies on every setText.  Wire the notification at the
    # PUBLISH half (the hazard: a path handed down by the shell being pushed
    # back up as a user choice); only `_suppress_publish` may stop it.
    ctrl.widget.output_file_path.on_changed = (
        lambda value: ctrl._publish_workspace_to_shell(value))
    ctrl._step1_migration_offered = set()
    ctrl._step1_sources_restored = set()
    ctrl._migration_offer_timer = _Timer()
    ctrl._suppress_publish = False
    ctrl._workflow_shell = None
    return ctrl


def _key(path):
    return os.path.normcase(os.path.normpath(path))


class TestAdoptWorkspace:
    def test_sets_the_path_box(self, tmp_path):
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.widget.output_file_path.text() == str(tmp_path)

    def test_no_op_when_step_1_already_has_a_workspace(self, tmp_path):
        ctrl = _controller()
        ctrl.widget.output_file_path.setText("D:/chosen")
        ctrl.widget.output_file_path.sets.clear()
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.widget.output_file_path.text() == "D:/chosen"
        assert ctrl.widget.output_file_path.sets == []

    def test_pre_seeds_the_migration_one_shot(self, tmp_path):
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        assert _key(str(tmp_path)) in ctrl._step1_migration_offered

    def test_migration_prompt_is_skipped_after_adopt(self, tmp_path,
                                                     monkeypatch):
        """`_maybe_offer_step1_migration` must return before any dialog."""
        ctrl = _controller()
        calls = []
        import napari_epicycle.data_preparation._migrate_step1_layout as m
        monkeypatch.setattr(
            m, "should_offer_migration",
            lambda ws: calls.append(ws) or True)
        ctrl.adopt_workspace(str(tmp_path))
        ctrl._maybe_offer_step1_migration()
        assert calls == []          # the decision check never ran → no modal

    def test_browse_clears_the_seed_so_migration_is_still_offered(
            self, tmp_path):
        ctrl = _controller()
        ctrl.adopt_workspace(str(tmp_path))
        ctrl.on_workspace_browsed(str(tmp_path))
        assert _key(str(tmp_path)) not in ctrl._step1_migration_offered

    def test_adopt_does_not_publish_back_to_the_shell(self, tmp_path):
        """Even though the fake box notifies, `_suppress_publish` holds."""
        published = []

        class _Shell:
            def set_workspace(self, path, source_step=""):
                published.append(path)

        ctrl = _controller()
        ctrl._workflow_shell = _Shell()
        ctrl.adopt_workspace(str(tmp_path))
        assert ctrl.widget.output_file_path.text() == str(tmp_path)
        assert published == []

    def test_the_notification_really_would_publish_without_the_flag(
            self, tmp_path):
        """Proves the previous test has teeth: same wiring, flag off."""
        published = []

        class _Shell:
            def set_workspace(self, path, source_step=""):
                published.append(path)

        ctrl = _controller()
        ctrl._workflow_shell = _Shell()
        ctrl.widget.output_file_path.setText(str(tmp_path))
        assert published == [str(tmp_path)]

    def test_empty_path_is_ignored(self):
        ctrl = _controller()
        ctrl.adopt_workspace("")
        assert ctrl.widget.output_file_path.text() == ""


class TestReAdopt:
    """The symmetric rule: the shared workspace can move under this step."""

    def test_re_adopts_when_the_box_still_holds_the_inherited_path(
            self, tmp_path):
        a, b = str(tmp_path / "A"), str(tmp_path / "B")
        ctrl = _controller()
        ctrl.adopt_workspace(a)
        ctrl.adopt_workspace(b, previous=a)
        assert ctrl.widget.output_file_path.text() == b

    def test_does_not_re_adopt_a_path_the_user_chose_here(self, tmp_path):
        a, b = str(tmp_path / "A"), str(tmp_path / "B")
        ctrl = _controller()
        ctrl.widget.output_file_path.setText("D:/chosen-by-hand")
        ctrl.adopt_workspace(b, previous=a)
        assert ctrl.widget.output_file_path.text() == "D:/chosen-by-hand"

    def test_re_adopt_without_previous_is_refused(self, tmp_path):
        a, b = str(tmp_path / "A"), str(tmp_path / "B")
        ctrl = _controller()
        ctrl.adopt_workspace(a)
        ctrl.adopt_workspace(b)          # no `previous` -> not ours to touch
        assert ctrl.widget.output_file_path.text() == a

    def test_re_adopt_to_the_same_path_is_a_no_op(self, tmp_path):
        a = str(tmp_path / "A")
        ctrl = _controller()
        ctrl.adopt_workspace(a)
        ctrl.widget.output_file_path.sets.clear()
        ctrl.adopt_workspace(a, previous=a)
        assert ctrl.widget.output_file_path.sets == []

    def test_previous_matches_across_case_and_slashes(self, tmp_path):
        a = str(tmp_path / "A")
        b = str(tmp_path / "B")
        ctrl = _controller()
        ctrl.adopt_workspace(a)
        ctrl.adopt_workspace(b, previous=a.replace("\\", "/"))
        assert ctrl.widget.output_file_path.text() == b

    def test_re_adopt_seeds_the_migration_flag_for_the_new_workspace(
            self, tmp_path):
        a, b = str(tmp_path / "A"), str(tmp_path / "B")
        ctrl = _controller()
        ctrl.adopt_workspace(a)
        ctrl.adopt_workspace(b, previous=a)
        assert _key(b) in ctrl._step1_migration_offered


class TestBrowsePublish:
    def test_browse_publishes_with_the_step_id(self, tmp_path):
        published = []

        class _Shell:
            def set_workspace(self, path, source_step=""):
                published.append((path, source_step))

        ctrl = _controller()
        ctrl._workflow_shell = _Shell()
        ctrl.on_workspace_browsed(str(tmp_path))
        # Browse now writes the box itself (before the publish, so the shell
        # cannot see Step 1 as diverged from itself), and THIS fake
        # deliberately routes its `setText` notification at the publish
        # handler to prove adopt cannot leak — so the hostile wiring fires
        # once alongside the real publish.  The app has no
        # `textChanged -> publish` connection, so there is no double publish
        # in the product; what matters is that every publish carries the new
        # path and this step's id.
        assert published
        assert set(published) == {(str(tmp_path), "project_setup")}

    def test_publish_without_a_shell_does_not_raise(self, tmp_path):
        ctrl = _controller()
        ctrl.on_workspace_browsed(str(tmp_path))

    def test_publish_is_suppressed_while_suppressed(self, tmp_path):
        published = []

        class _Shell:
            def set_workspace(self, path, source_step=""):
                published.append(path)

        ctrl = _controller()
        ctrl._workflow_shell = _Shell()
        ctrl._suppress_publish = True
        ctrl.on_workspace_browsed(str(tmp_path))
        assert published == []
