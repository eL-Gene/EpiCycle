"""Step 2's patch default.

With no patch folder chosen, Step 2 lists (and loads) from the patches Step 1
extracted into the shared workspace, ``artifacts/step1/patches/``.  The rules
this file pins:

* the default applies only when the folder exists **and** holds >= 1 TIFF —
  there is deliberately no workspace-root scan, so a project where Step 1 has
  not extracted yet shows nothing at all;
* a folder the user picked always wins;
* the default is **never persisted**.  ``loaded_stack_path`` keeps meaning
  "the user's explicit choice", so a fallback is re-resolved on the next load
  (after the workspace is copied to another machine, say) rather than frozen
  into the state file as if the user had chosen it;
* every read goes through ``_effective_stack_path()`` — the list scan, the
  load join, the saved-patch reload and the "is a folder set?" guards.

Qt-free: the controller is built with ``__new__`` and stub collaborators.
"""
from __future__ import annotations

import os

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (  # noqa: E501
    TrainingAnnotationController,
)


class _Line:
    def __init__(self, text=""):
        self._text = text

    def text(self):
        return self._text

    def setText(self, value):
        self._text = value


class _Label:
    def __init__(self):
        self.visible = None

    def setVisible(self, v):
        self.visible = bool(v)


class _Item:
    def __init__(self, name):
        self._name = name

    def text(self):
        return self._name

    def data(self, role):
        return self._name


class _PatchList:
    def __init__(self):
        self._items = []

    def count(self):
        return len(self._items)

    def item(self, i):
        return self._items[i]

    def clear(self):
        self._items = []

    def addItem(self, item):
        self._items.append(item)

    def currentItem(self):
        return None

    def selectedItems(self):
        return list(self._items[:1])


class _UI:
    def __init__(self):
        self.load_training_path_patch = _Line()
        self.patch_dir_qualifier = _Label()
        self.train_patch_list = _PatchList()


def _controller(ws="", chosen=""):
    ctrl = TrainingAnnotationController.__new__(TrainingAnnotationController)
    ctrl.ui = _UI()
    ctrl.working_dir = ws
    ctrl.loaded_stack_path = chosen
    ctrl.loaded_patch_name = ""
    ctrl.logs = []
    ctrl._append_ui_log = lambda msg: ctrl.logs.append(msg)
    ctrl._refresh_patch_list_badges = lambda: None
    ctrl._refresh_sample_queue = lambda: None
    ctrl._clean_patch_filename = lambda item: item.text()
    return ctrl


def _patches(tmp_path, names=("p1.tiff", "p2.tiff")):
    d = tmp_path / "artifacts" / "step1" / "patches"
    d.mkdir(parents=True)
    for n in names:
        (d / n).write_bytes(b"II*\x00")
    return d


class TestEffectiveStackPath:
    def test_default_is_used_when_nothing_is_chosen(self, tmp_path):
        d = _patches(tmp_path)
        ctrl = _controller(str(tmp_path))
        assert ctrl._effective_stack_path() == str(d)

    def test_the_users_choice_wins(self, tmp_path):
        _patches(tmp_path)
        elsewhere = tmp_path / "elsewhere"
        elsewhere.mkdir()
        ctrl = _controller(str(tmp_path), chosen=str(elsewhere))
        assert ctrl._effective_stack_path() == str(elsewhere)

    def test_no_default_when_the_folder_is_missing(self, tmp_path):
        ctrl = _controller(str(tmp_path))
        assert ctrl._effective_stack_path() == ""

    def test_no_default_when_the_folder_is_empty(self, tmp_path):
        (tmp_path / "artifacts" / "step1" / "patches").mkdir(parents=True)
        ctrl = _controller(str(tmp_path))
        assert ctrl._effective_stack_path() == ""

    def test_a_non_tiff_file_does_not_count_as_a_patch(self, tmp_path):
        d = tmp_path / "artifacts" / "step1" / "patches"
        d.mkdir(parents=True)
        (d / "notes.txt").write_text("x")
        ctrl = _controller(str(tmp_path))
        assert ctrl._effective_stack_path() == ""

    def test_a_tif_suffix_counts_too(self, tmp_path):
        d = tmp_path / "artifacts" / "step1" / "patches"
        d.mkdir(parents=True)
        (d / "one.TIF").write_bytes(b"II*\x00")
        ctrl = _controller(str(tmp_path))
        assert ctrl._effective_stack_path() == str(d)

    def test_no_workspace_means_no_default(self):
        ctrl = _controller("")
        assert ctrl._effective_stack_path() == ""

    def test_there_is_no_workspace_root_scan(self, tmp_path):
        """A TIFF loose in the workspace root is NOT a patch source."""
        (tmp_path / "stray.tiff").write_bytes(b"II*\x00")
        ctrl = _controller(str(tmp_path))
        assert ctrl._effective_stack_path() == ""


class TestTheDefaultIsNeverPersisted:
    def test_listing_from_the_default_leaves_loaded_stack_path_empty(
            self, tmp_path):
        _patches(tmp_path)
        ctrl = _controller(str(tmp_path))
        ctrl._populate_patch_list_from_disk()
        assert ctrl.loaded_stack_path == "", (
            "the default must not masquerade as a user choice")

    def test_the_saved_state_records_no_path_when_only_the_default_applies(
            self, tmp_path):
        """The field the state file writes stays empty (`:10877`)."""
        _patches(tmp_path)
        ctrl = _controller(str(tmp_path))
        ctrl._populate_patch_list_from_disk()
        ctrl._auto_reload_saved_patch = lambda: None
        assert str(getattr(ctrl, "loaded_stack_path", "") or "") == ""

    def test_a_browsed_folder_still_wins_and_is_kept(self, tmp_path):
        _patches(tmp_path)
        mine = tmp_path / "mine"
        mine.mkdir()
        ctrl = _controller(str(tmp_path))
        ctrl.loaded_stack_path = str(mine)
        assert ctrl._effective_stack_path() == str(mine)
        ctrl._populate_patch_list_from_disk()
        assert ctrl.loaded_stack_path == str(mine)


class TestTheListAndTheLoadJoinBothUseIt:
    def test_the_list_populates_from_the_default(self, tmp_path):
        d = _patches(tmp_path)
        ctrl = _controller(str(tmp_path))
        ctrl._populate_patch_list_from_disk()
        assert ctrl.ui.train_patch_list.count() == 2
        assert ctrl.ui.load_training_path_patch.text() == str(d)

    def test_one_log_line_when_the_default_kicks_in(self, tmp_path, capsys):
        _patches(tmp_path)
        ctrl = _controller(str(tmp_path))
        ctrl._populate_patch_list_from_disk()
        ctrl._populate_patch_list_from_disk()      # idempotent, still one line
        out = capsys.readouterr().out
        assert out.count("[WORKSPACE] Step 2 patch folder defaulted") == 1, out

    def test_no_log_line_when_there_is_no_default(self, tmp_path, capsys):
        ctrl = _controller(str(tmp_path))
        ctrl._populate_patch_list_from_disk()
        assert "defaulted" not in capsys.readouterr().out

    def test_the_qualifier_shows_only_when_defaulted(self, tmp_path):
        _patches(tmp_path)
        ctrl = _controller(str(tmp_path))
        ctrl._populate_patch_list_from_disk()
        assert ctrl.ui.patch_dir_qualifier.visible is True

    def test_the_qualifier_hides_for_a_chosen_folder(self, tmp_path):
        d = _patches(tmp_path)
        ctrl = _controller(str(tmp_path), chosen=str(d))
        ctrl._populate_patch_list_from_disk()
        assert ctrl.ui.patch_dir_qualifier.visible is False

    def test_nothing_at_all_when_step1_has_not_extracted(self, tmp_path):
        ctrl = _controller(str(tmp_path))
        ctrl._populate_patch_list_from_disk()
        assert ctrl.ui.train_patch_list.count() == 0
        assert ctrl.ui.patch_dir_qualifier.visible is False

    def test_the_load_join_goes_through_the_effective_path(self):
        """`_load_patch_prelude` builds `file_path` from the effective folder.

        Structural rather than textual: the name must appear among the
        function's referenced attributes, so wiring it back to the raw
        `loaded_stack_path` fails here.
        """
        code = TrainingAnnotationController._load_patch_prelude.__code__
        assert "_effective_stack_path" in code.co_names


class TestGuardsSeeTheDefault:
    def test_the_click_guard_passes_with_only_the_default(self, tmp_path):
        _patches(tmp_path)
        ctrl = _controller(str(tmp_path))
        assert bool(ctrl._effective_stack_path()) is True

    def test_the_saved_patch_reload_uses_the_default(self, tmp_path):
        d = _patches(tmp_path, names=("saved.tiff",))
        ctrl = _controller(str(tmp_path))
        ctrl.loaded_patch_name = "saved"
        loaded = []
        ctrl._select_patch_item = lambda name: (loaded.append(name), True)[1]
        ctrl._load_patch = lambda *a, **k: loaded.append("LOAD")
        ctrl._auto_reload_saved_patch()
        assert "LOAD" in loaded, (
            f"the saved patch under {d} should have reloaded; got {loaded}")

    def test_the_reload_still_no_ops_without_any_folder(self, tmp_path):
        ctrl = _controller(str(tmp_path))
        ctrl.loaded_patch_name = "saved"
        loaded = []
        ctrl._select_patch_item = lambda name: (loaded.append(name), True)[1]
        ctrl._load_patch = lambda *a, **k: loaded.append("LOAD")
        ctrl._auto_reload_saved_patch()
        assert loaded == []


class TestAChosenFolderThatIsGoneFallsBack:
    """Audit note 1 — a chosen folder that no longer exists must not win.

    Step 3 re-resolves its folder on every load; Step 2 used to hand back a
    dead path forever, which is exactly the wrong answer after a workspace is
    copied to another machine (the old absolute path travels in the state
    file) or after the folder is deleted.  Falling through to the Step 1
    default makes the two steps behave the same way.
    """

    def test_a_missing_chosen_folder_falls_through_to_the_default(self, tmp_path):
        d = _patches(tmp_path)
        ctrl = _controller(str(tmp_path), chosen="C:/gone/patches")
        assert ctrl._effective_stack_path() == str(d)

    def test_the_dead_choice_is_still_REMEMBERED(self, tmp_path):
        """Falling back is a read-time decision; nothing is overwritten, so
        re-attaching the drive brings the user's folder back."""
        _patches(tmp_path)
        ctrl = _controller(str(tmp_path), chosen="C:/gone/patches")
        ctrl._effective_stack_path()
        assert ctrl.loaded_stack_path == "C:/gone/patches"

    def test_a_missing_choice_with_no_default_gives_nothing(self, tmp_path):
        ctrl = _controller(str(tmp_path), chosen="C:/gone/patches")
        assert ctrl._effective_stack_path() == ""

    def test_the_list_shows_the_default_and_says_so(self, tmp_path):
        _patches(tmp_path)
        ctrl = _controller(str(tmp_path), chosen="C:/gone/patches")
        ctrl._populate_patch_list_from_disk()
        assert ctrl.ui.train_patch_list.count() == 2
        assert ctrl.ui.patch_dir_qualifier.visible is True

    def test_a_live_choice_still_wins(self, tmp_path):
        d = _patches(tmp_path)
        mine = tmp_path / "mine"
        mine.mkdir()
        ctrl = _controller(str(tmp_path), chosen=str(mine))
        assert ctrl._effective_stack_path() == str(mine)
        assert ctrl._effective_stack_path() != str(d)

    def test_a_dead_folder_with_no_fallback_still_says_so(self, tmp_path):
        """Losing the folder must not fail silently — the pre-B3 'not found
        at' line has to survive the fallback."""
        ctrl = _controller(str(tmp_path), chosen=str(tmp_path / "vanished"))
        ctrl.loaded_patch_name = "saved"
        ctrl._select_patch_item = lambda name: True
        ctrl._load_patch = lambda *a, **k: ctrl.logs.append("LOAD")
        ctrl._auto_reload_saved_patch()
        assert "LOAD" not in ctrl.logs
        assert any("not found at" in m for m in ctrl.logs), ctrl.logs
