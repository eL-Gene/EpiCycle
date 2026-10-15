"""Tests for Step 1 canonical path helpers.

Pure path-construction tests — no Qt, no napari, no filesystem mutation.
"""
import os
import tempfile
from pathlib import Path

from napari_epicycle.data_preparation._paths import (
    step1_artifacts_dir,
    step1_patches_dir,
    step1_state_path,
)


class TestStep1Paths:
    def test_patches_dir(self):
        ws = os.path.join("some", "workspace")
        expected = Path(ws) / "artifacts" / "step1" / "patches"
        assert step1_patches_dir(ws) == expected

    def test_artifacts_dir(self):
        ws = os.path.join("some", "workspace")
        expected = Path(ws) / "artifacts" / "step1"
        assert step1_artifacts_dir(ws) == expected

    def test_state_path(self):
        ws = os.path.join("some", "workspace")
        expected = Path(ws) / "step1_state.json"
        assert step1_state_path(ws) == expected

    def test_accepts_str_and_path(self):
        ws_str = os.path.join("a", "b")
        ws_path = Path(ws_str)
        assert step1_patches_dir(ws_str) == step1_patches_dir(ws_path)
        assert step1_state_path(ws_str) == step1_state_path(ws_path)

    def test_returns_path_objects(self):
        ws = os.path.join("a", "b")
        assert isinstance(step1_patches_dir(ws), Path)
        assert isinstance(step1_artifacts_dir(ws), Path)
        assert isinstance(step1_state_path(ws), Path)

    def test_pure_no_filesystem_mutation(self):
        """Helpers must not create any directories or files."""
        with tempfile.TemporaryDirectory() as ws:
            step1_artifacts_dir(ws)
            step1_patches_dir(ws)
            step1_state_path(ws)
            # Nothing should have been created under the workspace.
            assert os.listdir(ws) == []

    def test_patches_dir_nested_under_artifacts(self):
        ws = os.path.join("a", "b")
        assert step1_artifacts_dir(ws) in step1_patches_dir(ws).parents
