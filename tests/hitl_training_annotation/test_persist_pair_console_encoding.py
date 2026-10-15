r"""Regression test for the real 2026-09-07 incident: a console-encoding
raise silently redirected accepted training pairs to the LEGACY folder.

``_persist_curated_training_pair`` resolves ``out_dir`` inside a ``try``
whose ``except Exception`` falls back to the legacy
``{ws}/training_data/round_NN/accepted_pairs/{spatial,axial}`` path.  The
"NEW layout" diagnostic ``print`` sat INSIDE that ``try`` and contained a
"→" (U+2192).  On the user's Windows box, with stdout on cp1252, that
print raised ``UnicodeEncodeError`` (a ``ValueError`` subclass) — the
``except Exception`` caught it and two accepted pairs landed in
``training_data/`` on a v3 (new-layout) workspace.

This test installs a cp1252 stdout, runs the real persist path against a
v3 workspace, and asserts the pair lands under
``artifacts/step2/runs/<run>/lateral/round_NN/accepted/`` with NO
``training_data/`` folder created.

Pre-fix this FAILS (pair goes to ``training_data/``).  It stays green
regardless of the global stdout guard in ``napari_epicycle/__init__.py``
because the cp1252 stream is installed AFTER import — so it locks the
independent "console strings are ASCII" defense.

The controller class is too heavy to instantiate (Qt + napari), so the
method is invoked unbound against a Qt-free fake ``self`` carrying only
the attributes it reads — the same approach as
``tests/hitl_training_annotation/controller/test_persist_tiff_perf.py``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import types

import numpy as np
import pytest

from napari_epicycle.hitl_training_annotation import _paths as p
from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController,
)


class _FakeSegmentationManager:
    """Minimal stand-in: returns a 3D image stack + a 2D mask."""

    def __init__(self, img, mask):
        self._img = img
        self._mask = mask

    def get_input_stack(self, slice_label):
        return self._img

    def get_masks(self, slice_label):
        return self._mask


def _make_fake_controller(ws):
    """A Qt-free object exposing exactly the attributes the persist path
    reads.  Not a controller subclass — the method is called unbound."""
    img = np.zeros((2, 8, 8), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=np.int32)
    mask[2:5, 2:5] = 1

    fake = types.SimpleNamespace(
        working_dir=str(ws),
        active_run_id="default",
        current_round_by_view={"lateral": 0, "axial": 0},
        sample_records={},
        segmentation_manager=_FakeSegmentationManager(img, mask),
        _persisted_pair_sigs={},
        _pair_writer=None,
    )
    fake._current_round = lambda: 0
    fake._current_round_key = lambda: "lateral"
    return fake


@pytest.fixture
def v3_workspace(tmp_path):
    """A minimal v3 (new-layout) Step 2 workspace: state JSON present with
    a ``runs`` envelope, so ``is_new_layout_state()`` is True."""
    ws = tmp_path / "ws_v3"
    ws.mkdir(parents=True)
    p.state_json_path(ws).write_text(
        json.dumps(
            {
                "schema_version": 3,
                "active_run_id": "default",
                "runs": {
                    "default": {
                        "display_name": "Default",
                        "target": None,
                        "backend": "cellpose",
                        "created_at": "2026-09-07T00:00:00Z",
                        "created_from": None,
                        "current_round": 0,
                        "current_round_by_view": {"lateral": 0, "axial": 0},
                        "training_data_by_round": {},
                        "losses_by_round": {},
                        "model_checkpoint_paths": {},
                        "round_backend_info": {},
                        "sample_records": {},
                        "current_config": {},
                    }
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ws


@contextlib.contextmanager
def cp1252_stdout():
    """Install a strict cp1252 stdout — exactly the user's environment
    when napari's output is redirected to a file on Windows.

    NOTE: this MUST be applied inside the test body, not via a fixture.
    pytest's global capture re-assigns ``sys.stdout`` between the setup
    and call phases, so a fixture-level ``monkeypatch.setattr`` is
    silently clobbered before the test runs.
    """
    stream = io.TextIOWrapper(io.BytesIO(), encoding="cp1252", newline="")
    prior = sys.stdout
    sys.stdout = stream
    try:
        yield stream
    finally:
        sys.stdout = prior


def _assert_stream_rejects_arrows(stream):
    """Sanity guard: the fake stream really is the hostile environment."""
    with pytest.raises(UnicodeEncodeError):
        stream.write("→")


def test_persist_pair_lands_in_new_layout_under_cp1252_stdout(v3_workspace):
    """The accepted pair must land in the v3 run folder even when every
    diagnostic print has to survive a cp1252 stdout."""
    fake = _make_fake_controller(v3_workspace)

    with cp1252_stdout() as stream:
        _assert_stream_rejects_arrows(stream)
        result = TrainingAnnotationController._persist_curated_training_pair(
            fake, "patchA_XY_3", async_write=False
        )

    expected_dir = p.round_accepted_dir(v3_workspace, "default", "lateral", 0)
    assert result, "persist returned no uris"
    img_path = expected_dir / "patchA_XY_z0003_img.tif"
    mask_path = expected_dir / "patchA_XY_z0003_mask.tif"
    assert img_path.is_file(), f"image not written to {expected_dir}"
    assert mask_path.is_file(), f"mask not written to {expected_dir}"

    legacy = v3_workspace / "training_data"
    assert not legacy.exists(), (
        "legacy training_data/ was created on a v3 workspace — the "
        "new-layout resolution fell through its except Exception"
    )


def test_persist_pair_lands_in_new_layout_under_utf8_stdout(v3_workspace):
    """Control: the same call on a UTF-8 stdout must behave identically —
    the fix must not change routing."""
    fake = _make_fake_controller(v3_workspace)

    utf8 = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="")
    prior = sys.stdout
    sys.stdout = utf8
    try:
        TrainingAnnotationController._persist_curated_training_pair(
            fake, "patchA_XY_3", async_write=False
        )
    finally:
        sys.stdout = prior

    expected_dir = p.round_accepted_dir(v3_workspace, "default", "lateral", 0)
    assert (expected_dir / "patchA_XY_z0003_mask.tif").is_file()
    assert not (v3_workspace / "training_data").exists()
