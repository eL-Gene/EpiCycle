"""Brick R1 -- the pure guide-mask loader.

Pins ``napari_epicycle/annotator/reshape_guide.py``: which guide sources are listed.

No Qt, no napari viewer.  ``list_guide_sources`` does lazily import the Step-3
path/run helpers, which live under ``napari_epicycle`` (whose package
``__init__`` imports napari) -- that is unavoidable and is exactly why the
module's own imports stay lazy.  The purity of the MODULE ITSELF is proven by
``test_module_imports_without_napari_or_qt``, which loads it by file path in a
subprocess, bypassing the package ``__init__``.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import tifffile

from napari_epicycle.annotator import reshape_guide as rg


REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    REPO_ROOT / "napari_epicycle" / "annotator" / "reshape_guide.py"
)


# --------------------------------------------------------------------------- #
# workspace fabrication
# --------------------------------------------------------------------------- #
PATCH = "patch_001"


def _write_labels(path: Path, shape=(4, 6, 6), dtype=np.int32) -> np.ndarray:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(shape, dtype=dtype)
    arr[1:3, 1:3, 1:3] = 1
    arr[1:3, 4:6, 4:6] = 2
    tifffile.imwrite(str(path), arr)
    return arr


def _make_workspace(tmp_path: Path) -> Path:
    """A workspace with:

    * ``default`` -- FLAT layout, reconstruction only,
    * ``nuclei_v2``  -- named run with BOTH curated and reconstruction,
    * ``old_run``    -- named run, archived,
    * ``recon_only`` -- named run, reconstruction only.
    """
    ws = tmp_path / "ws"
    step3 = ws / "artifacts" / "step3"

    # default run = flat: runs/{patch}/labels_3d.tif
    _write_labels(step3 / "runs" / PATCH / "labels_3d.tif")

    for slug, archived in (
        ("nuclei_v2", False),
        ("old_run", True),
        ("recon_only", False),
    ):
        run_root = step3 / "runs" / slug
        run_root.mkdir(parents=True, exist_ok=True)
        (run_root / "run_manifest.json").write_text(
            json.dumps(
                {
                    "run_id": slug,
                    "display_name": slug.replace("_", " ").title(),
                    "target": "nuclei" if slug == "nuclei_v2" else "",
                    "archived_at": "2026-01-01T00:00:00Z" if archived else None,
                    "backend": "cellpose",
                }
            ),
            encoding="utf-8",
        )
        _write_labels(run_root / "reconstructions" / PATCH / "labels_3d.tif")

    # nuclei_v2 + old_run are curated; recon_only is not.
    for slug in ("nuclei_v2", "old_run"):
        _write_labels(
            step3 / "runs" / slug / "curation" / PATCH / "labels_curated.tif"
        )

    state = {
        "schema_version": 3,
        "active_run_id": "default",
        "runs": {
            "default": {
                "display_name": "Default",
                "patches": {PATCH: {"status": "reconstructed"}},
            },
            "nuclei_v2": {
                "display_name": "Nuclei V2",
                "patches": {PATCH: {"status": "curating"}},
            },
            "old_run": {"patches": {PATCH: {"status": "accepted"}}},
            "recon_only": {"patches": {PATCH: {"status": "reconstructed"}}},
        },
    }
    (ws / "artifacts" / "step3" / "step3_state.json").write_text(
        json.dumps(state), encoding="utf-8"
    )
    # state_json_path lives at the workspace root under the v2 layout; write
    # both so the helper finds it wherever it points.
    from napari_epicycle.fusion_reconstruction import _paths as step3_paths

    real = Path(step3_paths.state_json_path(ws))
    real.parent.mkdir(parents=True, exist_ok=True)
    real.write_text(json.dumps(state), encoding="utf-8")
    return ws


def _by_run(sources):
    return {s.run_id: s for s in sources}


# --------------------------------------------------------------------------- #
# list_guide_sources
# --------------------------------------------------------------------------- #
def test_lists_runs_that_have_labels_for_this_patch(tmp_path):
    ws = _make_workspace(tmp_path)
    sources = rg.list_guide_sources(ws, PATCH)
    ids = set(_by_run(sources))
    assert "default" in ids
    assert "nuclei_v2" in ids
    assert "recon_only" in ids


def test_curated_file_is_preferred_over_the_reconstruction(tmp_path):
    ws = _make_workspace(tmp_path)
    src = _by_run(rg.list_guide_sources(ws, PATCH))["nuclei_v2"]
    assert src.which_file == "labels_curated.tif"
    assert src.path.endswith("labels_curated.tif")


def test_reconstruction_is_used_when_there_is_no_curated_file(tmp_path):
    ws = _make_workspace(tmp_path)
    src = _by_run(rg.list_guide_sources(ws, PATCH))["recon_only"]
    assert src.which_file == "labels_3d.tif"
    assert src.path.endswith("labels_3d.tif")


def test_default_run_resolves_through_the_flat_layout(tmp_path):
    ws = _make_workspace(tmp_path)
    src = _by_run(rg.list_guide_sources(ws, PATCH))["default"]
    # flat == runs/{patch}/labels_3d.tif, NOT runs/default/reconstructions/...
    assert Path(src.path) == (
        tmp_path / "ws" / "artifacts" / "step3" / "runs" / PATCH / "labels_3d.tif"
    )


def test_archived_runs_are_skipped(tmp_path):
    ws = _make_workspace(tmp_path)
    assert "old_run" not in _by_run(rg.list_guide_sources(ws, PATCH))


def test_exclude_run_id_is_skipped(tmp_path):
    ws = _make_workspace(tmp_path)
    ids = set(_by_run(rg.list_guide_sources(ws, PATCH, exclude_run_id="nuclei_v2")))
    assert "nuclei_v2" not in ids
    assert "default" in ids


def test_label_carries_status_target_file_and_the_trusting_sentence(tmp_path):
    ws = _make_workspace(tmp_path)
    src = _by_run(rg.list_guide_sources(ws, PATCH))["nuclei_v2"]
    assert "curating" in src.label
    assert "labels_curated.tif" in src.label
    assert "nuclei" in src.label
    assert src.label.endswith("you are trusting these labels.")


def test_label_says_target_not_set_when_the_manifest_has_none(tmp_path):
    ws = _make_workspace(tmp_path)
    src = _by_run(rg.list_guide_sources(ws, PATCH))["recon_only"]
    assert "target not set" in src.label
    assert src.label.endswith("you are trusting these labels.")


def test_a_corrupt_run_is_skipped_not_raised(tmp_path):
    ws = _make_workspace(tmp_path)
    manifest = (
        ws / "artifacts" / "step3" / "runs" / "nuclei_v2" / "run_manifest.json"
    )
    manifest.write_text("{ this is not json", encoding="utf-8")
    sources = rg.list_guide_sources(ws, PATCH)  # must not raise
    assert "default" in _by_run(sources)


def test_corrupt_state_json_leaves_status_blank_but_still_lists(tmp_path):
    ws = _make_workspace(tmp_path)
    from napari_epicycle.fusion_reconstruction import _paths as step3_paths

    Path(step3_paths.state_json_path(ws)).write_text("not json", encoding="utf-8")
    src = _by_run(rg.list_guide_sources(ws, PATCH))["recon_only"]
    assert src.status == ""
    assert "no status here" in src.label


def test_list_guide_sources_never_raises_on_garbage(tmp_path):
    assert rg.list_guide_sources(None, PATCH) == []
    assert rg.list_guide_sources(tmp_path, "") == []
    assert rg.list_guide_sources(tmp_path / "nope", PATCH) == []


def test_a_patch_with_no_labels_anywhere_lists_nothing(tmp_path):
    ws = _make_workspace(tmp_path)
    assert rg.list_guide_sources(ws, "patch_999") == []


# --------------------------------------------------------------------------- #
# load_guide_array
# --------------------------------------------------------------------------- #
def test_load_refuses_a_missing_file_in_plain_words(tmp_path):
    with pytest.raises(rg.GuideRefused) as exc:
        rg.load_guide_array(tmp_path / "gone.tif")
    text = str(exc.value)
    assert "no file at" in text
    assert "pick the mask file again" in text.lower()


def test_load_refuses_an_empty_path():
    with pytest.raises(rg.GuideRefused):
        rg.load_guide_array("")


def test_load_refuses_a_float_tiff_with_plain_wording(tmp_path):
    path = tmp_path / "float.tif"
    tifffile.imwrite(str(path), np.zeros((3, 4, 4), dtype=np.float32))
    with pytest.raises(rg.GuideRefused) as exc:
        rg.load_guide_array(path)
    text = str(exc.value)
    assert "isn't a label mask" in text
    assert "decimal numbers" in text
    assert "dtype" not in text.lower()


def test_load_refuses_a_bool_tiff(tmp_path):
    path = tmp_path / "mask.tif"
    tifffile.imwrite(str(path), np.zeros((3, 4, 4), dtype=bool))
    with pytest.raises(rg.GuideRefused) as exc:
        rg.load_guide_array(path)
    assert "yes/no values" in str(exc.value)


def test_load_refuses_a_2d_tiff_with_plain_wording(tmp_path):
    path = tmp_path / "flat.tif"
    tifffile.imwrite(str(path), np.zeros((8, 8), dtype=np.int32))
    with pytest.raises(rg.GuideRefused) as exc:
        rg.load_guide_array(path)
    text = str(exc.value)
    assert "3D stack" in text
    assert "shape" not in text.lower()


def test_load_refuses_an_unreadable_file(tmp_path):
    path = tmp_path / "junk.tif"
    path.write_bytes(b"not a tiff at all")
    with pytest.raises(rg.GuideRefused) as exc:
        rg.load_guide_array(path)
    assert "could not be opened" in str(exc.value)


def test_load_squeezes_size_one_axes(tmp_path):
    path = tmp_path / "guide.tif"
    tifffile.imwrite(str(path), np.zeros((1, 4, 6, 6), dtype=np.int32))
    array = rg.load_guide_array(path)
    assert array.shape == (4, 6, 6)
    assert array.dtype.kind in ("i", "u")


def test_load_reads_a_real_label_mask(tmp_path):
    path = tmp_path / "guide.tif"
    expected = _write_labels(path)
    array = rg.load_guide_array(path)
    assert np.array_equal(array, expected)


# --------------------------------------------------------------------------- #
# validate_and_match_guide
# --------------------------------------------------------------------------- #
def test_matching_guide_passes_through_unchanged():
    guide = np.zeros((4, 6, 6), dtype=np.int32)
    guide[1, 1, 1] = 7
    matched, notes = rg.validate_and_match_guide(guide, (4, 6, 6))
    assert np.array_equal(matched, guide)
    assert matched.flags["C_CONTIGUOUS"]
    assert any("MB" in n for n in notes)


def test_notes_name_the_final_kind_and_megabytes():
    guide = np.zeros((4, 6, 6), dtype=np.int32)
    guide[0, 0, 0] = 40000            # too large for the small kind
    _, notes = rg.validate_and_match_guide(guide, (4, 6, 6))
    assert any("int32" in n and "MB" in n for n in notes)


def test_two_times_guide_is_majority_downsampled_in_a_two_times_session():
    rng = np.random.default_rng(0)
    guide = rng.integers(0, 4, size=(8, 12, 12)).astype(np.int32)
    matched, notes = rg.validate_and_match_guide(
        guide, (4, 6, 6), downsample_factor=2
    )
    assert matched.shape == (4, 6, 6)
    # majority pooling never invents an id
    assert set(np.unique(matched)).issubset(set(np.unique(guide)))
    assert any("half size" in n for n in notes)
    assert any("Small nuclei may merge" in n for n in notes)
    assert any("QC Settings" in n for n in notes)


def test_downsample_matching_never_touches_the_file_on_disk(tmp_path):
    path = tmp_path / "guide.tif"
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 4, size=(8, 12, 12)).astype(np.int32)
    tifffile.imwrite(str(path), arr)
    before = hashlib.sha256(path.read_bytes()).hexdigest()

    guide = rg.load_guide_array(path)
    matched, _ = rg.validate_and_match_guide(guide, (4, 6, 6), downsample_factor=2)

    after = hashlib.sha256(path.read_bytes()).hexdigest()
    assert before == after, "the guide file on disk must never be rewritten"
    assert matched.shape == (4, 6, 6)
    # and the original in-memory array is untouched too
    assert np.array_equal(rg.load_guide_array(path), arr)


def test_downsample_tolerates_the_floor_rounding_downsample_labels_uses():
    # 9 // 2 == 4 -- the helper drops the trailing remainder, so a 9-deep
    # guide legitimately matches a 4-deep loaded volume.
    guide = np.ones((9, 13, 13), dtype=np.int32)
    matched, _ = rg.validate_and_match_guide(
        guide, (4, 6, 6), downsample_factor=2
    )
    assert matched.shape == (4, 6, 6)


def test_a_guide_already_at_session_size_is_not_downsampled_again():
    guide = np.ones((4, 6, 6), dtype=np.int32)
    matched, notes = rg.validate_and_match_guide(
        guide, (4, 6, 6), downsample_factor=2
    )
    assert matched.shape == (4, 6, 6)
    assert not any("half size" in n for n in notes)


def test_mismatch_message_names_downsampling_when_the_factor_is_two():
    guide = np.ones((5, 7, 9), dtype=np.int32)
    with pytest.raises(rg.GuideRefused) as exc:
        rg.validate_and_match_guide(guide, (4, 6, 6), downsample_factor=2)
    text = str(exc.value)
    assert "downsampling is on" in text
    assert "QC Settings" in text
    assert "reload the patch" in text
    assert "5 x 7 x 9" in text and "4 x 6 x 6" in text


def test_mismatch_message_does_not_name_downsampling_at_factor_one():
    guide = np.ones((5, 7, 9), dtype=np.int32)
    with pytest.raises(rg.GuideRefused) as exc:
        rg.validate_and_match_guide(guide, (4, 6, 6))
    text = str(exc.value)
    assert "downsampl" not in text.lower()
    assert "5 x 7 x 9" in text and "4 x 6 x 6" in text
    assert "different size from this patch" in text


def test_validate_refuses_a_non_integer_array():
    with pytest.raises(rg.GuideRefused):
        rg.validate_and_match_guide(np.zeros((4, 6, 6), np.float32), (4, 6, 6))


def test_validate_refuses_a_non_array():
    with pytest.raises(rg.GuideRefused):
        rg.validate_and_match_guide([[1, 2], [3, 4]], (4, 6, 6))


def test_validate_refuses_a_two_dimensional_target():
    with pytest.raises(rg.GuideRefused):
        rg.validate_and_match_guide(np.zeros((4, 6, 6), np.int32), (6, 6))


# --- byte budget ----------------------------------------------------------- #
def test_over_budget_downcasts_to_small_whole_numbers_when_ids_allow():
    guide = np.zeros((10, 40, 40), dtype=np.int32)  # 64000 bytes
    guide[0, 0, 0] = 30000  # fits in int16
    matched, notes = rg.validate_and_match_guide(
        guide, (10, 40, 40), byte_budget=40_000
    )
    assert matched.dtype == np.int16
    assert matched.nbytes == 32_000
    assert int(matched.max()) == 30000
    assert any("smaller whole numbers" in n for n in notes)
    assert any("int16" in n for n in notes)


def test_over_budget_refuses_with_a_megabyte_number_when_ids_are_too_big():
    guide = np.zeros((10, 40, 40), dtype=np.int32)
    guide[0, 0, 0] = 40000  # does NOT fit in int16
    with pytest.raises(rg.GuideRefused) as exc:
        rg.validate_and_match_guide(guide, (10, 40, 40), byte_budget=40_000)
    text = str(exc.value)
    assert "MB" in text
    assert "0.1 MB" in text  # 64000 bytes
    assert "QC Settings" in text


def test_over_budget_still_refuses_when_the_downcast_is_not_enough():
    guide = np.zeros((10, 40, 40), dtype=np.int32)
    guide[0, 0, 0] = 100  # would fit int16, but 32000 bytes is still too big
    with pytest.raises(rg.GuideRefused) as exc:
        rg.validate_and_match_guide(guide, (10, 40, 40), byte_budget=1000)
    assert "MB" in str(exc.value)


def test_small_ids_are_downcast_even_when_the_mask_is_under_budget():
    """The guide is held for the whole session AND handed to napari as a
    layer; that layer add is main-thread work proportional to its size, so
    halving it halves the only part of the load that can stall the window
    (AUDIT 2026-09-09).  The ids are unchanged."""
    guide = np.zeros((4, 6, 6), dtype=np.int32)
    guide[0, 0, 0] = 32767
    matched, notes = rg.validate_and_match_guide(guide, (4, 6, 6))
    assert matched.dtype == np.int16
    assert int(matched.max()) == 32767, "the ids must survive the downcast"
    assert any("smaller whole numbers" in n for n in notes)


def test_ids_too_large_for_the_small_kind_keep_the_wide_one():
    guide = np.zeros((4, 6, 6), dtype=np.int32)
    guide[0, 0, 0] = 40000
    matched, _ = rg.validate_and_match_guide(guide, (4, 6, 6))
    assert matched.dtype == np.int32
    assert int(matched.max()) == 40000


def test_default_byte_budget_is_512_mb():
    assert rg.GUIDE_BYTE_BUDGET == 512 * 1024 * 1024


# --------------------------------------------------------------------------- #
# seed_count / guide_layer_name
# --------------------------------------------------------------------------- #
def test_seed_count_counts_distinct_non_zero_ids():
    roi = np.array([[[0, 1, 1], [2, 2, 0]], [[0, 0, 5], [5, 1, 0]]], dtype=np.int32)
    assert rg.seed_count(roi) == 3


def test_seed_count_is_zero_for_empty_and_garbage():
    assert rg.seed_count(np.zeros((3, 3, 3), np.int32)) == 0
    assert rg.seed_count(np.array([], dtype=np.int32)) == 0
    assert rg.seed_count(None) == 0


def test_guide_layer_name_starts_with_the_guide_prefix_for_a_run():
    src = rg.GuideSource(
        kind="run",
        run_id="nuclei_v2",
        display_name="Nuclei V2",
        target="nuclei",
        status="curating",
        path="/ws/labels_curated.tif",
        which_file="labels_curated.tif",
        label="x",
    )
    name = rg.guide_layer_name("patch_001", src)
    assert name.startswith("guide_")
    assert name == "guide_nuclei_v2_patch_001"


def test_guide_layer_name_uses_the_file_stem_for_a_file_source(tmp_path):
    src = rg.guide_source_for_file(tmp_path / "my nuclei mask.tif")
    name = rg.guide_layer_name("patch_001", src)
    assert name.startswith("guide_")
    assert "my_nuclei_mask" in name


def test_guide_layer_name_sanitises_and_never_loses_the_prefix():
    weird = rg.GuideSource(
        kind="run",
        run_id="a b/c:d*",
        display_name="",
        target="",
        status="",
        path="",
        which_file="",
        label="",
    )
    name = rg.guide_layer_name("patch #1", weird)
    assert name.startswith("guide_")
    assert all(ch.isalnum() or ch in "._-" for ch in name)


def test_guide_layer_name_survives_empty_everything():
    name = rg.guide_layer_name("", object())
    assert name.startswith("guide_")


def test_guide_source_for_file_label_ends_with_the_trusting_sentence(tmp_path):
    src = rg.guide_source_for_file(tmp_path / "nuclei.tif")
    assert src.kind == "file"
    assert src.which_file == "external file"
    assert src.status == ""
    assert src.run_id == ""
    assert src.label.endswith("you are trusting these labels.")


# --------------------------------------------------------------------------- #
# hygiene
# --------------------------------------------------------------------------- #
def test_module_source_is_cp1252_encodable():
    """The Windows console is cp1252; a raising print() is a control-flow bug."""
    text = MODULE_PATH.read_text(encoding="utf-8")
    text.encode("cp1252")  # must not raise


def test_module_source_is_pure_ascii():
    text = MODULE_PATH.read_text(encoding="utf-8")
    offenders = sorted({ch for ch in text if ord(ch) > 127})
    assert offenders == [], offenders


def test_module_imports_without_napari_or_qt():
    """Loaded BY FILE PATH so the napari-importing package __init__ is bypassed.

    ``napari_epicycle/__init__.py`` imports ``_widget`` (and therefore napari),
    so a normal ``import napari_epicycle.annotator.reshape_guide`` can never be
    napari-free.  What this test pins is the thing that is actually in our
    control: the module's OWN import graph.
    """
    code = (
        "import sys, importlib.util\n"
        "spec = importlib.util.spec_from_file_location('rg_pure', r'{}')\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "sys.modules['rg_pure'] = mod\n"
        "spec.loader.exec_module(mod)\n"
        "banned = [m for m in ('napari', 'qtpy', 'vispy', 'PyQt5', 'PyQt6',\n"
        "                      'PySide2', 'PySide6', 'magicgui')\n"
        "          if m in sys.modules]\n"
        "assert not banned, banned\n"
        "assert mod.GUIDE_BYTE_BUDGET == 512 * 1024 * 1024\n"
        "assert mod.guide_layer_name('p', object()).startswith('guide_')\n"
    ).format(str(MODULE_PATH))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_public_surface_is_exactly_what_the_orchestrator_expects():
    for name in (
        "GUIDE_BYTE_BUDGET",
        "GuideSource",
        "GuideRefused",
        "list_guide_sources",
        "load_guide_array",
        "validate_and_match_guide",
        "seed_count",
        "guide_layer_name",
    ):
        assert hasattr(rg, name), name
    assert issubclass(rg.GuideRefused, Exception)
    fields = rg.GuideSource.__dataclass_fields__
    assert list(fields) == [
        "kind",
        "run_id",
        "display_name",
        "target",
        "status",
        "path",
        "which_file",
        "label",
    ]


def test_guide_source_is_frozen():
    src = rg.guide_source_for_file("a.tif")
    with pytest.raises(Exception):
        src.kind = "run"


def test_no_user_facing_message_uses_jargon(tmp_path):
    """Every refusal a user can trip is checked for the banned words."""
    banned = ("dtype", "ndim", "shape mismatch", "ValueError", "array")
    messages = []

    for call in (
        lambda: rg.load_guide_array(tmp_path / "missing.tif"),
        lambda: rg.validate_and_match_guide(
            np.ones((5, 7, 9), np.int32), (4, 6, 6)
        ),
        lambda: rg.validate_and_match_guide(
            np.ones((5, 7, 9), np.int32), (4, 6, 6), downsample_factor=2
        ),
        lambda: rg.validate_and_match_guide(
            np.zeros((4, 6, 6), np.float32), (4, 6, 6)
        ),
    ):
        with pytest.raises(rg.GuideRefused) as exc:
            call()
        messages.append(str(exc.value))

    path = tmp_path / "float.tif"
    tifffile.imwrite(str(path), np.zeros((3, 4, 4), dtype=np.float32))
    with pytest.raises(rg.GuideRefused) as exc:
        rg.load_guide_array(path)
    messages.append(str(exc.value))

    for msg in messages:
        low = msg.lower()
        for word in banned:
            assert word.lower() not in low, (word, msg)
        msg.encode("cp1252")
