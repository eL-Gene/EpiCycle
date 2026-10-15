"""Tests for ReconstructionStateManager."""

import json
import os
from pathlib import Path

import pytest

from napari_epicycle.fusion_reconstruction.state_manager import (
    ReconstructionStateManager,
    VALID_STATUSES,
)


def test_empty_state(tmp_path):
    mgr = ReconstructionStateManager()
    state = mgr.load(str(tmp_path))
    # Schema 3 == the runs-envelope is now the on-disk format; the flat
    # internal state re-stamps the loaded schema (empty state = 3).
    assert state["schema_version"] == 3
    assert state["patches"] == {}
    assert state["active_patch"] is None


def test_register_patches(tmp_path):
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.register_patches(["a.tif", "b.tif", "c.tif"])
    assert mgr.get_patch_status("a.tif") == "unprocessed"
    assert mgr.get_patch_status("b.tif") == "unprocessed"
    assert mgr.get_patch_status("d.tif") == "unprocessed"  # not registered


def test_set_and_get_status(tmp_path):
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.set_patch_status("patch.tif", "predicted", predicted_at="2026-01-01T00:00:00Z")
    assert mgr.get_patch_status("patch.tif") == "predicted"


def test_invalid_status_raises(tmp_path):
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    with pytest.raises(ValueError, match="Invalid status"):
        mgr.set_patch_status("patch.tif", "bogus")


def test_status_counts(tmp_path):
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.register_patches(["a.tif", "b.tif", "c.tif"])
    mgr.set_patch_status("a.tif", "predicted")
    mgr.set_patch_status("b.tif", "reconstructed")
    counts = mgr.get_status_counts()
    assert counts["predicted"] == 1
    assert counts["reconstructed"] == 1
    assert counts["unprocessed"] == 1


def test_save_and_reload(tmp_path):
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.register_patches(["x.tif"])
    mgr.set_patch_status("x.tif", "finalized")
    mgr.save(str(tmp_path))

    # Reload in a fresh manager
    mgr2 = ReconstructionStateManager()
    mgr2.load(str(tmp_path))
    assert mgr2.get_patch_status("x.tif") == "finalized"


def test_active_patch(tmp_path):
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    assert mgr.active_patch is None
    mgr.active_patch = "foo.tif"
    assert mgr.active_patch == "foo.tif"


def test_corrupt_json_recovery(tmp_path):
    # Post-2026-05-20 cutover: state JSON filename is step3_state.json.
    # Previously this test wrote to "reconstruction_state.json" which
    # state_manager no longer reads — the recovery code path was no
    # longer exercised (the test passed against the empty default).
    state_file = tmp_path / "step3_state.json"
    state_file.write_text("NOT JSON", encoding="utf-8")

    mgr = ReconstructionStateManager()
    state = mgr.load(str(tmp_path))
    assert state["patches"] == {}


def test_all_patch_statuses(tmp_path):
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.register_patches(["a.tif", "b.tif"])
    mgr.set_patch_status("a.tif", "accepted")
    statuses = mgr.get_all_patch_statuses()
    assert statuses["a.tif"] == "accepted"
    assert statuses["b.tif"] == "unprocessed"


def _write_recon_manifest(
    workspace: Path,
    run_id: str,
    source_name: str,
    *,
    mtime: int,
) -> None:
    """Write a reconstruction manifest in the NEW (post-restructure) layout.

    Post-2026-05-20: each patch owns its own ``runs/{patch_name}/`` dir.
    Re-reconstruction overwrites in place; there is no longer a notion
    of multiple historical runs per patch.  ``run_id`` is still recorded
    in the manifest content for backward-compat reading but does not
    appear in the directory path.
    """
    run_dir = workspace / "artifacts" / "step3" / "runs" / source_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": run_id,
                "source_stack_path": str(workspace / "stacks" / source_name),
                "labels_path": str(run_dir / "labels_3d.tif"),
            }
        ),
        encoding="utf-8",
    )
    os.utime(manifest_path, (mtime, mtime))


def test_scan_existing_artifacts_finds_single_manifest_per_patch(tmp_path):
    """Post-restructure each patch has ONE run dir (re-reconstruction
    overwrites in place — Q2).  The previous
    "prefers latest of multiple manifests" test no longer applies."""
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.register_patches(["patch_a.tif"])

    _write_recon_manifest(tmp_path, "20260331T120000Z", "patch_a.tif", mtime=200)

    mgr.scan_existing_artifacts(str(tmp_path))

    record = mgr._state["patches"]["patch_a.tif"]
    assert record["status"] == "reconstructed"
    assert record["reconstruction_run_id"] == "20260331T120000Z"


def test_scan_existing_artifacts_backfills_missing_run_id_without_downgrading_status(tmp_path):
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.register_patches(["patch_a.tif"])
    mgr.set_patch_status("patch_a.tif", "accepted", accepted_at="2026-03-31T00:00:00Z")

    _write_recon_manifest(tmp_path, "20260331T120000Z", "patch_a.tif", mtime=200)

    mgr.scan_existing_artifacts(str(tmp_path))

    record = mgr._state["patches"]["patch_a.tif"]
    assert record["status"] == "accepted"
    assert record["reconstruction_run_id"] == "20260331T120000Z"


# ===========================================================================
# Phase 1b-i — scan_existing_artifacts scans the ACTIVE run's tree
# ===========================================================================

from napari_epicycle.fusion_reconstruction import _paths as _p3


def _write_run_scoped_recon_manifest(workspace, run_id, source_name, *, mtime):
    """Write a recon manifest under ``reconstructions_dir(ws, path_id)``.

    ``path_id=None`` → flat ``runs/{patch}/``; a slug → the named run's
    ``runs/{slug}/reconstructions/{patch}/``.
    """
    recon_root = _p3.reconstructions_dir(workspace, run_id)
    run_dir = recon_root / source_name
    run_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = run_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "run_id": "20260401T000000Z",
                "source_stack_path": str(workspace / "stacks" / source_name),
                "labels_path": str(run_dir / "labels_3d.tif"),
            }
        ),
        encoding="utf-8",
    )
    os.utime(manifest_path, (mtime, mtime))


def test_scan_active_named_run_finds_run_scoped_manifest(tmp_path):
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.register_patches(["patch_a.tif"])
    mgr.active_run_id = "cells"

    # Manifest under the named run's reconstruction tree.
    _write_run_scoped_recon_manifest(tmp_path, "cells", "patch_a.tif", mtime=200)

    mgr.scan_existing_artifacts(str(tmp_path))

    record = mgr._state["patches"]["patch_a.tif"]
    assert record["status"] == "reconstructed"


def test_scan_named_run_ignores_flat_tree(tmp_path):
    """A named run must NOT pick up the flat (default-run) manifests."""
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.register_patches(["patch_a.tif"])
    mgr.active_run_id = "cells"

    # Only a FLAT manifest exists (default run's tree); the named run is empty.
    _write_run_scoped_recon_manifest(tmp_path, None, "patch_a.tif", mtime=200)

    mgr.scan_existing_artifacts(str(tmp_path))

    record = mgr._state["patches"]["patch_a.tif"]
    assert record["status"] == "unprocessed"


def test_scan_default_run_reads_flat_tree(tmp_path):
    """The default run (active_run_id == DEFAULT_RUN_ID) scans flat — the
    byte-identical-to-today path (reconstructions_dir(ws, None) == runs_dir)."""
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.register_patches(["patch_a.tif"])
    # active_run_id defaults to DEFAULT_RUN_ID; assert it stays flat.

    _write_run_scoped_recon_manifest(tmp_path, None, "patch_a.tif", mtime=200)

    mgr.scan_existing_artifacts(str(tmp_path))

    record = mgr._state["patches"]["patch_a.tif"]
    assert record["status"] == "reconstructed"


def test_scan_active_named_run_finds_prediction_cache(tmp_path):
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.register_patches(["patch_a.tif"])
    mgr.active_run_id = "cells"

    cache_dir = _p3.prediction_cache_dir(tmp_path, "cells")
    cache_dir.mkdir(parents=True, exist_ok=True)
    import numpy as _np
    _np.save(str(cache_dir / "patch_a_xy_flows.npy"), _np.zeros((2, 2, 2, 2)))

    mgr.scan_existing_artifacts(str(tmp_path))

    record = mgr._state["patches"]["patch_a.tif"]
    assert record["status"] == "predicted"


# ===========================================================================
# Phase 1a — run-aware envelope (v3 runs-nested on disk; flat internal state)
# ===========================================================================

from napari_epicycle.fusion_reconstruction._migrate_step3_runs import (
    DEFAULT_RUN_ID,
)

STATE_FILENAME = "step3_state.json"


def _read_raw(tmp_path) -> dict:
    """Read the raw on-disk state JSON (the v3 envelope)."""
    return json.loads((Path(tmp_path) / STATE_FILENAME).read_text(encoding="utf-8"))


def _write_v1_flat(tmp_path, **patches_status) -> None:
    """Write a flat schema-v1 state file (the legacy single-context shape)."""
    state = {
        "schema_version": 1,
        "patches": {
            name: {"status": status}
            for name, status in patches_status.items()
        },
        "active_patch": next(iter(patches_status), None),
        "loaded_stack_dir": "/some/stack/dir",
        "downsample_mode": True,
        "last_updated": "2026-04-01T00:00:00+00:00",
    }
    (Path(tmp_path) / STATE_FILENAME).write_text(
        json.dumps(state), encoding="utf-8"
    )


def _write_v3_envelope(tmp_path, runs: dict, active_run_id=DEFAULT_RUN_ID,
                       **workspace_keys) -> None:
    """Write a v3 runs-envelope directly to disk."""
    envelope = {
        "schema_version": 3,
        "active_run_id": active_run_id,
        "runs": runs,
    }
    envelope.update(workspace_keys)
    (Path(tmp_path) / STATE_FILENAME).write_text(
        json.dumps(envelope), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# Upgrade-on-load (flat v1 / v2 → v3)
# ---------------------------------------------------------------------------

def test_v1_flat_loads_and_upgrades_on_save(tmp_path):
    """A flat v1 file loads losslessly; the next save writes schema-3 with
    runs.default.patches populated + workspace keys at top-level."""
    _write_v1_flat(tmp_path, **{"a.tif": "accepted", "b.tif": "reconstructed"})

    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))

    # All patch statuses + workspace keys survived the upgrade.
    assert mgr.get_patch_status("a.tif") == "accepted"
    assert mgr.get_patch_status("b.tif") == "reconstructed"
    assert mgr.get_downsample_mode() is True
    assert mgr.loaded_stack_dir == "/some/stack/dir"

    mgr.save(str(tmp_path))
    raw = _read_raw(tmp_path)

    # On-disk is now the v3 runs-envelope.
    assert raw["schema_version"] == 3
    assert raw["active_run_id"] == DEFAULT_RUN_ID
    assert "runs" in raw and DEFAULT_RUN_ID in raw["runs"]
    default_patches = raw["runs"][DEFAULT_RUN_ID]["patches"]
    assert default_patches["a.tif"]["status"] == "accepted"
    assert default_patches["b.tif"]["status"] == "reconstructed"
    # Workspace-level keys stay top-level (NOT buried under runs.default).
    assert raw["downsample_mode"] is True
    assert raw["loaded_stack_dir"] == "/some/stack/dir"
    assert "patches" not in raw  # patches moved under runs.default


def test_v2_flat_loads_losslessly(tmp_path):
    """The layout-migrated flat-v2 shape (schema_version 2, layout_version 2,
    top-level patches) loads without loss."""
    state = {
        "schema_version": 2,
        "layout_version": 2,
        "patches": {"p.tif": {"status": "curating"}},
        "active_patch": "p.tif",
        "downsample_mode": False,
    }
    (tmp_path / STATE_FILENAME).write_text(json.dumps(state), encoding="utf-8")

    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    assert mgr.get_patch_status("p.tif") == "curating"
    assert mgr.active_patch == "p.tif"

    # layout_version is workspace-level: it survives a save round-trip.
    mgr.save(str(tmp_path))
    raw = _read_raw(tmp_path)
    assert raw["schema_version"] == 3
    assert raw.get("layout_version") == 2
    assert raw["runs"][DEFAULT_RUN_ID]["patches"]["p.tif"]["status"] == "curating"


def test_v3_envelope_loads_active_run_not_empty(tmp_path):
    """A v3 envelope file loads as the active run's flat view — NOT empty."""
    _write_v3_envelope(
        tmp_path,
        runs={
            DEFAULT_RUN_ID: {
                "display_name": "Default",
                "patches": {"a.tif": {"status": "finalized"}},
            },
        },
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    assert mgr.get_patch_status("a.tif") == "finalized"
    # Manifest-only keys must NOT leak into the flat patch view.
    assert "display_name" not in mgr._state


# ---------------------------------------------------------------------------
# No-clobber save
# ---------------------------------------------------------------------------

def test_save_does_not_clobber_other_runs(tmp_path):
    """Saving while active on ``default`` preserves the ``cells`` run
    (patches + manifest fields)."""
    _write_v3_envelope(
        tmp_path,
        runs={
            DEFAULT_RUN_ID: {
                "display_name": "Default",
                "patches": {"a.tif": {"status": "reconstructed"}},
            },
            "cells": {
                "display_name": "Cell membranes",
                "target": "cells",
                "created_from": "model_x",
                "patches": {"b.tif": {"status": "accepted"}},
            },
        },
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))  # active = default
    assert mgr.get_active_run_id() == DEFAULT_RUN_ID

    mgr.set_patch_status("a.tif", "curating")
    mgr.save(str(tmp_path))

    raw = _read_raw(tmp_path)
    # cells run survives intact.
    assert "cells" in raw["runs"]
    assert raw["runs"]["cells"]["patches"]["b.tif"]["status"] == "accepted"
    assert raw["runs"]["cells"]["display_name"] == "Cell membranes"
    assert raw["runs"]["cells"]["target"] == "cells"
    assert raw["runs"]["cells"]["created_from"] == "model_x"
    # default run got the mutation.
    assert raw["runs"][DEFAULT_RUN_ID]["patches"]["a.tif"]["status"] == "curating"


# ---------------------------------------------------------------------------
# switch_active_run
# ---------------------------------------------------------------------------

def test_switch_active_run_swaps_flat_view(tmp_path):
    """Switching runs swaps which run's patches the flat state mirrors."""
    _write_v3_envelope(
        tmp_path,
        runs={
            DEFAULT_RUN_ID: {"patches": {"patchA": {"status": "accepted"}}},
            "cells": {"patches": {"patchB": {"status": "reconstructed"}}},
        },
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))  # default
    assert mgr.get_patch_status("patchA") == "accepted"
    assert mgr.get_patch_status("patchB") == "unprocessed"  # not in default

    mgr.switch_active_run("cells", str(tmp_path))
    assert mgr.get_active_run_id() == "cells"
    assert mgr.get_patch_status("patchB") == "reconstructed"
    assert mgr.get_patch_status("patchA") == "unprocessed"  # not in cells

    mgr.switch_active_run(DEFAULT_RUN_ID, str(tmp_path))
    assert mgr.get_patch_status("patchA") == "accepted"
    assert mgr.get_patch_status("patchB") == "unprocessed"


def test_switch_active_run_flushes_pending_save_no_cross_run_clobber(tmp_path):
    """A pending debounced save is flushed BEFORE the switch mutates
    active_run_id — the cross-run clobber guard.

    (No event loop in unit tests, so the QTimer never actually fires —
    we assert the ORDER directly: ``flush`` must run while
    ``active_run_id`` is still the OLD run, and the current-run state must
    persist to the OLD run's slot, never the new one.)
    """
    _write_v3_envelope(
        tmp_path,
        runs={
            DEFAULT_RUN_ID: {"patches": {"patchA": {"status": "accepted"}}},
            "cells": {"patches": {"patchB": {"status": "reconstructed"}}},
        },
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))  # default

    # Mutate default — schedules a debounced save.
    mgr.set_patch_status("patchA", "curating")

    # Spy: capture active_run_id at the moment flush() runs.  Must still be
    # the OLD run, proving flush happens BEFORE active_run_id is mutated.
    seen = {}
    real_flush = mgr.flush

    def spy_flush():
        seen["active_at_flush"] = mgr.active_run_id
        return real_flush()

    mgr.flush = spy_flush

    mgr.switch_active_run("cells", str(tmp_path))

    assert seen["active_at_flush"] == DEFAULT_RUN_ID

    # The current-run mutation landed in the OLD (default) run, not cells.
    raw = _read_raw(tmp_path)
    assert raw["runs"][DEFAULT_RUN_ID]["patches"]["patchA"]["status"] == "curating"
    assert raw["runs"]["cells"]["patches"]["patchB"]["status"] == "reconstructed"


def test_switch_to_nonexistent_run_starts_empty_no_raise(tmp_path):
    """Switching to a run with no on-disk entry yields empty patches and
    never raises."""
    _write_v3_envelope(
        tmp_path,
        runs={DEFAULT_RUN_ID: {"patches": {"patchA": {"status": "accepted"}}}},
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))

    mgr.switch_active_run("nuclei", str(tmp_path))  # no such run
    assert mgr.get_active_run_id() == "nuclei"
    assert mgr.get_all_patch_statuses() == {}


def test_switch_to_a_run_absent_on_disk_keeps_every_workspace_key(tmp_path):
    """B MED-3: a run with no on-disk entry starts with empty patches but
    keeps EVERY workspace-level key of the envelope (not just
    ``patch_spacing``), so the next save does not write them away."""
    spacing = {"a.tif": {"zyx": [2.0, 0.5, 0.5], "source": "user"}}
    _write_v3_envelope(
        tmp_path,
        runs={DEFAULT_RUN_ID: {"patches": {"a.tif": {"status": "accepted"}}}},
        default_run_nested=True, downsample_mode=True,
        loaded_stack_dir="D:/patches", patch_spacing=spacing,
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.switch_active_run("nuclei", str(tmp_path))     # absent on disk
    assert mgr.get_all_patch_statuses() == {}
    assert mgr._state["default_run_nested"] is True
    assert mgr._state["downsample_mode"] is True
    assert mgr._state["loaded_stack_dir"] == "D:/patches"
    assert mgr._state["patch_spacing"] == spacing
    # the envelope's own keys are not copied into the flat view
    assert mgr._state["schema_version"] == \
        mgr._empty_state()["schema_version"]
    assert "active_run_id" not in mgr._state
    mgr.save(str(tmp_path))
    raw = _read_raw(tmp_path)
    assert raw["default_run_nested"] is True
    assert raw["downsample_mode"] is True
    assert raw["loaded_stack_dir"] == "D:/patches"
    assert raw["patch_spacing"] == spacing
    assert raw["active_run_id"] == "nuclei"


# ---------------------------------------------------------------------------
# list_run_ids / has_run / create_run
# ---------------------------------------------------------------------------

def test_list_run_ids_flat_or_absent_returns_default(tmp_path):
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))  # no file
    assert mgr.list_run_ids(str(tmp_path)) == [DEFAULT_RUN_ID]

    _write_v1_flat(tmp_path, **{"a.tif": "accepted"})
    assert mgr.list_run_ids(str(tmp_path)) == [DEFAULT_RUN_ID]


def test_list_run_ids_envelope_returns_sorted(tmp_path):
    _write_v3_envelope(
        tmp_path,
        runs={
            DEFAULT_RUN_ID: {"patches": {}},
            "cells": {"patches": {}},
            "nuclei": {"patches": {}},
        },
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    assert mgr.list_run_ids(str(tmp_path)) == ["cells", "default", "nuclei"]
    assert mgr.has_run("cells", str(tmp_path)) is True
    assert mgr.has_run("missing", str(tmp_path)) is False


def test_create_run_adds_empty_run_without_switching(tmp_path):
    """create_run adds runs.nuclei (empty patches + manifest) without
    changing active_run_id or clobbering other runs."""
    _write_v3_envelope(
        tmp_path,
        runs={
            DEFAULT_RUN_ID: {"patches": {"a.tif": {"status": "accepted"}}},
            "cells": {
                "display_name": "Cells",
                "patches": {"b.tif": {"status": "reconstructed"}},
            },
        },
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))  # active = default
    assert mgr.get_active_run_id() == DEFAULT_RUN_ID

    mgr.create_run("nuclei", str(tmp_path), display_name="Nuclei")

    # active_run_id unchanged; in-memory state still the default run.
    assert mgr.get_active_run_id() == DEFAULT_RUN_ID
    assert mgr.get_patch_status("a.tif") == "accepted"

    raw = _read_raw(tmp_path)
    assert "nuclei" in raw["runs"]
    assert raw["runs"]["nuclei"]["patches"] == {}
    assert raw["runs"]["nuclei"]["display_name"] == "Nuclei"
    # Other runs survive.
    assert raw["runs"]["cells"]["patches"]["b.tif"]["status"] == "reconstructed"
    assert raw["runs"][DEFAULT_RUN_ID]["patches"]["a.tif"]["status"] == "accepted"


def test_create_run_existing_run_id_is_noop_for_entry(tmp_path):
    """create_run on an existing run id leaves that run's patches intact."""
    _write_v3_envelope(
        tmp_path,
        runs={
            DEFAULT_RUN_ID: {"patches": {}},
            "cells": {"patches": {"b.tif": {"status": "accepted"}}},
        },
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.create_run("cells", str(tmp_path), display_name="Renamed")

    raw = _read_raw(tmp_path)
    # Existing cells run preserved (patches not blanked).
    assert raw["runs"]["cells"]["patches"]["b.tif"]["status"] == "accepted"


# ---------------------------------------------------------------------------
# delete_run (Phase 4b)
# ---------------------------------------------------------------------------

def test_delete_run_drops_entry(tmp_path):
    """delete_run removes the target run entry from the on-disk envelope."""
    _write_v3_envelope(
        tmp_path,
        runs={
            DEFAULT_RUN_ID: {"patches": {"a.tif": {"status": "accepted"}}},
            "cells": {
                "display_name": "Cells",
                "patches": {"b.tif": {"status": "reconstructed"}},
            },
        },
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))  # active = default
    mgr.delete_run("cells", str(tmp_path))

    raw = _read_raw(tmp_path)
    assert "cells" not in raw["runs"]
    # active_run_id pointer untouched.
    assert raw["active_run_id"] == DEFAULT_RUN_ID


def test_delete_run_merge_forward_safe(tmp_path):
    """Deleting one run preserves every OTHER run (patches intact)."""
    _write_v3_envelope(
        tmp_path,
        runs={
            DEFAULT_RUN_ID: {"patches": {"a.tif": {"status": "accepted"}}},
            "cells": {"patches": {"b.tif": {"status": "reconstructed"}}},
            "nuclei": {
                "display_name": "Nuclei",
                "patches": {"c.tif": {"status": "finalized"}},
            },
        },
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))  # active = default
    mgr.delete_run("cells", str(tmp_path))

    raw = _read_raw(tmp_path)
    assert "cells" not in raw["runs"]
    # Sibling runs survive untouched.
    assert raw["runs"]["nuclei"]["patches"]["c.tif"]["status"] == "finalized"
    assert raw["runs"]["nuclei"]["display_name"] == "Nuclei"
    assert raw["runs"][DEFAULT_RUN_ID]["patches"]["a.tif"]["status"] == "accepted"


def test_delete_run_refuses_default(tmp_path):
    """delete_run raises ValueError for the implicit FLAT default run."""
    _write_v3_envelope(
        tmp_path,
        runs={DEFAULT_RUN_ID: {"patches": {}}},
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    with pytest.raises(ValueError):
        mgr.delete_run(DEFAULT_RUN_ID, str(tmp_path))


def test_delete_run_refuses_active(tmp_path):
    """delete_run raises ValueError when the target is the active run."""
    _write_v3_envelope(
        tmp_path,
        runs={
            DEFAULT_RUN_ID: {"patches": {}},
            "cells": {"patches": {"b.tif": {"status": "accepted"}}},
        },
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))
    mgr.switch_active_run("cells", str(tmp_path))
    assert mgr.get_active_run_id() == "cells"
    with pytest.raises(ValueError):
        mgr.delete_run("cells", str(tmp_path))
    # cells survives the refused delete.
    raw = _read_raw(tmp_path)
    assert "cells" in raw["runs"]


def test_delete_run_missing_entry_is_noop(tmp_path):
    """delete_run on an absent run id writes nothing surprising (idempotent)."""
    _write_v3_envelope(
        tmp_path,
        runs={
            DEFAULT_RUN_ID: {"patches": {}},
            "cells": {"patches": {"b.tif": {"status": "accepted"}}},
        },
    )
    mgr = ReconstructionStateManager()
    mgr.load(str(tmp_path))  # active = default
    mgr.delete_run("ghost", str(tmp_path))  # no such run
    raw = _read_raw(tmp_path)
    assert "cells" in raw["runs"]
    assert "ghost" not in raw["runs"]


# ---------------------------------------------------------------------------
# Phase 7: default_run_nested workspace-level flag + resolver mapping
# ---------------------------------------------------------------------------


class TestDefaultRunNestedFlag:
    """Workspace-level unified-layout flag — getter/setter/round-trip +
    the _resolve_path_id mapping it gates."""

    def test_default_false(self, tmp_path):
        mgr = ReconstructionStateManager()
        mgr.load(str(tmp_path))
        assert mgr.get_default_run_nested() is False

    def test_setter_round_trips_through_disk(self, tmp_path):
        mgr = ReconstructionStateManager()
        mgr.load(str(tmp_path))
        mgr.set_default_run_nested(True)
        assert mgr.get_default_run_nested() is True
        mgr.save()  # bypass QTimer debounce
        # Survives the v3 wrap → stays top-level (workspace-level key).
        raw = _read_raw(tmp_path)
        assert raw["default_run_nested"] is True
        assert "default_run_nested" not in raw["runs"][DEFAULT_RUN_ID]
        # Reload picks it up.
        mgr2 = ReconstructionStateManager()
        mgr2.load(str(tmp_path))
        assert mgr2.get_default_run_nested() is True

    def test_old_state_file_loads_with_default_false(self, tmp_path):
        _write_v1_flat(tmp_path, a="curating")
        mgr = ReconstructionStateManager()
        mgr.load(str(tmp_path))
        assert mgr.get_default_run_nested() is False

    def test_resolve_path_id_flat_default(self, tmp_path):
        mgr = ReconstructionStateManager()
        mgr.load(str(tmp_path))
        assert mgr._resolve_path_id(DEFAULT_RUN_ID) is None
        assert mgr._resolve_path_id("cells") == "cells"

    def test_resolve_path_id_nested_default(self, tmp_path):
        mgr = ReconstructionStateManager()
        mgr.load(str(tmp_path))
        mgr.set_default_run_nested(True)
        assert mgr._resolve_path_id(DEFAULT_RUN_ID) == "default"
        # Named runs unaffected.
        assert mgr._resolve_path_id("cells") == "cells"


class TestScanExistingArtifactsLayoutAware:
    """scan_existing_artifacts must read flat paths when the flag is False
    and the nested runs/default/ paths when it is True."""

    def _seed_recon(self, ws, run_id):
        recon = _p3.reconstructions_dir(ws, run_id)
        d = recon / "patch_001.tif"
        d.mkdir(parents=True, exist_ok=True)
        (d / "manifest.json").write_text(
            json.dumps({"run_id": "R1", "source_stack_path": "patch_001.tif"}),
            encoding="utf-8",
        )

    def test_scan_flat_when_flag_false(self, tmp_path):
        mgr = ReconstructionStateManager()
        mgr.load(str(tmp_path))
        mgr.register_patches(["patch_001.tif"])
        # Flat reconstruction dir (run_id=None).
        self._seed_recon(tmp_path, None)
        mgr.scan_existing_artifacts(str(tmp_path))
        assert mgr.get_patch_status("patch_001.tif") == "reconstructed"

    def test_scan_nested_when_flag_true(self, tmp_path):
        mgr = ReconstructionStateManager()
        mgr.load(str(tmp_path))
        mgr.set_default_run_nested(True)
        mgr.register_patches(["patch_001.tif"])
        # Nested reconstruction dir (run_id="default").
        self._seed_recon(tmp_path, DEFAULT_RUN_ID)
        mgr.scan_existing_artifacts(str(tmp_path))
        assert mgr.get_patch_status("patch_001.tif") == "reconstructed"

    def test_scan_flat_does_not_see_nested_when_flag_false(self, tmp_path):
        """With the flag False the scanner reads flat paths only — a nested
        recon dir is invisible (the un-migrated invariant)."""
        mgr = ReconstructionStateManager()
        mgr.load(str(tmp_path))
        mgr.register_patches(["patch_001.tif"])
        self._seed_recon(tmp_path, DEFAULT_RUN_ID)  # only nested
        mgr.scan_existing_artifacts(str(tmp_path))
        assert mgr.get_patch_status("patch_001.tif") == "unprocessed"
