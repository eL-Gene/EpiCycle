"""`ReconstructionStateManager` — the two B2 additions.

The shared-workspace hand-over:

* **`suppress_save`** — Step 3's adopt is READ-ONLY for a workspace
  whose `step3_state.json` predates the v3 runs envelope.  `load` upgrades
  v1/v2 → v3 *in memory*, and the next `save` would rewrite the file with no
  backup — so a user who merely walked into Step 3 would have had their old
  state file silently rewritten.  While the flag is set, neither `save` nor
  `schedule_save` may touch the disk.
* **the blank-`active_run_id` peek** (invariant I4) — a workspace switch used
  to keep the PREVIOUS workspace's active run, because nothing ever re-read
  it.  Blanking the attribute now makes `load` resolve the run from the file
  it is reading.
"""
from __future__ import annotations

import json

import pytest

from napari_epicycle.fusion_reconstruction.state_manager import (
    ReconstructionStateManager,
)


def _write(ws, payload):
    (ws / "step3_state.json").write_text(json.dumps(payload), encoding="utf-8")


def _read(ws):
    return json.loads((ws / "step3_state.json").read_text(encoding="utf-8"))


class TestSuppressSave:
    def test_defaults_to_off(self):
        assert ReconstructionStateManager().suppress_save is False

    def test_save_writes_when_not_suppressed(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.set_patch_status("p1.tif", "predicted")
        sm.save()
        assert (tmp_path / "step3_state.json").is_file()

    def test_save_is_a_noop_while_suppressed(self, tmp_path):
        pre_v3 = {"schema_version": 2, "patches": {"old.tif": {"status": "predicted"}}}
        _write(tmp_path, pre_v3)
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.suppress_save = True
        sm.set_patch_status("new.tif", "reconstructed")
        sm.save()
        assert _read(tmp_path) == pre_v3, "adopt must not rewrite a pre-v3 file"

    def test_a_pre_v3_file_is_byte_identical_after_a_suppressed_load(self, tmp_path):
        _write(tmp_path, {"schema_version": 1, "patches": {}})
        before = (tmp_path / "step3_state.json").read_bytes()
        sm = ReconstructionStateManager()
        sm.suppress_save = True
        sm.load(str(tmp_path))
        sm.register_patches(["a.tif", "b.tif"])   # ends in schedule_save
        sm.save()
        assert (tmp_path / "step3_state.json").read_bytes() == before

    def test_schedule_save_starts_no_timer_while_suppressed(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.suppress_save = True
        sm.schedule_save()
        assert sm._save_timer is None or not sm._save_timer.isActive()

    def test_clearing_the_flag_restores_writing(self, tmp_path):
        _write(tmp_path, {"schema_version": 2, "patches": {}})
        sm = ReconstructionStateManager()
        sm.load(str(tmp_path))
        sm.suppress_save = True
        sm.save()
        assert _read(tmp_path).get("schema_version") == 2
        sm.suppress_save = False
        sm.save()
        assert _read(tmp_path).get("schema_version") == 3


class TestBlankActiveRunIdPeek:
    def test_blank_run_id_resolves_from_the_file(self, tmp_path):
        _write(tmp_path, {
            "schema_version": 3,
            "active_run_id": "cells",
            "runs": {
                "default": {"patches": {"d.tif": {"status": "predicted"}}},
                "cells": {"patches": {"c.tif": {"status": "reconstructed"}}},
            },
        })
        sm = ReconstructionStateManager()
        sm.active_run_id = ""
        sm.load(str(tmp_path))
        assert sm.active_run_id == "cells"
        assert sm.get_patch_status("c.tif") == "reconstructed"
        assert sm.get_patch_status("d.tif") == "unprocessed"

    def test_blank_run_id_falls_back_to_default(self, tmp_path):
        _write(tmp_path, {"schema_version": 3,
                          "runs": {"default": {"patches": {}}}})
        sm = ReconstructionStateManager()
        sm.active_run_id = ""
        sm.load(str(tmp_path))
        assert sm.active_run_id == "default"

    def test_blank_run_id_with_no_state_file_falls_back_to_default(self, tmp_path):
        sm = ReconstructionStateManager()
        sm.active_run_id = ""
        sm.load(str(tmp_path))
        assert sm.active_run_id == "default"

    def test_an_explicit_run_id_is_never_overridden(self, tmp_path):
        _write(tmp_path, {
            "schema_version": 3,
            "active_run_id": "cells",
            "runs": {"default": {"patches": {}},
                     "cells": {"patches": {}}},
        })
        sm = ReconstructionStateManager()
        sm.active_run_id = "default"
        sm.load(str(tmp_path))
        assert sm.active_run_id == "default"
