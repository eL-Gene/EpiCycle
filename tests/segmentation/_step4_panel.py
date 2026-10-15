"""Helpers for the B1a panel tests: a small workspace on disk and the REAL
Step 4 panel (``SegmentStack`` with its controller, ``viewer=None``).

The workspace has a Step 1 state whose patch registry lists TIFF stacks
under ``artifacts/step1/patches``, plus optional extra
folders.  ``panel(ws)`` opens the panel on it the way a Browse does.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

#: the live dock: 578 px wide, a section viewport of 572 (566 with a
#: scrollbar), ② right content 322, ② / ③ left rows 218.
DOCK_W = 578
DOCK_H = 968


def write_stack(path: Path, *, channels=2, z=6, y=24, x=24,
                spacing_um=None, zcyx=False) -> Path:
    import tifffile
    path.parent.mkdir(parents=True, exist_ok=True)
    if channels == 1:
        data = np.zeros((z, y, x), np.uint8)
        axes = "ZYX"
    elif zcyx:
        data = np.zeros((z, channels, y, x), np.uint8)
        axes = "ZCYX"
    else:
        data = np.zeros((channels, z, y, x), np.uint8)
        axes = "CZYX"
    data[..., y // 3:y // 2, x // 3:x // 2] = 200
    if spacing_um is None:
        tifffile.imwrite(str(path), data, metadata={"axes": axes})
    else:
        zz, yy, xx = spacing_um
        tifffile.imwrite(str(path), data, imagej=True,
                         resolution=(1.0 / xx, 1.0 / yy),
                         metadata={"spacing": zz, "unit": "um",
                                   "axes": axes})
    return path


def make_workspace(tmp_path: Path, *, patches=("P1.tif", "P2.tif", "P3.tif"),
                   voxel=(2.0, 0.5, 0.5), channels=2,
                   one_channel=(), zcyx=()) -> Path:
    """A workspace whose Step 1 registry lists *patches* (their source has
    voxel size *voxel*, or none when ``voxel`` is None)."""
    ws = tmp_path / "ws"
    ws.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in patches:
        rel = f"artifacts/step1/patches/{name}"
        write_stack(ws / rel,
                    channels=1 if name in one_channel else channels,
                    zcyx=name in zcyx)
        rows.append({"patch_name": Path(name).stem, "relative_path": rel,
                     "source_id": "src_0", "source_kind": "raw"})
    state = {
        "schema_version": 2,
        "sources": [{"source_id": "src_0", "path": "raw.tif",
                     "voxel_size": list(voxel) if voxel else None,
                     "voxel_size_source": "user" if voxel else "unknown"}],
        "patches": rows,
    }
    (ws / "step1_state.json").write_text(json.dumps(state), encoding="utf-8")
    return ws


def panel(qapp, ws=None):
    """The real panel with its controller, opened on *ws* as a Browse does
    (write mode), and shown at the live dock size."""
    from napari_epicycle.segmentation.ui import SegmentStack

    w = SegmentStack(viewer=None)
    w.resize(DOCK_W, DOCK_H)
    w.show()
    qapp.processEvents()
    w.controller._step4_active = True     # K6: the user is in Step 4
    if ws is not None:
        open_workspace(w, ws)
    pump(qapp)
    return w


def open_workspace(w, ws):
    c = w.controller
    c.working_dir = str(ws)
    w.workdir_line.setText(str(ws))
    c.state_manager.load(str(ws))
    c._adopt_only = False
    c._restore_session_from_state(persist=True)


def pump(qapp, n=5):
    for _ in range(n):
        qapp.processEvents()


def close(w):
    try:
        w.controller.shutdown()
    except Exception:      # noqa: BLE001
        pass
    w.hide()
    w.deleteLater()


def tick(w, qapp, *names, on=True):
    """Tick rows in ② the way a click does (the item's check state)."""
    from qtpy.QtCore import Qt
    lw = w.stacks_list
    for i in range(lw.count()):
        it = lw.item(i)
        if it.data(Qt.UserRole) in names:
            it.setCheckState(Qt.Checked if on else Qt.Unchecked)
    pump(qapp)


def refined_bundle(ws: Path, run_id=None, bundle="wound-v2",
                   channels=(2, 1)) -> Path:
    """A trained Step 3 refined bundle on disk (both buckets) with its
    training config, the way ``list_cross_run_refined_models`` finds it."""
    from napari_epicycle.fusion_reconstruction import _paths as S3P
    root = S3P.refined_models_root(ws, run_id) / bundle
    root.mkdir(parents=True, exist_ok=True)
    for bucket in ("lateral", "axial"):
        wp = S3P.refined_model_weights_path(ws, bundle, bucket, run_id)
        wp.parent.mkdir(parents=True, exist_ok=True)
        wp.write_bytes(b"weights")
        (wp.parent / f"{bucket}_training_config.json").write_text(
            json.dumps({"hyperparameters": {"channels": list(channels)}}),
            encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({
        "schema_version": "2.0", "created_at": "2026-09-28T10:00:00",
        "lateral": {"weights_path": "lateral", "n_train_pairs": 12},
        "axial": {"weights_path": "axial", "n_train_pairs": 9},
    }), encoding="utf-8")
    return root
