"""Training-pair extraction reads the run folder's own labels, not a stored absolute path."""

import json
import shutil

import numpy as np

from napari_epicycle.fusion_reconstruction import _paths
from napari_epicycle.fusion_reconstruction import training_pair_extractor as tpe


def _build(base, marker_id):
    import tifffile

    ws = base / "ws"
    recon = _paths.reconstructions_dir(ws, None)
    img_dir = ws / "images"
    img_dir.mkdir(parents=True)
    shape = (8, 16, 16)
    image = np.full(shape, 7, dtype=np.uint8)
    labels = np.zeros(shape, dtype=np.int32)
    labels[2:6, 2:6, 2:6] = marker_id
    tifffile.imwrite(str(img_dir / "P1.tif"), image)
    run = recon / "P1"
    run.mkdir(parents=True)
    tifffile.imwrite(str(run / "labels_3d.tif"), labels)
    return ws, img_dir, run


def _spy(monkeypatch):
    seen = []
    real = tpe.extract_training_pairs_from_3d

    def spy(image, labels, out, **kw):
        seen.append(set(np.unique(labels).tolist()) - {0})
        return real(image, labels, out, **kw)

    monkeypatch.setattr(tpe, "extract_training_pairs_from_3d", spy)
    return seen


def test_labels_come_from_the_copied_run_folder(tmp_path, monkeypatch):
    import tifffile

    ws_a, img_a, run_a = _build(tmp_path / "A", 5)
    # the manifest stores an absolute path into workspace A
    (run_a / "manifest.json").write_text(json.dumps({
        "labels_path": str(run_a / "labels_3d.tif"),
        "source_stack_path": str(img_a / "P1.tif"),
    }), encoding="utf-8")
    ws_b = tmp_path / "B" / "ws"
    shutil.copytree(ws_a, ws_b)
    run_b = _paths.reconstructions_dir(ws_b, None) / "P1"
    edited = np.zeros((8, 16, 16), dtype=np.int32)
    edited[2:6, 2:6, 2:6] = 42
    tifffile.imwrite(str(run_b / "labels_3d.tif"), edited)
    shutil.rmtree(tmp_path / "A")

    seen = _spy(monkeypatch)
    tpe.extract_from_workspace(ws_b, image_dir=ws_b / "images", min_labels_per_slice=1)
    assert seen == [{42}]


def test_every_skip_is_logged(tmp_path, capsys):
    ws, img_dir, run = _build(tmp_path, 5)
    (run / "manifest.json").write_text(json.dumps({
        "labels_path": str(tmp_path / "gone.tif"),
        "source_stack_path": str(img_dir / "nope.tif"),
    }), encoding="utf-8")
    (run / "labels_3d.tif").unlink()
    tpe.extract_from_workspace(ws, image_dir=img_dir, min_labels_per_slice=1)
    out = capsys.readouterr().out
    assert "[REFINE-EXTRACT] skip" in out
