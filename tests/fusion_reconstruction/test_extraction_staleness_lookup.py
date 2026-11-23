"""The extraction staleness check finds a patch's labels in the run folder,
which is named by the patch FILE name, from the manifest's stem key."""
from __future__ import annotations

from napari_epicycle.fusion_reconstruction import _paths
from napari_epicycle.fusion_reconstruction.controller import (
    FusionReconstructionController as C,
    _active_run_path_id,
)


class _SM:
    def get_all_patch_statuses(self):
        return {"P.tiff": "curating"}

    def get_active_run_id(self):
        return "default"


def test_staleness_reads_the_run_folder_named_by_the_file_name(tmp_path):
    c = C.__new__(C)
    c.working_dir = str(tmp_path)
    c.state_manager = _SM()
    c._load_extraction_manifest = lambda: {
        "per_patch": {"P": {"source_mtime": 1.0, "source_path": ""}}}
    d = _paths.patch_run_dir(c.working_dir, "P.tiff",
                             run_id=_active_run_path_id(c))
    d.mkdir(parents=True)
    (d / "labels_3d.tif").write_bytes(b"x")       # mtime far after 1.0
    assert c._check_extraction_staleness() == {"P"}
