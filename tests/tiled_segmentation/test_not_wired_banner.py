"""The SLURM script comment points at the notebook and the command-line entry of the whole-tissue engine."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGIC = REPO_ROOT / "napari_epicycle" / "tiled_segmentation" / "logic.py"
COMMAND = "python -m epicycle_scaledseg.run_job"


def test_comment_sits_above_the_slurm_command():
    lines = LOGIC.read_text(encoding="utf-8").splitlines()
    idx = [i for i, ln in enumerate(lines) if COMMAND in ln]
    assert len(idx) == 1
    above = "\n".join(lines[idx[0] - 2 : idx[0]])
    assert above.count("#") >= 2
    assert "NOT WIRED" not in above
    assert "tutorials/segmentation/01_run_whole_tissue_segmentation.ipynb" in above
    assert "epicycle-scaledseg run" in above
