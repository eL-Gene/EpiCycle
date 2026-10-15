"""Checkpoint/resume: every finished stage is recorded; a skipped stage re-derives its outputs."""

import json
from pathlib import Path

from packages.epicycle_scaledseg.core.pipeline import EpiCyclePipeline, Stage, StageResult
from packages.epicycle_scaledseg.core.tiling import TilingStage

RAN = []


class _S(Stage):
    def __init__(self, name):
        super().__init__(name)
        self.seen = None

    def validate_inputs(self, tile_dirs):
        return True

    def run(self, tile_dirs, n_gpus, log_dir=None):
        RAN.append(self.name)
        self.seen = [Path(t).name for t in tile_dirs]
        sr = StageResult()
        for t in tile_dirs:
            sr.add_success(Path(t).name)
        return sr


def _tiles(tmp_path, names=("t0", "t1")):
    out = []
    for n in names:
        d = tmp_path / "tiles" / n
        d.mkdir(parents=True)
        out.append(str(d))
    return out


def _pipe(tmp_path, names):
    p = EpiCyclePipeline(tmp_path / "work", n_gpus=1, log_dir=str(tmp_path / "logs"))
    for n in names:
        p.add_stage(_S(n))
    return p


def _done(tmp_path):
    f = tmp_path / "logs" / "pipeline_checkpoint_coarse.json"
    return set(json.loads(f.read_text())["completed_stages"])


def test_all_stages_recorded(tmp_path):
    RAN.clear()
    _pipe(tmp_path, "abc").run(_tiles(tmp_path))
    assert _done(tmp_path) == {"a", "b", "c"}


def test_resume_skips_all(tmp_path):
    tiles = _tiles(tmp_path)
    _pipe(tmp_path, "abc").run(tiles)
    RAN.clear()
    _pipe(tmp_path, "abc").run(tiles, resume=True)
    assert RAN == []


def test_resume_runs_only_unfinished(tmp_path):
    tiles = _tiles(tmp_path)
    _pipe(tmp_path, "ab").run(tiles)
    RAN.clear()
    _pipe(tmp_path, "abc").run(tiles, resume=True)
    assert RAN == ["c"]
    assert _done(tmp_path) == {"a", "b", "c"}


def test_skipped_tiling_stage_hands_tile_dirs_on(tmp_path):
    out = tmp_path / "tiles_out"
    names = ["tile_r0_c0", "tile_r0_c1"]
    for n in names:
        (out / n).mkdir(parents=True)
    (tmp_path / "in.tif").write_bytes(b"x")
    (out / "tiling_config.json").write_text(json.dumps({"tiles": names}))

    def build():
        p = EpiCyclePipeline(tmp_path / "work", n_gpus=1, log_dir=str(tmp_path / "logs"))
        p.add_stage(TilingStage(str(tmp_path / "in.tif"), output_dir=str(out)))
        nxt = _S("next")
        p.add_stage(nxt)
        return p, nxt

    p1, _ = build()
    p1.run([str(tmp_path)])
    assert {"tiling", "next"} <= _done(tmp_path)

    # Drop "next" from the checkpoint so only tiling is skipped on resume.
    f = tmp_path / "logs" / "pipeline_checkpoint_coarse.json"
    cp = json.loads(f.read_text())
    cp["completed_stages"] = ["tiling"]
    f.write_text(json.dumps(cp))

    p2, nxt = build()
    RAN.clear()
    p2.run([str(tmp_path)], resume=True)
    assert RAN == ["next"]
    assert sorted(nxt.seen) == sorted(names)
