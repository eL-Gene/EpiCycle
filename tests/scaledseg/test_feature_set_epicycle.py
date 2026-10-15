"""The ``feature_set`` gate: ``"epicycle"`` makes the vendored
pipeline measure the EpiCycle feature catalogue and nothing
else; ``"full"`` (the in-code default) is the engine's unchanged ~400-column output.

The catalogue is read from ``tests/scaledseg/fixtures/FEATURE_CATALOG.csv`` (the lab zip drops
``docs/research``); a sync test compares the fixture with ``docs/research`` and SKIPS when the docs
copy is absent.  Everything that needs a channel / probability volume runs SERIAL (n_workers=1) with
the ``raw_roi`` readers monkeypatched, so it needs no tiles on disk; the Pool path is covered with
the labels-only + adjacency + qc_score columns.
"""

import csv
import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import packages.epicycle_scaledseg.core.feature_stage as fs
from packages.epicycle_scaledseg.core import qc_features as qf
from packages.epicycle_scaledseg.core.feature_stage import FeatureStage
from packages.epicycle_scaledseg.reconcile import raw_roi
from packages.epicycle_scaledseg.reconcile.features import extract_label_features
from tests.scaledseg.test_feature_stage import _build_fixture, _td

_HERE = Path(__file__).parent
FIXTURE_CSV = _HERE / "fixtures" / "FEATURE_CATALOG.csv"
DOCS_CSV = _HERE.parents[1] / "docs" / "research" / "FEATURE_CATALOG.csv"
PLUGIN_QCF = _HERE.parents[1] / "napari_epicycle" / "fusion_reconstruction" / "qc_features.py"

# --- the M8 parity constants (grill 1 M8) ------------------------------------------
PROVENANCE = {"label", "voxel_count", "is_border_cell", "n_tiles", "seam_status", "min_match_iou"}
# The plugin's extract_features emits all three principal-axis unit vectors (major / mid / minor,
# fusion_reconstruction/qc_features.py extract_features), so all three are plugin extras.
PLUGIN_EXTRA_PREFIXES = ("major_axis_", "mid_axis_", "minor_axis_", "inertia_eigval_", "centroid-", "bbox-")
# Q8 default: `compactness` equals `extent` (the engine documents it), the engine drops it on
# purpose, the epicycle set does NOT emit it; `feature_state` is not a catalogue row, not emitted.
COMPACTNESS_EXEMPT = {"compactness"}
# The 8 unprefixed single-channel intensity rows: the tutorial always names the channel.
UNPREFIXED_EXEMPT = {
    "intensity_mean", "intensity_std", "intensity_min", "intensity_max",
    "boundary_intensity_mean", "interior_intensity_mean", "boundary_interior_ratio",
    "z_intensity_variation",
}
# Engine-only per-cell columns: none may appear under "epicycle", all appear under "full".
ENGINE_ONLY_PROBES = {
    "triaxiality", "axis_mid_length",            # compute_triaxial
    "cross_section_area_cv",                     # cross_section_profile_stats
}


def _load_catalogue(path=FIXTURE_CSV):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


CATALOGUE = _load_catalogue()


def expected_columns(*, channel="mem", prob=True, adjacency=True):
    """EXPECTED = catalogue rows grouped by ``requires_optional_input`` (``<channel>`` -> ``channel``)."""
    out = set()
    for r in CATALOGUE:
        name, req = r["feature"], r["requires_optional_input"]
        if name in COMPACTNESS_EXEMPT or name in UNPREFIXED_EXEMPT:
            continue
        if req == "none":
            out.add(name)
        elif req == "per_channel_image":
            if channel:
                out.add(name.replace("<channel>", channel))
        elif req == "probability_volume":
            if prob:
                out.add(name)
        elif req.startswith("optional"):
            if adjacency:
                out.add(name)
        else:                                              # pragma: no cover - a new category
            pytest.fail(f"unknown requires_optional_input {req!r} for {name}")
    return out


def all_catalogue_names(channel="mem"):
    """Every catalogue name whatever the optional input (e.g. boundary_confidence is a named
    composite that is emitted, all-NaN, without a probability volume)."""
    return {r["feature"].replace("<channel>", channel) for r in CATALOGUE}


def _is_plugin_extra(col):
    return col.startswith(PLUGIN_EXTRA_PREFIXES)


def _spy(monkeypatch, module, names, called):
    for name in names:
        orig = getattr(module, name)

        def _wrap(*a, _o=orig, _n=name, **k):
            called.append(_n)
            return _o(*a, **k)
        monkeypatch.setattr(module, name, _wrap)


# --- helpers -----------------------------------------------------------------------------------

@pytest.fixture
def fake_readers(monkeypatch):
    """Patch the raw/prob readers (in-process only: serial runs) with deterministic synthetic volumes."""
    rng = np.random.default_rng(7)
    shape = (24, 40, 40)
    ch = (rng.random(shape) * 200 + 10).astype(np.float32)
    pr = rng.random(shape).astype(np.float32)

    def _read_global_roi(tiles_dir, bbox, channel, halo=0, halo_z=None):
        z0, z1, y0, y1, x0, x1 = bbox
        return ch[z0:z1, y0:y1, x0:x1]

    def _read_prob_roi(tiles_dir, recon_dir, bbox, halo=0, halo_z=None, apply_sigmoid=False):
        z0, z1, y0, y1, x0, x1 = bbox
        return pr[z0:z1, y0:y1, x0:x1]

    monkeypatch.setattr(raw_roi, "read_global_roi", _read_global_roi)
    monkeypatch.setattr(raw_roi, "read_prob_roi", _read_prob_roi)


def _run(tmp_path, *, n_workers=1, **kw):
    stitch, vol = _build_fixture(tmp_path)
    res = FeatureStage(input_dir=str(stitch), n_workers=n_workers, **kw).run(_td(), n_gpus=0)
    assert res.n_failed == 0
    df = pd.read_csv(stitch / "cell_features.csv")
    man = json.loads((stitch / "feature_manifest.json").read_text())
    return df, man, stitch, vol


def _with_samplers(**kw):
    return dict(tiles_dir="FAKE_TILES", channels=[(0, "mem")], prob=True, **kw)


def _tiny_labels():
    lab = np.zeros((12, 20, 20), dtype=np.int32)
    lab[2:8, 3:10, 3:10] = 1
    return lab


# --- the catalogue fixture ---------------------------------------------------------------------

def test_catalogue_fixture_has_the_86_rows():
    assert len(CATALOGUE) == 86
    assert len({r["feature"] for r in CATALOGUE}) == 86


def test_catalogue_fixture_in_sync_with_docs():
    if not DOCS_CSV.exists():
        pytest.skip("docs/research/FEATURE_CATALOG.csv absent (the lab zip drops docs/research)")
    assert _load_catalogue(DOCS_CSV) == CATALOGUE, (
        "tests/scaledseg/fixtures/FEATURE_CATALOG.csv drifted from docs/research/FEATURE_CATALOG.csv: "
        "copy the docs file over the fixture and re-check the parity test")


# --- the M8 parity assertion -------------------------------------------------------------------

def test_epicycle_stage_emits_every_catalogue_name_and_nothing_else(tmp_path, fake_readers):
    df, man, *_ = _run(tmp_path, feature_set="epicycle", **_with_samplers())
    got = set(df.columns)
    expected = expected_columns()
    assert expected - got == set(), f"catalogue names missing: {sorted(expected - got)}"
    extra = {c for c in got - expected - PROVENANCE if not _is_plugin_extra(c)}
    assert extra == set(), f"engine-only / non-catalogue columns under 'epicycle': {sorted(extra)}"
    assert "compactness" not in got and "feature_state" not in got
    assert not (got & ENGINE_ONLY_PROBES)
    assert man["params"]["feature_set"] == "epicycle"


def test_epicycle_labels_only_stage_matches_the_none_rows(tmp_path):
    df, *_ = _run(tmp_path, feature_set="epicycle")
    got = set(df.columns)
    expected = expected_columns(channel=None, prob=False)
    assert expected - got == set(), sorted(expected - got)
    # the shared extract_features output is not trimmed: the plugin emits these too
    assert {f"{a}_axis_{d}" for a in ("major", "mid", "minor") for d in "zyx"} <= got
    extra = {c for c in got - all_catalogue_names() - PROVENANCE if not _is_plugin_extra(c)}
    assert extra == set(), sorted(extra)


def test_epicycle_channel_columns_are_exactly_the_catalogue_suffixes(tmp_path, fake_readers):
    df, *_ = _run(tmp_path, feature_set="epicycle", **_with_samplers())
    suffixes = {r["feature"][len("<channel>"):] for r in CATALOGUE if r["feature"].startswith("<channel>")}
    assert len(suffixes) == 11
    assert {c[len("mem"):] for c in df.columns if c.startswith("mem_")} == suffixes


def test_epicycle_reconcile_path_parity(fake_readers):
    lab = np.zeros((24, 40, 40), dtype=np.int32)
    lab[4:12, 5:13, 5:13] = 1
    lab[4:9, 5:8, 20:32] = 2
    lab[14:20, 22:30, 22:28] = 3
    df = extract_label_features(
        lab, feature_set="epicycle", n_workers=1, tiles_dir="FAKE_TILES", recon_dir="FAKE_RECON",
        channels=[(0, "mem")], prob=True)
    got = set(df.columns)
    expected = expected_columns()
    assert expected - got == set(), sorted(expected - got)
    extra = {c for c in got - expected - {"label"} if not _is_plugin_extra(c)}
    assert extra == set(), sorted(extra)


def test_epicycle_shared_numeric_columns_match_the_plugin(tmp_path):
    if not PLUGIN_QCF.exists():
        pytest.skip("napari_epicycle/fusion_reconstruction/qc_features.py absent")
    spec = importlib.util.spec_from_file_location("_plugin_qc_features_parity", PLUGIN_QCF)
    plug = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = plug
    try:
        spec.loader.exec_module(plug)
    finally:
        sys.modules.pop(spec.name, None)
    df, _man, _stitch, vol = _run(tmp_path, feature_set="epicycle")
    p = plug.extract_features(vol)
    plug.compute_surface_features(vol, p)
    plug.compute_discrete_compactness(vol, p)
    plug.compute_skeleton_features(vol, p)
    plug.compute_cross_section_features(vol, p)
    p = p.set_index("label")
    e = df.set_index("label")
    shared = [c for c in e.columns if c in p.columns
              and not c.startswith(("centroid-", "prob_")) and e[c].dtype.kind in "fiu"
              and p[c].dtype.kind in "fiu"]
    # the plugin's own extraction must overlap the engine on the core + shape families
    assert {"area", "surface_area", "skeleton_length", "discrete_compactness"} <= set(shared)
    for c in shared:
        a = e.loc[p.index, c].to_numpy(dtype=float)
        b = p[c].to_numpy(dtype=float)
        assert np.allclose(a, b, rtol=1e-4, atol=1e-6, equal_nan=True), c


# --- "full" is unchanged -----------------------------------------------------------------------

def test_in_code_default_is_full_and_emits_engine_only_columns(tmp_path):
    df, man, *_ = _run(tmp_path)                                    # no feature_set argument
    assert man["params"]["feature_set"] == "full"
    assert ENGINE_ONLY_PROBES <= set(df.columns)
    assert {"is_border_cell", "n_tiles"} <= set(df.columns)


def test_full_does_not_force_adjacency_or_qc_score():
    st = FeatureStage(input_dir="x")
    assert st.feature_set == "full" and st.adjacency is False and st.qc_score is False


def test_reconcile_default_is_full():
    df = extract_label_features(_tiny_labels(), n_workers=1)
    assert ENGINE_ONLY_PROBES <= set(df.columns)


# --- the flag resolution -----------------------------------------------------------------------

def test_epicycle_forces_adjacency_and_qc_score_unless_explicitly_false():
    st = FeatureStage(input_dir="x", feature_set="epicycle")
    assert st.adjacency is True and st.qc_score is True
    st = FeatureStage(input_dir="x", feature_set="epicycle", adjacency=False, qc_score=False)
    assert st.adjacency is False and st.qc_score is False


def test_epicycle_skips_the_engine_only_opt_ins():
    st = FeatureStage(input_dir="x", feature_set="epicycle", tissue_depth=True, territory=True, lumen=True)
    assert (st.tissue_depth, st.territory, st.lumen) == (False, False, False)
    assert st.get_params()["feature_set"] == "epicycle"


def test_unknown_feature_set_raises():
    with pytest.raises(ValueError, match="feature_set"):
        FeatureStage(input_dir="x", feature_set="everything")
    with pytest.raises(ValueError, match="feature_set"):
        extract_label_features(np.ones((4, 4, 4), np.int32), feature_set="everything")


def test_feature_set_skip_is_logged(tmp_path, caplog):
    with caplog.at_level(logging.INFO, logger="epicycle.features"):
        _run(tmp_path, feature_set="epicycle", tissue_depth=True)
    assert any("[FEATURE-SET]" in r.getMessage() for r in caplog.records)


# --- worker safety: pool == serial, and the global is reset ------------------------------------

def test_pool_and_serial_paths_give_the_same_epicycle_columns(tmp_path):
    ser, *_ = _run(tmp_path / "serial", feature_set="epicycle", n_workers=1)
    pool, *_ = _run(tmp_path / "pool", feature_set="epicycle", n_workers=2)
    assert list(ser.columns) == list(pool.columns)
    assert not (set(pool.columns) & ENGINE_ONLY_PROBES)
    a = ser.select_dtypes("number").to_numpy(dtype=float)
    b = pool.select_dtypes("number").to_numpy(dtype=float)
    assert np.allclose(a, b, equal_nan=True)


def test_serial_epicycle_run_does_not_leak_into_a_following_full_run(tmp_path):
    ep, *_ = _run(tmp_path / "ep", feature_set="epicycle", n_workers=1)
    assert fs._WORKER_FEATURE_SET == "full"                         # reset in the serial finally
    full, *_ = _run(tmp_path / "full", feature_set="full", n_workers=1)
    assert ENGINE_ONLY_PROBES <= set(full.columns)
    assert not (set(ep.columns) & ENGINE_ONLY_PROBES)
    # and the bare worker default after the epicycle run is "full" again
    fs._init_feature_worker(str(tmp_path / "ep" / "stage4_full_stitch" / "global_labels_clean.npy"))
    try:
        row = fs._extract_one((1, (4, 12, 5, 13, 5, 13), None, False, True))
    finally:
        fs._WORKER_SRC = None
    assert ENGINE_ONLY_PROBES <= set(row)


def test_reconcile_serial_epicycle_run_is_reset_too():
    lab = _tiny_labels()
    ep = extract_label_features(lab, feature_set="epicycle", n_workers=1)
    assert fs._WORKER_FEATURE_SET == "full"
    full = extract_label_features(lab, n_workers=1)
    assert not (set(ep.columns) & ENGINE_ONLY_PROBES)
    assert ENGINE_ONLY_PROBES <= set(full.columns)


def test_extract_one_task_tuple_shape_is_frozen(tmp_path):
    # tests call _extract_one((cell, bbox, spacing, invert_z, include_surface)) at 10 sites
    stitch, _ = _build_fixture(tmp_path)
    fs._init_feature_worker(str(stitch / "global_labels_clean.npy"), None, None, None, False, "epicycle")
    try:
        row = fs._extract_one((1, (4, 12, 5, 13, 5, 13), None, False, True))
    finally:
        fs._WORKER_SRC = None
        fs._WORKER_FEATURE_SET = "full"
    assert isinstance(row, dict) and not (set(row) & ENGINE_ONLY_PROBES)


def test_epicycle_skips_the_engine_only_calls(tmp_path, fake_readers, monkeypatch):
    called = []
    _spy(monkeypatch, qf, (
        "compute_triaxial", "compute_edt_features", "compute_curvature_features",
        "compute_channel_texture", "compute_channel_puncta", "compute_prob_geometry",
        "compute_centroid_spacing", "compute_reliability_meta", "compute_contact_geometry",
        "compute_graph_topology", "compute_local_comparative"), called)
    _run(tmp_path, feature_set="epicycle", **_with_samplers())
    assert called == [], called


# --- the manifest knows the feature set --------------------------------------------------------

def test_a_full_manifest_is_not_complete_for_an_epicycle_rerun(tmp_path):
    full, man, stitch, _ = _run(tmp_path, feature_set="full")
    assert man["status"] == "complete" and ENGINE_ONLY_PROBES <= set(full.columns)
    FeatureStage(input_dir=str(stitch), n_workers=1, feature_set="epicycle").run(_td(), n_gpus=0)
    again = pd.read_csv(stitch / "cell_features.csv")
    man2 = json.loads((stitch / "feature_manifest.json").read_text())
    assert man2["params"]["feature_set"] == "epicycle"
    assert not (set(again.columns) & ENGINE_ONLY_PROBES)


def test_a_manifest_without_feature_set_counts_as_full(tmp_path):
    _df, _man, stitch, _ = _run(tmp_path, feature_set="full")
    mp = stitch / "feature_manifest.json"
    m = json.loads(mp.read_text())
    m["params"].pop("feature_set")
    mp.write_text(json.dumps(m))
    FeatureStage(input_dir=str(stitch), n_workers=1, feature_set="epicycle").run(_td(), n_gpus=0)
    assert json.loads(mp.read_text())["params"]["feature_set"] == "epicycle"


def test_same_feature_set_rerun_still_skips(tmp_path):
    _df, _man, stitch, _ = _run(tmp_path, feature_set="epicycle")
    t0 = (stitch / "cell_features.csv").stat().st_mtime_ns
    FeatureStage(input_dir=str(stitch), n_workers=1, feature_set="epicycle").run(_td(), n_gpus=0)
    assert (stitch / "cell_features.csv").stat().st_mtime_ns == t0


# --- reconcile: the engine-only calls are skipped under "epicycle" -----------------------------

def test_reconcile_epicycle_skips_reliability_meta(monkeypatch):
    calls = []
    _spy(monkeypatch, qf, ("compute_reliability_meta",), calls)
    extract_label_features(_tiny_labels(), feature_set="epicycle", n_workers=1)
    assert calls == []
    extract_label_features(_tiny_labels(), n_workers=1)
    assert calls == ["compute_reliability_meta"]


# --- the CLI -----------------------------------------------------------------------------------

_BASE = ["run", "--work-dir", "x", "--lateral-model", "a", "--axial-model", "b"]


def test_cli_prog_is_epicycle_scaledseg():
    from packages.epicycle_scaledseg.cli.main import build_parser
    p = build_parser()
    assert p.prog == "epicycle-scaledseg"
    assert p.format_usage().startswith("usage: epicycle-scaledseg")


def test_cli_feature_set_default_is_epicycle_and_full_is_selectable():
    from packages.epicycle_scaledseg.cli.main import build_parser
    assert build_parser().parse_args(_BASE).feat_feature_set == "epicycle"
    assert build_parser().parse_args(_BASE + ["--feat-feature-set", "full"]).feat_feature_set == "full"
    with pytest.raises(SystemExit):
        build_parser().parse_args(_BASE + ["--feat-feature-set", "bogus"])


def test_cli_feat_all_under_epicycle_is_redefined_to_the_epicycle_set():
    """Feat-all means 'all of the epicycle set' (adjacency + qc_score); the engine-only
    opt-ins it used to add are skipped, never raised."""
    from packages.epicycle_scaledseg.cli.main import _build_feature_stage, build_parser
    st = _build_feature_stage(build_parser().parse_args(_BASE + ["--stages", "features", "--feat-all"]))
    assert st.feature_set == "epicycle"
    assert st.adjacency is True and st.qc_score is True
    assert (st.tissue_depth, st.territory, st.lumen) == (False, False, False)


def test_cli_default_stage_is_epicycle_with_adjacency_and_qc_score_forced():
    from packages.epicycle_scaledseg.cli.main import _build_feature_stage, build_parser
    st = _build_feature_stage(build_parser().parse_args(_BASE + ["--stages", "features"]))
    assert (st.feature_set, st.adjacency, st.qc_score) == ("epicycle", True, True)


def test_cli_feat_all_under_full_keeps_the_engine_meaning():
    from packages.epicycle_scaledseg.cli.main import _build_feature_stage, build_parser
    st = _build_feature_stage(build_parser().parse_args(
        _BASE + ["--stages", "features", "--feat-feature-set", "full", "--feat-all"]))
    assert st.feature_set == "full"
    assert (st.adjacency, st.tissue_depth, st.territory, st.lumen, st.qc_score) == (True,) * 5


# --- the pipeline checkpoint knows the feature set (resume) ------------------------------------

class _CountingStage:
    """Factory for a minimal Stage that counts its runs; ``feature_set`` mirrors FeatureStage."""

    @staticmethod
    def make(name, feature_set=None):
        from packages.epicycle_scaledseg.core.pipeline import Stage, StageResult

        class _S(Stage):
            def __init__(self):
                super().__init__(name)
                self.runs = 0
                if feature_set is not None:
                    self.feature_set = feature_set

            def validate_inputs(self, tile_dirs):
                return True

            def run(self, tile_dirs, n_gpus, log_dir=None):
                self.runs += 1
                r = StageResult()
                for td in tile_dirs:
                    r.add_success(td.name)
                return r

        return _S()


def _resume(tmp_path, stage, resume=True):
    from packages.epicycle_scaledseg.core.pipeline import EpiCyclePipeline
    tile = tmp_path / "tile_0"
    tile.mkdir(exist_ok=True)
    p = EpiCyclePipeline(work_dir=str(tmp_path), n_gpus=1, log_dir=str(tmp_path / "logs"), mode="full")
    p.add_stage(stage)
    p.run([str(tile)], resume=resume)
    return stage.runs


def test_resume_reruns_a_features_stage_completed_under_another_feature_set(tmp_path):
    assert _resume(tmp_path, _CountingStage.make("features", "full"), resume=False) == 1
    assert _resume(tmp_path, _CountingStage.make("features", "epicycle")) == 1      # NOT skipped
    assert _resume(tmp_path, _CountingStage.make("features", "epicycle")) == 0      # same set: skipped
    assert _resume(tmp_path, _CountingStage.make("features", "full")) == 1          # back to full: rerun


def test_resume_treats_a_pre_gate_checkpoint_as_full(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "pipeline_checkpoint_full.json").write_text(
        json.dumps({"completed_stages": ["features"], "failed_tiles": {}}))
    assert _resume(tmp_path, _CountingStage.make("features", "full")) == 0
    assert _resume(tmp_path, _CountingStage.make("features", "epicycle")) == 1


def test_resume_of_a_stage_without_feature_set_is_unchanged(tmp_path):
    assert _resume(tmp_path, _CountingStage.make("seg"), resume=False) == 1
    assert _resume(tmp_path, _CountingStage.make("seg")) == 0
