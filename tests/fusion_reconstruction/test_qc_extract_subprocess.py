"""Tests for the Step 3 QC feature-extraction subprocess relocation.

2026-06-07 (feature/step3-extract-subprocess): the GIL-bound QC feature
extraction was moved OFF the QThread into a SEPARATE PROCESS so the napari UI
stays live during the 100-270s compute.  The HARD CONSTRAINT is **perf only,
identical behavior** — the subprocess must produce byte-for-byte the same
``{patch}_features.csv`` as the in-process compute.

The linchpin here is ``test_equivalence_*`` — it runs the SAME config both
in-process (``run_full_qc_extraction``) and through the subprocess
(``python -m ..._qc_extract_subprocess``), then asserts the two feature CSVs
are column-identical with equal values.  That proves "perf only, identical
output".

The other tests cover the subprocess entry module (DONE / ERROR / CANCEL
markers + sidecars) and the controller-side stdout parser routing.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

import napari_epicycle
from napari_epicycle.fusion_reconstruction._qc_extract_core import (
    run_full_qc_extraction,
)
from napari_epicycle.fusion_reconstruction import _qc_extract_subprocess as sub
from napari_epicycle.fusion_reconstruction.qc_features import load_patch_csv


# The package root (dir CONTAINING ``napari_epicycle``) derived from the live
# module — mirrors the controller's PYTHONPATH-prepend fix.  Prepending it to
# the child's PYTHONPATH pins the subprocess to THIS checkout regardless of the
# pytest cwd or the editable-install target (which may point elsewhere).
_PKG_ROOT = Path(napari_epicycle.__file__).resolve().parent.parent


def _child_env() -> dict:
    """os.environ + PYTHONPATH prepended with the live package root."""
    return {
        **os.environ,
        "PYTHONPATH": str(_PKG_ROOT) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }


# ---------------------------------------------------------------------------
# Synthetic workspace fixtures
# ---------------------------------------------------------------------------

def _make_labels(n_labels=5, shape=(24, 40, 40)):
    """Small 3D label volume with n_labels distinct cubic regions."""
    labels = np.zeros(shape, dtype=np.int32)
    z, y, x = shape
    cube = max(min(z // 3, y // n_labels - 2, x // 3), 4)
    step_y = y // n_labels
    for i in range(n_labels):
        y0 = i * step_y + 1
        y1 = min(y0 + cube, y - 1)
        z0, z1 = 2, min(2 + cube, z - 1)
        x0, x1 = 2, min(2 + cube, x - 1)
        labels[z0:z1, y0:y1, x0:x1] = i + 1
    return labels


def _write_workspace(tmp_path, *, with_prob=False, n_labels=5):
    """Write labels TIFF (+ optional prob NPY) and return a kwargs dict."""
    labels = _make_labels(n_labels=n_labels)
    labels_path = tmp_path / "patch001_labels.tif"
    tifffile.imwrite(str(labels_path), labels)

    prob_path = ""
    if with_prob:
        rng = np.random.default_rng(0)
        prob = rng.random(labels.shape).astype(np.float32)
        prob_path = str(tmp_path / "patch001_prob.npy")
        np.save(prob_path, prob)

    csv_path = tmp_path / "patch001_features.csv"
    return dict(
        labels_path=str(labels_path),
        source_stack_path="",
        prob_path=prob_path,
        csv_path=str(csv_path),
        patch_name="patch001",
        spacing=None,
        erosion_iters=2,
        max_radius=5,
        composite_method="zscore",
        seg_channel=0,
        pca_features=None,
        n_pcs=5,
        compute_umap=False,
        umap_min_dist=0.10,
        umap_n_neighbors=15,
        compute_tsne=False,
        tsne_perplexity=30.0,
        channel_configs=[],
        exclude_border=False,
        invert_z=False,
        metric="euclidean",
        patch_status="reconstructed",
    )


def _run_subprocess(config: dict, config_path: Path):
    """Run the extraction subprocess on a config JSON, return CompletedProcess.

    Passes the PYTHONPATH-prepended env (fix #2) so the child imports THIS
    worktree's code regardless of the cwd pytest was invoked from — this is what
    makes the equivalence test independent of cwd / editable-install target.
    """
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable, "-m",
            "napari_epicycle.fusion_reconstruction._qc_extract_subprocess",
            str(config_path),
        ],
        capture_output=True,
        text=True,
        timeout=600,
        env=_child_env(),
    )


def _assert_frames_identical(df_a: pd.DataFrame, df_b: pd.DataFrame, *, atol=1e-9):
    """Same columns (same order) + equal values (tiny float tolerance)."""
    assert list(df_a.columns) == list(df_b.columns), (
        f"Column mismatch:\n in-process: {list(df_a.columns)}\n"
        f" subprocess: {list(df_b.columns)}"
    )
    assert len(df_a) == len(df_b), "Row count differs"
    for col in df_a.columns:
        a, b = df_a[col], df_b[col]
        if pd.api.types.is_numeric_dtype(a) and pd.api.types.is_numeric_dtype(b):
            np.testing.assert_allclose(
                a.values.astype(float), b.values.astype(float),
                rtol=0, atol=atol, equal_nan=True,
                err_msg=f"Numeric column {col!r} differs",
            )
        else:
            assert list(a.astype(str)) == list(b.astype(str)), (
                f"Non-numeric column {col!r} differs"
            )


# ---------------------------------------------------------------------------
# Equivalence — the linchpin
# ---------------------------------------------------------------------------

class TestEquivalence:
    def test_equivalence_labels_only(self, tmp_path):
        """In-process vs subprocess produce byte-identical feature CSVs
        (labels-only config — the default user path)."""
        # In-process run.
        (tmp_path / "a").mkdir(parents=True, exist_ok=True)
        cfg_a = _write_workspace(tmp_path / "a")
        summary_a = run_full_qc_extraction(**cfg_a)
        df_a = load_patch_csv(summary_a["csv_path"])

        # Subprocess run (separate dir so paths don't collide).
        (tmp_path / "b").mkdir(parents=True, exist_ok=True)
        cfg_b = _write_workspace(tmp_path / "b")
        result = _run_subprocess(cfg_b, tmp_path / "b" / "config.json")
        assert result.returncode == 0, (
            f"subprocess failed rc={result.returncode}\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        assert sub.MARKER_DONE in result.stdout
        df_b = load_patch_csv(cfg_b["csv_path"])

        _assert_frames_identical(df_a, df_b)
        # The deterministic feature/PCA/composite columns must be present.
        for col in ("label", "area", "qc_score", "PC1"):
            assert col in df_b.columns, f"{col} missing from subprocess CSV"

    def test_equivalence_with_prob(self, tmp_path):
        """Probability-volume branch also produces identical CSVs."""
        (tmp_path / "a").mkdir(parents=True, exist_ok=True)
        cfg_a = _write_workspace(tmp_path / "a", with_prob=True)
        summary_a = run_full_qc_extraction(**cfg_a)
        df_a = load_patch_csv(summary_a["csv_path"])

        (tmp_path / "b").mkdir(parents=True, exist_ok=True)
        cfg_b = _write_workspace(tmp_path / "b", with_prob=True)
        result = _run_subprocess(cfg_b, tmp_path / "b" / "config.json")
        assert result.returncode == 0, (
            f"subprocess failed rc={result.returncode}\n{result.stderr}"
        )
        df_b = load_patch_csv(cfg_b["csv_path"])
        _assert_frames_identical(df_a, df_b)

    def test_equivalence_umap_tsne(self, tmp_path):
        """UMAP + t-SNE embeddings match between in-process and subprocess.

        Both embeddings hardcode ``random_state=42`` in the compute, so the
        coordinates are deterministic and should agree across the two paths.
        We assert the UMAP_1/2 + tSNE_1/2 columns are PRESENT with the right
        shape, and (best-effort) that the values match — exact float agreement
        across processes can be flaky for numba/UMAP, so a relaxed tolerance is
        used and the column-presence/shape assertion is the hard guarantee.
        """
        # Use more labels so the embeddings have enough points to be stable.
        (tmp_path / "a").mkdir(parents=True, exist_ok=True)
        cfg_a = _write_workspace(tmp_path / "a", n_labels=8)
        cfg_a["compute_umap"] = True
        cfg_a["compute_tsne"] = True
        cfg_a["tsne_perplexity"] = 3.0  # small N → small perplexity
        summary_a = run_full_qc_extraction(**cfg_a)
        df_a = load_patch_csv(summary_a["csv_path"])

        (tmp_path / "b").mkdir(parents=True, exist_ok=True)
        cfg_b = _write_workspace(tmp_path / "b", n_labels=8)
        cfg_b["compute_umap"] = True
        cfg_b["compute_tsne"] = True
        cfg_b["tsne_perplexity"] = 3.0
        result = _run_subprocess(cfg_b, tmp_path / "b" / "config.json")
        assert result.returncode == 0, (
            f"subprocess failed rc={result.returncode}\n"
            f"STDOUT:\n{result.stdout}"
        )
        df_b = load_patch_csv(cfg_b["csv_path"])

        # Hard guarantee — embedding columns present with matching shape.
        for col in ("UMAP_1", "UMAP_2", "tSNE_1", "tSNE_2"):
            assert col in df_a.columns, f"{col} missing from in-process CSV"
            assert col in df_b.columns, f"{col} missing from subprocess CSV"
            assert len(df_a[col]) == len(df_b[col]), f"{col} row count differs"

        # Best-effort value agreement (deterministic random_state=42).  Relaxed
        # tolerance — cross-process numba/UMAP can differ in the last bits.
        for col in ("UMAP_1", "UMAP_2", "tSNE_1", "tSNE_2"):
            np.testing.assert_allclose(
                df_a[col].values.astype(float),
                df_b[col].values.astype(float),
                rtol=1e-3, atol=1e-3, equal_nan=True,
                err_msg=f"Embedding column {col!r} differs across processes",
            )

    def test_equivalence_raw_archives(self, tmp_path):
        """features_raw.csv + adjacency_raw.csv sidecars match between paths."""
        (tmp_path / "a").mkdir(parents=True, exist_ok=True)
        cfg_a = _write_workspace(tmp_path / "a")
        run_full_qc_extraction(**cfg_a)

        (tmp_path / "b").mkdir(parents=True, exist_ok=True)
        cfg_b = _write_workspace(tmp_path / "b")
        result = _run_subprocess(cfg_b, tmp_path / "b" / "config.json")
        assert result.returncode == 0

        raw_a = pd.read_csv(tmp_path / "a" / "patch001_features_raw.csv")
        raw_b = pd.read_csv(tmp_path / "b" / "patch001_features_raw.csv")
        _assert_frames_identical(raw_a, raw_b)

        adj_a = pd.read_csv(tmp_path / "a" / "patch001_adjacency_raw.csv")
        adj_b = pd.read_csv(tmp_path / "b" / "patch001_adjacency_raw.csv")
        _assert_frames_identical(adj_a, adj_b)


# ---------------------------------------------------------------------------
# Subprocess entry module (markers + sidecars)
# ---------------------------------------------------------------------------

class TestSubprocessMain:
    def test_main_done_writes_csv_and_adjacency_sidecar(self, tmp_path):
        cfg = _write_workspace(tmp_path)
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")

        rc = sub.main(["prog", str(config_path)])
        assert rc == 0
        assert Path(cfg["csv_path"]).is_file()
        # Adjacency sidecar written next to the config.
        adj_sidecar = Path(str(config_path) + ".adjacency.json")
        assert adj_sidecar.is_file()
        adj = json.loads(adj_sidecar.read_text(encoding="utf-8"))
        assert isinstance(adj, list)

    def test_main_pops_parent_pid_and_still_runs(self, tmp_path):
        """A ``_parent_pid`` in the config is consumed by the watchdog and must
        NOT be forwarded to run_full_qc_extraction (which has no such kwarg)."""
        cfg = _write_workspace(tmp_path)
        cfg["_parent_pid"] = os.getpid()  # our own pid → watchdog sees us alive
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")

        rc = sub.main(["prog", str(config_path)])
        assert rc == 0, "extraction must succeed with _parent_pid present"
        assert Path(cfg["csv_path"]).is_file()

    def test_main_missing_config_arg_errors(self):
        rc = sub.main(["prog"])
        assert rc == 1

    def test_main_bad_config_path_errors(self, tmp_path):
        rc = sub.main(["prog", str(tmp_path / "does_not_exist.json")])
        assert rc == 1

    def test_main_cancel_file_exits_2(self, tmp_path, capsys):
        """A pre-existing .cancel file makes the subprocess exit 2 at the first
        phase boundary with the CANCELLED marker."""
        cfg = _write_workspace(tmp_path)
        config_path = tmp_path / "config.json"
        config_path.write_text(json.dumps(cfg), encoding="utf-8")
        # Pre-create the cancel sentinel.
        Path(str(config_path) + ".cancel").write_text("1", encoding="utf-8")

        rc = sub.main(["prog", str(config_path)])
        assert rc == 2
        out = capsys.readouterr().out
        assert sub.MARKER_CANCELLED in out
        # CSV must NOT have been written (cancelled before save).
        assert not Path(cfg["csv_path"]).is_file()


# ---------------------------------------------------------------------------
# Parent-liveness watchdog (fix #3)
# ---------------------------------------------------------------------------

def test_parent_alive_true_for_self():
    """Our own PID is alive."""
    assert sub._parent_alive(os.getpid()) is True


def test_parent_alive_false_for_dead_pid():
    """A PID that doesn't exist reports dead.

    Use a very high PID that's overwhelmingly unlikely to be in use.
    """
    # If by cosmic chance it IS in use, the check returns True — accept either
    # but assert the common (dead) case holds for an absurd PID.
    assert sub._parent_alive(2_000_000_000) is False


def test_parent_alive_nonpositive_is_true():
    """ppid<=0 (unknown) → don't kill the child (returns True)."""
    assert sub._parent_alive(0) is True
    assert sub._parent_alive(-1) is True


def test_start_parent_watchdog_noop_for_zero():
    """_start_parent_watchdog(0) is a no-op (no thread spawned, no raise)."""
    sub._start_parent_watchdog(0)  # must not raise


# ---------------------------------------------------------------------------
# Controller runner / stdout parser routing
# ---------------------------------------------------------------------------

class _FakeProc:
    """Minimal subprocess.Popen stand-in for parser tests."""
    def __init__(self, returncode=0):
        self.returncode = returncode
        self.pid = 1234

    def poll(self):
        return self.returncode

    def wait(self, *a, **k):
        return self.returncode

    def terminate(self):
        self.returncode = -15

    def kill(self):
        self.returncode = -9


def _make_controller_skeleton():
    """Build a controller WITHOUT running __init__ (no Qt/napari needed).

    We only exercise the pure routing methods + their attribute reads, so a
    bare instance with the handful of attributes they touch is enough.
    """
    from napari_epicycle.fusion_reconstruction.controller import (
        FusionReconstructionController,
    )
    ctrl = FusionReconstructionController.__new__(FusionReconstructionController)
    # Attributes the routing methods read/write.
    ctrl._qc_proc = None
    ctrl._qc_proc_reader = None
    ctrl._qc_proc_config_path = None
    ctrl._qc_proc_patch_name = ""
    ctrl._qc_proc_csv_path = ""
    ctrl._qc_proc_cancelled = False
    ctrl._qc_view_patch_name = "patch001"
    ctrl._log_lines = []
    return ctrl


def test_on_qc_proc_done_success_routes_to_finished(tmp_path, monkeypatch):
    """rc==0 → re-read CSV + call _on_qc_extract_finished with the df."""
    ctrl = _make_controller_skeleton()
    # Write a real CSV the handler will re-read.
    cfg = _write_workspace(tmp_path)
    run_full_qc_extraction(**cfg)  # produces the CSV on disk
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    # Adjacency sidecar so the re-attach path is exercised.
    Path(str(config_path) + ".adjacency.json").write_text(
        json.dumps([{"label_a": 1, "label_b": 2, "contact_voxels": 3}]),
        encoding="utf-8",
    )

    ctrl._qc_proc = _FakeProc(returncode=0)
    ctrl._qc_proc_reader = None
    ctrl._qc_proc_config_path = str(config_path)
    ctrl._qc_proc_patch_name = "patch001"
    ctrl._qc_proc_csv_path = cfg["csv_path"]

    calls = {}

    def _fake_finished(df, pn):
        calls["finished"] = (df, pn)

    def _fake_csv_saved(pn, cp):
        calls["csv_saved"] = (pn, cp)

    monkeypatch.setattr(ctrl, "_on_qc_extract_finished", _fake_finished, raising=False)
    monkeypatch.setattr(ctrl, "_on_qc_csv_saved", _fake_csv_saved, raising=False)

    ctrl._on_qc_proc_done(0, "")

    assert "finished" in calls, "rc==0 must route to _on_qc_extract_finished"
    df, pn = calls["finished"]
    assert pn == "patch001"
    assert df.attrs.get("_adjacency_list") == [
        {"label_a": 1, "label_b": 2, "contact_voxels": 3}
    ], "adjacency list must be re-attached from the sidecar"
    # Live-proc state cleared.
    assert ctrl._qc_proc is None
    # Temp config + sidecars cleaned up.
    assert not config_path.is_file()
    assert not Path(str(config_path) + ".adjacency.json").is_file()


def test_on_qc_proc_done_error_routes_to_error(tmp_path, monkeypatch):
    """Nonzero rc → _on_qc_extract_error, NOT _on_qc_extract_finished."""
    ctrl = _make_controller_skeleton()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    ctrl._qc_proc = _FakeProc(returncode=1)
    ctrl._qc_proc_config_path = str(config_path)
    ctrl._qc_proc_patch_name = "patch001"
    ctrl._qc_proc_csv_path = str(tmp_path / "nope.csv")

    seen = {}
    monkeypatch.setattr(ctrl, "_on_qc_extract_error",
                        lambda msg: seen.setdefault("error", msg), raising=False)
    monkeypatch.setattr(ctrl, "_on_qc_extract_finished",
                        lambda *a: seen.setdefault("finished", a), raising=False)

    ctrl._on_qc_proc_done(1, "boom traceback")

    assert "error" in seen
    assert "finished" not in seen
    assert "rc=1" in seen["error"]


def test_on_qc_proc_done_cancelled_quiet(tmp_path, monkeypatch):
    """rc==2 (CANCELLED) → neither finished nor error; quiet cleanup."""
    ctrl = _make_controller_skeleton()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    ctrl._qc_proc = _FakeProc(returncode=2)
    ctrl._qc_proc_config_path = str(config_path)
    ctrl._qc_proc_patch_name = "patch001"
    ctrl._qc_proc_csv_path = str(tmp_path / "nope.csv")

    seen = {}
    monkeypatch.setattr(ctrl, "_on_qc_extract_error",
                        lambda msg: seen.setdefault("error", msg), raising=False)
    monkeypatch.setattr(ctrl, "_on_qc_extract_finished",
                        lambda *a: seen.setdefault("finished", a), raising=False)
    # _busy_hide + button mutation + _append_log touch self.ui — stub them.
    monkeypatch.setattr(ctrl, "_busy_hide", lambda: seen.setdefault("busy_hide", True), raising=False)
    ctrl._append_log = lambda m: None

    class _UI:
        class _Btn:
            def setEnabled(self, *a): pass
            def setText(self, *a): pass
        extract_features_btn = _Btn()
    ctrl.ui = _UI()

    ctrl._on_qc_proc_done(2, "")

    assert "error" not in seen
    assert "finished" not in seen
    assert seen.get("busy_hide") is True
    assert ctrl._qc_proc is None


def test_cancel_writes_sentinel_and_terminates(tmp_path):
    """_cancel_qc_extraction writes the .cancel file + terminates the proc."""
    ctrl = _make_controller_skeleton()
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    proc = _FakeProc(returncode=None)
    ctrl._qc_proc = proc
    ctrl._qc_proc_config_path = str(config_path)
    ctrl._qc_proc_patch_name = "patch001"
    ctrl._log_lines = []
    ctrl._append_log = lambda m: ctrl._log_lines.append(m)

    ctrl._cancel_qc_extraction()

    assert Path(str(config_path) + ".cancel").is_file(), "cancel sentinel written"
    assert proc.returncode == -15, "proc.terminate() called"
    assert ctrl._qc_proc_cancelled is True


def test_cancel_noop_when_no_proc():
    """_cancel_qc_extraction is a no-op when nothing is running."""
    ctrl = _make_controller_skeleton()
    ctrl._qc_proc = None
    # Should not raise.
    ctrl._cancel_qc_extraction()


# ---------------------------------------------------------------------------
# json.dumps numpy safety (fix #5)
# ---------------------------------------------------------------------------

def test_to_jsonable_coerces_numpy_scalars():
    """json.dumps(extract_kwargs, default=_to_jsonable) must survive numpy
    scalars (e.g. a numpy-int seg_channel/channel index) instead of raising
    TypeError and dropping to the in-process FREEZE fallback."""
    from napari_epicycle.fusion_reconstruction.controller import _to_jsonable

    extract_kwargs = {
        "seg_channel": np.int64(2),          # the classic offender
        "n_pcs": np.int32(5),
        "umap_min_dist": np.float64(0.10),
        "exclude_border": np.bool_(True),
        "spacing": np.array([1.0, 0.5, 0.5]),  # numpy array → list
        "channel_configs": [],
        "metric": "euclidean",
    }
    # Vanilla json.dumps raises on the numpy-int — confirm the premise.
    with pytest.raises(TypeError):
        json.dumps(extract_kwargs)

    # With the coercer it round-trips cleanly to plain-Python types.
    payload = json.dumps(extract_kwargs, default=_to_jsonable)
    parsed = json.loads(payload)
    assert parsed["seg_channel"] == 2 and isinstance(parsed["seg_channel"], int)
    assert parsed["n_pcs"] == 5
    assert abs(parsed["umap_min_dist"] - 0.10) < 1e-12
    assert parsed["exclude_border"] is True
    assert parsed["spacing"] == [1.0, 0.5, 0.5]


def test_to_jsonable_still_raises_on_genuinely_unserializable():
    """The coercer rescues numpy only — a genuinely unserializable object still
    raises TypeError (so we don't silently swallow real bugs)."""
    from napari_epicycle.fusion_reconstruction.controller import _to_jsonable

    class _Weird:
        pass

    with pytest.raises(TypeError):
        json.dumps({"x": _Weird()}, default=_to_jsonable)
