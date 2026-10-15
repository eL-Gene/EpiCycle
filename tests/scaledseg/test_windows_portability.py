"""Windows / Linux portability guards for the vendored scaled-seg engine.

Qt-free.  Fake ``sys.platform`` where a probe branches on it.
"""
import re
import sys
from pathlib import Path

import pytest

PKG = Path(__file__).resolve().parents[2] / "packages" / "epicycle_scaledseg"


def _sources():
    return [p for p in PKG.rglob("*.py") if "__pycache__" not in p.parts]


def test_no_fork_signal_or_proc_in_unguarded_code():
    """No os.fork, no signal module; /proc only inside resources.py (guarded by try)."""
    bad = []
    for p in _sources():
        txt = p.read_text(encoding="utf-8", errors="replace")
        if re.search(r"\bos\.fork\b|^\s*import signal\b|^\s*from signal import", txt, re.M):
            bad.append(str(p))
        if "/proc/" in txt and p.name != "resources.py":
            bad.append(str(p))
    assert not bad, bad


def test_pool_contexts_are_spawn_or_default_windows():
    """Every explicit get_context() is 'spawn' (never 'fork'/'forkserver')."""
    for p in _sources():
        txt = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"get_context\(\s*['\"](\w+)['\"]", txt):
            assert m.group(1) == "spawn", (p, m.group(0))


def test_resources_probe_sane_here_and_without_proc():
    from packages.epicycle_scaledseg.utils import resources

    gb, src = resources.available_ram_gb_and_source()
    assert gb > 0.5 and isinstance(src, str)
    assert resources.get_allocated_cpus() >= 1
    assert resources.get_allocated_gpus() >= 0


def test_resources_falls_back_to_psutil_when_cgroup_absent(monkeypatch):
    from packages.epicycle_scaledseg.utils import resources

    for v in ("SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "SLURM_MEM_PER_GPU"):
        monkeypatch.delenv(v, raising=False)
    real_open = open

    def fake_open(path, *a, **k):
        # only the cgroup files: psutil itself reads /proc/meminfo through open()
        if str(path).startswith("/sys/fs/cgroup/"):
            raise FileNotFoundError(path)
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)
    gb, src = resources.available_ram_gb_and_source()
    assert gb > 0 and src.startswith("psutil")


@pytest.mark.parametrize("platform", ["win32", "linux"])
def test_peak_rss_probes_by_platform(monkeypatch, platform):
    from packages.epicycle_scaledseg.core import reconstruction

    if platform == "win32" and sys.platform != "win32":
        pytest.skip("peak_wset only exists on Windows psutil")
    if platform == "linux" and sys.platform == "win32":
        pytest.skip("resource module is POSIX only")
    monkeypatch.setattr(sys, "platform", platform)
    v = reconstruction._peak_rss_gb()
    assert v is None or v > 0
    if sys.platform == platform:
        assert v is not None


def test_peak_rss_returns_none_not_raises_when_probe_breaks(monkeypatch):
    from packages.epicycle_scaledseg.core import reconstruction

    monkeypatch.setattr(sys, "platform", "win32" if sys.platform != "win32" else "linux")
    # wrong-platform branch must degrade to None, never raise
    assert reconstruction._peak_rss_gb() is None or reconstruction._peak_rss_gb() > 0
