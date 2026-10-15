"""Importing the engine core must stay light: no heavy / GUI dependency at import time."""
import subprocess
import sys
from pathlib import Path

FORBIDDEN = ("torch", "cellpose", "segment3D", "zarr", "plotly", "napari", "qtpy")

# The child runs from THIS checkout's root: with ``-c`` its sys.path[0] is its cwd, so it imports
# this tree's ``packages`` (e.g. a worktree), not whatever the editable install points at.
REPO_ROOT = Path(__file__).resolve().parents[2]


def test_core_import_loads_no_heavy_modules():
    code = (
        "import sys\n"
        "from pathlib import Path\n"
        "import packages.epicycle_scaledseg.core as core\n"
        f"assert Path(core.__file__).resolve().is_relative_to(Path({str(REPO_ROOT)!r})), core.__file__\n"
        f"bad = [m for m in {FORBIDDEN!r} if m in sys.modules or any(k.startswith(m + '.') for k in sys.modules)]\n"
        "print('LOADED:' + ','.join(bad))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300,
                         cwd=str(REPO_ROOT))
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().endswith("LOADED:"), out.stdout + out.stderr[-1000:]
