import shutil
from pathlib import Path

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py

# Name/version/metadata live in setup.cfg. setup.py is here to express the
# find_packages include-list, which covers BOTH `napari_epicycle/` (the UI layer)
# AND `packages/` (the business-logic packages), and to ship the help guides.
#
# If `packages/` is ever renamed (e.g. to `epicycle_core/`), update the include
# list below.


class BuildPyWithGuides(build_py):
    """Copy docs/guides/*.md into the BUILD dir as napari_epicycle/guides/.

    The source tree stays clean (docs/guides/ is the single source of truth);
    the installed copy is found by napari_epicycle.ui_shell._guides.guides_dir.
    """

    def run(self):
        super().run()
        src = Path(__file__).resolve().parent / "docs" / "guides"
        dst = Path(self.build_lib) / "napari_epicycle" / "guides"
        # build/ persists between in-tree builds: start empty so a guide
        # deleted or renamed in docs/guides does not ship from a stale copy.
        if dst.is_dir():
            shutil.rmtree(dst)
        if not src.is_dir():
            self.warn("docs/guides not found: the built package has NO help "
                      "guides (the ? buttons will show a placeholder)")
            return
        dst.mkdir(parents=True, exist_ok=True)
        for f in src.iterdir():
            if f.is_file() and f.suffix.lower() in {".md", ".png", ".gif", ".jpg", ".svg"}:
                shutil.copy2(f, dst / f.name)


setup(
    packages=find_packages(include=[
        'napari_epicycle', 'napari_epicycle.*',
        'packages', 'packages.*',
    ]),
    cmdclass={'build_py': BuildPyWithGuides},
)
