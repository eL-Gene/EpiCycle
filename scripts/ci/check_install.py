"""Check that an INSTALLED (non-editable) copy of the wheel works.

Usage, from a folder that is NOT the source checkout:

    python /path/to/scripts/ci/check_install.py dist/napari_epicycle-X.whl

It uses only the standard library plus what the installed package pulls in, and
imports nothing from the checkout.  Four checks:

  a) ``napari_epicycle`` is imported from a site-packages folder.
  b) every ``.py`` file the wheel carries under ``napari_epicycle/`` and
     ``packages/`` imports (``__main__`` modules are the only ones skipped).
  c) the napari plugin manifest is found and its widget callables resolve.
  d) the guides folder exists and holds at least 25 ``.md`` files.

Exit code 0 when all four pass, 1 otherwise.
"""
import glob
import importlib
import sys
import traceback
import zipfile
from pathlib import Path

MIN_GUIDES = 25
ROOTS = ("napari_epicycle/", "packages/")


def _wheel_path(arg: str) -> Path:
    hits = sorted(glob.glob(arg)) or [arg]
    path = Path(hits[0])
    if not path.is_file():
        sys.exit(f"wheel not found: {arg}")
    return path


def _module_names(wheel: Path):
    names = []
    for member in zipfile.ZipFile(wheel).namelist():
        if not member.endswith(".py") or not member.startswith(ROOTS):
            continue
        parts = member[:-3].split("/")
        if parts[-1] == "__main__":
            continue
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return sorted(set(names))


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 1
    wheel = _wheel_path(sys.argv[1])
    failed = False

    # a) the package must come from an installed copy, not a source tree
    import napari_epicycle

    where = Path(napari_epicycle.__file__).resolve()
    in_site = any(p in ("site-packages", "dist-packages") for p in where.parts)
    print(f"[a] napari_epicycle imported from {where}")
    if not in_site:
        print("[a] FAIL: not under a site-packages folder")
        failed = True

    # b) every module the wheel carries must import
    modules = _module_names(wheel)
    errors = {}
    for name in modules:
        try:
            importlib.import_module(name)
        except BaseException:  # noqa: BLE001 - report everything, even SystemExit
            errors[name] = traceback.format_exc()
    print(f"[b] imported {len(modules) - len(errors)} of {len(modules)} modules")
    for name, tb in errors.items():
        print(f"[b] FAIL {name}\n{tb}")
    if errors:
        failed = True

    # c) the plugin manifest resolves
    try:
        from npe2 import PluginManager

        pm = PluginManager.instance()
        pm.discover()
        manifest = pm.get_manifest("napari-epicycle")
        widgets = list(manifest.contributions.widgets or [])
        if not widgets:
            raise RuntimeError("manifest lists no widgets")
        for w in widgets:
            print(f"[c] widget {w.command!r} -> {w.get_callable()!r}")
    except BaseException:  # noqa: BLE001
        print("[c] FAIL\n" + traceback.format_exc())
        failed = True

    # d) the guides ship inside the package
    try:
        from napari_epicycle.ui_shell._guides import guides_dir

        gd = guides_dir()
        n = len(list(gd.glob("*.md"))) if gd.is_dir() else 0
        print(f"[d] guides dir {gd}: {n} .md files")
        if n < MIN_GUIDES or Path(gd).resolve().parent != where.parent:
            print(f"[d] FAIL: need >= {MIN_GUIDES} guides inside the installed package")
            failed = True
    except BaseException:  # noqa: BLE001
        print("[d] FAIL\n" + traceback.format_exc())
        failed = True

    print("RESULT:", "FAIL" if failed else "OK")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
