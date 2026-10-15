"""u-Segment3D -- a **lazy, optional** import with a clear message.

``segment3D`` is imported inside the function, in a ``try``/``except``, at every
call site.  On ImportError the tool degrades -- watershed instead of label
diffusion, no-op instead of guided refine -- with a **visible message**, never a
silent difference and never an import-time failure.

Caveats worth recording: ``diffuse_labels3D``'s docstring is a stub,
it carries an upstream author comment questioning its correctness, and it
allocates a dense ``(n_voxels, n_labels)`` float32 -- it is only viable on a
per-cell crop, which is exactly how this engine uses it (and the pre-flight size
guard in :mod:`~.engine` is what keeps it that way).
"""

from __future__ import annotations

USEGMENT3D_MISSING = (
    "u-Segment3D is not installed, so {what} is unavailable -- {fallback}. "
    "Install the 'segment3D' package to enable it."
)


def import_guidedfilter():
    """Return ``segment3D.filters.guidedfilter``.

    Raises ``ImportError`` with a user-facing message when u-Segment3D is absent.
    """
    try:
        from segment3D.filters import guidedfilter
    except Exception as exc:  # ImportError, or a broken install
        raise ImportError(
            USEGMENT3D_MISSING.format(
                what="the guided-filter boundary refine",
                fallback="the split is kept unrefined",
            )
        ) from exc
    return guidedfilter


def import_diffuse_labels3D():
    """Return ``segment3D.filters.diffuse_labels3D``.

    Raises ``ImportError`` with a user-facing message when u-Segment3D is absent.
    """
    try:
        from segment3D.filters import diffuse_labels3D
    except Exception as exc:
        raise ImportError(
            USEGMENT3D_MISSING.format(
                what="the label-diffusion solver",
                fallback="watershed was used instead",
            )
        ) from exc
    return diffuse_labels3D


def have_usegment3d() -> bool:
    """Whether u-Segment3D can be imported right now (for a UI availability hint)."""
    try:
        import_guidedfilter()
    except ImportError:
        return False
    return True
