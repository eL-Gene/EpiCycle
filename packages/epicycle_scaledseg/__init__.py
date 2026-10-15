"""EpiCycle scaled segmentation: the large-scale 3D cell segmentation engine.

Vendored from the whole-tissue engine (top-level package ``epicycle`` renamed to
``packages.epicycle_scaledseg``; no behaviour change beyond the documented
portability fixes).

Licence: GPL-3.0-or-later, like the rest of EpiCycle (see THIRD_PARTY_NOTICES.md).

Status: Step 5 runs from the tutorials and the command line (`epicycle-scaledseg`);no plugin panel calls this package. Standalone leaf: it never imports ``napari_epicycle`` and has no Qt.
"""

__version__ = "0.1.0"
