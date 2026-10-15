"""epicycle_largedata — napari-free large-data mechanics.

The shared, Qt/napari-free layer for handling data too large to fit in RAM:
axis resolution + canonicalization, lazy on-disk open / crop, and (added in
later bricks) TIFF -> multiscale OME-Zarr conversion + a lazy multiscale reader.

napari-free BY DESIGN — this is what lets Step 1 (GUI), Step 5 (GUI), notebooks,
and the headless HPC engine all consume the same code.  The napari interaction
(feeding a lazy multiscale array to the viewer, progressive display) lives in the
UI layer on top, never here.

Shares its axis-resolution + zarr conventions with the standalone scaled-seg
whole-tissue engine (`utils/{axes,io}.py`) by PORTING +
parity (the engine must run standalone on HPC, so it cannot import this package).
"""
