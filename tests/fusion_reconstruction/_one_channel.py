"""A one-channel (ZYX) patch for the channel tests of Steps 3 and 4.

A patch with a single channel is a real case (a nuclear stain saved on its
own), and Cellpose reads it as gray whatever the channel boxes say, so the
channel tests list one beside the usual two-channel patches.  The same
patch can also be written with a channel axis of length 1 (ZCYX with
C = 1), which must count as one channel too.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

#: (Z, Y, X) of the tiny patch
ZYX_SHAPE = (4, 8, 8)
#: the same patch with a channel axis of length 1, (Z, C, Y, X)
ZCYX_C1_SHAPE = (4, 1, 8, 8)


def write_zyx_patch(folder, name: str = "one_ch.tif", c1: bool = False
                    ) -> Path:
    """Write a tiny ``uint16`` one-channel TIFF into *folder*: ZYX
    ``(4, 8, 8)``, or with *c1* ZCYX ``(4, 1, 8, 8)``; returns its path."""
    import tifffile

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    data = (np.arange(int(np.prod(ZYX_SHAPE)), dtype=np.uint16)
            .reshape(ZYX_SHAPE))
    if c1:
        tifffile.imwrite(str(path), data.reshape(ZCYX_C1_SHAPE),
                         photometric="minisblack",
                         metadata={"axes": "ZCYX"})
    else:
        tifffile.imwrite(str(path), data, photometric="minisblack",
                         metadata={"axes": "ZYX"})
    return path


def write_two_channel_patch(folder, name: str = "two_ch.tif") -> Path:
    """Write a tiny ``uint16`` two-channel CZYX TIFF ``(2, 4, 8, 8)``."""
    import tifffile

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / name
    data = np.zeros((2,) + ZYX_SHAPE, dtype=np.uint16)
    tifffile.imwrite(str(path), data, photometric="minisblack",
                     metadata={"axes": "CZYX"})
    return path
