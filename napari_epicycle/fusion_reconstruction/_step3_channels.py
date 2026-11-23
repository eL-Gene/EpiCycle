"""Which channels a new Step 3 run starts with, and which patches have only
one channel (no Qt of its own; the channel count is the Patches rows').

Two small rules shared by the controller's callers:

* ``seed_codes`` -- a new run's two channel codes, from the codes its model
  was trained on.  Both views must record the same codes; anything else (a
  Cellpose built-in records none, or the two views disagree) leaves
  "Channel to segment" blank (``None``) and the helper at 0, so the user
  picks the channel rather than inheriting a guess.
* ``is_one_channel`` -- a patch has one channel when its axes have no C, or
  its C length is 1.  The foot's line (from the cached header shape, no
  file opened) and Predict (from the loaded image) use this ONE rule, so
  the two can never disagree about the same patch.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

#: Cellpose's channel codes: 0 gray / none, 1 red, 2 green, 3 blue
CODE_MIN = 0
CODE_MAX = 3
#: the two roles a channel can have for the model (keys, not words)
ROLE_SEGMENT = "segment"
ROLE_HELPER = "helper"


def channel_roles(codes, n_channels) -> Dict[int, str]:
    """``{channel index: ROLE_SEGMENT | ROLE_HELPER}`` for an image with
    *n_channels* channels run with the two Cellpose *codes*.  A Cellpose
    code is not an array index: code 0 is gray / none, code k >= 1 is the
    image's channel k - 1.  Empty when the codes are not two Cellpose
    codes, the image has one channel (it ran as gray), or the channel to
    segment is 0 (every channel averaged, the helper not used).  A code
    past the last channel is left out; a helper on the channel to segment
    is not named twice."""
    got = valid_codes(codes)
    try:
        n = int(n_channels)
    except (TypeError, ValueError):
        return {}
    if got is None or n <= 1:
        return {}
    c1, c2 = got
    if c1 == 0:
        return {}
    out: Dict[int, str] = {}
    if c1 - 1 < n:
        out[c1 - 1] = ROLE_SEGMENT
    if c2 >= 1 and c2 - 1 < n and (c2 - 1) not in out:
        out[c2 - 1] = ROLE_HELPER
    return out


def valid_codes(codes) -> Optional[Tuple[int, int]]:
    """``(c1, c2)`` when *codes* is two Cellpose codes 0-3, else None."""
    if not isinstance(codes, (list, tuple)) or len(codes) != 2:
        return None
    out = []
    for v in codes:
        if isinstance(v, bool):
            return None
        try:
            i = int(v)
        except (TypeError, ValueError):
            return None
        if i != v or not CODE_MIN <= i <= CODE_MAX:
            return None
        out.append(i)
    return out[0], out[1]


def seed_codes(lateral_codes, axial_codes) -> Tuple[Optional[int], int]:
    """A new run's ``(channel to segment, helper)``: the trained codes when
    both views record the same two; else ``(None, 0)`` -- the first box
    blank, the helper at "0: none"."""
    lat = valid_codes(lateral_codes)
    ax = valid_codes(axial_codes)
    if lat is not None and lat == ax:
        return lat
    return None, 0


def is_one_channel(axes: Optional[str],
                   shape: Optional[Sequence[int]] = None) -> bool:
    """True when the patch has a single channel: a 2D image, *axes* with no
    ``C``, or a ``C`` length of 1 in *shape* -- the channel count of
    ``_step3_rows.channel_count``, the rule the Patches rows use.  Unknown
    axes, or a C axis whose length cannot be read from *shape*, is False
    (the patch is not exempted)."""
    from ._step3_rows import channel_count
    try:
        if shape is not None and len(shape) == 2:
            return True
    except TypeError:
        pass
    return channel_count(axes, shape) == 1


def one_channel_names(names: Iterable[str],
                      axes_by_name: Dict[str, str],
                      shapes_by_name: Dict[str, Sequence[int]]) -> List[str]:
    """The patches of *names* (order kept) that have one channel, by the
    cached axes and header shapes; a patch with no cached axes is not
    counted."""
    out = []
    for name in names:
        if is_one_channel((axes_by_name or {}).get(name),
                          (shapes_by_name or {}).get(name)):
            out.append(name)
    return out


def recorded_codes(record) -> Optional[Tuple[int, int]]:
    """The two codes a prediction or reconstruction manifest says were
    ASKED for (``seg_config.channels``), or None: no record, a blank
    channel to segment (``null``), or codes that are not two Cellpose
    codes.  A one-channel patch's record counts: it ran as gray, but its
    ``seg_config.channels`` keeps the codes the run asked for."""
    if not isinstance(record, dict):
        return None
    seg = record.get("seg_config")
    if not isinstance(seg, dict):
        return None
    return valid_codes(seg.get("channels"))


def record_time(record, key: str, fallback: Optional[float]):
    """When a manifest was written: its own *key* (``predicted_at`` /
    ``generated_at``, ISO 8601; no zone read as UTC) as a POSIX time, else
    *fallback* (the file's mtime)."""
    from datetime import datetime, timezone
    when = record.get(key) if isinstance(record, dict) else None
    if isinstance(when, str) and when.strip():
        try:
            dt = datetime.fromisoformat(when.strip().replace("Z", "+00:00"))
        except ValueError:
            dt = None
        if dt is not None:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
    return fallback


def newest_recorded_codes(entries: Iterable):
    """``((c1, c2), kind)`` of the NEWEST entry that has codes, or None.
    *entries* are ``(time, kind, record)``; prediction and reconstruction
    records are pooled in one time order, and on a tie a reconstruction
    wins (it was built from that prediction)."""
    best = None
    for when, kind, record in entries:
        if when is None:
            continue
        codes = recorded_codes(record)
        if codes is None:
            continue
        key = (float(when), 1 if kind == "reconstruction" else 0)
        if best is None or key > best[0]:
            best = (key, codes, kind)
    return None if best is None else (best[1], best[2])


def first_recorded_codes(records: Iterable) -> Optional[Tuple[int, int]]:
    """The codes of the first record of *records* (newest first) that has
    them (:func:`recorded_codes`), or None.  *records* may be lazy: it is
    read only as far as the first answer."""
    for record in records:
        got = recorded_codes(record)
        if got is not None:
            return got
    return None
