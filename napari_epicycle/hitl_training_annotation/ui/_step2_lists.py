"""Step 2's painted lists.

BRICK L: the Session list (wide rows, 128 px picture), Curation's Patches
list (narrow rows, 80 px picture), the per-patch tally behind both, the
Run summary card and the Up / Down key-handler list.  BRICK A (the end of
this file): the Slices rows and Training's staged rows (56 px pictures of
the actual slice), the filter-chip counts, the Slices sort order and the
``SlicePictures`` manager.

This is the ONLY module that knows the shape of a ``sample_records`` entry
(``sample_id -> {patch_id, bucket, status, round_tag, ...}``, built by the
controller's ``_touch_sample``).  The controller never builds a row: its
``_refresh_patch_list_badges`` hands this module the read-only payload of
``on_session_patches`` and the widget calls :class:`PatchLists`, which
fills BOTH patch lists from ONE tally -- Session's wide rows and Curation's
narrow ones -- and keeps their pictures.

Layers: the builders (``tally_records`` ... ``curation_patch_record``) are
pure; the Qt pieces (the delegate-painted lists, ``PatchPictures``,
``RunSummaryCard``) import ``qtpy`` at module top like the shared delegate
does.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from qtpy.QtCore import QObject, QTimer, Qt, Signal
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (QFrame, QGridLayout, QHBoxLayout, QLabel,
                            QListWidget, QVBoxLayout, QWidget)

from ...ui_shell._stack_row_delegate import (
    MODE_NARROW, MODE_WIDE, ROLE_RECORD, ROLE_THUMB, ROLE_THUMB_TIP,
    StackListWidget, StackRowDelegate, list_cap_px)
from ...ui_shell.theme import (
    DIM_LABEL_STYLE, PANEL_CARD_BG, PANEL_CARD_BORDER, PANEL_CHIP_CHECKED_BG,
    PANEL_CHIP_CHECKED_BORDER, PANEL_COLHEAD_STYLE, PANEL_GOLD,
    PANEL_SLICES_HEADER_STYLE, PANEL_VALUE_STYLE, PANEL_ZERO_COLOR)
from . import _step2_text as T

BUCKETS = ("lateral", "axial")
#: the strip's grey for the part of a patch nobody has touched yet
STRIP_REST = "#333333"
#: what the strip draws first (left) to last
STRIP_ORDER = ("committed", "staged", "accepted", "curating", "predicted")
#: Step 3's row palette, restated here (its constant lives in a 4000-line
#: Step 3 module): gold selection tint, a dark tick glyph, the warn amber
ROW_PALETTE = {
    "sel_bg": QColor(212, 168, 67, 64),
    "chk_mark": QColor("#1d1e24"),
    "warn": QColor("#e6c26a"),
}
ROW_ACCENT = PANEL_GOLD
#: the property the training lock sets on a list so the painter dims its rows
LOCK_PROPERTY = "step2Locked"
#: the key-burst wait before Up / Down load the patch (D18)
KEY_LOAD_DELAY_MS = 250
#: header probes arriving close together refill the lines ONCE
META_COALESCE_MS = 50


# ---------------------------------------------------------------------------
# The tally (pure)
# ---------------------------------------------------------------------------

def patch_of_record(sample_id: str, record: Mapping[str, Any]) -> str:
    """The patch stem a record belongs to: its ``patch_id`` / ``roi_id``,
    else derived from the sample id (``{patch}_{view}_{index}``) -- the
    controller's own rule (``_refresh_patch_list_badges``)."""
    patch = str(record.get("patch_id") or record.get("roi_id") or "").strip()
    if not patch or patch == "-":
        parts = str(sample_id).rsplit("_", 2)
        patch = parts[0] if len(parts) >= 3 else str(sample_id)
    return patch


def _empty_counts() -> Dict[str, int]:
    return {s: 0 for s in T.STATUS_ORDER}


def tally_records(sample_records: Optional[Mapping[str, Mapping[str, Any]]]
                  ) -> Dict[str, Dict[str, Dict[str, int]]]:
    """``{patch: {"lateral": {status: n}, "axial": {status: n}, "rounds":
    {bucket: highest committed round index}}}`` over every record, BOTH
    buckets (today's tally counts the active one only).  A status outside
    the table (``rejected``) is not counted."""
    out: Dict[str, Dict[str, Any]] = {}
    for sample_id, rec in (sample_records or {}).items():
        if not isinstance(rec, Mapping):
            continue
        bucket = rec.get("bucket")
        if bucket not in BUCKETS:
            continue
        status = rec.get("status", "")
        if status not in T.STATUS_WORDS:
            continue
        patch = patch_of_record(sample_id, rec)
        slot = out.setdefault(patch, {"lateral": _empty_counts(),
                                      "axial": _empty_counts(),
                                      "rounds": {}})
        slot[bucket][status] += 1
        if status == "committed":
            n = T.display_round(rec.get("round_tag"))
            if n is not None:
                prior = slot["rounds"].get(bucket)
                slot["rounds"][bucket] = n if prior is None else max(prior, n)
    return out


def counts_for(tally, patch: str, bucket: str) -> Dict[str, int]:
    """The per-status counts of *patch* in *bucket* (all zero for none)."""
    slot = (tally or {}).get(patch)
    return dict(slot[bucket]) if slot else _empty_counts()


def total_records(counts: Mapping[str, int]) -> int:
    return int(sum(int(v) for v in counts.values()))


def has_records(tally, patch: str) -> bool:
    """True when the patch has a record in EITHER bucket (D22)."""
    slot = (tally or {}).get(patch)
    return bool(slot) and (total_records(slot["lateral"])
                           + total_records(slot["axial"])) > 0


def latest_round(tally, patch: str, bucket: Optional[str] = None
                 ) -> Optional[int]:
    """The highest round a trained slice of *patch* is in -- in *bucket*,
    or (``None``) either bucket -- or ``None``: the round chip's number."""
    slot = (tally or {}).get(patch)
    if not slot or not slot["rounds"]:
        return None
    if bucket is not None:
        return slot["rounds"].get(bucket)
    return max(slot["rounds"].values())


def classify_patch(tally, patch: str) -> str:
    """``not_started`` / ``trained`` / ``finalized`` / ``in_progress`` --
    exclusive, in that priority, over both buckets (the Session summary)."""
    slot = (tally or {}).get(patch)
    if not has_records(tally, patch):
        return "not_started"
    if slot["lateral"]["committed"] + slot["axial"]["committed"] > 0:
        return "trained"
    if slot["lateral"]["staged"] + slot["axial"]["staged"] > 0:
        return "finalized"
    return "in_progress"


def session_summary(tally, patch_names: Iterable[str]) -> str:
    """"15 patches . 4 in a trained round . 2 finalized . 8 in progress .
    1 not started"; ``""`` for no patches."""
    names = list(patch_names)
    if not names:
        return ""
    tot = {"trained": 0, "finalized": 0, "in_progress": 0, "not_started": 0}
    for stem in names:
        tot[classify_patch(tally, stem)] += 1
    return T.SESSION_PATCHES_SUMMARY.format(
        n=len(names), t=tot["trained"], f=tot["finalized"],
        p=tot["in_progress"], s=tot["not_started"])


def strip_segments(counts: Mapping[str, int], total: int):
    """The proportion strip: ``((fraction, colour), ...)`` for the statuses
    in :data:`STRIP_ORDER` plus the grey rest; every fraction is
    ``count / total`` and they sum to at most 1 (a patch with more records
    than slices is scaled to fit)."""
    total = max(int(total), 1)
    curated = total_records(counts)
    denom = max(total, curated)
    segs = []
    for status in STRIP_ORDER:
        n = int(counts.get(status, 0))
        if n > 0:
            segs.append((n / denom, T.STATUS_COLOURS[status]))
    rest = (denom - curated) / denom
    if rest > 1e-9:
        segs.append((rest, STRIP_REST))
    return tuple(segs)


def slice_total(shape, bucket: str, curated: int) -> int:
    """The denominator of a patch's strip: lateral = Z, axial = Y + X (the
    controller's rule); unknown shape -> ``max(curated, 1)``."""
    try:
        z, y, x = (int(v) for v in list(shape)[-3:])
        return z if bucket == "lateral" else y + x
    except (TypeError, ValueError):
        return max(int(curated), 1)


def status_slots(counts: Mapping[str, int]):
    """``((icon, colour, n, word), ...)`` of the five statuses, in the
    counts line's order (the delegate drops the zeros of a wide row)."""
    return tuple((T.STATUS_ICONS[s], T.STATUS_COLOURS[s],
                  int(counts.get(s, 0)), T.STATUS_WORDS[s])
                 for s in T.STATUS_ORDER)


# ---------------------------------------------------------------------------
# The rows (pure)
# ---------------------------------------------------------------------------

def _stem(name: str) -> str:
    return os.path.splitext(str(name))[0]


def meta_line(meta, shape) -> str:
    """``CZYX . 124 x 240 x 240 . 2 ch`` -- each part only when known
    (the header probe's axes / channels, else the cached shape)."""
    parts: List[str] = []
    axes = getattr(meta, "axes", "") if meta is not None else ""
    if axes:
        parts.append(axes)
    dims = T.dims_text(shape)
    if dims:
        parts.append(dims)
    if meta is not None and not getattr(meta, "reason", ""):
        parts.append(T.ROW_CHANNELS.format(c=int(meta.n_channels)))
    return T.SEP.join(parts)


def tail_keep_chars(stem: str) -> int:
    """How many trailing characters of a patch name must survive a middle
    elision: ``WholeScale_patch_8`` -> ``patch_8`` (7).  The last two
    ``_`` parts; the last six characters when there is no such pair."""
    stem = str(stem)
    last = stem.rfind("_")
    prev = stem.rfind("_", 0, last) if last > 0 else -1
    if prev > 0:
        return len(stem) - prev - 1
    return min(6, max(len(stem) - 1, 1))


def line2_variants(line2: str) -> tuple:
    """Shorter spellings of a row's line 2, longest first: the full line,
    then no spaces around the times signs, then no spaces around the dots
    either (``CZYX . 124x240x240 . 2 ch`` at 250 px)."""
    full = str(line2)
    tight = full.replace(" × ", "×")
    tighter = tight.replace(" · ", "·")
    out = []
    for v in (full, tight, tighter):
        if v and v not in out:
            out.append(v)
    return tuple(out)


def _voxel_tip(spacing) -> str:
    text = T.voxel_text(spacing)
    return T.ROW_VOXEL_TIP_NONE if text == T.ROW_VOXEL_NONE else text


def _round_chips(tally, patch: str, bucket: str) -> list:
    """The round chips: the active bucket's round plain (``round_01``), the
    OTHER bucket's prefixed with its family (``ax round_00``) -- the counts
    follow the View, so a round of the other bucket must not read as theirs."""
    out = []
    n = latest_round(tally, patch, bucket)
    if n is not None:
        out.append((T.ROW_CHIP_ROUND.format(n=n), "round"))
    other = "axial" if bucket == "lateral" else "lateral"
    m = latest_round(tally, patch, other)
    if m is not None:
        prefix = "ax" if other == "axial" else "lat"
        out.append((f"{prefix} {T.ROW_CHIP_ROUND.format(n=m)}", "round"))
    return out


def run_head_tail(desc, pairs) -> str:
    """The Run summary head after the run name: ``whole cell . 83 pairs`` --
    each part only when known (``pairs`` must be a number)."""
    parts: List[str] = []
    d = str(desc or "").strip()
    if d:
        parts.append(d)
    try:
        n = None if isinstance(pairs, bool) else int(pairs)
    except (TypeError, ValueError):
        n = None
    if n is not None:
        parts.append(f"{n} {T.RUN_SUMMARY_PAIRS_UNIT}")
    return T.SEP.join(parts)


def session_row_record(name: str, *, tally, shape=None, meta=None,
                       spacing=None, bucket: str = "lateral") -> Dict[str, Any]:
    """The ``MODE_WIDE`` record of one Session row.

    Line 2 ``CZYX . 124 x 240 x 240 . 2 ch . 2 of 124 lateral slices
    curated``; line 3 the voxel size (the amber "voxels" when none); the
    icon-counts line (one icon + number + word per non-zero status, the
    ``grey_word`` "no records yet" for a patch with no record, dimmed slots
    for a patch with records only in the OTHER bucket, D22) followed by the
    chips ``lat n`` / ``ax n`` and the round chip; the proportion strip.
    No status word, no tick, no eye (read-only)."""
    stem = _stem(name)
    counts = counts_for(tally, stem, bucket)
    curated = total_records(counts)
    # "curated" = touched by the user: a merely predicted slice is not
    curated_k = max(curated - int(counts.get("predicted", 0)), 0)
    total = slice_total(shape, bucket, curated)
    head = meta_line(meta, shape)
    tail = T.ROW_CURATED.format(k=curated_k, z=total, bucket=bucket)
    line2 = T.SEP.join(p for p in (head, tail) if p)
    vox = T.voxel_text(spacing)
    rec: Dict[str, Any] = {
        "name": name, "word": "", "dots": "",
        "colour": T.NOT_STARTED_COLOUR,
        "axes": getattr(meta, "axes", "") if meta is not None else "",
        "line2": line2, "line3": vox, "line3_warn": vox == T.ROW_VOXEL_NONE,
        "row_tip": _voxel_tip(spacing) if vox == T.ROW_VOXEL_NONE else None,
    }
    slot = (tally or {}).get(stem)
    anywhere = has_records(tally, stem)
    if not anywhere:
        rec["grey_word"] = T.ROW_NO_RECORDS
    else:
        rec["status_counts"] = status_slots(counts)
        if curated == 0:
            rec["dim_slots"] = True
        chips = [(T.ROW_CHIP_LATERAL.format(n=total_records(slot["lateral"])),
                  "" if total_records(slot["lateral"]) else "dim"),
                 (T.ROW_CHIP_AXIAL.format(n=total_records(slot["axial"])),
                  "" if total_records(slot["axial"]) else "dim")]
        chips.extend(_round_chips(tally, stem, bucket))
        rec["chips_inline"] = tuple(chips)
        rec["strip"] = strip_segments(counts, total)
    return rec


#: a Patches row's picture in a short dock (the room buys Slices a row)
SHORT_PATCH_THUMB_PX = 56


def curation_patch_record(name: str, *, tally, shape=None, meta=None,
                          spacing=None, bucket: str = "lateral",
                          loaded: bool = False,
                          thumb_px: Optional[int] = None) -> Dict[str, Any]:
    """The ``MODE_NARROW`` record of one Curation Patches row: name, line 2
    ``CZYX . 81 x 240 x 240 . 2 ch``, line 3 the five icon slots with their
    numbers (counts of the ACTIVE bucket -- they follow the View switch, the
    picture does not), the proportion strip; the loaded row carries the
    delegate's amber bar + glow (``"loaded"``).  A patch with no record
    shows the grey "not started" in the counts' place; one with records
    only in the other bucket shows the dimmed slots (D22).  The voxel size
    is the row's hover text."""
    stem = _stem(name)
    counts = counts_for(tally, stem, bucket)
    curated = total_records(counts)
    total = slice_total(shape, bucket, curated)
    vox = T.voxel_text(spacing)
    rec: Dict[str, Any] = {
        # fix 2: no extension on a narrow row, and the tail (patch_8) always
        # survives the elision; line 2 has shorter spellings; the counts
        # line drops the zero slots when five would wrap
        "name": stem, "name_keep_tail": tail_keep_chars(stem),
        "word": "", "dots": "", "colour": T.NOT_STARTED_COLOUR,
        "no_icon": True, "line2_plain": meta_line(meta, shape),
        "line2_variants": line2_variants(meta_line(meta, shape)),
        "counts_compact": True,
        "loaded": bool(loaded),
        "row_tip": vox if vox != T.ROW_VOXEL_NONE else T.ROW_VOXEL_TIP_NONE,
    }
    if thumb_px:
        rec["thumb_px"] = int(thumb_px)
    if not has_records(tally, stem):
        rec["grey_word"] = T.ROW_NOT_STARTED
    else:
        rec["status_counts"] = status_slots(counts)
        if curated == 0:
            rec["dim_slots"] = True
        rec["strip"] = strip_segments(counts, total)
    return rec


# ---------------------------------------------------------------------------
# The Run summary numbers (pure, D21)
# ---------------------------------------------------------------------------

def run_summary_rounds(round_by_view: Optional[Mapping[str, Any]],
                       run_id: str) -> Dict[str, Dict[str, str]]:
    """``{bucket: {"last", "next", "model"}}`` as texts.  ``round_by_view``
    is the controller's ``current_round_by_view`` -- per bucket the 0-based
    round a Train would write NEXT (disk numbering, D16); "Last round" is
    the one before it, none yet at 0.  Never parsed from the "Lat 3 | Axi 2"
    label (D21)."""
    out: Dict[str, Dict[str, str]] = {}
    for bucket in BUCKETS:
        try:
            nxt = int(round_by_view.get(bucket, 0))
        except (AttributeError, TypeError, ValueError):
            nxt = 0
        nxt_n = T.display_round(nxt)
        last_n = T.display_round(nxt - 1)
        out[bucket] = {
            "next": str(nxt_n) if nxt_n is not None else T.RUN_SUMMARY_NONE,
            "last": str(last_n) if last_n is not None
            else T.RUN_SUMMARY_NONE,
            "model": (T.RUN_SUMMARY_MODEL.format(run=run_id or "default",
                                                 n=last_n)
                      if last_n is not None else T.RUN_SUMMARY_NONE),
        }
    return out


# ---------------------------------------------------------------------------
# The key-handler list (D18)
# ---------------------------------------------------------------------------

class PatchKeyList(StackListWidget):
    """Curation's Patches list: a ``StackListWidget`` whose Up / Down move
    the current row and, after ``KEY_LOAD_DELAY_MS`` of silence, emit
    ``loadRequested(item)`` -- the controller connects that to the same
    slot as ``itemClicked``.  There is NO ``currentItemChanged`` hookup: a
    mouse press fires ``currentItemChanged`` and the release fires
    ``itemClicked``, which would load the patch twice (~6 s each with the
    annotator open)."""

    loadRequested = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._key_timer = QTimer(self)
        self._key_timer.setSingleShot(True)
        self._key_timer.setInterval(KEY_LOAD_DELAY_MS)
        self._key_timer.timeout.connect(self._fire_load)
        self._burst_row = -1          # the row the current key burst began on
        self.loaded_name = ""         # the patch already loaded: never reloaded

    def set_loaded_name(self, name: str) -> None:
        self.loaded_name = str(name or "")

    def mousePressEvent(self, event):
        # a click loads through ``itemClicked``; a pending key load must not
        # fire a second time after it
        self._key_timer.stop()
        super().mousePressEvent(event)

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key_Up, Qt.Key_Down) and self.isEnabled():
            delta = -1 if key == Qt.Key_Up else 1
            n = self.count()
            if n:
                before = self.currentRow()
                row = 0 if before < 0 else min(max(before + delta, 0), n - 1)
                if row != before:
                    if not self._key_timer.isActive():
                        self._burst_row = before
                    self.setCurrentRow(row)
                    self._key_timer.start()
            event.accept()
            return
        super().keyPressEvent(event)

    def _fire_load(self):
        item = self.currentItem()
        if item is None or not self.isEnabled():
            return
        if self.currentRow() == self._burst_row:
            return                      # the burst ended where it began
        name = _item_name(item)
        if self.loaded_name and self.loaded_name in (name, _stem(name)):
            return                      # that patch is already loaded
        self.loadRequested.emit(item)


# ---------------------------------------------------------------------------
# Pictures on the rows (main thread)
# ---------------------------------------------------------------------------

def _item_name(item) -> str:
    """The row's name: its ``UserRole``; for a row with none -- Training's
    staged rows, added by ``addItem(sample_id)`` -- its text, which IS the
    sample id there."""
    if item is None:
        return ""
    v = item.data(Qt.UserRole)
    if v is None and hasattr(item, "text"):
        v = item.text()
    return str(v or "")


def find_item(list_widget, name: str):
    """The row whose ``UserRole`` is *name*, or ``None``."""
    for i in range(list_widget.count()):
        it = list_widget.item(i)
        if _item_name(it) == name:
            return it
    return None


class PatchPictures:
    """The patch pictures of the two lists: asks the worker for the misses
    only, keeps the in-memory ``ThumbCache`` and sets ``ROLE_THUMB`` /
    ``ROLE_THUMB_TIP`` on the rows of every registered list SYNCHRONOUSLY
    from the cache (a refilled list keeps its pictures), and stores each
    patch's header probe (``meta``).  Bound methods only; main thread only.
    """

    def __init__(self, on_meta: Optional[Callable[[], None]] = None):
        from ...ui_shell._stack_thumbs import ThumbCache
        from .. import _step2_thumbs as TH

        self._TH = TH
        self.cache = ThumbCache()
        self.state = TH.Step2ThumbState()
        self.lists: List[Any] = []
        self.meta: Dict[str, Any] = {}
        self._keys: Dict[str, tuple] = {}     # name -> (path, mtime, size)
        self._pending: set = set()
        self._req_keys: Dict[str, tuple] = {}  # name -> the key REQUESTED
        self._failed: Dict[str, tuple] = {}   # name -> (key, tip)
        self._seen: Optional[tuple] = None     # (workspace, stack_dir)
        self._workspace = ""
        self._stack_dir = ""
        self._swept_for: Optional[tuple] = None
        self._on_meta = on_meta

    # -- the request side -------------------------------------------------
    def refresh(self, workspace: str, stack_dir: str, names: List[str]) -> int:
        """Stat the listed patches, apply cached pictures to the rows, and
        start a batch for the misses not already in flight -> how many
        requested.  Cheap enough for every list refresh: one ``os.stat``
        per patch, no file opened."""
        TH = self._TH
        now = (workspace or "", stack_dir or "")
        if self._seen is not None and self._seen != now:
            self.forget_workspace()       # names are only unique per folder
        self._seen = now
        self._workspace, self._stack_dir = now
        reqs = TH.patch_thumb_requests(self._workspace, self._stack_dir,
                                       names)
        self._keys = {r.name: (r.path, r.mtime_ns, r.st_size) for r in reqs}
        misses = []
        for r in reqs:
            key = self._keys[r.name]
            if self.cache.get(*key) is not None:
                continue
            f = self._failed.get(r.name)
            if f is not None and f[0] == key:
                continue                      # failed for this very file
            misses.append(r)
        self.apply_to_rows()
        self._sweep_if_changed(names)
        wanted = {r.name for r in misses}
        if not wanted or wanted <= self._pending:
            return 0
        if not self._workspace:
            return 0                      # nothing starts: nothing pends
        self._pending = set(wanted)
        for r in misses:
            self._req_keys[r.name] = self._keys[r.name]
        TH.start_patch_thumbs(misses, TH.patch_thumbs_dir(self._workspace),
                              self._on_ready, self._on_failed,
                              self._on_meta_json, state=self.state,
                              on_finished=self._on_finished)
        return len(misses)

    def _sweep_if_changed(self, names) -> None:
        sig = (self._workspace, tuple(sorted(str(n) for n in names)))
        if sig == self._swept_for or not self._workspace:
            return
        self._swept_for = sig
        try:
            self._TH.sweep_unlisted_patches(self._workspace, names)
        except Exception as exc:      # noqa: BLE001 -- a cache sweep
            print(f"[STEP2-THUMB] sweep failed: {exc}", flush=True)

    # -- the row side -------------------------------------------------------
    def apply_to_rows(self) -> None:
        """ROLE_THUMB (the cached pixmap) and ROLE_THUMB_TIP on every row of
        every list, from memory."""
        for lst in self.lists:
            for i in range(lst.count()):
                self._apply(lst.item(i))
            lst.viewport().update()

    def _apply(self, item) -> None:
        name = _item_name(item)
        key = self._keys.get(name)
        pm = self.cache.get(*key) if key is not None else None
        item.setData(ROLE_THUMB, pm)
        tip = None
        if pm is not None:
            tip = T.THUMB_TIP
        elif name in self._failed and self._failed[name][0] == key:
            tip = self._failed[name][1]
        elif name in self._pending:
            tip = T.THUMB_PENDING
        item.setData(ROLE_THUMB_TIP, tip)

    # -- worker callbacks (main thread; bound methods) ----------------------
    def _on_ready(self, name: str, path: str, png_path: str,
                  data: bytes) -> None:
        # cached under the key the picture was REQUESTED with: a patch
        # rewritten during the render then misses at the next refresh
        key = self._req_keys.get(name) or self._keys.get(name)
        if key is None or os.path.normcase(os.path.abspath(path)) != \
                os.path.normcase(os.path.abspath(key[0])):
            return
        self.cache.load(*key, png_path=png_path or None, png_bytes=data)
        self._pending.discard(name)
        self._failed.pop(name, None)
        for lst in self.lists:
            it = find_item(lst, name)
            if it is not None:
                self._apply(it)
                lst.viewport().update()

    def _on_failed(self, name: str, path: str, reason: str) -> None:
        key = self._keys.get(name)
        self._pending.discard(name)
        if key is None:
            return
        self._failed[name] = (key, self._TH.thumb_tooltip_for_failure(reason))
        for lst in self.lists:
            it = find_item(lst, name)
            if it is not None:
                self._apply(it)
                lst.viewport().update()

    def _on_meta_json(self, name: str, path: str, text: str) -> None:
        try:
            self.meta[name] = self._TH.PatchMeta.from_json(text)
        except (ValueError, TypeError):
            return
        if self._on_meta is not None:
            self._on_meta()

    def _on_finished(self) -> None:
        self._pending.clear()

    # -- the end -------------------------------------------------------------
    def stop(self, wait_ms: int = 3000) -> bool:
        """``on_shutdown`` / a workspace close: stop the batch (bounded)."""
        self._pending.clear()
        return self._TH.stop_thumbs(self.state, wait_ms=wait_ms)

    def forget_patch(self, workspace: str, patch_name: str) -> None:
        """Forget a patch (file name or stem): drop its picture from memory
        and its PNG from the patches folder."""
        stem = os.path.splitext(str(patch_name))[0]
        for nm in list(self._keys):
            if nm == patch_name or os.path.splitext(nm)[0] == stem:
                key = self._keys.pop(nm)
                self.cache.discard_path(key[0])
                self.meta.pop(nm, None)
                self._failed.pop(nm, None)
        self._TH.remove_patch_pngs(workspace, patch_name)

    def forget_workspace(self) -> None:
        """A different workspace: drop everything of the old one."""
        self.stop(wait_ms=0)
        self.cache.clear()
        self.meta.clear()
        self._keys.clear()
        self._req_keys.clear()
        self._failed.clear()
        self._swept_for = None


# ---------------------------------------------------------------------------
# The two lists, filled from one tally
# ---------------------------------------------------------------------------

def build_session_list(parent, *, help_lines=None) -> StackListWidget:
    """Session's read-only wide-row list (Step 3's Session list shape):
    a ``StackListWidget`` on the shared delegate in ``MODE_WIDE``, no tick,
    no eye, no menu, nothing selects."""
    lst = StackListWidget(parent)
    lst.setSelectionMode(QListWidget.NoSelection)
    lst.setFocusPolicy(Qt.NoFocus)
    lst.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    deleg = StackRowDelegate(
        MODE_WIDE, lst, palette=ROW_PALETTE, accent=ROW_ACCENT,
        show_tick=False, show_eye=False, lock_property=LOCK_PROPERTY,
        eye_lock_property=LOCK_PROPERTY, dim_when_locked=True,
        keep_aspect=True, help_lines=help_lines or {})
    lst.setItemDelegate(deleg)
    lst._epicycle_delegate = deleg           # keep alive with the list
    lst.set_empty_lines(T.SESSION_PATCHES_EMPTY_LINES)
    return lst


def build_curation_patch_list(parent, *, help_lines=None) -> PatchKeyList:
    """Curation's narrow-row Patches list with the Up / Down handler."""
    lst = PatchKeyList(parent)
    lst.setSelectionMode(QListWidget.SingleSelection)
    deleg = StackRowDelegate(
        MODE_NARROW, lst, palette=ROW_PALETTE, accent=ROW_ACCENT,
        show_tick=False, show_eye=False, show_thumb=True, keep_aspect=True,
        lock_property=LOCK_PROPERTY, eye_lock_property=LOCK_PROPERTY,
        dim_when_locked=True, help_lines=help_lines or {})
    lst.setItemDelegate(deleg)
    lst._epicycle_delegate = deleg
    lst.set_empty_lines(T.CURATION_PATCHES_EMPTY_LINES)
    return lst


class PatchLists:
    """Fills Session's list and Curation's Patches list from ONE refresh.

    ``session`` is a ``StackListWidget`` this class owns the rows of (it is
    rebuilt each refresh from the Curation list's names, so both lists
    always list the same patches); ``curation`` is the controller's
    ``train_patch_list`` whose rows the controller populates -- this class
    only sets roles on them, it never adds, removes or retexts a row
    (``item.text()`` and ``UserRole`` stay as the controller reads them).
    """

    def __init__(self, session, curation, pictures: PatchPictures, *,
                 on_summary: Optional[Callable[[str], None]] = None,
                 session_floor_rows: int = 2):
        self.session = session
        self.curation = curation
        self.pictures = pictures
        self.on_summary = on_summary
        self._floor_rows = session_floor_rows
        pictures.lists = [session, curation]
        self._last: Optional[Mapping[str, Any]] = None
        self._short = False
        self._spacing_key: Optional[tuple] = None
        self._spacing_entries: Dict[str, Any] = {}
        self._meta_timer = QTimer()
        self._meta_timer.setSingleShot(True)
        self._meta_timer.setInterval(META_COALESCE_MS)
        self._meta_timer.timeout.connect(self._refill_now)

    # -- the one entry point (hook 4) -----------------------------------------
    def fill(self, payload: Mapping[str, Any]) -> None:
        self._last = payload
        names = self._names_from_curation()
        self._sync_session_rows(names)
        self.pictures.refresh(str(payload.get("workspace") or ""),
                              str(payload.get("stack_dir") or ""), names)
        self._apply_records(payload, names)
        self.fit_session_height()

    def set_short(self, short: bool) -> None:
        """A short dock draws Curation's Patches rows with the 56 px
        picture (Session's wide rows are unchanged); redraws them at once."""
        short = bool(short)
        if short == self._short:
            return
        self._short = short
        if self._last is not None:
            self._refill_now()

    def refill_last(self) -> None:
        """A header probe arrived: rebuild the lines with the new meta --
        once for a burst of probes (``META_COALESCE_MS``)."""
        if self._last is not None and not self._meta_timer.isActive():
            self._meta_timer.start()

    def _refill_now(self) -> None:
        if self._last is not None:
            self._apply_records(self._last, self._names_from_curation())
            self.fit_session_height()

    def stop(self) -> None:
        self._meta_timer.stop()

    def _names_from_curation(self) -> List[str]:
        out = []
        for i in range(self.curation.count()):
            n = _item_name(self.curation.item(i))
            if n:
                out.append(n)
        return out

    def _sync_session_rows(self, names: List[str]) -> None:
        from qtpy.QtWidgets import QListWidgetItem

        have = [_item_name(self.session.item(i))
                for i in range(self.session.count())]
        if have == names:
            return
        self.session.clear()
        for n in names:
            it = QListWidgetItem(n)
            it.setData(Qt.UserRole, n)
            it.setFlags(Qt.ItemIsEnabled)          # read-only: nothing selects
            self.session.addItem(it)

    def _load_spacing(self, workspace: str) -> None:
        """Stat ``patch_spacing.json`` ONCE per fill; re-read it only when
        its (workspace, mtime) changed.  Never raises."""
        if not workspace:
            self._spacing_key, self._spacing_entries = None, {}
            return
        try:
            from napari_epicycle.data_preparation import _voxel_spacing as VS

            path = VS.patch_spacing_path(workspace)
            try:
                mtime = os.stat(path).st_mtime_ns
            except OSError:
                mtime = None
            if self._spacing_key != (workspace, mtime):
                self._spacing_key = (workspace, mtime)
                self._spacing_entries = (VS.read_patch_spacing_file(
                    workspace, quarantine=False) if mtime is not None else {})
        except Exception:          # noqa: BLE001 -- decoration only
            self._spacing_key, self._spacing_entries = None, {}

    def _spacing_for(self, workspace: str, stack_dir: str, name: str):
        """The patch's ``patch_spacing`` entry from the entries that
        :meth:`_load_spacing` read, or ``None``.  No file access."""
        if not workspace or not self._spacing_entries:
            return None
        try:
            from pathlib import Path

            from napari_epicycle.data_preparation import _voxel_spacing as VS

            key = VS.patch_spacing_key(str(Path(stack_dir) / name), workspace)
            return VS.find_patch_spacing(self._spacing_entries, key)
        except Exception:          # noqa: BLE001 -- decoration only
            return None

    def _apply_records(self, payload, names: List[str]) -> None:
        tally = tally_records(payload.get("records"))
        shapes = payload.get("shapes") or {}
        bucket = "axial" if str(payload.get("view") or "").upper() in (
            "XZ", "YZ") else "lateral"
        loaded = str(payload.get("loaded") or "")
        ws = str(payload.get("workspace") or "")
        sd = str(payload.get("stack_dir") or "")
        metas = self.pictures.meta
        self._load_spacing(ws)
        if hasattr(self.curation, "set_loaded_name"):
            self.curation.set_loaded_name(loaded)
        for lst, wide in ((self.session, True), (self.curation, False)):
            prev_block = lst.blockSignals(True)
            try:
                for i in range(lst.count()):
                    it = lst.item(i)
                    name = _item_name(it)
                    stem = _stem(name)
                    kw = dict(tally=tally, shape=shapes.get(stem),
                              meta=metas.get(name),
                              spacing=self._spacing_for(ws, sd, name),
                              bucket=bucket)
                    rec = (session_row_record(name, **kw) if wide else
                           curation_patch_record(
                               name, loaded=bool(loaded) and loaded in (
                                   name, stem),
                               thumb_px=(SHORT_PATCH_THUMB_PX
                                         if self._short else None), **kw))
                    it.setData(ROLE_RECORD, rec)
            finally:
                lst.blockSignals(prev_block)
            lst.doItemsLayout()
            lst.viewport().update()
        self.pictures.apply_to_rows()
        if self.on_summary is not None:
            self.on_summary(session_summary(tally, [_stem(n) for n in names]))

    # -- the Session list's height ---------------------------------------------
    def fit_session_height(self) -> None:
        """Content height, at least two wide rows, at most four (the list
        scrolls inside itself above that, ``LIST_CAP_ROWS``)."""
        lst = self.session
        floor = list_cap_px(MODE_WIDE, self._floor_rows)
        cap = list_cap_px(MODE_WIDE)
        content = sum(max(lst.sizeHintForRow(i), 0)
                      for i in range(lst.count())) + 6
        lst.setFixedHeight(max(floor, min(cap, content)))


# ---------------------------------------------------------------------------
# The Run summary card
# ---------------------------------------------------------------------------

# the card is the panel card (one card look for Session and Curation)
_CARD_STYLE = (
    "QFrame#step2RunSummaryCard {"
    f" background: {PANEL_CARD_BG};"
    f" border: 1px solid {PANEL_CARD_BORDER}; border-radius: 4px; }}")
_DIM_STYLE = DIM_LABEL_STYLE + " background: transparent;"
_ROW_STYLE = PANEL_VALUE_STYLE
_VAL_STYLE = PANEL_VALUE_STYLE
_COLHEAD_STYLE = PANEL_COLHEAD_STYLE


class RunSummaryCard(QFrame):
    """The Run summary: ``Run: {name} . {n} pairs``, a
    two-column table (Lateral | Axial: Last round, Next round, Pairs in the
    training set, Last model) and one dim sentence; an empty state ("No
    workspace yet.") instead of a column of zeros.

    The run name and the pair counts are the controller's own labels
    (``status_run_value``, ``status_total_pairs_value``,
    ``status_pairs_lateral_value``, ``status_pairs_axial_value``), RE-PARENTED
    into the card so the controller keeps writing them; the round and model
    cells are written here from ``current_round_by_view`` (D21), never parsed
    from the controller's "Lat 3 | Axi 2" string."""

    def __init__(self, parent, *, run_value: QLabel, total_value: QLabel,
                 lateral_pairs: QLabel, axial_pairs: QLabel):
        super().__init__(parent)
        self.setObjectName("step2RunSummaryCard")
        self.setStyleSheet(_CARD_STYLE)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 6, 8, 6)
        root.setSpacing(4)

        self._empty = QLabel("\n".join(T.RUN_SUMMARY_EMPTY_LINES), self)
        self._empty.setAlignment(Qt.AlignCenter)
        self._empty.setWordWrap(True)
        self._empty.setStyleSheet(_DIM_STYLE)
        self._empty.setMinimumHeight(48)
        root.addWidget(self._empty)

        self._body = QWidget(self)
        body = QVBoxLayout(self._body)
        body.setContentsMargins(0, 0, 0, 0)
        body.setSpacing(4)
        head = QHBoxLayout()
        head.setSpacing(5)
        lab = QLabel(T.RUN_SUMMARY_RUN, self._body)
        lab.setStyleSheet(_ROW_STYLE)
        head.addWidget(lab)
        for w in (run_value,):
            w.setParent(self._body)
            w.setStyleSheet(_VAL_STYLE + " font-weight: bold;")
            head.addWidget(w)
        # The controller writes ``status_total_pairs_value`` with the pairs
        # already in trained rounds -- NOT the run's total.  It stays inside
        # the card (the controller keeps writing it) but hidden; the head
        # reads the run's own ``run_pairs`` / ``run_desc`` from the payload.
        total_value.setParent(self._body)
        total_value.setVisible(False)
        self._head_tail = QLabel("", self._body)
        self._head_tail.setStyleSheet(_DIM_STYLE)
        self._head_tail.setContentsMargins(0, 0, 0, 0)
        self._head_tail.setVisible(False)
        head.addWidget(self._head_tail)
        head.addStretch(1)
        body.addLayout(head)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(2)
        grid.setColumnStretch(0, 1)
        for col, text in ((1, T.RUN_SUMMARY_COL_LATERAL),
                          (2, T.RUN_SUMMARY_COL_AXIAL)):
            h = QLabel(text, self._body)
            h.setStyleSheet(_COLHEAD_STYLE)
            h.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
            grid.addWidget(h, 0, col)
        self._cells: Dict[str, QLabel] = {}
        self._pair_cells: Dict[str, QLabel] = {}
        self._pair_dash: Dict[str, QLabel] = {}
        self._active_bucket = "lateral"
        rows = (("last", T.RUN_SUMMARY_ROW_LAST),
                ("next", T.RUN_SUMMARY_ROW_NEXT),
                ("pairs", T.RUN_SUMMARY_ROW_PAIRS),
                ("model", T.RUN_SUMMARY_ROW_MODEL))
        for r, (key, text) in enumerate(rows, start=1):
            rl = QLabel(text, self._body)
            rl.setStyleSheet(_ROW_STYLE)
            grid.addWidget(rl, r, 0)
            for col, bucket in ((1, "lateral"), (2, "axial")):
                if key == "pairs":
                    cell = lateral_pairs if bucket == "lateral" \
                        else axial_pairs
                    cell.setParent(self._body)
                    cell.setStyleSheet(_VAL_STYLE)
                    self._pair_cells[bucket] = cell
                    dash = QLabel(T.RUN_SUMMARY_PAIRS_ELSEWHERE, self._body)
                    dash.setStyleSheet(_VAL_STYLE)
                    dash.setToolTip(T.RUN_SUMMARY_PAIRS_ELSEWHERE_TIP)
                    dash.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                    dash.setVisible(False)
                    grid.addWidget(dash, r, col)
                    self._pair_dash[bucket] = dash
                else:
                    cell = QLabel(T.RUN_SUMMARY_NONE, self._body)
                    cell.setStyleSheet(_VAL_STYLE)
                    self._cells[f"{bucket}:{key}"] = cell
                cell.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
                grid.addWidget(cell, r, col)
        body.addLayout(grid)
        foot = QLabel(T.RUN_SUMMARY_FOOT, self._body)
        foot.setWordWrap(True)
        foot.setStyleSheet(_DIM_STYLE)
        body.addWidget(foot)
        root.addWidget(self._body)
        self.set_has_workspace(False)
        self.set_active_bucket("lateral")

    def set_has_workspace(self, has: bool) -> None:
        self._empty.setVisible(not has)
        self._body.setVisible(bool(has))

    def set_active_bucket(self, bucket: str) -> None:
        """The controller counts a run's pairs for the ACTIVE bucket only, so
        the other bucket's cell is a dash ("shown for the active view") rather
        than a 0 that reads as a count."""
        bucket = bucket if bucket in BUCKETS else "lateral"
        self._active_bucket = bucket
        for b in BUCKETS:
            on = b == bucket
            self._pair_cells[b].setVisible(on)
            self._pair_dash[b].setVisible(not on)

    def update_from(self, payload: Mapping[str, Any]) -> None:
        """Fill the round and model cells from the hook's payload."""
        self.set_has_workspace(bool(payload.get("workspace")))
        self.set_active_bucket("axial" if str(payload.get("view") or "")
                               .upper() in ("XZ", "YZ") else "lateral")
        tail = run_head_tail(payload.get("run_desc"), payload.get("run_pairs"))
        self._head_tail.setText(f"{T.SEP}{tail}" if tail else "")
        self._head_tail.setVisible(bool(tail))
        texts = run_summary_rounds(payload.get("round_by_view"),
                                   str(payload.get("run_id") or ""))
        for bucket in BUCKETS:
            for key in ("last", "next", "model"):
                self._cells[f"{bucket}:{key}"].setText(texts[bucket][key])

    def head_tail_text(self) -> str:
        """The head after the run name without the leading separator."""
        text = self._head_tail.text()
        return text[len(T.SEP):] if text.startswith(T.SEP) else text

    def cell_text(self, bucket: str, key: str) -> str:
        return self._cells[f"{bucket}:{key}"].text()


# ===========================================================================
# BRICK A -- Slices rows and Training's staged rows
# ===========================================================================
#
# The controller keeps adding the rows (``_add_colored_item``: today's
# ``item.text()`` and ``UserRole = sample_id``) and the staged list
# (``addItem(sample_id)``); this module only sets ``ROLE_RECORD`` and
# ``ROLE_THUMB`` on them, never retexts a row (``_train_model`` reads the
# staged list's ``item(i).text()``).  The record shape it reads is a
# ``sample_records`` entry (``view``, ``slice_index``, ``patch_id``,
# ``status``, ``round_tag``, ``img_uri``, ``mask_uri``, ``updated_at``,
# ``inherited_from_pair`` / ``inherited_from_external``, ``bucket``).

#: the picture's edge on both lists (the narrow row's, D3 / D4)
SLICE_ROW_THUMB_PX = 56
#: the order of the Slices list, first to last (D8, D24: trained, finalized,
#: accepted, curating, predicted); a status outside the table sorts last
SLICE_RANK = {"committed": 5, "staged": 4, "accepted": 3, "curating": 2,
              "predicted": 1}
#: which stored statuses each filter chip counts (the Finalized chip covers
#: trained, as the controller's ``_get_active_status_filters`` does)
FILTER_CHIPS = (("all", None),
                ("predicted", ("predicted",)),
                ("curating", ("curating",)),
                ("accepted", ("accepted",)),
                ("staged", ("staged", "committed")))
#: the glyphs of today's ``item.text()`` (the controller's ``_STATUS_DOTS``)
_QUEUE_DOTS = {"predicted": "▧", "curating": "✏",
               "accepted": "✔", "staged": "⬥",
               "committed": "⬥", "rejected": "✕"}
UNKNOWN_STATUS_COLOUR = "#777777"
UNKNOWN_STATUS_ICON = "mdi6.circle-outline"


def record_view_index(sample_id: str, record: Mapping[str, Any]):
    """``(VIEW, index)`` of a record: its own fields, else parsed from the
    ``{patch}_{VIEW}_{index}`` id; ``("", -1)`` when neither says."""
    view = str(record.get("view") or "").upper()
    idx = record.get("slice_index")
    if view not in ("XY", "XZ", "YZ") or idx is None:
        from .. import _step2_thumbs as TH

        parsed = TH.parse_sample_id(sample_id)
        if parsed is not None:
            view = view if view in ("XY", "XZ", "YZ") else parsed[1]
            idx = parsed[2] if idx is None else idx
    try:
        return view, int(idx)
    except (TypeError, ValueError):
        return view, -1


def slice_name(sample_id: str, record: Mapping[str, Any]) -> str:
    """``XY slice 31`` -- the row's name (mono)."""
    view, idx = record_view_index(sample_id, record)
    if not view or idx < 0:
        return str(sample_id)
    return T.ROW_SLICE_NAME.format(view=view, n=idx)


def format_queue_row(record: Mapping[str, Any]) -> str:
    """Today's ``item.text()`` of a Slices row, character for character
    (the controller's ``_format_queue_row``): ``"  XY  slice 31     patch
    ⬥  round_01"``.  The controller still builds the text; this copy is for
    the tests that hold the two equal and for any caller without the
    controller."""
    view = str(record.get("view", "-")).upper()
    idx = record.get("slice_index", "-")
    patch = str(record.get("patch_id") or record.get("roi_id") or "").strip()
    if not patch or patch == "-":
        sid = str(record.get("sample_id", ""))
        parts = sid.rsplit("_", 2)
        patch = parts[0] if len(parts) >= 3 else sid
    status = record.get("status", "predicted")
    dots = _QUEUE_DOTS.get(status, "")
    round_tag = str(record.get("round_tag", "")).strip()
    if round_tag and round_tag != "-":
        round_str = (f"  {round_tag}" if round_tag.startswith("round")
                     else f"  r{round_tag}")
    else:
        round_str = ""
    prefix = ("↳ " if record.get("inherited_from_pair")
              or record.get("inherited_from_external") else "  ")
    return f"{prefix}{view:<3s} slice {str(idx):<5s}  {patch:<16s} " \
           f"{dots}{round_str}"


def slice_sort_key(record: Mapping[str, Any]):
    """The Slices order: status rank (trained first ... predicted last, a
    status outside the table after them), then ``updated_at`` -- today's
    key with ``curating`` ranked (D24)."""
    return (-SLICE_RANK.get(str(record.get("status", "")), 0),
            str(record.get("updated_at", "")))


def sorted_slice_ids(records: Mapping[str, Mapping[str, Any]],
                     ids: Optional[Iterable[str]] = None) -> List[str]:
    """*ids* (default: every key) of *records* in the Slices order."""
    keys = list(records) if ids is None else [i for i in ids if i in records]
    return sorted(keys, key=lambda sid: slice_sort_key(records[sid]))


def _status_look(status: str):
    """``(word, colour, icon)`` of a stored status (D8); a status outside
    the table (``rejected``) is its own word in grey."""
    if status in T.STATUS_WORDS:
        return (T.STATUS_WORDS[status], T.STATUS_COLOURS[status],
                T.STATUS_ICONS[status])
    return str(status or ""), UNKNOWN_STATUS_COLOUR, UNKNOWN_STATUS_ICON


def slice_chips(record: Mapping[str, Any]):
    """The row's chips, ``((text, warn), ...)``: ``round_00`` on a trained
    slice only (the disk's number, D16), ``imported`` on an imported
    pair."""
    chips = []
    if record.get("status") == "committed":
        n = T.display_round(record.get("round_tag"))
        if n is not None:
            chips.append((T.ROW_CHIP_ROUND.format(n=n), False))
    if record.get("inherited_from_pair") or \
            record.get("inherited_from_external"):
        chips.append((T.ROW_CHIP_IMPORTED, False))
    return tuple(chips)


def build_slice_row(sample_id: str, record: Mapping[str, Any]
                    ) -> Dict[str, Any]:
    """The ``MODE_NARROW`` delegate record of one Slices row: a 56 px
    picture (``thumb_px``; the picture itself is ``ROLE_THUMB``, set by
    :class:`SlicePictures`), the status icon in its colour, the name
    ``XY slice 31`` in the row's mono font, the status word in its colour
    under it, and the chips (``round_00``, ``imported``).  Pure over the
    record."""
    word, colour, icon = _status_look(str(record.get("status", "")))
    rec: Dict[str, Any] = {
        "name": slice_name(sample_id, record), "word": word, "dots": "",
        "colour": colour, "icon": icon, "icon_colour": colour, "axes": "",
        "thumb_px": SLICE_ROW_THUMB_PX,
    }
    chips = slice_chips(record)
    if chips:
        rec["chips"] = chips
    return rec


def labels_text(n: int) -> str:
    return T.ROW_LABELS_ONE if int(n) == 1 else T.ROW_LABELS.format(n=int(n))


def staged_shape_line(meta) -> str:
    """``240 × 240 · 2 ch · 57 labels`` from the worker's ``SliceMeta``;
    each part only when the worker knew it, ``""`` before it arrived."""
    if meta is None:
        return ""
    parts: List[str] = []
    shape = tuple(getattr(meta, "shape", ()) or ())
    if len(shape) == 2:
        parts.append(T.ROW_PLANE_DIMS.format(h=int(shape[0]), w=int(shape[1])))
    if int(getattr(meta, "n_channels", 0) or 0) > 0 and shape:
        parts.append(T.ROW_CHANNELS.format(c=int(meta.n_channels)))
    if int(getattr(meta, "n_labels", -1)) >= 0 and \
            getattr(meta, "has_mask", False):
        parts.append(labels_text(meta.n_labels))
    return T.SEP.join(parts)


def build_staged_row(sample_id: str, record: Mapping[str, Any], *,
                     meta=None) -> Dict[str, Any]:
    """The ``MODE_NARROW`` delegate record of one Training staged row: a
    56 px picture of the finalized slice with its labels, the finalized icon,
    the name ``XY slice 26``, line 2 the patch, line 3 ``240 × 240 · 2 ch ·
    57 labels``.  The label count rides the worker's ``meta`` (never counted
    on the main thread); until it arrives line 3 is empty.  Pure."""
    _, colour, icon = _status_look("staged")
    patch = patch_of_record(sample_id, record)
    rec: Dict[str, Any] = {
        "name": slice_name(sample_id, record), "word": "", "dots": "",
        "colour": colour, "icon": icon, "icon_colour": colour,
        "line2_plain": patch, "thumb_px": SLICE_ROW_THUMB_PX,
    }
    line3 = staged_shape_line(meta)
    if line3:
        rec["shape_text"] = line3
        # polish: one line, fitted from shorter spellings -- never a wrap
        rec["shape_variants"] = line2_variants(line3)
    return rec


#: the loaded patch beside the Slices header: the grey mono line (the accent
#: is never the colour of a sentence, look rule 1)
SLICES_HEADER_PATCH_STYLE = PANEL_SLICES_HEADER_STYLE


def filter_chip_style(colour: str) -> str:
    """One filter chip: a pill with its status colour; a checked chip is
    amber (Step 3's checked chip); a chip whose count is zero (``zero``
    property) dims."""
    return (
        "QPushButton { padding: 2px 5px; font-size: 10px; color: %s;"
        " border: 1px solid rgba(255,255,255,0.15); border-radius: 10px;"
        " background: transparent; }"
        # the v8 pill, but the checked state is Step 3's amber (PANEL_CHIP_STYLE)
        "QPushButton:checked { background: %s; border-color: %s; }"
        "QPushButton[zero=\"true\"] { color: %s; }"
        % (colour, PANEL_CHIP_CHECKED_BG, PANEL_CHIP_CHECKED_BORDER,
           PANEL_ZERO_COLOR))


def filter_chip_counts(records: Mapping[str, Mapping[str, Any]], *,
                       bucket: str, patch: str = "") -> Dict[str, int]:
    """The five filter chips' numbers for the rows the list WOULD show
    without a status filter: the active *bucket* and the loaded *patch*
    (``""`` = every patch).  ``all`` is the sum of the other four; the
    ``staged`` (Finalized) chip counts trained slices too."""
    out = {key: 0 for key, _ in FILTER_CHIPS}
    for sid, rec in (records or {}).items():
        if not isinstance(rec, Mapping) or rec.get("bucket") != bucket:
            continue
        if patch and patch_of_record(sid, rec) != patch:
            continue
        status = rec.get("status", "predicted")
        for key, statuses in FILTER_CHIPS:
            if statuses is not None and status in statuses:
                out[key] += 1
        if status in T.STATUS_WORDS:
            out["all"] += 1
    return out


def filter_chip_texts(counts: Mapping[str, int]) -> Dict[str, str]:
    """``{"all": "All 7", "predicted": "Predicted 3", ...}`` (the chip
    buttons' texts; the words are the status words, capitalised)."""
    words = {"all": T.FILTER_ALL_WORD,
             "predicted": T.STATUS_WORDS["predicted"].capitalize(),
             "curating": T.STATUS_WORDS["curating"].capitalize(),
             "accepted": T.STATUS_WORDS["accepted"].capitalize(),
             "staged": T.STATUS_WORDS["staged"].capitalize()}
    return {key: T.FILTER_CHIP_FORMAT.format(word=words[key],
                                           n=int(counts.get(key, 0)))
            for key, _ in FILTER_CHIPS}


# ---------------------------------------------------------------------------
# The two lists (Qt)
# ---------------------------------------------------------------------------

def _narrow_list(parent, selection, help_lines) -> StackListWidget:
    lst = StackListWidget(parent)
    lst.setSelectionMode(selection)
    deleg = StackRowDelegate(
        MODE_NARROW, lst, palette=ROW_PALETTE, accent=ROW_ACCENT,
        show_tick=False, show_eye=False, show_thumb=True, keep_aspect=True,
        lock_property=LOCK_PROPERTY, eye_lock_property=LOCK_PROPERTY,
        dim_when_locked=True, help_lines=help_lines or {})
    lst.setItemDelegate(deleg)
    lst._epicycle_delegate = deleg
    # a short viewport shows the first empty-state line, not clipped dots
    lst.empty_one_line_when_short = True
    return lst


def build_slice_list(parent, *, help_lines=None) -> StackListWidget:
    """The Slices list (bound to ``segment_slice_data_list``): narrow rows
    with the 56 px picture, extended selection as today (Accept / Discard /
    Finalize / Revert read ``selectedItems()``), no tick, no eye."""
    lst = _narrow_list(parent, QListWidget.ExtendedSelection, help_lines)
    lst.set_empty_lines(T.SLICES_EMPTY_LINES)
    return lst


def build_staged_list(parent, *, help_lines=None) -> StackListWidget:
    """Training's staged list (bound to ``data_to_add_list``): the same
    narrow 56 px rows; nothing reads its selection."""
    lst = _narrow_list(parent, QListWidget.SingleSelection, help_lines)
    lst.set_empty_lines(T.STAGED_EMPTY_LINES)
    return lst


def slice_failure_tip(reason) -> str:
    """The row's hover when no picture could be drawn, in the slice's
    words (the shared worker's reasons speak of a "stack")."""
    from ...ui_shell import _stack_thumbs as S

    text = str(reason or "").strip()
    for const, words in T.SLICE_THUMB_REASONS.items():
        if text and text == getattr(S, const, None):
            text = words
            break
    return T.SLICE_THUMB_FAILED.format(
        reason=text or T.SLICE_THUMB_REASONS["THUMB_FAIL_UNREADABLE"])


def apply_slice_records(lst, records: Mapping[str, Mapping[str, Any]], *,
                        staged: bool = False,
                        metas: Optional[Mapping[str, Any]] = None) -> int:
    """Set ``ROLE_RECORD`` on every row of *lst* from *records* (by the
    row's ``UserRole`` sample id) -> how many rows.  Never adds, removes or
    retexts a row.  An item's own tooltip (the imported rows' provenance) is
    carried as the record's ``row_tip``.  ``staged``: Training's staged rows (patch line + shape
    line from the worker's meta) instead of Slices rows."""
    n = 0
    lst.blockSignals(True)
    try:
        for i in range(lst.count()):
            it = lst.item(i)
            sid = _item_name(it)
            rec = records.get(sid)
            if rec is None:
                continue
            row = (build_staged_row(sid, rec, meta=(metas or {}).get(sid))
                   if staged else build_slice_row(sid, rec))
            # the controller's provenance tooltip (imported rows) lives on the
            # item; the delegate's ``helpEvent`` shows ``row_tip`` instead of
            # it, so carry it over
            tip = it.toolTip() if hasattr(it, "toolTip") else ""
            if tip and "row_tip" not in row:
                row["row_tip"] = tip
            it.setData(ROLE_RECORD, row)
            n += 1
    finally:
        lst.blockSignals(False)
    lst.doItemsLayout()
    lst.viewport().update()
    return n


# ---------------------------------------------------------------------------
# Pictures on the slice rows (main thread)
# ---------------------------------------------------------------------------

class _SliceBatch:
    """The PNG names one worker batch owns.  ``finished`` is the (bound)
    method the worker's ``finished`` signal calls; it clears only these
    names.  Held in ``SlicePictures._batches`` until it fires."""

    __slots__ = ("owner", "names", "worker", "__weakref__")   # PyQt weak-refs a bound-method slot's owner

    def __init__(self, owner):
        self.owner = owner
        self.names: set = set()
        self.worker = None

    def finished(self) -> None:
        self.owner._batch_finished(self)


class SlicePictures:
    """The slice pictures of the Slices list and the staged list.

    What it does and does not do: every fill calls :meth:`refresh`, which
    stats each listed slice's files (``os.stat`` only), sets ``ROLE_THUMB``
    SYNCHRONOUSLY from the in-memory ``ThumbCache`` (a rebuilt list keeps its
    pictures; while a new picture draws, the row keeps the slice's previous
    one), and requests ONLY the misses -- never one whose PNG name is already
    in flight.  The loaded patch's ``curating`` rows whose mask changed are
    NOT requested per stroke: :meth:`note_mask_written` (called after a mask
    write) feeds a trailing 1.5 s debounce, and one request goes out after
    the last write.  Bound methods only; main thread only."""

    def __init__(self, on_meta: Optional[Callable[[], None]] = None, *,
                 timer_factory=None):
        from ...ui_shell._stack_thumbs import ThumbCache
        from .. import _step2_thumbs as TH

        self._TH = TH
        self.cache = ThumbCache()
        self.state = TH.Step2ThumbState()
        self.lists: List[Any] = []
        self.meta: Dict[str, Any] = {}              # sample id -> SliceMeta
        self.inflight = TH.InFlight()
        self.debounce = TH.TrailingDebounce(self._on_debounced,
                                            timer_factory=timer_factory)
        self._reqs: Dict[str, Any] = {}             # sid -> SliceThumbRequest
        self._png_of: Dict[str, str] = {}           # sid -> png name in flight
        self._batches: List[Any] = []               # live per-worker batches
        self._latest: Dict[str, Any] = {}           # sid -> last good pixmap
        self._failed: Dict[str, tuple] = {}         # sid -> (key, tip)
        self._workspace = ""
        self._records: Mapping[str, Any] = {}
        self._stack_paths: Mapping[str, str] = {}
        self._loaded_patch = ""
        self._on_meta = on_meta

    # -- the request side -------------------------------------------------
    def refresh(self, workspace: str, records: Mapping[str, Mapping[str, Any]],
                sample_ids: Iterable[str], *,
                stack_paths: Optional[Mapping[str, str]] = None,
                loaded_patch: str = "") -> int:
        """Pictures for *sample_ids* (the rows of the lists now) -> how many
        requests went out."""
        self._workspace = workspace or ""
        self._records = records or {}
        self._stack_paths = stack_paths or {}
        self._loaded_patch = str(loaded_patch or "")
        sids = list(dict.fromkeys(str(s) for s in sample_ids))
        n = self._request(sids, deferred_ok=True)
        self.apply_to_rows()
        return n

    def _request(self, sids: List[str], *, deferred_ok: bool) -> int:
        TH = self._TH
        if not self._workspace:
            return 0
        folder = TH.slice_thumbs_dir(self._workspace)
        misses = []
        for sid in sids:
            rec = self._records.get(sid)
            req = (TH.slice_thumb_request(
                sid, rec, stack_paths=self._stack_paths,
                workspace=self._workspace) if rec is not None else None)
            if req is None:
                self._reqs.pop(sid, None)
                continue
            self._reqs[sid] = req
            key = TH.cache_key_for(req)
            if self.cache.get(*key) is not None:
                continue
            f = self._failed.get(sid)
            if f is not None and f[0] == key:
                continue                          # failed for these very files
            if (deferred_ok and req.status == "curating"
                    and self._in_loaded_patch(sid, rec)
                    and sid in self._latest):
                self.debounce.poke(sid)           # re-curated: wait for silence
                continue
            name = TH.slice_png_path(folder, req).name
            if not self.inflight.add(name):
                continue                          # already queued / running
            self._png_of[sid] = name
            misses.append((req, name))
        if not misses:
            return 0
        # one submit per request, so the worker that ends up holding it is
        # known (``state.worker`` right after the call): a batch's ``finished``
        # then clears only ITS names (audit L2) -- a second batch started
        # because the first was closing keeps its in-flight marks
        for req, name in misses:
            batch = _SliceBatch(self)
            TH.submit_slice_thumbs(
                [req], folder, self._on_ready, self._on_failed,
                self._on_meta_json, state=self.state,
                on_finished=batch.finished)
            self._attribute(batch, name)
        return len(misses)

    def _attribute(self, fresh, name: str) -> None:
        """Record *name* with the batch of the worker now holding it: an
        already known worker's batch (the request was enqueued) or *fresh*
        (it started a new worker)."""
        w = self.state.worker
        if w is None:
            return                                # a stand-in submit (tests)
        for b in self._batches:
            if b.worker is w:
                b.names.add(name)
                return
        fresh.worker = w
        fresh.names.add(name)
        self._batches.append(fresh)

    def _batch_finished(self, batch) -> None:
        """A worker batch ended: its names (and only its) are no longer in
        flight."""
        if batch in self._batches:
            self._batches.remove(batch)
        for name in batch.names:
            self.inflight.discard(name)
        for sid, nm in list(self._png_of.items()):
            if nm in batch.names:
                del self._png_of[sid]

    def _in_loaded_patch(self, sid: str, rec) -> bool:
        return bool(self._loaded_patch) and \
            patch_of_record(sid, rec) == self._loaded_patch

    def note_mask_written(self, sample_id: str) -> None:
        """A mask file of *sample_id* was (re)written: re-request its picture
        1.5 s after the LAST write."""
        self.debounce.poke(str(sample_id))

    def _on_debounced(self, sids) -> None:
        self._request(sorted(sids), deferred_ok=False)
        self.apply_to_rows()

    # -- the row side -------------------------------------------------------
    def apply_to_rows(self) -> None:
        """``ROLE_THUMB`` (the cached pixmap, else the slice's previous one)
        and ``ROLE_THUMB_TIP`` on every row of every list, from memory."""
        for lst in self.lists:
            for i in range(lst.count()):
                self._apply(lst.item(i))
            lst.viewport().update()

    def _apply(self, item) -> None:
        sid = _item_name(item)
        req = self._reqs.get(sid)
        key = self._TH.cache_key_for(req) if req is not None else None
        pm = self.cache.get(*key) if key is not None else None
        if pm is None:
            pm = self._latest.get(sid)            # keep the old picture
        item.setData(ROLE_THUMB, pm)
        tip = None
        if pm is not None:
            tip = T.SLICE_THUMB_TIP
        elif sid in self._failed and self._failed[sid][0] == key:
            tip = self._failed[sid][1]
        elif sid in self._png_of:
            tip = T.SLICE_THUMB_PENDING
        item.setData(ROLE_THUMB_TIP, tip)

    # -- worker callbacks (main thread; bound methods) ----------------------
    def _on_ready(self, name: str, path: str, png_path: str,
                  data: bytes) -> None:
        req = self._reqs.get(name)
        answered = Path(png_path).name if png_path else ""
        if req is not None and answered:
            expected = self._TH.slice_png_path(
                self._TH.slice_thumbs_dir(self._workspace), req).name
            if answered != expected:
                # the answer of an OLDER request (the mask changed again):
                # its png is done, but the newer one is still in flight and
                # this picture must not be cached under the new key (L1)
                self.inflight.discard(answered)
                return
        self.inflight.discard(self._png_of.pop(name, ""))
        if req is None or path != req.path:
            return                                # a stale answer
        key = self._TH.cache_key_for(req)
        pm = self.cache.load(req.path, key[1], key[2], png_path=png_path or None,
                             png_bytes=data, extra=key[3])
        self._failed.pop(name, None)
        if pm is not None:
            self._latest[name] = pm
        for lst in self.lists:
            it = find_item(lst, name)
            if it is not None:
                self._apply(it)
                lst.viewport().update()

    def _on_failed(self, name: str, path: str, reason: str) -> None:
        req = self._reqs.get(name)
        self.inflight.discard(self._png_of.pop(name, ""))
        if req is None:
            return
        self._failed[name] = (self._TH.cache_key_for(req),
                              slice_failure_tip(reason))
        for lst in self.lists:
            it = find_item(lst, name)
            if it is not None:
                self._apply(it)
                lst.viewport().update()

    def _on_meta_json(self, name: str, path: str, text: str) -> None:
        try:
            meta = self._TH.SliceMeta.from_json(text)
        except (ValueError, TypeError):
            return
        if meta.busy:
            # locked twice: nothing drawn, the old picture stays, try again
            # at the next refresh
            self.inflight.discard(self._png_of.pop(name, ""))
            return
        self.meta[name] = meta
        if self._on_meta is not None:
            self._on_meta()

    # -- the end -------------------------------------------------------------
    def stop(self, wait_ms: int = 3000) -> bool:
        """``on_shutdown`` / a workspace close: cancel the debounce, stop the
        batch (bounded)."""
        self.debounce.cancel()
        self.inflight.clear()
        self._png_of.clear()
        self._batches.clear()
        return self._TH.stop_thumbs(self.state, wait_ms=wait_ms)

    def sweep_at_open(self, workspace: str,
                      all_sample_ids: Iterable[str]) -> int:
        """Workspace open: delete slice PNGs of samples no run has any more
        (the slices folder only)."""
        return self._TH.sweep_unlisted_slices(workspace, all_sample_ids)

    def forget_patch(self, workspace: str, patch_stem: str) -> None:
        """Forget a patch: drop its slices' pictures from memory and disk."""
        for sid in list(self._reqs):
            parsed = self._TH.parse_sample_id(sid)
            if parsed is not None and parsed[0] == patch_stem:
                req = self._reqs.pop(sid)
                self.cache.discard_path(req.path)
                self._latest.pop(sid, None)
                self.meta.pop(sid, None)
                self._failed.pop(sid, None)
        self._TH.remove_patch_slice_pngs(workspace, patch_stem)

    def forget_workspace(self) -> None:
        """A different workspace: drop everything of the old one."""
        self.stop(wait_ms=0)
        self.cache.clear()
        for d in (self.meta, self._reqs, self._latest, self._failed):
            d.clear()
