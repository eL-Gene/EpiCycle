"""Step 3's Predict + Reconstruct rows, as records for the shared row painter.

Pure and Qt-free: :func:`build_row` turns one patch's facts into the
immutable record ``ui_shell._stack_row_delegate.StackRowDelegate`` paints in
``MODE_NARROW`` (tick · status icon · the mono name / the status word and
dots, the axis on the right / the shape line / notes / amber lines / chips).
:func:`shape_line` reads the shape from a reconstruction manifest payload
the caller already holds -- this module opens no file.

Brick R6 adds the "made by" chip under the shape line (``chips_top``,
amber-bordered when the recorded model is not the active one) and the ⓘ
(``info``) that opens the model details window:
:func:`made_by_fields` turns a provenance block into the chip's words.

Brick V2 adds the ``MODE_WIDE`` records of Session's patch list
(:func:`build_session_row`) and Curation's box list
(:func:`build_curation_row`): no tick, the picture in ``ROLE_THUMB``, the
status dots + word on line 1 and a line 2 of only what is KNOWN.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from ..ui_shell.theme import (
    PATCH_STATUS_COLORS,
    PATCH_STATUS_DOTS,
    PATCH_STATUS_ICONS,
    PROTECTED_STATUSES,
    ROW_RECONSTRUCTING,
    ROW_SEGMENTING,
)
from . import _step3_mr_text as M
from . import _step3_pr_text as T

#: unprocessed is an empty circle on these rows, not today's box
ROW_ICONS = dict(PATCH_STATUS_ICONS, unprocessed="mdi6.circle-outline")
RUNNING_ICON = "mdi6.timer-sand"
RUNNING_COLOUR = "#777777"
MISSING_ICON = "mdi6.alert-outline"
#: PANEL_WARN_STYLE's text colour: the row's amber lines and the sign
WARN_COLOUR = "#e6c26a"
UNKNOWN_COLOUR = "#9e9e9e"

#: the key of Session's inline "Set spacing…" button (``row_action``)
ROW_ACTION_SET_SPACING = "set_spacing"
#: the statuses whose patch has a reconstruction for the eye to view
EYE_STATUSES = ("reconstructed", "curating", "accepted", "finalized")

#: the running kinds ``refresh_patch_state_list`` passes
RUNNING_WORDS = {"segmenting": ROW_SEGMENTING,
                 "reconstructing": ROW_RECONSTRUCTING}


#: ``build_row(spacing=)``'s "not asked" (``None`` is a patch with no voxel
#: size)
_NOT_GIVEN = object()


def shape_line(payload: Optional[Mapping[str, Any]]) -> str:
    """``"{z}×{y}×{x} · {n} labels"`` from a reconstruction manifest
    (``labels_shape``, ``num_cells_final``); "" when it has no shape."""
    if not payload:
        return ""
    shape = payload.get("labels_shape")
    try:
        z, y, x = (int(v) for v in list(shape)[-3:])
    except (TypeError, ValueError):
        return ""
    n = payload.get("num_cells_final", payload.get("num_cells_raw"))
    try:
        n = int(n)
    except (TypeError, ValueError):
        return T.S3_ROW_SHAPE_NO_COUNT.format(z=z, y=y, x=x)
    return T.S3_ROW_SHAPE.format(z=z, y=y, x=x, n=f"{n:,}")


def made_by_fields(block, lateral_path, axial_path, workspace: str = ""
                   ) -> tuple:
    """``(chip text, active)`` of one row from its provenance *block*
    (``_model_provenance.pick_block``'s choice, or ``None``): "made by
    {short}" and whether it names the ACTIVE lateral / axial models
    (Step 4's ``same_model`` path rule).  A patch with no block reads the
    unknown sentence and counts as active -- nothing is known to differ, so
    no amber border."""
    from . import _model_provenance as MP
    text = MP.made_by_text(block)
    if not MP.short_name(block):
        return text, True
    return text, MP.is_active_model(block, lateral_path, axial_path,
                                    workspace)


def is_tickable(status: str, *, missing: bool = False) -> bool:
    """Can a tick be PUT on this row (it can always be taken
    off)?  Not for a protected status, nor a missing file."""
    return not missing and status not in PROTECTED_STATUSES


def build_row(name: str, status: str, axes: Optional[str] = None,
              manifest_info: Optional[Mapping[str, Any]] = None,
              ticked: bool = False, protected: Optional[bool] = None,
              missing: bool = False,
              running: Optional[str] = None,
              made_by: Optional[str] = None,
              made_by_active: bool = True,
              source_shape=None, spacing=_NOT_GIVEN,
              pixel_fallback: bool = True) -> Dict[str, Any]:
    """The delegate record of one patch row.

    *status* is the patch's pipeline status; *running* ``"segmenting"`` /
    ``"reconstructing"`` while a job works on it (the row then shows the
    running words); *manifest_info* the reconstruction manifest payload (or
    ``None``); *protected* defaults to the status being accepted or
    finalized; *missing* means the patch file is gone (row 27).

    *made_by* (R6) is the chip's text (:func:`made_by_fields`) for a
    patch that has been predicted, ``None`` for one that has not;
    *made_by_active* False draws it amber-bordered with "— not the active
    model".  A row with the chip carries the ⓘ (``info``) unless its file
    is missing (the ⋯ menu takes that slot).

    A3: given *source_shape* (the raw patch's recorded shape)
    the shape line is the metadata line ``Z×Y×X · N ch · k labels`` (axes
    stay on the status line); given *spacing* (the patch's ``patch_spacing``
    entry, or ``None`` for a patch with none) the row carries the voxel-size
    line ``vox_text`` -- amber "voxels" when there is none, never an inline
    button on a narrow row.  A row asked for neither is what it was."""
    status = status or "unprocessed"
    if protected is None:
        protected = status in PROTECTED_STATUSES
    rec: Dict[str, Any] = {
        "name": name,
        "status": status,
        "axes": (axes or "").strip(),
        "shape_text": shape_line(manifest_info),
        "tickable": is_tickable(status, missing=missing) and not protected,
        "ticked": bool(ticked),
        "missing": bool(missing),
        "chips": (),
        "warns_narrow": (),
        "notes_narrow": (),
        "dim_name": bool(missing),
        "row_tip": None,
    }
    if source_shape is not None:
        meta = session_line(axes, source_shape, manifest_info,
                            with_axes=False)
        if meta:
            rec["shape_text"] = meta
    if spacing is not _NOT_GIVEN and not missing:
        line3 = T.spacing_row_text(
            dict(spacing) if isinstance(spacing, Mapping) else None)
        rec.update(vox_text=line3, vox_warn=line3 == T.S3_ROW_SPACING_NONE)
        if rec["vox_warn"] and not protected and pixel_fallback:
            # B (board D): the amber "voxels" line is the row's amber line;
            # what it costs is its tooltip -- no second line, no height.  A
            # protected row is skipped by Predict, so the cell width never
            # reaches it.
            rec["vox_tip"] = T.S3_NO_VOXEL_WARNING
    if running in RUNNING_WORDS:
        rec.update(word=RUNNING_WORDS[running], dots="",
                   colour=RUNNING_COLOUR, icon=RUNNING_ICON)
    else:
        rec.update(word=status, dots=PATCH_STATUS_DOTS.get(status, ""),
                   colour=PATCH_STATUS_COLORS.get(status, UNKNOWN_COLOUR),
                   icon=ROW_ICONS.get(status, "mdi6.help-circle-outline"))
    if missing:
        rec.update(icon=MISSING_ICON, icon_colour=WARN_COLOUR,
                   warns_narrow=(T.S3_WARN_SIGN + T.S3_PATCH_MISSING,),
                   menu=True, row_tip=T.S3_PATCH_MISSING)
    elif protected:
        rec.update(notes_narrow=(T.S3_ROW_PROTECTED,),
                   row_tip=T.S3_ROW_PROTECTED_TIP.format(status=status))
    if ticked and status == "curating" and not missing:
        rec["chips"] = ((T.S3_CHIP_HAS_CURATION, True),)
    if made_by:
        text = made_by if made_by_active else             T.S3_MADE_BY_NOT_ACTIVE.format(made_by=made_by)
        rec["chips_top"] = ((text, not made_by_active),)
        # B: on a narrow row the chip is one line,
        # elided at the tail (the amber tail shown only when the whole chip
        # fits), the full text its tooltip
        rec["chips_top_elide"] = True
        if not made_by_active:
            rec["chips_top_tail"] = T.S3_MADE_BY_NOT_ACTIVE_TAIL
        rec["info"] = not missing
    return rec


def build_refine_row(name: str, status: str, axes: Optional[str] = None, *,
                     n_lateral: int = 0, n_axial: int = 0,
                     ticked: bool = False, stale: bool = False,
                     mixed: bool = False,
                     missing: bool = False,
                     manifest_info: Optional[Mapping[str, Any]] = None,
                     source_shape=None) -> Dict[str, Any]:
    """The delegate record of one Model Refinement Patch Selection row: :func:`build_row` with
    ``protected=False`` -- every listed patch is finalized, and a finalized
    row there is TICKABLE (the ticks are the extraction scope, not a
    protection) -- plus the chip line from the run's extraction manifest
    counts and the ⏱ stale / ⚠ mixed-settings markers as ``warns_narrow``
    lines (their whole sentences are the row's tooltip).

    The chips: ``"{n} lateral"`` and ``"{m} axial"`` once anything is
    extracted (else ``"no pairs yet"``); the lateral one is the warn
    (amber) chip when its count is under ``FEW_PAIRS_ROW``."""
    rec = build_row(name, status, axes, manifest_info, ticked=ticked,
                    protected=False, missing=missing,
                    source_shape=source_shape)
    rec["tickable"] = not missing
    # M9: a chip never wraps -- one line, elided at the tail, the whole
    # sentence the row's tooltip
    rec["chips_elide"] = True
    n_lat, n_ax = max(int(n_lateral), 0), max(int(n_axial), 0)
    total = n_lat + n_ax
    chips = []
    if total > 0:
        # the lateral chip is drawn amber when the count is few; the
        # sentence stays the row tooltip
        chips.append((M.MR_ROW_PAIRS_LATERAL.format(n=f"{n_lat:,}"),
                      n_lat < M.FEW_PAIRS_ROW))
        chips.append((M.MR_ROW_PAIRS_AXIAL.format(m=f"{n_ax:,}"), False))
    else:
        chips.append((M.MR_ROW_NOT_EXTRACTED, False))
    rec["chips"] = tuple(chips)
    warns = list(rec.get("warns_narrow") or ())
    tip = [M.MR_ROW_TIP_PATCH.format(name=name),
           M.MR_ROW_TIP_STATUS.format(status=status or "unprocessed")]
    if total > 0:
        tip.append(M.MR_ROW_TIP_PAIRS.format(n=n_lat, m=n_ax, t=total))
        if n_lat < M.FEW_PAIRS_ROW:
            tip.append(M.MR_ROW_FEW_PAIRS_TIP.format(n=n_lat))
    if stale:
        warns.append(M.MR_ROW_STALE)
        tip.append(M.MR_ROW_STALE_TIP)
    if mixed:
        warns.append(M.MR_ROW_MIXED)
        tip.append(M.MR_ROW_MIXED_TIP)
    rec["warns_narrow"] = tuple(warns)
    if missing and rec.get("row_tip"):
        tip.append(rec["row_tip"])
    rec["row_tip"] = "\n".join(tip)
    return rec


def trained_line(lateral_rec, axial_rec) -> tuple:
    """``(text, known)`` of the Model card's trained-channels line, from the
    two views' ``_model_provenance.view_record`` dicts (``kind``,
    ``trained_channels`` -- the Cellpose codes ``[c1, c2]`` -- and
    ``run_id``).

    A Cellpose built-in records no channels by nature, and a model whose
    file is gone has none to read, so either view being one gives
    ``("", False)`` -- the line is hidden.  Otherwise both models
    must have recorded channels, else the line is the grey
    ``S3_MODEL_TRAINED_UNKNOWN`` (nothing is invented); two models trained
    on different channels name BOTH sets (``S3_MODEL_TRAINED_DIFFER``); the
    names are the roles (``membrane`` = code c1 - 1, ``nuclei`` = code c2 -
    1, none for 0) with the array index in brackets (Step 4's rule); the
    ``Step 2 run`` tail only when both are Step 2 models of ONE run."""
    if any(isinstance(r, Mapping)
           and r.get("kind") in ("pretrained", "unknown")
           for r in (lateral_rec, axial_rec)):
        return "", False
    codes = []
    for rec in (lateral_rec, axial_rec):
        ch = rec.get("trained_channels") if isinstance(rec, Mapping)             else None
        try:
            codes.append((int(ch[0]), int(ch[1])))
        except (TypeError, ValueError, IndexError, KeyError):
            return T.S3_MODEL_TRAINED_UNKNOWN, False
    def names(code):
        c1, c2 = code
        # code 0 = grayscale: no membrane role and no index are claimed
        parts = [T.S3_MODEL_CHANNEL.format(
            name=T.S3_MODEL_ROLE_MEMBRANE, index=c1 - 1) if c1 > 0
            else T.S3_MODEL_ROLE_GRAY]
        if c2 > 0:
            parts.append(T.S3_MODEL_CHANNEL.format(
                name=T.S3_MODEL_ROLE_NUCLEI, index=c2 - 1))
        return parts

    if codes[0] != codes[1]:
        step2 = ("bootstrap", "domain_specialized")
        runs = ""
        if all(r.get("kind") in step2 and r.get("run_id")
               for r in (lateral_rec, axial_rec)):
            a, b = lateral_rec["run_id"], axial_rec["run_id"]
            runs = (T.S3_MODEL_TRAINED_DIFFER_RUN.format(run=a) if a == b
                    else T.S3_MODEL_TRAINED_DIFFER_RUNS.format(lat=a, ax=b))
        return T.S3_MODEL_TRAINED_DIFFER.format(
            lat=T.S3_MODEL_CHANNEL_JOIN.join(names(codes[0])),
            ax=T.S3_MODEL_CHANNEL_JOIN.join(names(codes[1])),
            runs=runs), True
    parts = names(codes[0])
    runs = {r.get("run_id") for r in (lateral_rec, axial_rec)
            if r.get("kind") in ("bootstrap", "domain_specialized")}
    both_step2 = all(r.get("kind") in ("bootstrap", "domain_specialized")
                     for r in (lateral_rec, axial_rec))
    run = ""
    if both_step2 and len(runs) == 1 and None not in runs:
        run = T.S3_MODEL_TRAINED_RUN.format(run=runs.pop())
    if codes[0] == (0, 0):
        return T.S3_MODEL_TRAINED_GRAY.format(run=run), True
    tpl = T.S3_MODEL_TRAINED_ON if len(parts) > 1 else T.S3_MODEL_TRAINED_ONE
    return tpl.format(n=len(parts), names=T.S3_MODEL_CHANNEL_JOIN.join(parts),
                      run=run), True


def channel_count(axes: Optional[str],
                  source_shape: Optional[Sequence[int]] = None
                  ) -> Optional[int]:
    """How many channels the raw patch has, when it is known WITHOUT opening
    the file: 1 for axes with no ``C``; else the ``C`` length of
    *source_shape* (the shape recorded at the header read that resolved the
    axes) when that shape has one entry per axis; else ``None``."""
    axes = (axes or "").strip().upper()
    if not axes:
        return None
    if "C" not in axes:
        return 1
    try:
        shape = [int(v) for v in source_shape]
    except (TypeError, ValueError):
        return None
    if len(shape) != len(axes) or any(v < 1 for v in shape):
        return None
    return shape[axes.index("C")]


def _zyx(axes: str, source_shape) -> Optional[tuple]:
    """``(z, y, x)`` of *source_shape* laid out as *axes*, when it fits."""
    axes = (axes or "").strip().upper()
    try:
        shape = [int(v) for v in source_shape]
    except (TypeError, ValueError):
        return None
    if len(shape) != len(axes) or not all(a in axes for a in "ZYX"):
        return None
    return tuple(shape[axes.index(a)] for a in "ZYX")


def session_line(axes: Optional[str], source_shape=None,
                 manifest_info: Optional[Mapping[str, Any]] = None, *,
                 with_axes: bool = True) -> str:
    """Session's second line: ``"{axes} · {z}×{y}×{x} · {c} ch · {n}
    labels"``, each part only when known (the spacing joins
    in R7).  The shape comes from the recorded source shape, else from the
    reconstruction manifest; the label count from the manifest.

    ``with_axes=False`` (A3, the narrow rows, whose axes sit on the status
    line) leaves the axes part out."""
    axes = (axes or "").strip()
    parts = [axes] if (axes and with_axes) else []
    zyx = _zyx(axes, source_shape) if source_shape is not None else None
    if zyx is None and manifest_info:
        try:
            zyx = tuple(int(v) for v in
                        list(manifest_info.get("labels_shape"))[-3:])
            if len(zyx) != 3:
                zyx = None
        except (TypeError, ValueError):
            zyx = None
    if zyx is not None:
        parts.append(T.S3_ROW_SHAPE_NO_COUNT.format(z=zyx[0], y=zyx[1],
                                                    x=zyx[2]))
    c = channel_count(axes, source_shape)
    if c is not None:
        parts.append(T.S3_SESSION_CHANNELS.format(c=c))
    if manifest_info:
        n = manifest_info.get("num_cells_final",
                              manifest_info.get("num_cells_raw"))
        try:
            parts.append(T.S3_SESSION_LABELS.format(n=f"{int(n):,}"))
        except (TypeError, ValueError):
            pass
    return T.S3_LINE_SEP.join(parts)


def _wide_base(name: str, status: str, *, missing: bool) -> Dict[str, Any]:
    """The parts every ``MODE_WIDE`` record of Step 3 shares: line 1 (the
    name, the status dots + word in its colour) and the missing line."""
    status = status or "unprocessed"
    rec: Dict[str, Any] = {
        "name": name,
        "status": status,
        "word": status,
        "dots": PATCH_STATUS_DOTS.get(status, ""),
        "colour": PATCH_STATUS_COLORS.get(status, UNKNOWN_COLOUR),
        "missing": bool(missing),
        "warns_wide": (),
        "dim_name": bool(missing),
        "row_tip": None,
    }
    if missing:
        rec.update(warns_wide=(T.S3_WARN_SIGN + T.S3_PATCH_MISSING,),
                   menu=True, row_tip=T.S3_PATCH_MISSING)
    return rec


def build_session_row(name: str, status: str, axes: Optional[str] = None,
                      *, manifest_info: Optional[Mapping[str, Any]] = None,
                      source_shape=None, missing: bool = False,
                      spacing: Optional[Mapping[str, Any]] = None,
                      tier_counts: Optional[Sequence] = None,
                      measured_spacing: Optional[Mapping[str, Any]] = None
                      ) -> Dict[str, Any]:
    """The ``MODE_WIDE`` record of one Session row.
    *source_shape* is the raw patch's shape as recorded by the header read
    that resolved *axes* (``None`` when not recorded).

    R7: *spacing* is the patch's ``patch_spacing`` entry
    (``{"zyx", "source"}``) or ``None``; its line (``spacing_row_text``)
    sits under line 2 as ``line3`` -- amber "voxels", with the no-tags hint
    as the row's tooltip, when there is none.  Every Session row has a
    ``⋯`` menu (Set spacing…, and Forget on a missing patch).

    A2: *tier_counts* -- the
    ``_step3_tier_counts.tier_items`` of the patch's CSV on disk, or
    ``None`` -- is the line of counts under the voxel size."""
    rec = _wide_base(name, status, missing=missing)
    line3 = T.spacing_row_text(
        dict(spacing) if isinstance(spacing, Mapping) else None)
    none = line3 == T.S3_ROW_SPACING_NONE
    rec.update(axes=(axes or "").strip(),
               n_channels=channel_count(axes, source_shape),
               line2=session_line(axes, source_shape, manifest_info),
               line3=line3, line3_warn=none, menu=True)
    if none and not missing:
        # (audit M3) no inline button: Set spacing… is on the
        # row's menu and its right-click only
        rec["row_tip"] = T.S3_SPACING_NO_TAGS_HINT
    # A2 fold: the eye only where there is a reconstruction to view
    rec["eye"] = rec["status"] in EYE_STATUSES and not missing
    if tier_counts:
        rec["tier_counts"] = tuple(tier_counts)
    # S1 (4.3b): the features were measured at another voxel size
    tail = _measured_tail(measured_spacing)
    if tail:
        rec["line3_tail"] = tail
        rec["row_tip"] = _tip_with_tail(rec.get("row_tip"), tail)
    return rec


def _measured_tail(measured) -> str:
    """The tail for a patch whose features were measured at *measured*; the
    CALLER decided it differs from the size in force (the effective one:
    own, else QC Settings' -- ``controller._session_measured``)."""
    return T.measured_tail(dict(measured) if isinstance(measured, Mapping)
                           else None)


def _tip_with_tail(tip, tail: str) -> str:
    """The row's tooltip carries the whole tail sentence (a narrow row
    elides it)."""
    full = tail[len(T.S3_LINE_SEP):] if tail.startswith(T.S3_LINE_SEP) \
        else tail
    return f"{tip}\n{full}" if tip else full


def build_curation_row(name: str, status: str, *, qc_extracted: bool = False,
                       edited: bool = False, missing: bool = False,
                       tier_counts: Optional[Sequence] = None,
                       axes: Optional[str] = None, source_shape=None,
                       manifest_info: Optional[Mapping[str, Any]] = None,
                       spacing=_NOT_GIVEN,
                       measured_spacing: Optional[Mapping[str, Any]] = None
                       ) -> Dict[str, Any]:
    """The ``MODE_WIDE`` record of one Curation box-list row: today's
    content -- the name, the status and its dots, the measured / edited
    marks (line 2) and the missing line -- for the picture beside it; A2:
    *tier_counts* as on Session's row."""
    rec = _wide_base(name, status or "reconstructed", missing=missing)
    tags = []
    if qc_extracted:
        tags.append(T.S3_CURATION_TAG_QC)
    if edited:
        tags.append(T.S3_CURATION_TAG_EDITED)
    rec["line2"] = T.S3_LINE_SEP.join(tags)
    # A3 fold: the same metadata line as Session / P+R rows,
    # the marks after it, and the voxel-size line (no inline button here)
    meta = ""
    if axes or source_shape is not None or manifest_info:
        meta = session_line(axes, source_shape, manifest_info)
    if meta:
        # (audit L6) Curation's own marks come FIRST, so a long line elides
        # the metadata, never the marks
        rec["line2"] = T.S3_LINE_SEP.join(tags + [meta])
    if spacing is not _NOT_GIVEN and not missing:
        line3 = T.spacing_row_text(
            dict(spacing) if isinstance(spacing, Mapping) else None)
        rec.update(line3=line3, line3_warn=line3 == T.S3_ROW_SPACING_NONE)
    if tier_counts:
        rec["tier_counts"] = tuple(tier_counts)
    tail = _measured_tail(measured_spacing)
    if tail and "line3" in rec:
        # S1 (4.3b): on the voxel-size line, in the warn colour
        rec["line3_tail"] = tail
        rec["row_tip"] = _tip_with_tail(rec.get("row_tip"), tail)
    return rec
