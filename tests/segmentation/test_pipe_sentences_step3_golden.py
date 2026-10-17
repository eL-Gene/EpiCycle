"""I-10: Step 3's pipeline sentences are byte-identical before and after D-e.

E.  Brick B1a moves Step 3's pipeline f-strings
(``fusion_reconstruction/controller.py``) and its two running words
(``fusion_reconstruction/ui.py``) onto one shared sentence table in
``ui_shell/theme.py``, rendered with ``unit="patch"``.  This file pins the
EXACT text Step 3 prints today, for fixed inputs.

How it is built so it survives the move:

* ``GOLDEN`` is the contract: (inputs, expected text).  It was written from
  the UNTOUCHED Step 3 code in brick B0's first commit and must never be
  edited to make a later brick pass.
* ``render(name, **inputs)`` produces the text.  Today it finds Step 3's own
  f-string in the Step 3 source (by method name and leading literal) and
  evaluates it with ``inputs`` as its locals.  When B1a moves a sentence
  onto the shared table, B1a rewrites ``render`` for that name to call the
  shared constant -- and ``GOLDEN`` stays exactly as it is.
"""
from __future__ import annotations

import ast
import functools
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
S3C = ROOT / "napari_epicycle" / "fusion_reconstruction" / "controller.py"
S3U = ROOT / "napari_epicycle" / "fusion_reconstruction" / "ui.py"


def _fmt_elapsed(seconds):
    """Step 3's own ``_format_elapsed``, taken from its source.

    Compiled from the AST rather than imported: importing the Step 3
    controller pulls in napari and costs over a minute per run.
    """
    fn = _functions(S3C)["_format_elapsed"]
    namespace: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(S3C), "exec"),
         namespace)  # noqa: S102 -- our own source, pinned
    return namespace["_format_elapsed"](seconds)


# name -> (method, leading literal of the f-string, inputs, expected)
GOLDEN = {
    "PIPE_RECONSTRUCTING_START": (
        "_reconstruct_selected", "Reconstructing ",
        {"configs": [1, 2, 3]},
        "Reconstructing 3 patches..."),
    "PIPE_SEGMENTING_START": (
        "_predict_all", "Segmenting ",
        {"configs": [1, 2], "total_slices": 120},
        "Segmenting 2 patches (120 slices)..."),
    "PIPE_TIME_HEAD": (
        "_on_serial_pred_started", "[",
        {"elapsed": 5.0},
        "[5s"),
    "PIPE_TIME_ETA": (
        "_on_serial_pred_started", " | ",
        {"eta": "~1m 00s remaining"},
        " | ~1m 00s remaining]"),
    "PIPE_SEGMENTING_ITEM": (
        "_on_serial_pred_started", "Segmenting ",
        {"idx": 0, "total": 3, "patch_name": "p1",
         "time_str": "[5s | ~1m 00s remaining]"},
        "Segmenting 1/3: p1  [5s | ~1m 00s remaining]"),
    "PIPE_SEGMENTING_VIEW": (
        "_on_serial_pred_view_progress", "Segmenting patch ",
        {"done_patches": 1, "n_patches": 4, "patch_name": "p2",
         "view": "xz", "current": 12, "total_slices": 80,
         "time_str": "[1m 05s]"},
        "Segmenting patch 2/4: p2 — XZ (12/80)  [1m 05s]"),
    "PIPE_SEGMENTATION_DONE": (
        "_on_serial_pred_complete", "Segmentation complete: ",
        {"succeeded": 3, "n_patches": 4},
        "Segmentation complete: 3/4 patches"),
    "PIPE_FAILED_SUFFIX_PRED": (
        "_on_serial_pred_complete", " (",
        {"self": types.SimpleNamespace(_pipeline_errors=1)},
        " (1 failed)"),
    "PIPE_TIME_SUFFIX_PRED": (
        "_on_serial_pred_complete", "  [",
        {"elapsed": 125.0},
        "  [2m 05s]"),
    "PIPE_ERROR_PREDICT": (
        "_on_serial_pred_error", "[ERROR] Prediction failed for ",
        {"patch_name": "p3", "error_msg": "boom"},
        "[ERROR] Prediction failed for p3: boom"),
    "PIPE_RECONSTRUCTING_ITEM": (
        "_on_serial_recon_started", "Reconstructing ",
        {"idx": 1, "total": 2, "patch_name": "p4", "time_str": "[9s]"},
        "Reconstructing 2/2: p4...  [9s]"),
    "PIPE_RECONSTRUCTION_DONE": (
        "_on_serial_recon_complete", "Reconstruction complete: ",
        {"succeeded": 2, "total": 2},
        "Reconstruction complete: 2/2 patches"),
    "PIPE_FAILED_SUFFIX_RECON": (
        "_on_serial_recon_complete", " (",
        {"self": types.SimpleNamespace(_pipeline_errors=2)},
        " (2 failed)"),
    "PIPE_TIME_SUFFIX_RECON": (
        "_on_serial_recon_complete", "  [",
        {"elapsed": 30.0},
        "  [30s]"),
    "PIPE_ERROR_RECON": (
        "_on_serial_recon_error", "[ERROR] Reconstruction failed for ",
        {"patch_name": "p5", "error_msg": "nope"},
        "[ERROR] Reconstruction failed for p5: nope"),
}

# Plain literals: (file, method or None, expected text).
LITERALS = {
    "PIPE_CANCELLING_PRED": (S3C, "_cancel_prediction",
                             "Cancelling... will stop after current slice "
                             "completes"),
    "PIPE_CANCELLING_RECON": (S3C, "_cancel_reconstruction",
                              "Cancelling... will stop after current patch "
                              "completes"),
    "ROW_SEGMENTING": (S3U, None, "segmenting..."),
    "ROW_RECONSTRUCTING": (S3U, None, "reconstructing..."),
}


@functools.lru_cache(maxsize=None)
def _parse(path):
    return ast.parse(path.read_text(encoding="utf-8"))


@functools.lru_cache(maxsize=None)
def _functions(path):
    tree = _parse(path)
    return {n.name: n for n in ast.walk(tree)
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}


def _lead(node: ast.JoinedStr) -> str:
    first = node.values[0] if node.values else None
    if isinstance(first, ast.Constant) and isinstance(first.value, str):
        return first.value
    return ""


def _find_fstring(method: str, lead: str) -> ast.JoinedStr:
    fn = _functions(S3C)[method]
    hits = [n for n in ast.walk(fn)
            if isinstance(n, ast.JoinedStr) and _lead(n).startswith(lead)]
    assert hits, f"{method}: no f-string starting {lead!r}"
    return hits[0]


def _shared():
    from napari_epicycle.ui_shell import theme
    return theme


def _render_shared(name: str, **inputs) -> str:
    """B1a: every sentence comes from the shared table (D-e), unit="patch"."""
    T = _shared()
    units = T.pipe_units("patch")
    fmt = _fmt_elapsed
    if name == "PIPE_RECONSTRUCTING_START":
        return T.PIPE_RECONSTRUCTING_START.format(
            n=len(inputs["configs"]), units=units)
    if name == "PIPE_SEGMENTING_START":
        return T.PIPE_SEGMENTING_START.format(
            n=len(inputs["configs"]), units=units,
            slices=inputs["total_slices"])
    if name == "PIPE_TIME_HEAD":
        # pipe_time's head: "[{elapsed}" (the closing bracket belongs to
        # the eta part, pinned by PIPE_TIME_ETA)
        return T.pipe_time(fmt(inputs["elapsed"]), "")[:-1]
    if name == "PIPE_TIME_ETA":
        whole = T.pipe_time("X", inputs["eta"])
        return whole[len("[X"):]
    if name == "PIPE_SEGMENTING_ITEM":
        return T.PIPE_SEGMENTING_ITEM.format(
            i=inputs["idx"] + 1, n=inputs["total"],
            name=inputs["patch_name"], time=inputs["time_str"])
    if name == "PIPE_SEGMENTING_VIEW":
        return T.PIPE_SEGMENTING_VIEW.format(
            unit="patch", i=inputs["done_patches"] + 1,
            n=inputs["n_patches"], name=inputs["patch_name"],
            VIEW=inputs["view"].upper(), c=inputs["current"],
            t=inputs["total_slices"], time=inputs["time_str"])
    if name == "PIPE_SEGMENTATION_DONE":
        return T.PIPE_SEGMENTATION_DONE.format(
            s=inputs["succeeded"], n=inputs["n_patches"], units=units)
    if name in ("PIPE_FAILED_SUFFIX_PRED", "PIPE_FAILED_SUFFIX_RECON"):
        return T.PIPE_FAILED_SUFFIX.format(k=inputs["self"]._pipeline_errors)
    if name in ("PIPE_TIME_SUFFIX_PRED", "PIPE_TIME_SUFFIX_RECON"):
        return T.PIPE_TIME_SUFFIX.format(t=fmt(inputs["elapsed"]))
    if name == "PIPE_ERROR_PREDICT":
        return T.PIPE_ERROR_PREDICT.format(
            name=inputs["patch_name"], msg=inputs["error_msg"])
    if name == "PIPE_RECONSTRUCTING_ITEM":
        return T.PIPE_RECONSTRUCTING_ITEM.format(
            i=inputs["idx"] + 1, n=inputs["total"],
            name=inputs["patch_name"], time=inputs["time_str"])
    if name == "PIPE_RECONSTRUCTION_DONE":
        return T.PIPE_RECONSTRUCTION_DONE.format(
            s=inputs["succeeded"], n=inputs["total"], units=units)
    if name == "PIPE_ERROR_RECON":
        return T.PIPE_ERROR_RECON.format(
            name=inputs["patch_name"], msg=inputs["error_msg"])
    raise KeyError(name)


def render(name: str, **inputs) -> str:
    """B0: evaluated Step 3's own f-string.  B1a: the shared constant
    (``GOLDEN`` is unchanged: the contract survives the move)."""
    return _render_shared(name, **inputs)


# The shared name each Step 3 method must now read (D-e: Step 3 renders
# through the table, so a Step 3 edit cannot drift from Step 4).
USES = {
    "_reconstruct_selected": ("PIPE_RECONSTRUCTING_START",),
    "_predict_all": ("PIPE_SEGMENTING_START",),
    "_on_serial_pred_started": ("PIPE_SEGMENTING_ITEM", "pipe_time"),
    "_on_serial_pred_view_progress": ("PIPE_SEGMENTING_VIEW", "pipe_time"),
    "_on_serial_pred_complete": ("PIPE_SEGMENTATION_DONE",
                                 "PIPE_FAILED_SUFFIX", "PIPE_TIME_SUFFIX"),
    "_on_serial_pred_error": ("PIPE_ERROR_PREDICT",),
    "_on_serial_recon_started": ("PIPE_RECONSTRUCTING_ITEM", "pipe_time"),
    "_on_serial_recon_complete": ("PIPE_RECONSTRUCTION_DONE",
                                  "PIPE_FAILED_SUFFIX", "PIPE_TIME_SUFFIX"),
    "_on_serial_recon_error": ("PIPE_ERROR_RECON",),
    "_cancel_prediction": ("PIPE_CANCELLING",),
    "_cancel_reconstruction": ("PIPE_CANCELLING",),
}


@pytest.mark.parametrize("method", sorted(USES))
def test_step3_reads_the_shared_table(method):
    fn = _functions(S3C)[method]
    names = {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
    for const in USES[method]:
        assert const in names, f"{method} no longer reads {const}"


def test_step3_rows_read_the_running_words():
    names = {n.attr for n in ast.walk(_parse(S3U))
             if isinstance(n, ast.Attribute)}
    assert {"ROW_SEGMENTING", "ROW_RECONSTRUCTING"} <= names


def test_pipe_time_is_step3s_time_part():
    T = _shared()
    assert T.pipe_time("5s", "~1m 00s remaining") == "[5s | ~1m 00s remaining]"
    assert T.pipe_time("5s", "") == "[5s]"
    assert T.pipe_units("patch") == "patches"
    assert T.pipe_units("stack") == "stacks"
    assert T.pipe_units("stack", 1) == "stack"


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_step3_sentence_is_byte_identical(name):
    _method, _lead_text, inputs, expected = GOLDEN[name]
    assert render(name, **inputs) == expected


def _literals_in(path, method):
    tree = _parse(path)
    scope = _functions(path)[method] if method else tree
    return {n.value for n in ast.walk(scope)
            if isinstance(n, ast.Constant) and isinstance(n.value, str)}


_LITERAL_CONST = {
    "PIPE_CANCELLING_PRED": lambda T: T.PIPE_CANCELLING.format(unit="slice"),
    "PIPE_CANCELLING_RECON": lambda T: T.PIPE_CANCELLING.format(unit="patch"),
    "ROW_SEGMENTING": lambda T: T.ROW_SEGMENTING,
    "ROW_RECONSTRUCTING": lambda T: T.ROW_RECONSTRUCTING,
}


@pytest.mark.parametrize("method,unit", [
    ("_cancel_prediction", "slice"),
    ("_cancel_reconstruction", "patch"),
])
def test_step3_cancel_formats_the_expected_unit(method, unit):
    """Predict stops after a slice, Reconstruct after a patch."""
    fn = _functions(S3C)[method]
    units = [kw.value.value
             for n in ast.walk(fn) if isinstance(n, ast.Call)
             for kw in n.keywords
             if kw.arg == "unit" and isinstance(kw.value, ast.Constant)]
    assert units == [unit]


@pytest.mark.parametrize("name", sorted(LITERALS))
def test_step3_literal_is_byte_identical(name):
    _path, _method, expected = LITERALS[name]
    assert _LITERAL_CONST[name](_shared()) == expected


def test_step3_running_rows_render(qapp):
    """B1a audit A1-1: Step 3's patch list draws its running rows from the
    shared words (a NameError here broke every Predict / Reconstruct
    redraw).  The row text is exactly today's padded text."""
    import types

    from qtpy.QtWidgets import QListWidget

    from napari_epicycle.fusion_reconstruction.ui import (
        FusionReconstructionCuration,
    )
    fake = types.SimpleNamespace(
        _patch_state_list=QListWidget(),
        # A3: a protected row carries its tier counts; the row TEXT this
        # test pins does not read them, so the record passes through
        _with_pr_tier_counts=lambda rec, name: rec)
    FusionReconstructionCuration.refresh_patch_state_list(
        fake, {"p": "predicted", "q": "predicted", "r": "unprocessed"},
        predicting={"p"}, reconstructing={"q"})
    lw = fake._patch_state_list
    texts = {lw.item(i).data(0x0100): lw.item(i).text()
             for i in range(lw.count())}
    assert texts["p"] == " p  " + "segmenting...".ljust(11)
    assert texts["q"] == " q  " + "reconstructing...".ljust(11)
