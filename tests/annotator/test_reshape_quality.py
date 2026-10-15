"""The quality gate -- four rules, one helper.

The sandbox showed three warnings after every cut; the panel showed none of
them, because the engine computes ``coverage``, ``piece_sizes`` and
``n_seeds_found`` and the widget threw them away.  ``quality_warnings`` is
the one place those rules live now.

Rule (c) -- the sliver -- was already implemented, three times over, inside
``update_split_guided_ui``.  It is **MOVED** here, not changed: the tests
below pin it against the same ``0.2 x median`` arithmetic the widget used,
so a reader can see the move was a move.

Pure: no Qt, no napari.
"""

from __future__ import annotations

import pytest


def _mod():
    from packages.epicycle_annotator.reshape import quality

    return quality


def _result(**kw):
    from packages.epicycle_annotator.reshape import ReshapeResult

    out = ReshapeResult()
    for key, value in kw.items():
        setattr(out, key, value)
    return out


def _stats(sizes, start=11):
    return [
        {"id": start + i, "voxels": int(v), "centroid_zyx": (0, 0, 0),
         "bbox": (0, 0, 0, 1, 1, 1)}
        for i, v in enumerate(sizes)
    ]


def _clean(**kw):
    """A result that trips NO rule, so a test can turn on exactly one."""
    base = dict(
        status="ok",
        n_seeds=2, n_seeds_found=2, n_seeds_dropped=0,
        n_pieces=2,
        piece_sizes=[5000, 4800],
        piece_stats=_stats([5000, 4800]),
        coverage=1.0,
    )
    base.update(kw)
    return _result(**base)


# --------------------------------------------------------------------------- #
# the shape of the answer
# --------------------------------------------------------------------------- #
class TestTheAnswerIsData:
    def test_a_clean_result_says_nothing(self):
        assert _mod().quality_warnings(_clean(), target_id=149) == []

    def test_every_note_carries_a_kind_a_sentence_and_its_pieces(self):
        notes = _mod().quality_warnings(
            _clean(piece_sizes=[5000, 4800, 100],
                   piece_stats=_stats([5000, 4800, 100])),
            target_id=149,
        )
        assert notes
        for note in notes:
            assert note.kind
            assert note.text.strip().endswith(".")
            assert isinstance(note.piece_ids, tuple)

    def test_a_refused_result_has_nothing_to_say(self):
        """A refusal already carries its own red reason; a second
        opinion about a cut that did not happen is noise."""
        out = _clean(status="refused", refusal_reason="no-seeds",
                     n_pieces=0, piece_sizes=[], piece_stats=[],
                     coverage=0.0)
        assert _mod().quality_warnings(out, target_id=149) == []

    def test_every_sentence_is_cp1252_safe(self):
        out = _clean(
            n_seeds=3, n_seeds_found=3, n_pieces=2,
            piece_sizes=[5000, 400], piece_stats=_stats([5000, 400]),
            coverage=0.94,
        )
        for note in _mod().quality_warnings(out, target_id=149,
                                            expected_count=3):
            note.text.encode("cp1252")


# --------------------------------------------------------------------------- #
# (a) a marker got no piece of its own
# --------------------------------------------------------------------------- #
class TestRuleALostMarker:
    def test_three_markers_two_pieces_is_a_lost_marker(self):
        out = _clean(n_seeds=3, n_seeds_found=3, n_pieces=2,
                     piece_sizes=[5000, 4800],
                     piece_stats=_stats([5000, 4800]))
        notes = _mod().quality_warnings(out, target_id=149)
        kinds = [n.kind for n in notes]
        assert "lost-marker" in kinds
        text = [n.text for n in notes if n.kind == "lost-marker"][0]
        assert "1 marker did not get a piece of its own" in text
        assert "Min marker gap" in text and "Marker blur" in text

    def test_the_count_is_the_difference(self):
        out = _clean(n_seeds=5, n_seeds_found=5, n_pieces=2,
                     piece_sizes=[5000, 4800],
                     piece_stats=_stats([5000, 4800]))
        text = [n.text for n in _mod().quality_warnings(out, target_id=149)
                if n.kind == "lost-marker"][0]
        assert "3 markers did not get a piece" in text

    def test_it_is_suppressed_when_merge_to_count_fired(self):
        """``merge_to_count`` absorbs marker-backed pieces ON PURPOSE and
        the engine already says so -- a second warning about it is a lie
        about a thing the user asked for (rule (a)."""
        out = _clean(n_seeds=3, n_seeds_found=3, n_pieces=2,
                     piece_sizes=[5000, 4800],
                     piece_stats=_stats([5000, 4800]))
        notes = _mod().quality_warnings(out, target_id=149,
                                        merged_to_count=True)
        assert [n for n in notes if n.kind == "lost-marker"] == []

    def test_more_pieces_than_markers_is_not_a_lost_marker(self):
        out = _clean(n_seeds=2, n_seeds_found=2, n_pieces=3,
                     piece_sizes=[5000, 4800, 4000],
                     piece_stats=_stats([5000, 4800, 4000]))
        assert [n for n in _mod().quality_warnings(out, target_id=149)
                if n.kind == "lost-marker"] == []


# --------------------------------------------------------------------------- #
# (b) coverage
# --------------------------------------------------------------------------- #
class TestRuleBCoverage:
    def test_full_coverage_says_nothing(self):
        assert [n for n in _mod().quality_warnings(
            _clean(coverage=1.0), target_id=149)
            if n.kind == "coverage"] == []

    def test_just_above_the_floor_says_nothing(self):
        assert [n for n in _mod().quality_warnings(
            _clean(coverage=0.985), target_id=149)
            if n.kind == "coverage"] == []

    def test_below_the_floor_names_the_percentage_and_the_cell(self):
        notes = _mod().quality_warnings(_clean(coverage=0.94), target_id=149)
        text = [n.text for n in notes if n.kind == "coverage"][0]
        assert "6 % of the object was not given to any piece" in text
        assert "stay with object 149 when you Accept" in text
        assert "nothing is lost" in text
        assert "Soft cut" in text
        assert "Snap to the wall signal" in text

    def test_the_floor_is_a_keyword(self):
        notes = _mod().quality_warnings(
            _clean(coverage=0.94), target_id=149, coverage_floor=0.5)
        assert [n for n in notes if n.kind == "coverage"] == []

    def test_it_never_unticks_anything(self):
        """There is no piece to untick -- the voxels belong to no piece."""
        notes = _mod().quality_warnings(_clean(coverage=0.5), target_id=149)
        assert [n.piece_ids for n in notes if n.kind == "coverage"] == [()]

    def test_the_number_is_the_one_S1_shows(self):
        """S1 and the block agree on ONE number, formatted twice: 94 % assigned <-> 6 % not given."""
        M = _mod()
        assert M.coverage_percent(0.94) == 94
        assert M.shortfall_percent(0.94) == 6
        assert M.coverage_percent(0.94) + M.shortfall_percent(0.94) == 100


# --------------------------------------------------------------------------- #
# (c) the sliver rule -- MOVED from the widget, not changed
# --------------------------------------------------------------------------- #
class TestRuleCSliver:
    def _widget_rule(self, sizes):
        """The arithmetic ``update_split_guided_ui`` used, verbatim."""
        svals = sorted(v for v in sizes if v)
        median = svals[len(svals) // 2] if svals else 0
        frag_cut = 0.2 * median if median > 0 else 0
        return [bool(frag_cut) and s < frag_cut for s in sizes]

    @pytest.mark.parametrize("sizes", [
        [5000, 4800, 410],
        [5000, 4800, 4700],
        [100, 100, 100],
        [1],
        [9000, 1000, 800, 30],
    ])
    def test_it_flags_exactly_what_the_widget_flagged(self, sizes):
        out = _clean(n_seeds=len(sizes), n_seeds_found=len(sizes),
                     n_pieces=len(sizes), piece_sizes=list(sizes),
                     piece_stats=_stats(sizes))
        notes = [n for n in _mod().quality_warnings(out, target_id=149)
                 if n.kind == "sliver"]
        flagged = {pid for n in notes for pid in n.piece_ids}
        want = {11 + i for i, is_frag in
                enumerate(self._widget_rule(sizes)) if is_frag}
        assert flagged == want

    def test_the_sentence_names_the_piece_its_size_and_the_fix(self):
        out = _clean(n_seeds=3, n_seeds_found=3, n_pieces=3,
                     piece_sizes=[5000, 4800, 410],
                     piece_stats=_stats([5000, 4800, 410]))
        text = [n.text for n in _mod().quality_warnings(out, target_id=149)
                if n.kind == "sliver"][0]
        assert "Piece 3 is very small (410 voxels)" in text
        assert "unticked for you" in text
        assert "Balance" in text

    def test_the_fraction_is_a_keyword(self):
        out = _clean(n_seeds=3, n_seeds_found=3, n_pieces=3,
                     piece_sizes=[5000, 4800, 410],
                     piece_stats=_stats([5000, 4800, 410]))
        notes = _mod().quality_warnings(out, target_id=149, sliver_frac=0.0)
        assert [n for n in notes if n.kind == "sliver"] == []

    def test_a_piece_is_numbered_by_its_position_not_its_id(self):
        out = _clean(n_seeds=2, n_seeds_found=2, n_pieces=2,
                     piece_sizes=[5000, 100],
                     piece_stats=_stats([5000, 100], start=904))
        note = [n for n in _mod().quality_warnings(out, target_id=149)
                if n.kind == "sliver"][0]
        assert "Piece 2" in note.text
        assert note.piece_ids == (905,)


# --------------------------------------------------------------------------- #
# (d) the expected count
# --------------------------------------------------------------------------- #
class TestRuleDExpectedCount:
    def test_matching_the_expectation_says_nothing(self):
        out = _clean(n_pieces=3, n_seeds=3, n_seeds_found=3,
                     piece_sizes=[5000, 4800, 4700],
                     piece_stats=_stats([5000, 4800, 4700]))
        assert [n for n in _mod().quality_warnings(
            out, target_id=149, expected_count=3) if n.kind == "count"] == []

    def test_fewer_pieces_than_expected_is_said_plainly(self):
        out = _clean(n_pieces=2, n_seeds=2, n_seeds_found=2)
        text = [n.text for n in _mod().quality_warnings(
            out, target_id=149, expected_count=3) if n.kind == "count"][0]
        assert "You expected 3 objects; 2 came out" in text
        assert "Nothing was forced" in text
        assert "Prominence" in text

    def test_no_expectation_means_no_rule(self):
        out = _clean(n_pieces=2, n_seeds=2, n_seeds_found=2)
        assert [n for n in _mod().quality_warnings(out, target_id=149)
                if n.kind == "count"] == []

    def test_more_pieces_than_expected_is_not_this_rule(self):
        """``merge_to_count`` handles that case and the engine says so."""
        out = _clean(n_pieces=4, n_seeds=4, n_seeds_found=4,
                     piece_sizes=[5000, 4800, 4700, 4600],
                     piece_stats=_stats([5000, 4800, 4700, 4600]))
        assert [n for n in _mod().quality_warnings(
            out, target_id=149, expected_count=3) if n.kind == "count"] == []


# --------------------------------------------------------------------------- #
# order -- the block is read top to bottom
# --------------------------------------------------------------------------- #
class TestOrder:
    def test_the_notes_come_back_in_a_fixed_order(self):
        out = _clean(
            n_seeds=4, n_seeds_found=4, n_pieces=3,
            piece_sizes=[5000, 4800, 410],
            piece_stats=_stats([5000, 4800, 410]),
            coverage=0.9,
        )
        kinds = [n.kind for n in _mod().quality_warnings(
            out, target_id=149, expected_count=5)]
        assert kinds == ["lost-marker", "coverage", "sliver", "count"]


# --------------------------------------------------------------------------- #
# the package re-exports it eagerly
# --------------------------------------------------------------------------- #
class TestItIsPartOfThePackage:
    def test_quality_warnings_is_importable_from_the_package(self):
        from packages.epicycle_annotator import reshape

        assert hasattr(reshape, "quality_warnings")
        assert hasattr(reshape, "QualityNote")
        assert "quality_warnings" in reshape.__all__
        assert "QualityNote" in reshape.__all__

    def test_it_stays_pure(self):
        import sys

        from packages.epicycle_annotator.reshape import quality  # noqa: F401

        assert "qtpy" not in str(
            getattr(sys.modules[quality.__name__], "__doc__", "") or "")
        src = open(quality.__file__, encoding="utf-8").read()
        assert "import napari" not in src
        assert "qtpy" not in src


# --------------------------------------------------------------------------- #
# UX round section 0a -- "object", not "cell", in the ENGINE too
#
# The two earlier passes were scoped to the sentences they cited, so the
# sweep was never widened: ten user-facing sentences on the Reshape path
# still said "cell" while the panel, the knob and the guide around them
# said "object".  Four of them are the amber quality block, which is the
# most-read text on the page.
#
# A sweep is only closed if a NEW sentence cannot re-open it, which is
# what this walks the AST for: every string literal that is not a
# docstring, in every module of the package that talks to the user.
# --------------------------------------------------------------------------- #
#: Every module of the engine that emits a sentence the panel shows.
#: `solvers.py` is here although UX-round 8.2 lists it as untouched: its
#: "Settling made cell(s) N disappear" is the same class as the ten the
#: table names, and a sweep with one hole in it is the bug the table
#: exists to close.
SPEAKING_MODULES = (
    "quality", "engine", "arbiter", "postprocess", "solvers",
)


def _string_literals(module_name):
    """Every non-docstring string constant in *module_name*, with its line.

    Comments are not AST nodes, so they are excluded by construction --
    and several of them legitimately still say "cell" about the biology.
    """
    import ast
    import importlib

    mod = importlib.import_module(
        "packages.epicycle_annotator.reshape.{}".format(module_name))
    tree = ast.parse(open(mod.__file__, encoding="utf-8").read())
    docstrings = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", None) or []
            first = body[0] if body else None
            if (isinstance(first, ast.Expr)
                    and isinstance(first.value, ast.Constant)
                    and isinstance(first.value.value, str)):
                docstrings.add(id(first.value))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and id(node) not in docstrings):
            out.append((getattr(node, "lineno", 0), node.value))
    return out


class TestTheEngineSpeaksOfObjects:
    @pytest.mark.parametrize("module_name", SPEAKING_MODULES)
    def test_no_sentence_says_cell(self, module_name):
        import re

        offenders = [
            (line, text) for line, text in _string_literals(module_name)
            if re.search(r"\bcells?\b", text, re.IGNORECASE)
        ]
        assert not offenders, (
            "reshape/{}.py still speaks the old vocabulary: {}".format(
                module_name, offenders)
        )

    def test_the_four_amber_rules_all_say_object(self):
        """The amber block by itself, because it is the text the user
        reads on every cut that is not perfect."""
        from packages.epicycle_annotator.reshape import ReshapeResult

        res = ReshapeResult(
            status="ok", coverage=0.94, n_seeds=3,
            piece_stats=[{"id": 1, "voxels": 4000},
                         {"id": 2, "voxels": 10}],
        )
        notes = _mod().quality_warnings(
            res, target_id=149, expected_count=5)
        text = " ".join(n.text for n in notes)
        assert "object" in text
        import re
        assert not re.search(r"\bcells?\b", text, re.IGNORECASE), text
