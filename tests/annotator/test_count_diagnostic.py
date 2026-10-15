"""Brick U2 section 6 -- "I expect exactly n objects", made legible.

The knob is **diagnostic only** in this brick: not one line of the cut
changes.  What changes is that you can tell the six different reasons the
count came out wrong apart, from the console, without a debugger -- which
is what the owner asked for after the knob "sometimes did nothing".

Two lines and one number:

* ``[COUNT-DIAG] recipe ...`` at recipe build -- what was ASKED for.
* ``[COUNT-DIAG] engine ...`` at result -- what the engine DID.
* ``ReshapeResult.n_pieces_before_merge``, the count the merge stage saw,
  recorded on **every** ok result including the ones where that stage
  never fires.  That is the point: "the stage never ran" and "the stage
  ran and merged nothing" look identical on screen and have completely
  different fixes.

Decision table at 6.3.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from packages.epicycle_annotator.reshape import (
    ReshapeResult,
    ReshapeSpec,
    reshape_region,
)
from packages.epicycle_annotator.reshape import engine as engine_mod

from .reshape_fixtures import two_columns
from .test_reshape_split_orchestration import (
    FakeAdv,
    FakeAnnotator,
    FakeLayer,
    bind,
)

ISO = (1.0, 1.0, 1.0)


def _spec(**kw):
    base = dict(target="label", seeds="labels", walls="intensity",
                params={"wall_channel": "mem", "clip": 0.0}, arbiter=False)
    params = dict(base["params"])
    params.update(kw.pop("params", {}))
    base.update(kw)
    base["params"] = params
    return ReshapeSpec(**base)


def _run(spec=None, contrast=0.4):
    labels, guide, mem, _region = two_columns(contrast)
    spec = spec if spec is not None else _spec()
    return reshape_region(labels, {"mem": mem}, spec, target_id=7,
                          seed_layer=guide, spacing=ISO)


# --------------------------------------------------------------------------- #
# 1. the field -- additive, defaulted, and always filled
# --------------------------------------------------------------------------- #
class TestTheField:
    def test_it_defaults_to_None_not_zero(self):
        """0 is a real count the engine could report, so a 0 default
        made "the engine never got that far" indistinguishable from a
        genuine answer (audit L5)."""
        assert ReshapeResult().n_pieces_before_merge is None

    def test_it_is_set_when_the_stage_is_NOT_IN_REFINE_AT_ALL(self):
        """The case section 6 exists for: nothing merged because the
        stage was never sent, which on screen is indistinguishable from
        a stage that ran and found nothing to do."""
        out = _run()
        assert out.ok
        assert out.merged_to_count is False
        assert out.n_pieces_before_merge == out.n_pieces == 2

    def test_it_is_set_when_the_stage_is_ON_but_does_not_fire(self):
        """Asked for 2, got 2 -- correct silence, and now visible."""
        out = _run(_spec(refine=("merge_to_count",),
                         params={"target_count": 2}))
        assert out.ok
        assert out.merged_to_count is False
        assert out.n_pieces_before_merge == 2
        assert out.n_pieces == 2

    def test_it_records_the_count_BEFORE_a_merge_that_did_fire(self):
        """Asked for 1 from a two-piece cut: the stage merges, and this
        field is the only record that there were ever two."""
        out = _run(_spec(refine=("merge_to_count",),
                         params={"target_count": 1}))
        assert out.ok
        assert out.merged_to_count is True
        assert out.n_pieces_before_merge == 2
        assert out.n_pieces == 1

    def test_it_never_disagrees_with_n_pieces_when_nothing_merged(self):
        for target in (None, 2, 5):
            params = {} if target is None else {"target_count": target}
            refine = () if target is None else ("merge_to_count",)
            out = _run(_spec(refine=refine, params=params))
            if not out.merged_to_count:
                assert out.n_pieces_before_merge == out.n_pieces, target


class TestTheAssignmentIsOutsideTheIf:
    """Audit M3, pinned in source.  Inside the ``if`` it would be
    recorded only when the stage fired -- i.e. never in the case section
    6 is there to diagnose."""

    def test_the_source_assigns_it_before_the_merge_branch(self):
        src = inspect.getsource(engine_mod.reshape_region)
        assign = src.index("n_pieces_before_merge = n_before")
        branch = src.index('if ("merge_to_count" in spec.refine')
        assert assign < branch

    def test_the_assignment_is_not_nested_under_a_conditional(self):
        for line in inspect.getsource(engine_mod.reshape_region).splitlines():
            if "n_pieces_before_merge = n_before" in line:
                indent = len(line) - len(line.lstrip())
                assert indent == 8, (
                    "indent {} -- the assignment moved under a branch"
                    .format(indent)
                )
                break
        else:  # pragma: no cover
            pytest.fail("the assignment is gone")


class TestNothingElseChanged:
    """Section 6 is diagnostic only: no piece moves."""

    def test_the_partition_is_deterministic_and_still_two_pieces(self):
        a, b = _run(), _run()
        assert np.array_equal(a.result, b.result)
        assert a.n_pieces == b.n_pieces == 2

    def test_a_refused_result_still_refuses_and_reports_zero(self):
        labels, _guide, mem, _ = two_columns(0.4)
        out = reshape_region(labels, {"mem": mem}, _spec(), target_id=7,
                             seed_layer=np.zeros_like(labels), spacing=ISO)
        assert out.status == "refused"
        assert out.n_pieces_before_merge is None, (
            "a refusal never reached the merge stage; reporting 0 would "
            "read as a real count"
        )


# --------------------------------------------------------------------------- #
# 2. line (A) -- what was asked for, printed at recipe build
# --------------------------------------------------------------------------- #
def _recipe_fake(**params):
    vol = np.zeros((6, 6, 6), np.int32)
    vol[2:4, 2:4, 2:4] = 7
    roi = (slice(0, 6),) * 3
    refine = FakeLayer(np.zeros((6, 6, 6), np.int32), "Refine")
    fake = FakeAnnotator(vol, roi, 7, adv=FakeAdv(params), refine=refine)
    fake.viewer = type("V", (), {"layers": []})()
    bind(fake, "_reshape_recipe")
    fake._split_seed_ids = [7, 101, 102]
    fake._split_painted_marker_ids = [101, 102]
    fake._refine_layer.data[0, 0, 0] = 101
    fake._refine_layer.data[5, 5, 5] = 102
    return fake


def _recipe_lines(capsys, **params):
    fake = _recipe_fake(**params)
    out = fake._reshape_recipe()
    printed = capsys.readouterr().out
    lines = [ln for ln in printed.splitlines()
             if "[COUNT-DIAG]" in ln and "recipe" in ln]
    return out, lines


class TestLineA:
    def test_it_prints_exactly_once_per_recipe(self, capsys):
        _out, lines = _recipe_lines(capsys)
        assert len(lines) == 1, lines

    @pytest.mark.parametrize("field", [
        "target_count=", "refine=", "guided_now=", "seeds=", "walls=",
        "min_size=", "arbiter=",
    ])
    def test_it_carries_every_documented_field(self, capsys, field):
        """Section 6.2 names seven.  The decision table at 6.3 reads
        three of them off this line alone, so a dropped field makes a
        row of that table unreachable."""
        _out, lines = _recipe_lines(capsys)
        assert field in lines[0], lines[0]

    def test_it_reports_the_count_as_None_when_the_tick_is_off(self, capsys):
        _out, lines = _recipe_lines(capsys, target_count=None)
        assert "target_count=None" in lines[0]

    def test_a_count_that_is_ON_shows_the_stage_in_refine(self, capsys):
        """Table row 3: a number with no `merge_to_count` beside it is a
        wiring bug, and this is the line that would show it."""
        _out, lines = _recipe_lines(capsys, target_count=3)
        assert "target_count=3" in lines[0]
        assert "merge_to_count" in lines[0]

    def test_the_count_being_off_keeps_the_stage_out_of_refine(self, capsys):
        _out, lines = _recipe_lines(capsys, target_count=None)
        assert "merge_to_count" not in lines[0]

    def test_it_names_the_engine_enums_not_the_combo_indices(self, capsys):
        """`seeds=0 walls=0` would be unreadable and would drift with
        the pickers; the engine words do not."""
        out, lines = _recipe_lines(capsys, seeds_from=0, walls_from=0)
        spec = out[0]
        assert "seeds={}".format(spec.seeds) in lines[0]
        assert "walls={}".format(spec.walls) in lines[0]

    def test_a_FIND_build_is_silent(self, capsys):
        """`for_find=True` builds the recipe a "Find markers" press
        needs.  The expected count plays no part in finding markers, and
        a stray line here would break the 6.3 table, whose first row
        reads "(A) absent AND (B) absent" as the legacy path."""
        fake = _recipe_fake(target_count=3)
        fake._reshape_recipe(for_find=True)
        printed = capsys.readouterr().out
        assert "[COUNT-DIAG]" not in printed


# --------------------------------------------------------------------------- #
# 3. line (B) -- what the engine did, printed at the result
# --------------------------------------------------------------------------- #
class _Arb:
    def __init__(self, merged=()):
        self.verdict = "cut"
        self.merged_pairs = list(merged)


class _Result:
    def __init__(self, n_seeds=3, n_before=3, n_pieces=3, merged=False,
                 arb_merged=()):
        self.n_seeds = n_seeds
        self.n_pieces_before_merge = n_before
        self.n_pieces = n_pieces
        self.merged_to_count = merged
        self.arbiter = _Arb(arb_merged)
        self.flags = {}
        self.piece_stats = [
            {"id": i + 1, "voxels": 100 * (i + 1),
             "centroid_zyx": (0, 0, 0), "bbox": (0, 0, 0, 1, 1, 1)}
            for i in range(n_pieces)
        ]


def _apply_lines(capsys, result, diag=None):
    from napari_epicycle.annotator.curation_3d_widget import (
        Curation3DAnnotator as K,
    )

    class F:
        pass

    f = F()
    f._reshape_count_diag = dict(diag or {})
    f._split_label_id = 7
    f._reshape_final_ids = {}
    f._reshape_flags = {}
    f._print_count_diag_result = K._print_count_diag_result.__get__(f, F)
    f._print_count_diag_result(result)
    printed = capsys.readouterr().out
    return [ln for ln in printed.splitlines()
            if "[COUNT-DIAG]" in ln and "engine" in ln]


class TestLineB:
    @pytest.mark.parametrize("field", [
        "n_seeds=", "n_before=", "target=", "stage_on=", "merged=",
        "arbiter_merged=", "n_pieces=",
    ])
    def test_it_carries_every_documented_field(self, capsys, field):
        lines = _apply_lines(capsys, _Result(),
                             {"target": 3, "stage_on": True})
        assert lines and field in lines[0], lines

    def test_it_reports_the_engine_numbers_verbatim(self, capsys):
        lines = _apply_lines(
            capsys, _Result(n_seeds=4, n_before=4, n_pieces=2, merged=True),
            {"target": 2, "stage_on": True})
        assert "n_seeds=4" in lines[0]
        assert "n_before=4" in lines[0]
        assert "n_pieces=2" in lines[0]
        assert "merged=True" in lines[0]

    def test_it_counts_the_ARBITER_merges_separately(self, capsys):
        """6.3's last two rows: absorb-slivers and the arbiter can each
        explain a count that came in low, and both can be true at once."""
        lines = _apply_lines(capsys, _Result(arb_merged=[(1, 2), (3, 4)]),
                             {"target": 3, "stage_on": True})
        assert "arbiter_merged=2" in lines[0]

    def test_a_missing_arbiter_report_reads_zero_not_a_crash(self, capsys):
        r = _Result()
        r.arbiter = None
        lines = _apply_lines(capsys, r, {"target": 3, "stage_on": True})
        assert "arbiter_merged=0" in lines[0]

    def test_it_reports_the_target_the_RECIPE_sent(self, capsys):
        """Not what the panel says now.  A knob changed after the solve
        was launched would otherwise make the line describe a cut that
        never ran."""
        lines = _apply_lines(capsys, _Result(), {"target": 9,
                                                 "stage_on": True})
        assert "target=9" in lines[0]

    def test_stage_off_is_reported_as_such(self, capsys):
        lines = _apply_lines(capsys, _Result(), {"target": None,
                                                 "stage_on": False})
        assert "stage_on=False" in lines[0]

    def test_a_result_that_never_reached_the_stage_prints_None(self,
                                                              capsys):
        r = _Result()
        r.n_pieces_before_merge = None
        lines = _apply_lines(capsys, r, {"target": 3, "stage_on": True})
        assert "n_before=None" in lines[0], lines[0]

    def test_it_survives_an_empty_diag_record(self, capsys):
        """The legacy path never builds a recipe, so the record can be
        empty; a diagnostic may not be the thing that raises."""
        lines = _apply_lines(capsys, _Result(), {})
        assert lines


class TestItIsWiredIn:
    def test_apply_reshape_result_prints_line_B(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator as K,
        )

        src = inspect.getsource(K._apply_reshape_result)
        assert "_print_count_diag_result" in src

    def test_the_recipe_records_what_line_B_reads_back(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator as K,
        )

        src = inspect.getsource(K._reshape_recipe)
        assert "_reshape_count_diag" in src

    def test_the_record_is_initialised_on_the_annotator(self):
        from napari_epicycle.annotator.curation_3d_widget import (
            Curation3DAnnotator as K,
        )

        assert "_reshape_count_diag" in inspect.getsource(K.__init__)
