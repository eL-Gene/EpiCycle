"""Brick T, seam (d) Training sites: the hint goes
to the setup column through the six ``_step2_text`` cases, the Writes labels
are neutral and say the real file name, the staged count and the idle stage
line come from ``_step2_text``.  Qt-free fakes."""
import inspect

from napari_epicycle.hitl_training_annotation.controller.training_annotation_controller import (
    TrainingAnnotationController as C,
)
from napari_epicycle.hitl_training_annotation.ui import _step2_text as T


class _Label:
    def __init__(self):
        self.text = ""
        self.style = ""

    def setText(self, t):
        self.text = t

    def setStyleSheet(self, s):
        self.style = s


class _Edit:
    def __init__(self, t=""):
        self._t = t

    def text(self):
        return self._t


class _Column:
    def __init__(self):
        self.calls = []

    def set_hint(self, severity, text):
        self.calls.append((severity, text))


class _Combo:
    def __init__(self, items, cur=0):
        self.items, self.cur = items, cur

    def currentText(self):
        return self.items[self.cur]

    def count(self):
        return len(self.items)

    def itemText(self, i):
        return self.items[i]


def _ctrl(**ui):
    c = C.__new__(C)
    c.ui = type("U", (), {})()
    for k, v in ui.items():
        setattr(c.ui, k, v)
    c.working_dir = "/ws"
    c.active_run_id = "Cell"
    c.view_target = "lateral"
    c.selected_view = "XY"
    c.current_round_by_view = {"lateral": 2, "axial": 0}
    c.sample_records = {}
    return c


class TestHintToTheColumn:
    def test_the_six_branches_become_the_six_cases(self):
        c = _ctrl(training_mode_combo=_Combo([T.MODE_CONTINUE, T.MODE_SCRATCH]),
                  model_selector=_Combo(["scratch", "cyto3"], 1),
                  starting_checkpoint_path=_Edit(""))
        c._init_spec_for_active_bucket = lambda: {"mode": "scratch"}
        c.current_round_by_view = {"lateral": 0, "axial": 0}
        assert c._init_hint_parts()[:2] == ("apply", "round0_scratch")
        c._init_spec_for_active_bucket = lambda: {"mode": "pretrained", "path": "cyto3"}
        assert c._init_hint_parts()[:2] == ("override", "round0_locked")
        c.current_round_by_view = {"lateral": 2, "axial": 0}
        c._resolve_prior_model_checkpoint = lambda bucket=None: {"spatial": "x"}
        assert c._init_hint_parts()[:2] == ("override", "cumulative_prior")
        c._resolve_prior_model_checkpoint = lambda bucket=None: {}
        assert c._init_hint_parts()[:2] == ("warn", "cumulative_missing")
        c.ui.training_mode_combo.cur = 1
        c.ui.starting_checkpoint_path = _Edit("/a/b/custom.model")
        assert c._init_hint_parts()[:2] == ("apply", "scratch_file")
        c.ui.starting_checkpoint_path = _Edit("")
        assert c._init_hint_parts()[:2] == ("apply", "scratch_base")

    def test_no_run_is_neutral_and_not_a_case(self):
        c = _ctrl()
        c.working_dir = ""
        assert c._compute_init_hint() == ("neutral", "(no run loaded)")

    def test_refresh_hands_severity_and_text_to_the_column(self):
        col = _Column()
        c = _ctrl(training_setup_column=col,
                  training_mode_combo=_Combo([T.MODE_CONTINUE, T.MODE_SCRATCH]),
                  model_selector=_Combo(["scratch", "cyto3"], 1),
                  starting_checkpoint_path=_Edit(""))
        c._resolve_prior_model_checkpoint = lambda bucket=None: {"spatial": "x"}
        c._refresh_init_hint()
        assert col.calls == [("override",
                              "Starts from Cell_lateral_round_01 — "
                              "round 1’s lateral model.")]

    def test_a_missing_start_file_joins_the_hint_as_a_warning(self, tmp_path):
        col = _Column()
        c = _ctrl(training_setup_column=col,
                  training_mode_combo=_Combo([T.MODE_CONTINUE, T.MODE_SCRATCH], 1),
                  model_selector=_Combo(["scratch", "cyto3"], 1),
                  starting_checkpoint_path=_Edit(str(tmp_path / "gone.model")))
        c._refresh_init_hint()
        sev, text = col.calls[0]
        assert sev == "warn" and text.endswith(T.STARTING_FILE_NOT_FOUND)
        # an existing file is not a warning
        real = tmp_path / "ok.model"
        real.write_text("x")
        c.ui.starting_checkpoint_path = _Edit(str(real))
        c._refresh_init_hint()
        assert col.calls[1][0] == "apply"

    def test_a_ui_without_the_column_keeps_the_old_label(self):
        lab = _Label()
        c = _ctrl(_init_hint_label=lab,
                  training_mode_combo=_Combo([T.MODE_CONTINUE, T.MODE_SCRATCH], 1),
                  model_selector=_Combo(["scratch", "cyto3"], 1),
                  starting_checkpoint_path=_Edit(""))
        c._refresh_init_hint()
        assert "every round" in lab.text

    def test_the_six_sentences_are_constants_not_literals(self):
        src = inspect.getsource(C._init_hint_parts)
        for old in ("Round 0 from scratch", "Cumulative will inherit",
                    "From-scratch with"):
            assert old not in src


class TestWritesLabels:
    def test_active_bucket_says_the_file_and_the_round_neutral_not_amber(self):
        lat, ax = _Label(), _Label()
        c = _ctrl(model_name_lateral_label=lat, model_name_axial_label=ax)
        c._refresh_model_name_display()
        assert lat.text == "Cell_lateral_round_02 · round 2"
        assert ax.text == "axial: not in this round"
        assert "#d4a843" not in lat.style and "#d4a843" not in ax.style
        assert "bold" in lat.style
        c.view_target = "axial"
        c._refresh_model_name_display()
        assert ax.text == "Cell_axial_round_00 · round 0"
        assert lat.text == "lateral: not in this round"


class TestStagedCount:
    def test_the_count_and_the_other_bucket_line(self):
        lab = _Label()
        col = type("Col", (), {"set_desc": lambda s, b: setattr(s, "b", b)})()
        c = _ctrl(dataset_staging=lab, training_staged_column=col)
        c.sample_records = {
            "a": {"status": "staged", "bucket": "axial"},
            "b": {"status": "staged", "bucket": "axial"}}
        # the count is read from the records (audit M3), five of them lateral
        c.sample_records.update({f"l{i}": {"status": "staged",
                                           "bucket": "lateral"}
                                 for i in range(5)})
        c._staged_prior_pairs = ("lateral", 8)
        c._write_staged_count_text("lateral")
        first, second = lab.text.split("\n")
        assert first == T.staged_count_text(5, 13, 2, "lateral")
        assert first == ("5 new · round 2 trains on 13 lateral pairs in all "
                         "(8 from rounds 0–1).")
        assert second == T.staged_other_text("axial", 2)
        assert col.b == "lateral"

    def test_the_idle_stage_line_names_the_file_pairs_and_run(self):
        foot = type("F", (), {"set_stage_line": lambda s, t: setattr(s, "t", t)})()
        c = _ctrl(training_foot=foot)
        c._train_ui_locked = False
        c.sample_records = {f"l{i}": {"status": "staged", "bucket": "lateral"}
                            for i in range(5)}
        c._staged_prior_pairs = ("lateral", 8)
        c._refresh_training_stage_line()
        assert foot.t == T.STAGE_IDLE.format(
            name="Cell_lateral_round_02", n=13, bucket="lateral", run="Cell")
        c.sample_records = {}
        c._refresh_training_stage_line()
        assert foot.t == T.STAGE_IDLE_BARE

    def test_the_stage_line_is_left_alone_while_a_round_trains(self):
        foot = type("F", (), {"set_stage_line": lambda s, t: setattr(s, "t", t)})()
        c = _ctrl(training_foot=foot)
        c._train_ui_locked = True
        c._refresh_training_stage_line()
        assert not hasattr(foot, "t")
