"""Tests for the Step-1 large-data "Prepare settings" — resource-adaptive
recommender, per-workspace persistence, the controller's effective-config
helper, and the config dialog's preset->fields + estimate wiring.

The adaptive math + validation are PURE (napari-free) and tested headless; the
persistence helpers ride the existing step1_state.json manager; the dialog +
controller tests use the shared Qt app fixture.
"""
import json
import os
from types import SimpleNamespace

import pytest


def _superset(got, expected) -> bool:
    """*got* carries every key/value of *expected* (Brick 4b adds keys)."""
    return all(got.get(k) == v for k, v in expected.items())


from packages.epicycle_largedata.config import (
    ALLOWED_CHUNKS,
    DEFAULT_PRESET,
    PRESET_NAMES,
    detect_machine_resources,
    estimate_peak_ram_bytes,
    normalize_convert_config,
    recommend_convert_settings,
)
from napari_epicycle.data_preparation._paths import step1_state_path
from napari_epicycle.data_preparation._state import (
    load_largedata_config,
    load_state,
    register_source,
    save_largedata_config,
    save_state,
)

_GB = 1024 ** 3
_MB = 1024 ** 2
_BIG_RAM = 500 * _GB  # a machine where the clamp never bites


# ===========================================================================
# 1) Adaptive recommender (pure)
# ===========================================================================
class TestRecommendConvertSettings:
    def _rec(self, preset, cpu, avail=_BIG_RAM, total=_BIG_RAM, itemsize=2):
        return recommend_convert_settings(
            preset, total_ram_bytes=total, avail_ram_bytes=avail,
            cpu_count=cpu, dtype_itemsize=itemsize)

    def test_preset_names(self):
        assert PRESET_NAMES == ("safe", "balanced", "fast")

    def test_safe_is_portable_current_behavior(self):
        # Brick 4b: every preset now uses the ANISOTROPIC (32, 128, 128) shape +
        # zstd; the presets differ ONLY in worker count.  `chunk` stays as the
        # legacy cubic mirror (the in-plane edge) for old readers.
        # `chunk` mirrors the Z axis (32), not the in-plane edge -- an older
        # checkout reading it prepares a small safe cube, not a cubic 128.
        r = self._rec("safe", cpu=32)
        assert r == {"preset": "safe", "num_workers": 1, "chunk": 32,
                     "chunk_zyx": (32, 128, 128), "compressor": "zstd",
                     "downsample": 2, "min_size": 256}

    def test_safe_num_workers_is_one_regardless_of_cpu(self):
        assert self._rec("safe", cpu=1)["num_workers"] == 1
        assert self._rec("safe", cpu=64)["num_workers"] == 1

    def test_balanced_scales_with_cpu_capped_at_4(self):
        assert self._rec("balanced", cpu=1)["num_workers"] == 1
        assert self._rec("balanced", cpu=2)["num_workers"] == 2
        assert self._rec("balanced", cpu=8)["num_workers"] == 4  # min(4, 8)
        assert self._rec("balanced", cpu=32)["num_workers"] == 4  # capped
        # Brick 4b: the shape is the same anisotropic (32, 128, 128) in every
        # preset - thin-Z for the scroll, big-in-plane for the write.
        r = self._rec("balanced", cpu=8)
        assert (tuple(r["chunk_zyx"]), r["downsample"], r["min_size"]) == (
            (32, 128, 128), 2, 256)

    def test_fast_uses_all_cores_on_big_ram(self):
        assert self._rec("fast", cpu=8)["num_workers"] == 8
        assert self._rec("fast", cpu=20)["num_workers"] == 20
        assert self._rec("fast", cpu=64)["num_workers"] == 64

    def test_fast_uses_the_same_shape_only_more_workers(self):
        r = self._rec("fast", cpu=8)
        assert tuple(r["chunk_zyx"]) == (32, 128, 128)
        assert (r["downsample"], r["min_size"]) == (2, 256)

    def test_each_preset_scales_with_cpu(self):
        # balanced + fast both grow with cpu (until balanced's cap); safe flat.
        for cpu in (1, 2, 4):
            assert self._rec("fast", cpu=cpu)["num_workers"] == cpu
            assert self._rec("balanced", cpu=cpu)["num_workers"] == min(4, cpu)
            assert self._rec("safe", cpu=cpu)["num_workers"] == 1

    def test_clamp_reduces_workers_on_low_avail_ram(self):
        # Brick 4b: per-worker = 32*128*128*2 = 1 MiB.  avail = 40 MiB ->
        # budget = 0.25*40 = 10 MiB -> max 10 workers, so cpu=8 is NOT clamped;
        # drop the budget until it is.
        r = self._rec("fast", cpu=8, avail=8 * _MB)
        assert r["num_workers"] == 2
        assert r["num_workers"] < 8  # the clamp bit

    def test_clamp_never_below_one(self):
        r = self._rec("fast", cpu=16, avail=1)  # 1 byte avail
        assert r["num_workers"] == 1

    def test_ram_unknown_skips_clamp(self):
        # psutil-less fallback path: avail = 0 (unknown) -> no clamp, all cores.
        r = self._rec("fast", cpu=8, avail=0)
        assert r["num_workers"] == 8

    def test_itemsize_affects_clamp(self):
        # A larger itemsize means a bigger per-worker footprint -> fewer workers
        # survive the same RAM budget.
        avail = 8 * _MB
        r2 = self._rec("fast", cpu=8, avail=avail, itemsize=2)
        r4 = self._rec("fast", cpu=8, avail=avail, itemsize=4)
        assert r4["num_workers"] < r2["num_workers"]
        assert r4["num_workers"] == 1  # per-worker 2 MiB -> budget 2 MiB -> 1

    def test_deterministic(self):
        a = self._rec("fast", cpu=8, avail=16 * _GB)
        b = self._rec("fast", cpu=8, avail=16 * _GB)
        assert a == b

    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError):
            self._rec("custom", cpu=8)
        with pytest.raises(ValueError):
            self._rec("turbo", cpu=8)

    def test_cpu_count_floored_at_one(self):
        assert self._rec("fast", cpu=0)["num_workers"] == 1


class TestEstimatePeakRam:
    def test_formula(self):
        assert estimate_peak_ram_bytes(1, 64, 2) == 64 ** 3 * 2
        assert estimate_peak_ram_bytes(4, 128, 2) == 4 * 128 ** 3 * 2

    def test_itemsize_respected(self):
        assert (estimate_peak_ram_bytes(1, 64, 4)
                == 2 * estimate_peak_ram_bytes(1, 64, 2))

    def test_floors_degenerate_inputs(self):
        # 0 workers floored to 1, and a nonsense chunk falls back to the default
        # shape, so the estimate is never a nonsense 0.
        assert estimate_peak_ram_bytes(0, 0, 0) > 0


class TestNormalizeConvertConfig:
    def _valid(self):
        return {"preset": "balanced", "num_workers": 4, "chunk": 64,
                "downsample": 2, "min_size": 256}

    def test_valid_roundtrips(self):
        # Brick 4b keys are ADDITIVE: every original key survives verbatim.
        out = normalize_convert_config(self._valid())
        assert out is not None
        for k, v in self._valid().items():
            assert out[k] == v

    def test_custom_preset_allowed(self):
        c = self._valid()
        c["preset"] = "custom"
        assert normalize_convert_config(c)["preset"] == "custom"

    def test_extra_keys_dropped(self):
        c = self._valid()
        c["junk"] = 999
        assert "junk" not in normalize_convert_config(c)

    def test_non_dict_none(self):
        assert normalize_convert_config(None) is None
        assert normalize_convert_config([1, 2]) is None
        assert normalize_convert_config("x") is None

    def test_missing_field_none(self):
        c = self._valid()
        del c["chunk"]
        assert normalize_convert_config(c) is None

    def test_empty_preset_none(self):
        c = self._valid()
        c["preset"] = "  "
        assert normalize_convert_config(c) is None

    def test_bool_knob_rejected(self):
        c = self._valid()
        c["num_workers"] = True  # bool is an int subclass — must be rejected
        assert normalize_convert_config(c) is None

    def test_non_positive_knob_none(self):
        c = self._valid()
        c["chunk"] = 0
        assert normalize_convert_config(c) is None
        c = self._valid()
        c["min_size"] = -64
        assert normalize_convert_config(c) is None

    def test_string_knob_none(self):
        c = self._valid()
        c["num_workers"] = "4"
        assert normalize_convert_config(c) is None


class TestDetectMachineResources:
    def test_shape_and_positivity(self):
        r = detect_machine_resources()
        assert set(r) == {"cpu_count", "total_ram_bytes", "avail_ram_bytes"}
        assert r["cpu_count"] >= 1
        assert r["total_ram_bytes"] >= 0
        assert r["avail_ram_bytes"] >= 0


# ===========================================================================
# 2) Per-workspace persistence (rides step1_state.json)
# ===========================================================================
class TestLargedataConfigPersistence:
    def _cfg(self):
        return {"preset": "fast", "num_workers": 6, "chunk": 128,
                "downsample": 2, "min_size": 256}

    def test_roundtrip(self, tmp_path):
        ws = str(tmp_path)
        save_largedata_config(ws, self._cfg())
        assert _superset(load_largedata_config(ws), self._cfg())

    def test_survives_state_normalize_roundtrip(self, tmp_path):
        # The config must round-trip through load_state/_normalize at the CURRENT
        # schema version (not be dropped as an unknown top-level key).
        ws = str(tmp_path)
        save_largedata_config(ws, self._cfg())
        state = load_state(ws)                 # normalizes on load
        save_state(ws, state)                  # re-persist normalized
        assert _superset(load_largedata_config(ws), self._cfg())

    def test_missing_returns_none(self, tmp_path):
        assert load_largedata_config(str(tmp_path)) is None

    def test_corrupt_state_returns_none(self, tmp_path):
        ws = str(tmp_path)
        step1_state_path(ws).write_text("{ this is not json", encoding="utf-8")
        assert load_largedata_config(ws) is None

    def test_garbage_config_value_returns_none(self, tmp_path):
        ws = str(tmp_path)
        bad = {"schema_version": 1, "sources": [], "patches": [],
               "settings": {},
               "largedata_convert_config": {"preset": "x",
                                            "num_workers": "bad"}}
        step1_state_path(ws).write_text(json.dumps(bad), encoding="utf-8")
        assert load_largedata_config(ws) is None

    def test_save_preserves_other_state(self, tmp_path):
        ws = str(tmp_path)
        state = load_state(ws)
        register_source(state, path=str(tmp_path / "src.tif"),
                        display_name="kidney", axes="CZYX")
        save_state(ws, state)
        save_largedata_config(ws, self._cfg())
        reloaded = load_state(ws)
        assert len(reloaded["sources"]) == 1          # source survived
        assert _superset(reloaded["largedata_convert_config"], self._cfg())

    def test_save_invalid_raises(self, tmp_path):
        with pytest.raises(ValueError):
            save_largedata_config(str(tmp_path), {"preset": "safe"})  # no knobs


# ===========================================================================
# 3) Controller: _current_largedata_config returns saved-else-adaptive
# ===========================================================================
class TestCurrentLargedataConfig:
    def _controller(self, workspace):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController)
        ctl = DataPreparationController.__new__(DataPreparationController)
        ctl.widget = SimpleNamespace(
            output_file_path=SimpleNamespace(text=lambda: str(workspace)))
        return ctl

    def test_returns_saved_config_when_present(self, tmp_path):
        cfg = {"preset": "custom", "num_workers": 3, "chunk": 256,
               "downsample": 3, "min_size": 128}
        save_largedata_config(str(tmp_path), cfg)
        ctl = self._controller(tmp_path)
        assert _superset(ctl._current_largedata_config(), cfg)

    def test_falls_back_to_adaptive_balanced_default(self, tmp_path):
        ctl = self._controller(tmp_path)          # nothing saved
        eff = ctl._current_largedata_config()
        assert eff["preset"] == DEFAULT_PRESET == "balanced"
        # full dict so callers can read every knob unconditionally
        assert set(eff) == {"preset", "num_workers", "chunk", "chunk_zyx",
                            "compressor", "downsample", "min_size"}

    def test_empty_workspace_uses_adaptive_default(self, tmp_path):
        ctl = self._controller("")                # no workspace at all
        eff = ctl._current_largedata_config()
        assert eff["preset"] == "balanced"

    def test_explicit_workspace_arg_overrides_widget(self, tmp_path):
        # `custom`, so the saved knobs come back verbatim and the assertion is
        # about WHICH WORKSPACE was read -- which is what this test is for.  A
        # named preset is deliberately re-derived (see TestNamedPresetIsReDerived),
        # so it could not distinguish "read the right workspace" from "recomputed".
        cfg = {"preset": "custom", "num_workers": 2, "chunk": 64,
               "downsample": 2, "min_size": 256}
        save_largedata_config(str(tmp_path), cfg)
        ctl = self._controller("")                # widget says no workspace
        assert _superset(ctl._current_largedata_config(str(tmp_path)), cfg)


# ===========================================================================
# 4) Config dialog: preset->fields mapping + live estimate readout
# ===========================================================================
@pytest.fixture(scope="module")
def qt_app():
    pytest.importorskip("qtpy")
    from qtpy.QtWidgets import QApplication

    existing = QApplication.instance()
    if existing is not None:
        yield existing
        return
    yield QApplication([])


class TestLargeDataConfigDialog:
    def _dlg(self, cpu=8, total=32 * _GB, avail=16 * _GB, initial=None):
        from napari_epicycle.data_preparation._largedata_config_dialog import (
            LargeDataConfigDialog)
        return LargeDataConfigDialog(
            None, cpu_count=cpu, total_ram_bytes=total, avail_ram_bytes=avail,
            initial_config=initial)

    def test_default_is_adaptive_balanced_and_readonly(self, qt_app):
        d = self._dlg()
        try:
            cfg = d.get_config()
            assert cfg["preset"] == "balanced"
            assert cfg["num_workers"] == 4      # min(4, 8)
            assert tuple(cfg["chunk_zyx"]) == (32, 128, 128)
            assert not d.workers_spin.isEnabled()   # named preset -> read-only
            assert not d.chunk_combo.isEnabled()
        finally:
            d.setParent(None)

    def test_selecting_fast_fills_fields_and_locks(self, qt_app):
        d = self._dlg(cpu=8, avail=16 * _GB)
        try:
            d.preset_combo.setCurrentText("Fast")
            cfg = d.get_config()
            assert cfg["preset"] == "fast"
            assert cfg["num_workers"] == 8       # all cores, 16 GB -> no clamp
            assert tuple(cfg["chunk_zyx"]) == (32, 128, 128)
            assert not d.workers_spin.isEnabled()
        finally:
            d.setParent(None)

    def test_safe_preset_maps_to_one_worker(self, qt_app):
        d = self._dlg(cpu=8)
        try:
            d.preset_combo.setCurrentText("Safe")
            cfg = d.get_config()
            assert _superset(cfg, {"preset": "safe", "num_workers": 1,
                                   "downsample": 2, "min_size": 256})
            assert tuple(cfg["chunk_zyx"]) == (32, 128, 128)
        finally:
            d.setParent(None)

    def test_custom_enables_editing(self, qt_app):
        d = self._dlg(cpu=8)
        try:
            d.preset_combo.setCurrentText("Custom")
            assert d.workers_spin.isEnabled()
            assert d.chunk_combo.isEnabled()
            d.workers_spin.setValue(2)
            d._select_chunk((64, 256, 256))
            d.downsample_spin.setValue(3)
            d.min_size_spin.setValue(128)
            cfg = d.get_config()
            assert _superset(cfg, {"preset": "custom", "num_workers": 2,
                                   "downsample": 3, "min_size": 128})
            assert tuple(cfg["chunk_zyx"]) == (64, 256, 256)
        finally:
            d.setParent(None)

    def test_estimate_readout_matches_formula(self, qt_app):
        d = self._dlg(cpu=8, avail=16 * _GB)
        try:
            d.preset_combo.setCurrentText("Fast")   # 8 workers x the default shape
            expect_mb = estimate_peak_ram_bytes(8, (32, 128, 128), 2) / _MB
            assert f"~{expect_mb:.0f} MB" in d.estimate_label.text()
            assert d.estimate_label.text().strip() != ""
        finally:
            d.setParent(None)

    def test_estimate_updates_live_in_custom(self, qt_app):
        d = self._dlg(cpu=16)
        try:
            d.preset_combo.setCurrentText("Custom")
            d.workers_spin.setValue(1)
            d._select_chunk((16, 128, 128))
            small = estimate_peak_ram_bytes(1, (16, 128, 128), 2) / _MB
            assert f"~{small:.0f} MB" in d.estimate_label.text()
            d.workers_spin.setValue(8)
            d._select_chunk((64, 256, 256))
            big = estimate_peak_ram_bytes(8, (64, 256, 256), 2) / _MB
            assert f"~{big:.0f} MB" in d.estimate_label.text()
        finally:
            d.setParent(None)

    def test_seed_named_preset_is_adaptive_to_current_machine(self, qt_app):
        # A "fast" config saved on a 64-core box re-seeds on an 8-core box as
        # 8 workers (recomputed), not the stale 64.
        d = self._dlg(cpu=8, initial={"preset": "fast", "num_workers": 64,
                                      "chunk": 128, "downsample": 2,
                                      "min_size": 256})
        try:
            assert d.get_config()["preset"] == "fast"
            assert d.get_config()["num_workers"] == 8
            assert not d.workers_spin.isEnabled()
        finally:
            d.setParent(None)

    def test_seed_custom_loads_raw_knobs(self, qt_app):
        d = self._dlg(cpu=8, initial={"preset": "custom", "num_workers": 3,
                                      "chunk": 256, "downsample": 3,
                                      "min_size": 128})
        try:
            assert d.workers_spin.isEnabled()          # custom -> editable
            cfg = d.get_config()
            assert _superset(cfg, {"preset": "custom", "num_workers": 3,
                                   "downsample": 3, "min_size": 128})
            # a persisted CUBIC custom shape is preserved verbatim, not
            # silently rewritten to the new anisotropic default
            assert tuple(cfg["chunk_zyx"]) == (256, 256, 256)
        finally:
            d.setParent(None)

    def test_header_reports_detected_resources(self, qt_app):
        d = self._dlg(cpu=12, total=32 * _GB, avail=16 * _GB)
        try:
            # header label is the first child QLabel; assert on the dialog text
            texts = [w.text() for w in d.findChildren(type(d.estimate_label))]
            joined = " ".join(texts)
            assert "12 cores" in joined
            assert "32 GB" in joined
        finally:
            d.setParent(None)



# ===========================================================================
# Phase 2 item 2: a saved NAMED preset is RE-DERIVED, not replayed verbatim
# ===========================================================================
class TestNamedPresetIsReDerived:
    """A saved `{"preset": "balanced", "chunk": 64, "num_workers": 6}` is a
    snapshot of what "balanced" meant on the machine that saved it.  Replaying it
    verbatim pinned every pre-Brick-4b workspace to the OLD cubic shape (and to a
    stale core count), so the user never received the fix.  A NAMED preset means
    "whatever the recommender currently says"; only `custom` is a literal."""

    def _controller(self, workspace):
        from napari_epicycle.data_preparation.controller import (
            DataPreparationController)
        ctl = DataPreparationController.__new__(DataPreparationController)
        ctl.widget = SimpleNamespace(
            output_file_path=SimpleNamespace(text=lambda: str(workspace)))
        return ctl

    def test_a_saved_balanced_preset_adopts_the_current_shape(self, tmp_path):
        save_largedata_config(str(tmp_path), {
            "preset": "balanced", "num_workers": 6, "chunk": 64,
            "downsample": 2, "min_size": 256})
        eff = self._controller(tmp_path)._current_largedata_config()
        assert eff["preset"] == "balanced"
        assert tuple(eff["chunk_zyx"]) == (32, 128, 128)

    def test_a_saved_preset_also_refreshes_the_core_count(self, tmp_path):
        """The saved 999 workers came from another machine; re-deriving must
        scale it to THIS one rather than trusting the number."""
        save_largedata_config(str(tmp_path), {
            "preset": "balanced", "num_workers": 999, "chunk": 64,
            "downsample": 2, "min_size": 256})
        eff = self._controller(tmp_path)._current_largedata_config()
        assert eff["num_workers"] <= max(1, (os.cpu_count() or 1))

    def test_a_saved_custom_config_is_honoured_verbatim(self, tmp_path):
        cfg = {"preset": "custom", "num_workers": 3, "chunk": 64,
               "downsample": 3, "min_size": 128}
        save_largedata_config(str(tmp_path), cfg)
        eff = self._controller(tmp_path)._current_largedata_config()
        assert eff["preset"] == "custom"
        assert eff["num_workers"] == 3
        assert tuple(eff["chunk_zyx"]) == (64, 64, 64)
        assert eff["downsample"] == 3 and eff["min_size"] == 128

    def test_a_saved_preset_with_an_explicit_shape_still_re_derives(self, tmp_path):
        """`chunk_zyx` on a NAMED preset was written by the recommender, not by
        the user, so it is still just a snapshot."""
        save_largedata_config(str(tmp_path), {
            "preset": "fast", "num_workers": 2, "chunk": 64,
            "downsample": 2, "min_size": 256, "chunk_zyx": [64, 64, 64]})
        eff = self._controller(tmp_path)._current_largedata_config()
        assert tuple(eff["chunk_zyx"]) == (32, 128, 128)

    def test_no_saved_config_still_gives_the_adaptive_default(self, tmp_path):
        eff = self._controller(tmp_path)._current_largedata_config()
        assert eff["preset"] == DEFAULT_PRESET
        assert tuple(eff["chunk_zyx"]) == (32, 128, 128)


# ===========================================================================
# Phase 2 items 3 + 4: compressor reaches the converter; the dialog gets the
# real source shape + read regime
# ===========================================================================
class TestCompressorAndDialogWiring:
    def test_the_prep_worker_forwards_the_compressor(self, tmp_path, monkeypatch):
        """A config choosing lz4 must actually produce an lz4 store -- the knob
        was collected by the dialog and persisted, but never threaded into
        convert_to_ome_zarr, so every store came out zstd regardless."""
        from napari_epicycle.data_preparation import controller as _ctl

        seen = {}

        def _fake_convert(src, out, **kw):
            seen.update(kw)
            return {"path": str(out)}

        monkeypatch.setattr(_ctl, "convert_to_ome_zarr", _fake_convert,
                            raising=False)
        w = _ctl._LargeDataPrepWorker(
            str(tmp_path / "s.tif"), str(tmp_path / "o.zarr"), None, "ZYX",
            num_workers=1, chunk=(32, 128, 128), compressor="lz4",
            downsample=2, min_size=256)
        w.run()
        assert seen.get("compressor") == "lz4"
        assert tuple(seen.get("chunk")) == (32, 128, 128)

    def test_an_unset_compressor_is_omitted_so_the_engine_default_stands(
            self, tmp_path, monkeypatch):
        from napari_epicycle.data_preparation import controller as _ctl

        seen = {}
        monkeypatch.setattr(
            _ctl, "convert_to_ome_zarr",
            lambda src, out, **kw: (seen.update(kw), {"path": str(out)})[1],
            raising=False)
        w = _ctl._LargeDataPrepWorker(
            str(tmp_path / "s.tif"), str(tmp_path / "o.zarr"), None, "ZYX",
            num_workers=1)
        w.run()
        assert "compressor" not in seen

    def test_an_lz4_config_really_writes_an_lz4_store(self, tmp_path):
        """End-to-end through the real converter, no fakes."""
        import numpy as np
        import tifffile
        import zarr
        from napari_epicycle.data_preparation import controller as _ctl

        a = (np.arange(8 * 64 * 64) % 900).astype(np.uint16).reshape(8, 64, 64)
        src = tmp_path / "real.tif"
        tifffile.imwrite(str(src), a)
        out = tmp_path / "real.zarr"
        w = _ctl._LargeDataPrepWorker(
            str(src), str(out), None, "ZYX", num_workers=1,
            chunk=(4, 32, 32), compressor="lz4")
        w.run()
        assert zarr.open_group(str(out), mode="r")["0"].compressor.cname == "lz4"

    def test_the_dialog_receives_the_source_shape_and_regime(self, tmp_path,
                                                             monkeypatch):
        from napari_epicycle.data_preparation import controller as _ctl

        captured = {}

        class _FakeDialog:
            def __init__(self, parent, **kw):
                captured.update(kw)

            def exec_(self):
                return 0

        # the controller imports the dialog INSIDE the method, so patch it at
        # its defining module rather than on the controller module
        from napari_epicycle.data_preparation import (
            _largedata_config_dialog as _dlgmod)
        monkeypatch.setattr(_dlgmod, "LargeDataConfigDialog", _FakeDialog)
        ctl = _ctl.DataPreparationController.__new__(_ctl.DataPreparationController)
        ctl.widget = SimpleNamespace(
            output_file_path=SimpleNamespace(text=lambda: str(tmp_path)))
        ctl._largedata_source_shape_for_estimate = lambda: (143, 1024, 1024)
        ctl._largedata_source_regime_for_estimate = lambda: "memmap"
        assert ctl.open_largedata_config_dialog() is None      # cancelled
        assert captured["source_shape"] == (143, 1024, 1024)
        assert captured["source_regime"] == "memmap"
