from __future__ import annotations

from pathlib import Path

import yaml


def test_napari_manifest_points_to_main_widget():
    manifest_path = Path("napari_epicycle") / "napari.yaml"
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))

    assert payload["name"] == "napari-epicycle"
    command = payload["contributions"]["commands"][0]
    assert command["id"] == "napari-epicycle.main_widget"
    assert command["python_name"] == "napari_epicycle._widget:EpiCycleMainWidget"
