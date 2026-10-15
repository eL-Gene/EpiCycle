# Contributing to EpiCycle

## Status

We are still actively documenting the codebase, and we expect to refactor parts of it,
including a few very large modules. A guide to the architecture and an instructions file for AI coding assistants are
on the way. Until they exist, open an issue before starting a large change so we can coordinate with you; things may change as we move forward.

## Ways to help

- **Report a bug** through the [bug report form](https://github.com/eL-Gene/EpiCycle/issues/new/choose). Include your
  operating system and GPU, the EpiCycle and napari versions, the step you were in, what you clicked, the full console
  output, and the shape of your data. A report that only says "it crashed" is very hard to act on.
- **Ask a question** by opening an issue. If the in-app **?** guides did not answer it, we want to know and further improve accordingly.
- **Improve the documentation.** The **?** guides are plain Markdown files in [`docs/guides/`](docs/guides/). They are
  written for biologists, and corrections are always useful.
- **Write code.** The sections below cover setup, tests and pull requests.

## Setting up a development copy

Python 3.12 and napari 0.8.0 (exactly that version) with PyQt6; the [README](README.md) lists everything else.

```
git clone https://github.com/eL-Gene/EpiCycle.git
cd EpiCycle
conda create -n epicycle python=3.12
conda activate epicycle
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[pyqt6]" -c requirements-lock.txt
napari -w napari-epicycle "EpiCycle Plugin"
```

## Tests

Run them from the repository root. The Qt-free set needs no display:

```
pytest tests/domain tests/io_registry tests/tiled_segmentation tests/smoke tests/scaledseg -p no:napari -o addopts=""
```

The tests in `tests/scaledseg` start process pools, so run the heavier files one at a time. The Step 3 tests
need Qt and are run one file at a time: `QT_QPA_PLATFORM=offscreen pytest tests/fusion_reconstruction/<file>.py -o addopts=""`.
A few pixel-layout tests run only on Windows because fonts differ between systems. CI runs on every pull request.

We write the failing test first whenever we can. It is not a requirement, but a change that comes with a test is much
easier to review and much less likely to break later.

## Pull requests

Branch from `main`, keep each pull request to one change, and describe what it does and why. If you changed a control,
update its guide in `docs/guides/`. We review every pull request and merge it when it is ready.

## Where the methods live

- Training: `packages/epicycle_training/`
- 2D-to-3D reconstruction: `packages/epicycle_reconstruction/` for the plugin, `packages/epicycle_scaledseg/core/` for whole tissue
- Feature extraction: `napari_epicycle/fusion_reconstruction/qc_features.py` (the plugin) and `packages/epicycle_scaledseg/core/qc_features.py` (the whole-tissue engine); the equations are in `docs/research/`
- Cell-nucleus reconcile: `packages/epicycle_scaledseg/reconcile/`
- Analysis helpers: `tutorials/analysis/_helpers.py`

If you are checking the paper's Methods against the code, start there.

## AI coding tools

You are welcome to use them! Clone the repository, let your assistant read it, experiment, and add or improve features. Though you remain responsible for what you submit, we'd appreciate due diligence to design and verify things thoroughly. A repository-level setup for agentic coding is currently in the works!

## Licence

EpiCycle is GPL-3.0-or-later (see [LICENSE](LICENSE)). Contributions are accepted under the same licence.
