# EpiCycle

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

**Bootstrapped, human-in-the-loop 3D cell segmentation for stratified epithelia.**

## What it does

EpiCycle is a [napari](https://napari.org) plugin for human-in-the-loop 3D segmentation of stratified epithelia. It bootstraps dataset-specific segmentation models from sparse or no ground truth: starting from a handful of annotated 2D slices, a cycle of prediction, correction and retraining builds lateral and axial [Cellpose](https://github.com/MouseLand/cellpose) models while the curated slices accumulate into a training set. The refined models are then applied across whole stacks in all three orthoviews and fused into 3D cell reconstructions with [u-Segment3D](https://github.com/DanuserLab/u-segment3D), which can be measured, triaged and curated in 3D, fed back as further training data, or scaled up to whole tissues. We developed it for skin, where cells are crowded, anisotropic and differ between layers, but nothing in the workflow is specific to one tissue.

EpiCycle is under active development. We are documenting the codebase, expect to refactor parts of it, and may change file formats between versions; a workspace from an earlier version is migrated when it opens. Keep a backup of workspaces you care about.

The plugin has one panel for each of Steps 1-4; Step 5 runs from a notebook or the command line. You can jump between steps; each one checks its prerequisites when you run it.

| Step | Name | What it does |
|------|------|--------------|
| 1 | **Project Setup** | Load a volume, preprocess its channels if you want to, and draw and extract the ROI patches you will work on. |
| 2 | **ROI + HITL 2D** | Annotate slices in the lateral (XY) and axial (XZ, YZ) views, train Cellpose, review its predictions, stage your corrections and retrain. Each round feeds the next. |
| 3 | **Reconstruct + Curate 3D** | Predict all slices and fuse them to 3D. Measure more than 50 features per cell. Sort cells into good, review, bad and garbage with plots and clustering, fix cells in the three-view editor, and train a refined model. |
| 4 | **3D Stack Segmentation** | Run the finished model over whole stacks, measure, tidy and export. |
| 5 | **Whole tissue** | Tile, segment, reconstruct, stitch, clean up and measure very large images with the same models. Runs from a notebook or the command line (`epicycle-scaledseg run`) in the same environment; see [`tutorials/segmentation/`](tutorials/segmentation/). |

## System requirements

EpiCycle was developed and tested in Windows 11. Linux is experimental: all tests pass there, the GUI has not been used there for real work (but notebooks for Whole Tissue reconstructions should run). macOS is untested.

An NVIDIA GPU with a current driver is recommended. Without one, only the curation steps run; training models and prediction may not run properly or may be very slow.

32 GB of RAM and an 8 GB GPU is the tested minimum. More of both helps. What matters is that your working volume fits in memory with room for copies.

You also need Miniconda or Anaconda, about 10 GB of free disk, and internet access during install. The environment file installs Python 3.12 for you.

If your data is large or high-resolution, you can optionally downsample it before training. In Step 1, use the lazy loader to extract the regions you want instead of loading the whole volume. Then keep the same downsampling for everything after that. A model trained at one scale must predict at that scale. If you plan to segment whole tissue later, the data you train on and the data you segment must be downsampled the same way.

## Install

EpiCycle currently needs these exact versions:

- Python 3.12
- napari 0.8.0 (exactly this version) with PyQt6
- PyTorch 2.7.1 with CUDA 11.8 for training, prediction and the whole-tissue engine (CPU-only runs only the curation steps)
- Cellpose 3.1.1.2 (NOT Cellpose 4)
- u-Segment3D at the pinned commit `5d11e1ef`
- An NVIDIA GPU with a driver that supports CUDA 11.8 or newer. Newer drivers are backward compatible, so any current driver works; the more recent, the better.

The options below install exactly these.

**Option A, one file (recommended)**

1. Install [Miniconda](https://docs.conda.io/en/latest/miniconda.html).
2. Download [`environment.yml`](environment.yml) and, in an Anaconda Prompt, run `conda env create -f environment.yml`.
3. Run `conda activate epicycle`, then `napari -w napari-epicycle "EpiCycle Plugin"`.

**Option B, step by step**

Same versions, installed by hand. Run this in an Anaconda Prompt:

```
conda create -n epicycle python=3.12
conda activate epicycle
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r https://raw.githubusercontent.com/eL-Gene/EpiCycle/v0.1.1/requirements-lock.txt
pip install --no-deps "napari-epicycle @ https://github.com/eL-Gene/EpiCycle/archive/refs/tags/v0.1.1.zip"
```

The PyTorch line is the CUDA 11.8 build we test with; any current NVIDIA driver runs it (a CUDA 13 driver works too). If you prefer a different CUDA build of the same PyTorch version, pick its line on pytorch.org and use it in place of ours; later CUDA builds are untested but expected to work. Without a GPU, use the CPU build; only the curation steps run.

The `requirements-lock.txt` line installs every other dependency at the exact version we test with. The last line installs EpiCycle itself.

We install u-Segment3D from the exact version we used for the paper, not from its latest release. The latest release works with EpiCycle too, but it expects a newer CUDA than the rest of our setup, and installing it alongside ours leaves two GPU libraries fighting over the same job. We have not tested that on a GPU, so for now we stay with the version we know. Updating everything to the newer CUDA is on the list.

To install a later release, put its tag (`vX.Y.Z`) where the commands above say `v0.1.1`.

**Option C, from a clone**

```
git clone https://github.com/eL-Gene/EpiCycle.git
cd EpiCycle
conda create -n epicycle python=3.12
conda activate epicycle
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu118
pip install -e ".[pyqt6]" -c requirements-lock.txt
napari -w napari-epicycle "EpiCycle Plugin"
```

This is the install for anyone who wants to change the code; edits take effect without reinstalling.

## Launching EpiCycle

Open an Anaconda Prompt, run `conda activate epicycle`, then `napari -w napari-epicycle "EpiCycle Plugin"`. The EpiCycle panel opens in napari.

The first run downloads the Cellpose model weights (see Licence and third-party software), so give it a minute.

Start in Step 1 and choose a project folder, the workspace. Every step reads and writes that folder, and the steps share it.

Every section of the panels has a **?** button. It opens a short guide in plain language. The same guides are Markdown files in [`docs/guides/`](docs/guides/). When a control puzzles you, read the guide before you click around.

## Tutorials

There are eight notebooks in [`tutorials/`](tutorials/). Open them with JupyterLab. [`tutorials/README.md`](tutorials/README.md) lists what each one needs.

**`tutorials/segmentation/`** has three. The first runs the whole-tissue segmentation on a large 3D stack with the two models you trained in Steps 2-3 (`lateral` and `axial`). The second compares a cell segmentation with a nuclei segmentation of the same stack. The third cleans the two up to a consistent consensus. These notebooks are meant for a GPU cluster or a large workstation: it needs an NVIDIA GPU, RAM at least the size of the input image and disk for several copies of it (we run it on a Linux cluster, for all reconstructions shown in our manuscript). It will run on a Windows PC for a small stack, and it is also available from the command line as `epicycle-scaledseg run`, but on a desktop PC check your RAM and disk against the input size first.

**`tutorials/analysis/`** has five. They start from a per-cell feature table, from Step 4's outputs. They cover cell-shape clusters and a UMAP, neighbourhood niches, figure panels for each, and a comparison of two groups of cells. Trained model checkpoints and example data are released with the paper in a separate reproducibility repository; the link will be added here when it is public.

## Measured features

EpiCycle measures more than 50 features per cell in Steps 3 and 4 and in the whole-tissue engine. They are listed and explained in [`docs/research/QC_FEATURES_BIOLOGIST_GUIDE.md`](docs/research/QC_FEATURES_BIOLOGIST_GUIDE.md), with the equations in [`docs/research/QC_FEATURE_REFERENCE.md`](docs/research/QC_FEATURE_REFERENCE.md) and the full table in [`docs/research/FEATURE_CATALOG.csv`](docs/research/FEATURE_CATALOG.csv).

## Getting help

Try the **?** buttons first within each UI section. If something breaks, please [open an issue on GitHub](https://github.com/eL-Gene/EpiCycle/issues). The bug-report form asks for your operating system, the EpiCycle and napari versions, the step you were in, what you clicked, and the console output.

## Contributing

Read [CONTRIBUTING.md](CONTRIBUTING.md): how to set up a development copy, run the tests and open a pull request. Where things live: `napari_epicycle/` is the plugin (panels, controllers, annotators), `packages/` the logic that does not need Qt, `docs/guides/` the in-app help text, `tutorials/` the notebooks, `docs/research/` the feature documents, `tests/` the test suite.

## Citing

To cite EpiCycle, see [CITATION.cff](CITATION.cff). GitHub's "Cite this repository" button reads it.

Please also cite the tools EpiCycle builds on:

- **napari**: napari contributors (2019). napari: a multi-dimensional image viewer for python. doi:10.5281/zenodo.3555620.
- **Cellpose**: Stringer, C., Wang, T., Michaelos, M. & Pachitariu, M. (2021). Cellpose: a generalist algorithm for cellular segmentation. *Nature Methods* 18(1), 100-106.
- **Cellpose 2.0**: Pachitariu, M. & Stringer, C. (2022). Cellpose 2.0: how to train your own model. *Nature Methods*. [doi:10.1038/s41592-022-01663-4](https://doi.org/10.1038/s41592-022-01663-4)
- **u-Segment3D**: Zhou, F.Y., Marin, Z., Yapp, C., Zhou, Q., Nanes, B.A., Daetwyler, S., Jamieson, A.R., Islam, M.T., Jenkins, E., Gihana, G.M., Lin, J., Borges, H.M., Chang, B.-J., Weems, A., Morrison, S.J., Sorger, P.K., Fiolka, R., Dean, K. & Danuser, G. Universal consensus 3D segmentation of cells from 2D segmented stacks. *Nature Methods* 22, 2386-2399 (2025). [doi:10.1038/s41592-025-02887-w](https://doi.org/10.1038/s41592-025-02887-w)

## Licence and third-party software

EpiCycle is free software under the GNU General Public License, version 3 or (at your option) any later version. See [LICENSE](LICENSE). The licences of the libraries it uses, and of the one component it copies, are in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

**Cellpose's pretrained models are not covered by this licence.** They were trained on CC-BY-NC data, and a model you fine-tune from them may carry that restriction. For commercial use, check with the Cellpose authors or train from scratch.

## What's coming (generally)

- A napari 3D view of niche clusters in the analysis notebooks
- A command-line entry for the reconcile stage + wiring of Step 5
- A results viewer
- Additional measurable features
- PyPI packaging
- The whole stack on CUDA 12, including the current u-Segment3D release
- Updated readme and better guides
- Quality of life changes from user suggestions
- Other potential bug fixes

