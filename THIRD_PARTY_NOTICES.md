# Third-party notices

EpiCycle's own code, including `packages/epicycle_scaledseg/` (the whole-tissue engine), is licensed under the
GNU General Public License, version 3 or (at your option) any later version (GPL-3.0-or-later). See [LICENSE](LICENSE).

**Model weights are not covered by EpiCycle's licence.** Cellpose's pretrained models (e.g. `cyto3`, `nuclei`) were trained on CC-BY-NC data, and a model you fine-tune from them may carry that restriction. EpiCycle grants no rights to them. For commercial use, check with the Cellpose authors or train from scratch.

EpiCycle calls, but does not include, the libraries below. They are installed separately by pip or conda and keep their own
licences. Licences were read from installed package metadata on 2026-10-13; this is a courtesy list.

## Code that EpiCycle includes

| Component | Licence | Notes |
|---|---|---|
| `packages/epicycle_largedata/_vendor_progressive/` | BSD-3-Clause | Copied from napari PR #9067 (head `kephale/napari@07e4f3fcfb602878a864aa0de4d9b08bf5a9a2f7`, fetched 2026-08-02). Copyright (c) 2018, Napari. Full text in that folder's `NOTICE`. A temporary copy, to be removed when napari ships the feature. |

## Called, not included: GPL and LGPL

| Library | Licence | Author / project |
|---|---|---|
| [u-Segment3D](https://github.com/DanuserLab/u-segment3D) | GPL-3.0 | Felix Y. Zhou, Danuser Lab. Installed from a pinned git commit (see `setup.cfg`). |
| PyQt6 | GPL-3.0 | Riverbank Computing. The Qt 6 libraries underneath are LGPL-3.0 (The Qt Company). |
| leidenalg | GPL-3.0-or-later | V.A. Traag |
| python-igraph | GPL-2.0-or-later | igraph project |
| edt | LGPL-3.0-or-later | pulled in via u-Segment3D / Cellpose |
| fastremap | LGPL-3.0 | Seung Lab (William Silversmith) |
| connected-components-3d | LGPL-3.0-or-later | Seung Lab (William Silversmith) |

Because EpiCycle is GPL-3.0-or-later, it can depend on these at runtime. u-Segment3D is GPL-3.0 (its LICENSE
file is the GPL version 3 text with no "or later" grant), so a combined installation is effectively GPL-3.0.

## Called, not included: permissive

| Library | Licence | Notes |
|---|---|---|
| [Cellpose](https://github.com/MouseLand/cellpose) 3.1.1.2 | BSD-3-Clause | Copyright 2020 Howard Hughes Medical Institute. Upstream's current README also shows a GPL v3 badge for the code; check upstream if this matters to you. Pretrained weights: see the weights notice above. |
| [PyTorch](https://github.com/pytorch/pytorch) 2.7.1 | BSD-3-Clause | |
| [segment-anything](https://github.com/facebookresearch/segment-anything) 1.0 | Apache-2.0 | Meta Platforms. Checkpoints are downloaded separately and carry their own terms; see the segment-anything repository. |
| [napari](https://github.com/napari/napari) 0.8.0 | BSD-3-Clause | Copyright (c) 2018, Napari |
| [qtawesome](https://github.com/spyder-ide/qtawesome) 1.4.1 | MIT | Copyright (c) 2015 The Spyder development team. EpiCycle uses its Material Design Icons (Apache-2.0), Remix Icon (Apache-2.0), Phosphor (MIT per upstream) and Font Awesome Free (SIL OFL for the fonts, CC-BY-4.0 for the icons per upstream) sets at run time. No fonts are bundled in this repository. |

MIT or BSD-style: qtpy, magicgui, superqt, pyqtgraph, vispy, numpy, scipy, scikit-image, scikit-learn, pandas,
matplotlib, tifffile, zarr, psutil, umap-learn, dask, trimesh. The whole-tissue engine also uses pyarrow (Apache-2.0) and tqdm (MPL-2.0 and MIT).

## Please cite

Cellpose, Cellpose 3 and u-Segment3D, as listed in the README and in [CITATION.cff](CITATION.cff).
