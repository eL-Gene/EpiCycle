# Tutorials

Notebooks that show how to use EpiCycle's engines outside the napari plugin. They live in two folders, numbered in the order you would meet them.


**Which notebooks reflect the paper.** `segmentation/01` and the five `analysis/` notebooks follow the workflow used in the manuscript. In the paper the cell-nucleus pairing (segmentation/02 and 03) is a demonstration: on the mouse salivary gland it was used to make each nucleus fit inside one cell and to count how many cells contain one. No conclusions in the paper rest on it, and its accuracy has not been evaluated against ground truth; use it to inspect and curate, and judge the pairing on your own data. `segmentation/03` applies the pairing rule the paper used; `segmentation/02` measures the same overlap with its own thresholds.

## segmentation/

Turn a large 3D image into labelled cells, and reconcile a cell segmentation with a nuclei segmentation.

| Notebook | What it does |
|---|---|
| `01_run_whole_tissue_segmentation.ipynb` | Run the whole-tissue segmentation engine (`packages/epicycle_scaledseg`) on one large 3D stack with the two models you trained in Steps 2-3: tile, segment, reconstruct, stitch, clean up, measure. Ends with a per-cell feature table. |
| `02_reconcile_cells_and_nuclei.ipynb` | Take a cell segmentation and a nuclei segmentation of the same stack (two runs of notebook 01) and measure how well they agree: the cell-by-nucleus overlap, and which cells hold two nuclei, no nucleus, or share a nucleus. Nothing is edited. |
| `03_cleanup_to_consensus.ipynb` | Make the cell and nuclei volumes consistent: cut straddling nuclei at the cell boundary, rescue missed nuclei, drop fragments, and write a consistent pair plus feature and triage tables. |

## analysis/

Explore the shapes and neighbourhoods of cells from a per-cell feature table.

| Notebook | What it does |
|---|---|
| `01_cell_shape_morphospace.ipynb` | PCA, clusters and a UMAP of cell shapes from a feature table. |
| `02_cell_neighbourhood_niches.ipynb` | Group cells by their neighbourhood, using the contact graph. |
| `03_figure_panels.ipynb` | Editable SVG and PNG figure panels from the shape clusters. |
| `04_niche_figure_panels.ipynb` | The same figure panels for the niche clusters, with 3D views. |
| `05_compare_two_groups_of_cells.ipynb` | Do two groups of cells differ in shape? Effect sizes with bootstrap confidence intervals. |

The analysis notebooks start from the per-cell feature table that `segmentation/01` ends with, or that Step 4's Export writes, together with the adjacency CSV beside it. The figure notebooks need a few extra plotting libraries (plotly, kaleido, pyvista): install them with `pip install -e ".[tutorials]"`.

Every feature the engine measures with `feature_set="epicycle"` is described, with its equation, in `docs/research/QC_FEATURE_REFERENCE.md` (plain-language version: `docs/research/QC_FEATURES_BIOLOGIST_GUIDE.md`; table: `docs/research/FEATURE_CATALOG.csv`). `analysis/FEATURES.md` lists the features the analysis notebooks use.

The whole-tissue engine is Step 5 of the workflow. It runs from a notebook or from the command line (`epicycle-scaledseg run --help`), in the same environment as the plugin.

## Hardware for segmentation/01

- **An NVIDIA GPU (CUDA) is required.** The segmentation stage refuses to run with zero GPUs.
- **RAM at least the size of your input image.** The input is read whole before it is tiled; one tile needs roughly 25 GB at 1024 px.
- **Disk about 3-4x the input size** (tiles, flows, probabilities and the stitched outputs).
- Tested on Linux GPU clusters (SLURM) and on one small run on Windows. Whole-tissue scale has only been tested on Linux.

Notebooks `segmentation/02` and `segmentation/03` need no GPU. Notebook 03 reads both label volumes into memory, so plan for a few times the size of one label volume.

## What you need before you start (segmentation/01)

- A ZCYX, CZYX or ZYX TIFF.
- The two model files you trained in EpiCycle, `lateral` and `axial` (no file extension). Step 3 writes them under `artifacts/step3/refinement/models/<name>/`; Step 2 writes them under `artifacts/step2/runs/<run>/<bucket>/round_NN/model/`. Use the same pair Step 4 shows (`artifacts/step4/step4_state.json`). The notebook explains the channel setting.
- The voxel size of your image in micrometres (z, y, x). Without it, every output is in voxel units.

## What you need before you start (segmentation/02 and 03)

Two run folders written by notebook 01, one for a **cell** segmentation and one for a **nuclei** segmentation of the same stack. A run folder is the folder notebook 01 prints as `Output:`, that is `WORK_DIR/OUTPUT_NAME`. Both need `stage4_full_stitch/global_labels_clean.npy`, and the two volumes must have the same shape. Notebook 02 also reads `stage4_full_stitch/label_provenance_clean.csv`.

Making the nuclei run: run notebook 01 a second time with the same input TIFF, a model pair trained on nuclei, `CHANNELS` on the nuclear stain, a different `OUTPUT_NAME`, and post-processing on (`RUN_POSTPROCESS = True`, `PP_DRY_RUN = False`) so that the `*_clean` files exist.

To rescue missed nuclei in notebook 03, the cell run's `tiles/` folder must still be there, and the stack that run was made from must include the nuclear stain as one of its channels (a membrane-only stack is skipped).

## How to open a notebook

```
conda activate epicycle
pip install jupyterlab ipywidgets
jupyter lab
```

Open the notebook from the JupyterLab file browser, fill in the cells marked EDIT, and Run All. Each segmentation notebook finds the repository by looking upwards from the folder it was opened in, so you can start JupyterLab in the repository root or in `tutorials/segmentation/`. (If you installed with `pip install -e .`, the engine is found from anywhere.)
