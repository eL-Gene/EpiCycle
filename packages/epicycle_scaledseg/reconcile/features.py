"""Per-cell + per-nucleus feature extraction for the cleanup consensus, joined with triage.

Reuses the engine's Wave-1 morphology core VERBATIM — ``core.feature_stage._init_feature_worker``
(per-worker memmap) + ``_extract_one`` (single-label morphology on a bbox crop) — so the feature
columns are identical to a normal ``FeatureStage`` run. We only add the orchestration (find_objects
-> tasks -> pool) and the cell<->nucleus join driven by the cleanup's triage records.

Output is a per-CELL table (one row per surviving cell): the cell's morphology + the cleanup status
+ aggregates of its nucleus/nuclei (count, volume, volume fraction, and — for a single-nucleus cell
— that nucleus's key shape features). A separate per-NUCLEUS table carries full nucleus morphology
tagged with its owning cell. This is the triage worklist: filter ``status`` for the merged/strung
residuals, or sort by a feature outlier (size, sphericity, nucleus/cell ratio) for the rest.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd


def _maybe_tqdm(it, total=None, desc=None, enable=False):
    if not enable:
        return it
    try:
        from tqdm.auto import tqdm
        return tqdm(it, total=total, desc=desc)
    except Exception:
        return it


def extract_label_features(
    labels,
    *,
    spacing: Optional[Tuple[float, float, float]] = None,
    invert_z: bool = False,
    include_surface: bool = True,
    tiles_dir: Optional[str] = None,
    recon_dir: Optional[str] = None,
    channels: Optional[List[Tuple[int, str]]] = None,
    prob: bool = False,
    marker_expectations: Optional[dict] = None,
    adjacency: Optional[bool] = None,
    max_radius: int = 5,
    tissue_depth: bool = False,
    lumen: bool = False,
    territory: bool = False,
    field_downsample: int = 1,
    qc_score: Optional[bool] = None,
    qc_score_method: str = "zscore",
    n_workers: Optional[int] = None,
    progress: bool = False,
    desc: str = "features",
    feature_set: str = "full",
) -> pd.DataFrame:
    """Per-label Wave-1 morphology for every object in ``labels`` (an int-label ``.npy`` path OR
    an in-memory array — an array is staged to a temp ``.npy`` so the memmap pool can read it).

    Mirrors ``FeatureStage._extract_parallel``: workers memmap the volume once and pull only
    ``(label, bbox)`` tasks. Returns a DataFrame with one row per label (``label`` column = id),
    columns identical to a normal feature run. ``spacing`` (z,y,x µm) -> physical units.

    M4 NOTE — entry points produce different column sets BY DESIGN. This function emits the
    per-cell INTRINSIC morphology (+ the global-pass columns when ``adjacency`` / ``tissue_depth``
    / ``lumen`` / ``territory`` / ``qc_score`` are enabled). The :class:`core.feature_stage.FeatureStage`
    DRIVER additionally emits the ``crowding`` family (a cKDTree neighbour-spacing pass over all
    centroids) and the ``provenance`` family (seam / IoU / tile bookkeeping from the provenance CSV),
    which this bare per-volume function does NOT produce. ``tests/test_manifest_completeness.py`` is
    therefore scoped to THIS function's labels-only column set.

    OPTIONAL M2 sampler inputs (all default off -> labels-only output unchanged):
        tiles_dir : raw ZCYX channel tiles dir (read at each cell's bbox).
        recon_dir : per-tile probability dir (``stage2_reconstruction``).
        channels  : ``[(array_index, prefix), ...]`` raw channels to sample
            (e.g. ``[(0, "nuc"), (1, "mem")]``) -> per-channel intensity columns.
        prob      : also sample the per-tile probability (sigmoid -> [0,1]) ->
            ``prob_*`` columns. ⚠ engine prob is a per-tile LOGIT, NOT a global
            consensus (see ``qc_features.compute_probability_features``).
    When ``tiles_dir`` is None and ``prob`` is False the worker runs the existing
    labels-only path byte-identically (the reconcile cleanup driver passes nothing).

    OPTIONAL P1b biology config (single-volume path is a no-op):
        marker_expectations : a BIOLOGY-CONFIG dict ``{channel_prefix:
            "nuclear"|"cytoplasmic"|"membrane"}`` declaring where each marker is
            EXPECTED to localise. The expected-compartment QC flags it drives need
            BOTH the cell mask AND the nucleus mask, so they are not emitted by
            this single-volume per-cell pool (there is no nucleus here to
            define the nuclear / cytoplasmic compartments). It is accepted on this
            signature for API symmetry + so a caller can thread one config object
            through; on the single-volume path it is a no-op (the output is unchanged)
            and a one-line note is logged if it is given with a ``channels`` list.
            ``None`` (default) -> nothing happens.

    OPTIONAL M3 adjacency GLOBAL pass (default off -> output unchanged):
        adjacency  : when True, after the per-cell pool assembles ``df``, run the
            adjacency-graph pass ONCE on the FULL label volume — neighbour /
            topology features (B4 + B5) via :func:`qc_features.compute_neighbor_features`
            then the named composites (B6) via :func:`qc_features.compute_named_composites`.
            Adjacency is INHERENTLY GLOBAL (a per-cell bbox crop cannot see its
            neighbours), so this loads the whole label volume into memory ONCE
            (the in-memory ``labels`` array if given, else ``np.load(path)``) and
            runs the proven graduated-dilation graph on it — exactly like the
            existing global :func:`compute_centroid_spacing` crowding pass. The 21
            neighbour/topology columns + the 4 composites are merged into ``df`` by
            ``label``; ``df.attrs["_adjacency_list"]`` (the edge list) is carried
            through. FUTURE scale optimisation: a chunked/halo adjacency scheme that
            avoids the one-time full-volume load (do NOT reinvent it here).
        max_radius : graduated-dilation search radius in VOXELS (no spacing — the
            gap width / radii are voxel counts). Only used when ``adjacency=True``.
    When ``adjacency`` is False the output is byte-identical to the M2 path (the
    reconcile cleanup driver passes nothing).

    OPTIONAL Wave-C tissue-depth GLOBAL pass (default off -> output unchanged):
        tissue_depth : when True, run :func:`qc_features.compute_tissue_depth`
            ONCE on the FULL label volume after the per-cell pool assembles ``df``.
            A single 3D EDT of the WHOLE-TISSUE mask (``full_labels > 0``) gives
            each cell its depth-from-surface ("in vs out") + the spatial-
            context position covariates + a per-cell OUTWARD normal (the sign basis
            later waves use for the apical/basal frame) — 14 columns. INHERENTLY
            GLOBAL (a per-cell crop can't see the tissue surface), so it shares the
            SAME one-time ``full_labels`` load as the adjacency pass (the volume is
            NOT read twice) — but it is INDEPENDENT of the adjacency graph (a
            whole-tissue field, not graph-derived), so it runs with
            ``adjacency=False`` too. GEOMETRY CAVEAT: a clean apical/basal depth
            proxy only for CURVED/CLOSED tissue (organoid / gland / bud); for a FLAT
            slab the "nearest surface" conflates top/bottom/cut-edges, so read
            ``normalized_tissue_depth`` as a thickness-position and EXCLUDE
            ``at_fov_border`` cells (see :func:`qc_features.compute_tissue_depth`).
    When ``tissue_depth`` is False the output is byte-identical to the adjacency
    path.

    OPTIONAL P1b lumen-cavity GLOBAL pass (default off -> output unchanged):
        lumen : when True, run :func:`qc_features.compute_lumen_features` ONCE on the
            FULL label volume after the per-cell pool assembles ``df``. Detects every
            ENCLOSED background cavity (a lumen — the inner surface of a hollow
            organoid/gland/cyst) via ``binary_fill_holes(foreground) AND NOT
            foreground``, DROPS any cavity component touching the volume border (that
            is the external medium, not a lumen), then gives each cell its
            ``distance_to_lumen`` (min physical EDT to the nearest lumen voxel) +
            ``lumen_facing_fraction`` (share of the cell surface lining a lumen) — 2
            columns. INHERENTLY GLOBAL (an internal-vs-external cavity test needs the
            whole tissue), so it shares the SAME one-time ``full_labels`` load as the
            adjacency / tissue-depth passes (the volume is NOT read twice) — but it is
            INDEPENDENT of both, so it runs with ``adjacency=False`` /
            ``tissue_depth=False`` too. HONEST CAVEAT: many explants / open epithelia
            have NO lumen -> both columns are all-NaN, which is the CORRECT answer (no
            cavity to measure a distance to), NOT a failure.
    When ``lumen`` is False the output is byte-identical to the adjacency path.

    OPTIONAL M7 composite qc_score GLOBAL pass (default off -> output unchanged):
        qc_score : when True, after the table is fully assembled run
            :func:`qc_features.compute_composite_score` ONCE over the WHOLE table
            (it standardizes each feature across the population, so it must run
            LAST — after the per-cell pool AND the adjacency/composite global
            pass). Adds a single ``qc_score`` column (a per-label outlier score:
            higher = more outlier-like). qc_score reads the curated
            :data:`qc_features.PCA_FEATURES` subset, several of which
            (``neighbor_count`` / ``nearest_neighbor_distance``) are produced ONLY
            by the adjacency pass — so qc_score is most meaningful with
            ``adjacency=True``; when ``adjacency=False`` it scores on the
            labels-only subset and a one-line warning is logged.
        qc_score_method : ``"zscore"`` (mean |robust-z| across the features;
            default, interpretable) or ``"pca"`` (L2 distance in PC space).
        LEAKAGE: qc_score is DERIVED from the other feature columns. If a
            downstream label/tier is itself derived from qc_score (or the named
            composites), hold qc_score OUT of a classifier trained on that label
            — including it is circular. See
            :func:`qc_features.compute_composite_score`.
    When ``qc_score`` is False the output is byte-identical to the adjacency path.

    ``feature_set``: ``"full"`` (default) = every engine family, flags
    as given (``None`` == off).  ``"epicycle"`` = the EpiCycle catalogue only: ``adjacency`` and
    ``qc_score`` are forced ON unless passed explicitly ``False``, the engine-only opt-ins
    (``tissue_depth`` / ``lumen`` / ``territory``), the engine-only per-cell families and
    ``compute_reliability_meta`` are SKIPPED (never raised).
    """
    import multiprocessing as mp
    import os
    import tempfile
    from scipy.ndimage import find_objects
    from packages.epicycle_scaledseg.core.feature_stage import (
        _check_feature_set, _init_feature_worker, _extract_one)
    from packages.epicycle_scaledseg.utils.resources import get_allocated_cpus

    feature_set = _check_feature_set(feature_set)
    if feature_set == "epicycle":
        if adjacency is None:
            adjacency = True
        if qc_score is None:
            qc_score = True
        tissue_depth = lumen = territory = False
    adjacency = bool(adjacency)
    qc_score = bool(qc_score)

    # marker_expectations needs the nucleus mask. On this single-volume per-cell pool
    # there is no nucleus, so it is a no-op here; note it once if a caller threaded it
    # through with channels.
    if marker_expectations and channels:
        import logging
        logging.getLogger(__name__).info(
            "marker_expectations given to extract_label_features (single-volume): "
            "no-op here - the expected-compartment flags need the nucleus mask.")

    # M2 initargs threaded to every worker (identical for all cells). Empty/off
    # when no sampler inputs were given -> labels-only path unchanged.
    chan = [(int(i), str(p)) for i, p in channels] if channels else None
    m2_initargs = (
        str(tiles_dir) if tiles_dir else None,
        str(recon_dir) if recon_dir else None,
        chan,
        bool(prob),
        feature_set,
    )

    tmp = None
    if isinstance(labels, (str, os.PathLike)):
        path = str(labels)
    else:                                          # in-memory array -> stage to a temp .npy
        fd, tmp = tempfile.mkstemp(suffix=".npy")
        os.close(fd)
        np.save(tmp, np.ascontiguousarray(labels))
        path = tmp
    full_labels = None   # resident copy — ONLY for the truly-global passes (S2/S3).
    try:
        mm = np.load(path, mmap_mode="r")
        objs = find_objects(mm)
        if tissue_depth or lumen or territory:
            # The truly-global passes (Wave-C tissue-depth field, Wave-G watershed
            # territory, lumen cavity) are INHERENTLY GLOBAL + CANNOT be tiled —
            # they need the WHOLE-tissue label volume RESIDENT. Load it ONCE here,
            # off the memmap, BEFORE ``del mm`` / the tmp-unlink so the (possibly
            # temp-staged) source is still valid. np.array (a COPY, not
            # np.asarray's view) so the materialised volume carries no handle on
            # the memmap/temp .npy — the Windows unlink in the finally below must
            # not be blocked by an open mmap. (S2/S3 will DOWNSAMPLE these; for now
            # they stay resident -- scaling stages S2/S3.)
            #
            # S1: the ADJACENCY graph family is deliberately EXCLUDED from this
            # load — it reads per-cell HALO CROPS off a (re-opened) memmap instead
            # (see the graph block below: ``graph_src`` + ``slices=objs``), so it
            # never needs the whole volume resident. That is the whole point of
            # S1 — peak RSS tracks crop x workers, not volume size.
            full_labels = np.array(mm)
        del mm                                     # release the parent memmap so the temp .npy can
                                                   # be unlinked on Windows (open mmap blocks unlink).
        # INVARIANT: nothing may retain a VIEW of ``mm`` past this point — ``objs``
        # is plain slice tuples and ``full_labels`` is a COPY, so the only handle is
        # gone. A future edit that keeps e.g. ``region = mm[sl]`` alive here would
        # re-break the Windows temp-unlink in the finally below.
        tasks = [
            (i + 1,
             (sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, sl[2].start, sl[2].stop),
             tuple(spacing) if spacing is not None else None, invert_z, include_surface)
            for i, sl in enumerate(objs) if sl is not None
        ]
        if not tasks:
            return pd.DataFrame()

        rows: List[dict] = []
        n_missing = n_failed = 0

        def _collect(res):
            nonlocal n_missing, n_failed
            if res is None:
                return
            if isinstance(res, tuple):
                if res[0] == "MISSING":
                    n_missing += 1
                    return
                if res[0] == "ERROR":
                    n_failed += 1
                    return
            rows.append(res)

        nw = n_workers or max(1, get_allocated_cpus() - 1)
        nw = max(1, min(int(nw), len(tasks)))
        if nw <= 1:
            import packages.epicycle_scaledseg.core.feature_stage as _fs
            _init_feature_worker(path, *m2_initargs)
            try:
                for t in _maybe_tqdm(tasks, total=len(tasks), desc=desc, enable=progress):
                    _collect(_extract_one(t))
            finally:
                # Reset the module-level worker globals (in-process serial path
                # mutates them; leaving M2 config set would leak into a later
                # labels-only call in the same process).
                _fs._WORKER_SRC = None
                _fs._WORKER_TILES_DIR = None
                _fs._WORKER_RECON_DIR = None
                _fs._WORKER_CHANNELS = None
                _fs._WORKER_PROB = False
                _fs._WORKER_FEATURE_SET = "full"
        else:
            with mp.Pool(nw, initializer=_init_feature_worker,
                         initargs=(path, *m2_initargs)) as pool:
                for res in _maybe_tqdm(pool.imap_unordered(_extract_one, tasks, chunksize=16),
                                       total=len(tasks), desc=f"{desc} ({nw}w)", enable=progress):
                    _collect(res)
    finally:
        if tmp is not None:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("label").reset_index(drop=True)
        # Wave B — reliability / confidence meta. A pure TABLE pass over
        # the assembled per-label frame, reading existing columns
        # (major_axis_z_angle / z_span; is_border_cell + prob_mean if present).
        # ALWAYS run (degrades gracefully) so the reconcile cell/nucleus tables
        # carry the same trust filters the FeatureStage emits. Additive: it only
        # appends columns, so the labels-only output is otherwise unchanged.
        if feature_set != "epicycle":
            from packages.epicycle_scaledseg.core.qc_features import compute_reliability_meta
            compute_reliability_meta(df)

    # --- M3 OPTIONAL global adjacency pass ------------------------------------
    # Runs ONCE after the per-cell pool assembles the table (like the FeatureStage
    # cKDTree crowding pass). The graph IS global (every cell sees every neighbour)
    # but is computed from per-cell HALO CROPS off the memmap (S1) — NOT a resident
    # full-volume copy. compute_neighbor_features indexes the precomputed ``objs``
    # bboxes by label and writes its neighbour/topology columns back positionally
    # (merges by label onto the assembled rows); compute_named_composites (B6) then
    # reads the existing columns (degrades gracefully). Default off -> df is the M2
    # output unchanged.
    if adjacency and not df.empty:
        from packages.epicycle_scaledseg.core.qc_features import run_adjacency_graph_passes
        # --- S1 scaling: the graph family reads per-cell HALO CROPS off a memmap
        # instead of a resident full-volume copy (scaling stage S1). Two
        # ingredients, both byte-identical to the prior resident path:
        #   (1) ``slices=objs`` — reuse the bboxes we already computed for the
        #       per-cell task list. ``objs = find_objects(mm)`` ran ONCE inside the
        #       try above and survives ``del mm`` (plain slice tuples, no array
        #       handle); passing it skips each graph function's redundant
        #       full-volume find_objects + int32 copy. ``find_objects(mm)`` ==
        #       ``find_objects(labels_3d.astype(int32))`` for int-label volumes with
        #       ids < 2**31 (always true here — cell counts are ~1e6 << 2e9; larger
        #       ids would ALIAS under the OLD int32 cast, so the injected int64
        #       memmap path is strictly MORE correct), so the contact set is unchanged.
        #   (2) ``graph_src`` — a LAZY source for the per-cell crops:
        #       - a truly-global pass already loaded ``full_labels`` resident → reuse
        #         it (still benefits from (1));
        #       - else a PATH input → re-open the on-disk volume as a memmap so each
        #         crop ``labels_3d[expanded]`` streams only its bbox from disk (peak
        #         RSS tracks crop x workers, NOT volume size);
        #       - else an in-memory array → the caller already holds it resident, so
        #         use a no-copy view (np.asarray) — no extra RAM.
        if full_labels is not None:
            graph_src = full_labels
        elif isinstance(labels, (str, os.PathLike)):
            graph_src = np.load(str(labels), mmap_mode="r")
        else:
            graph_src = np.asarray(labels)
        # `label` round-trips through the worker's per-row Series as a float;
        # the graph functions index find_objects with `label_id - 1`, which
        # requires an int (FeatureStage casts likewise before its passes).
        df["label"] = df["label"].astype(np.int64)
        # The CANONICAL graph sequence (neighbour -> contact geometry -> tissue
        # frame -> graph topology -> axis persistence -> local-comparative ->
        # channel neighbour context -> graph motifs -> named composites), SHARED
        # with FeatureStage so both production paths are byte-identical.
        #
        # S1.5 (Phase B): the three VOLUME passes parallelise CHUNK-WISE when
        # n_workers > 1 AND a FILE PATH is available for the workers' per-process
        # memmap. ``labels`` is a PATH -> pass it; an in-memory array -> None so the
        # graph family falls back to SERIAL (the array is not pickled per worker —
        # ``path`` here is a TEMP .npy that the finally-block unlinks, so it is NOT a
        # durable handle for spawned workers). Byte-identical to serial either way.
        graph_labels_path = str(labels) if isinstance(labels, (str, os.PathLike)) else None
        run_adjacency_graph_passes(
            df, graph_src, slices=objs, spacing=spacing, max_radius=max_radius,
            n_workers=n_workers, labels_path=graph_labels_path, feature_set=feature_set)

    # --- Wave C OPTIONAL tissue-depth ("in/out") global pass -----------------
    # Runs ONCE on the full label volume (reusing ``full_labels``, loaded above for
    # the adjacency/tissue-depth global block — NOT re-read) after the per-cell
    # pool assembles the table. A single 3D EDT of the WHOLE-TISSUE mask gives each
    # cell its depth-from-surface ("in vs out"), the spatial-context position
    # covariates, and a per-cell OUTWARD normal (the sign basis later waves use for
    # the apical/basal frame). INDEPENDENT of the adjacency graph (a whole-tissue
    # field, not graph-derived), so it can run with ``adjacency=False`` too. Default
    # off -> df is the prior output unchanged. GEOMETRY CAVEAT: meaningful for
    # CURVED/CLOSED tissue; for a FLAT slab read normalized_tissue_depth as a
    # thickness-position + exclude at_fov_border cells (see compute_tissue_depth).
    if tissue_depth and full_labels is not None and not df.empty:
        from packages.epicycle_scaledseg.core.qc_features import (
            compute_tissue_depth, compute_directional_contact,
            compute_surface_partition, compute_ma_apical_basal,
            compute_apical_junction, compute_tissue_frame_features,
        )
        # ``label`` round-trips through the worker as a float; compute_tissue_depth
        # indexes find_objects with the int id (the adjacency block above casts it
        # when adjacency=True, but tissue_depth can run standalone — cast here too).
        df["label"] = df["label"].astype(np.int64)
        sp = tuple(spacing) if spacing is not None else None
        # field_downsample > 1 (S2/S3): compute_tissue_depth runs its EDT field on a
        # foreground downsampled by N (an APPROXIMATION; N=1 byte-identical). The
        # SIGNED-frame families below still crop the FULL-RES ``full_labels`` per cell
        # (they read each cell's voxels, not a global field), so they are unchanged.
        compute_tissue_depth(full_labels, df, spacing=sp, downsample=field_downsample)
        # --- P1a — SIGN the local tissue frame -------------------
        # Now that compute_tissue_depth wrote the per-cell OUTWARD normal
        # (``tissue_outward_*``), RE-CALL compute_tissue_frame_features so it ADDS the
        # SIGNED apical/basal frame columns (gated on the outward columns existing).
        # The UNSIGNED frame columns it (re)writes are byte-identical (same df +
        # adjacency inputs as the adjacency-block call above — when adjacency ran; if
        # adjacency was off they were never written, and stay all-NaN with a missing
        # graph, the same default). Passing ``full_labels`` lets the SIGNED MASS-MOMENT
        # members (cell_tilt_polarity / apical_vs_basal_elongation /
        # tangential_asymmetry_magnitude) read each cell's voxels; the axis-only signed
        # members need only the principal axes + the outward normal.
        #
        # C1 FIX — thread ``slices=objs`` (the bbox list already computed at the top
        # of extract_label_features) into the 5 SIGNED families so they SKIP their
        # internal ``find_objects(full_labels.astype(int32))`` — that ``.astype``
        # forces a full resident int32 copy. ``full_labels`` is resident here (loaded
        # above for the field passes), so this is a redundant-scan / consistency win;
        # at scale (the FeatureStage memmap path) the same threading is what avoids
        # the OOM. ``objs`` == ``find_objects(full_labels.astype(int32))`` for int-label
        # ids < 2**31, so the per-cell bboxes are unchanged (byte-identical output).
        compute_tissue_frame_features(
            df, adjacency=df.attrs.get("_adjacency_list"),
            spacing=sp, full_labels=full_labels, slices=objs)
        # --- Wave D — signed-frame families -------------
        # Both consume the per-cell OUTWARD normal compute_tissue_depth just wrote
        # (the ``tissue_outward_*`` columns -> the apical/basal/lateral sign basis),
        # so they run AFTER it, inside this same tissue_depth block. They self-skip
        # (leave their columns absent) when the outward normal is missing, so a
        # tissue_depth pass that degraded (no centroids -> NaN outward) cleanly
        # yields NaN columns rather than crashing. Directional contact
        # reuses the M3/M4 graduated-dilation contact set, so it needs the
        # ``_adjacency_list`` — pass the SAME max_radius as the adjacency pass so
        # the contact set is consistent; it NaNs every cell when adjacency was off
        # (no contacts to direction-classify). Surface partition is mesh-
        # only (no graph) so it runs regardless of adjacency.
        compute_directional_contact(full_labels, df, spacing=sp, max_radius=max_radius,
                                     slices=objs)
        compute_surface_partition(full_labels, df, spacing=sp, slices=objs)
        # --- Wave H — signed-frame REMAINDERS (the feature catalogue ma_apical/basal +
        # apical junction) -----------------------------------------------------
        # Both consume the SAME Wave C outward normal (``tissue_outward_*``) the
        # passes above use, so they run AFTER compute_tissue_depth inside this same
        # tissue_depth block. ma_apical_basal needs only the per-cell major axis +
        # the outward sign (no graph); apical_junction reuses the M3/M4 graduated-
        # dilation contact set, so it needs the ``_adjacency_list`` — pass the SAME
        # max_radius so its contact set is consistent; it NaNs every cell when
        # adjacency was off (no contacts to direction-classify). Both self-skip
        # (leave their columns absent) when the outward normal is missing.
        compute_ma_apical_basal(full_labels, df, spacing=sp, slices=objs)
        compute_apical_junction(full_labels, df, spacing=sp, max_radius=max_radius,
                                slices=objs)

    # --- P1b OPTIONAL lumen-cavity global pass --------------------------------
    # Runs ONCE on the full label volume (reusing ``full_labels``, loaded above for
    # the adjacency/tissue-depth/lumen/territory global block — NOT re-read) after the
    # per-cell pool assembles the table. Detects the ENCLOSED background cavities (the
    # lumen — the inner surface of a hollow organoid/gland/cyst) and gives each cell
    # its distance to the nearest lumen + the share of its surface lining one.
    # INHERENTLY GLOBAL (the internal-vs-external cavity test needs the whole tissue)
    # + INDEPENDENT of the adjacency graph AND the tissue-depth field, so it runs with
    # ``adjacency=False`` / ``tissue_depth=False`` too. Default off -> df unchanged.
    # HONEST: a tissue with no enclosed cavity -> both columns all-NaN (the correct
    # answer, not a failure).
    if lumen and full_labels is not None and not df.empty:
        from packages.epicycle_scaledseg.core.qc_features import compute_lumen_features
        # ``label`` round-trips through the worker as a float; compute_lumen_features
        # indexes find_objects with the int id (cast here so this can run standalone,
        # like the tissue_depth / territory blocks above).
        df["label"] = df["label"].astype(np.int64)
        sp = tuple(spacing) if spacing is not None else None
        compute_lumen_features(full_labels, df, spacing=sp, downsample=field_downsample)

    # --- Wave G OPTIONAL watershed-territory ("crowding") global pass ---------
    # Runs ONCE on the full label volume (reusing ``full_labels``, loaded above for
    # the adjacency/tissue-depth/territory global block — NOT re-read) after the
    # per-cell pool assembles the table. A single background-EDT seeded watershed
    # floods each cell into the surrounding space up to the inter-cell midline
    # (capped at a max distance), giving each cell a Voronoi-ish TERRITORY whose
    # volume relative to the cell = the local breathing-room / crowding signal, plus
    # the territory's anisotropy + solidity. INHERENTLY GLOBAL (a per-cell crop can't
    # see where neighbouring territories meet) + INDEPENDENT of the adjacency graph
    # AND the tissue-depth field (a fresh watershed), so it runs with
    # ``adjacency=False`` / ``tissue_depth=False`` too. Default off -> df is the prior
    # output unchanged.
    if territory and full_labels is not None and not df.empty:
        from packages.epicycle_scaledseg.core.qc_features import compute_watershed_territory
        # ``label`` round-trips through the worker as a float; compute_watershed_
        # territory indexes find_objects with the int id (cast here so this can run
        # standalone, like the tissue_depth block above).
        df["label"] = df["label"].astype(np.int64)
        sp = tuple(spacing) if spacing is not None else None
        compute_watershed_territory(full_labels, df, spacing=sp, downsample=field_downsample)

    # --- M7 OPTIONAL composite qc_score global pass --------------------------
    # Runs DEAD LAST (after the per-cell pool AND the adjacency/composite global
    # pass) because it standardizes each feature ACROSS the population, so the
    # whole assembled table must exist first. Adds a single `qc_score` column.
    # Default off -> df is the prior output unchanged. qc_score reads the curated
    # PCA_FEATURES subset; several of those (neighbor_count /
    # nearest_neighbor_distance) come ONLY from the adjacency pass, so warn (don't
    # block) when adjacency was off — qc_score still scores on the labels-only
    # subset. LEAKAGE: qc_score is derived from the other columns; hold it out of
    # a classifier trained on a label that itself derives from qc_score.
    if qc_score and not df.empty:
        from packages.epicycle_scaledseg.core.qc_features import compute_composite_score, PCA_FEATURES
        if not adjacency:
            import logging
            present = [c for c in PCA_FEATURES if c in df.columns]
            logging.getLogger(__name__).warning(
                "qc_score requested with adjacency=False: scoring on the "
                "labels-only subset (%d/%d curated features present; the "
                "neighbour columns require adjacency=True for the full score).",
                len(present), len(PCA_FEATURES),
            )
        compute_composite_score(df, method=qc_score_method)
    return df


def assemble_cell_table(
    triage: List[dict],
    cell_feats: pd.DataFrame,
    nuc_feats: pd.DataFrame,
) -> pd.DataFrame:
    """Join cell morphology + cleanup status + nucleus aggregates into the per-cell triage table.

    ``triage`` = ``cleanup_volume(..., return_triage=True)[3]``. Adds, per cell: ``status``,
    ``n_nuclei``, ``nucleus_ids``, ``nucleus_voxels`` (summed ``area`` of its nuclei),
    ``nucleus_cell_frac`` (nucleus voxels / cell voxels), and — for a single-nucleus cell — that
    nucleus's ``nuc_sphericity`` / ``nuc_aspect_ratio`` / ``nuc_area``. Multi-nucleus (merged)
    cells get the summed volume + ``NaN`` single-nucleus shape (ambiguous by definition).
    """
    tri = pd.DataFrame(triage) if triage else pd.DataFrame(
        columns=["cell_id", "status", "n_nuclei", "nucleus_ids", "carved_nucleus_ids", "cell_voxels"])
    nf = nuc_feats.set_index("label") if not nuc_feats.empty else nuc_feats

    def _nuc_agg(ids):
        if not ids or nf.empty:
            return (0.0, np.nan, np.nan, np.nan, 0)
        present = [i for i in ids if i in nf.index]
        if not present:
            return (0.0, np.nan, np.nan, np.nan, 0)
        vol = float(nf.loc[present, "area"].sum())
        if len(present) == 1:
            r = nf.loc[present[0]]
            return (vol, float(r.get("sphericity", np.nan)),
                    float(r.get("aspect_ratio", np.nan)), float(r.get("area", np.nan)), 1)
        return (vol, np.nan, np.nan, np.nan, len(present))   # merged: single-nucleus shape undefined

    aggs = tri["nucleus_ids"].apply(_nuc_agg) if not tri.empty else []
    tri["nucleus_voxels"] = [a[0] for a in aggs]
    tri["nuc_sphericity"] = [a[1] for a in aggs]
    tri["nuc_aspect_ratio"] = [a[2] for a in aggs]
    tri["nuc_area"] = [a[3] for a in aggs]
    tri["n_nuclei_with_feats"] = [a[4] for a in aggs]
    tri["nucleus_ids"] = tri["nucleus_ids"].apply(lambda v: ";".join(map(str, v)))  # CSV-safe
    if "carved_nucleus_ids" in tri.columns:                                          # same CSV-safe form
        tri["carved_nucleus_ids"] = tri["carved_nucleus_ids"].apply(
            lambda v: ";".join(map(str, v)) if isinstance(v, (list, tuple)) else "")

    # OUTER join: a cell that survived cleanup but whose feature extraction MISSING/ERROR'd must NOT
    # vanish from the worklist (those degenerate cells are exactly the ones to curate) — keep its
    # status, leave the morphology columns NaN.
    out = cell_feats.merge(tri, left_on="label", right_on="cell_id", how="outer")
    if "label" in out and "cell_id" in out:
        out["label"] = out["label"].fillna(out["cell_id"])     # canonical id for feature-less cells
    out["features_extracted"] = out["area"].notna() if "area" in out else False
    if "area" in out:
        denom = out["area"].replace(0, np.nan)
        out["nucleus_cell_frac"] = out["nucleus_voxels"] / denom
    # cells with no triage row (cell_feats without a survivor row — shouldn't happen) -> explicit
    out["status"] = out["status"].fillna("unknown")
    out["n_nuclei"] = out["n_nuclei"].fillna(0).astype(int)
    out["n_nuclei_with_feats"] = out["n_nuclei_with_feats"].fillna(0).astype(int)
    # surface (don't hide) the case where a counted nucleus is missing from nuc_feats: then
    # nucleus_voxels / nucleus_cell_frac UNDERCOUNT. consistent == every counted nucleus measured.
    out["nucleus_feats_complete"] = out["n_nuclei_with_feats"] == out["n_nuclei"]
    return out


def assemble_nucleus_table(nuc_feats: pd.DataFrame, triage: List[dict]) -> pd.DataFrame:
    """Per-nucleus morphology tagged with its owning ``cell_id`` (from the triage records)."""
    nuc_to_cell = {nid: rec["cell_id"] for rec in triage for nid in rec.get("nucleus_ids", [])}
    out = nuc_feats.copy()
    if not out.empty:
        out["cell_id"] = out["label"].map(nuc_to_cell).astype("Int64")
    return out


__all__ = ["extract_label_features", "assemble_cell_table", "assemble_nucleus_table"]
