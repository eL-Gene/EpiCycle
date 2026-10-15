# What the measurements mean

**Which measurements the notebooks use.** The shape notebook (01) uses the 31 standard shape measurements that your table has: `area`, `area_filled`, `equivalent_diameter`, `extent`, `euler_number`, `bbox_volume`, `fill_ratio`, `sphericity`, `aspect_ratio`, `z_span`, `z_elongation`, `xy_elongation`, `ellipsoid_solidity`, `inertia_elongation`, `inertia_flatness`, `axis_major_length`, `axis_minor_length`, `major_axis_z_angle`, `surface_area`, `true_sphericity`, `solidity`, `skeleton_branches`, `skeleton_nodes`, `skeleton_length`, `apical_area`, `basal_area`, `apical_basal_area_ratio`, `max_cross_section_z_relative`, `cross_section_taper`, `cross_section_bulge`, `roughness`. A measurement missing from your table is skipped (and one that is empty for more than 10% of the cells is dropped), so a table without, for example, the apical and basal measurements still runs. The optional size correction treats these as size measurements: `area`, `area_filled`, `equivalent_diameter`, `bbox_volume`, `surface_area`, `axis_major_length`, `axis_minor_length`. The neighbourhood notebook (02) does not use measurements directly: it averages the principal components (`PC1`, `PC2`...) of the shape notebook over each cell's neighbours, so it uses the same 31 measurements indirectly. The figure and comparison notebooks (03 to 05) colour, plot and compare any of these measurements, and the comparison notebook also compares `PC1` to `PC3`.

**Units.** Lengths and volumes are in voxels unless a voxel size was given when the table was made, in which case they are in the units of that voxel size; ratios, fractions and angles have no length unit.

**Orientation.** `apical_area`, `basal_area`, `apical_basal_area_ratio`, `max_cross_section_z_relative`, `cross_section_taper`, `cross_section_bulge` and the depth column `centroid-0` depend on which side of the tissue is "up". Every stack in one table must use the same orientation.

**Where to read more.**

- [A plain-language guide to every measurement](../../docs/research/QC_FEATURES_BIOLOGIST_GUIDE.md)
- [The equations](../../docs/research/QC_FEATURE_REFERENCE.md)
- [The full table of measurements, with units and definitions](../../docs/research/FEATURE_CATALOG.csv)
