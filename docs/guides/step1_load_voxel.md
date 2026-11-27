# Voxel size

The voxel size is the physical size of one pixel of your image, in micrometres
(µm), along each axis: **Z** (the distance between two slices), **Y** and **X**
(the pixel size within a slice). Most microscopes take slices further apart than
the pixel size, so Z is usually larger than Y and X.

EpiCycle uses it in three places:

- **The viewer.** The image is drawn at its true proportions, in µm, so a stack
  with coarse slices does not look squashed.
- **Boxes and resampling.** Box sizes and the even-grid resample step in
  Preprocess are measured with it.
- **Every patch you extract.** The spacing is recorded with the source the
  patch came from, and the panel always shows the recorded value. A spacing
  read from the file is recorded the first time you extract from a new source,
  or as soon as you select a source you have already worked with that has none
  yet. Step 3 and Step 4 start from it for each patch cut from that source, so
  their measurements come out in µm; when the file gave no Z step (Z was
  filled in: the pixel size, or ImageJ's default of 1 µm), they show the size
  in amber, starting with "Z assumed". Patches made from a prepared
  (processed) copy use that copy's own scale instead. A spacing you set for a
  patch in Step 3 or Step 4 takes its place there.

A pixel size over 50 µm read from a file is never used or recorded: it is
almost always a print resolution (dots per inch), not a microscope's. The
boxes keep their default values; type the real ones.

## Where the numbers come from

The line under the three boxes says where the values came from:

| What it says | What it means |
|---|---|
| detected from file | The values were read from the image file's own header (ImageJ or OME metadata). |
| detected from file (Z assumed) | A stack saved by ImageJ or Fiji with no slice spacing: Z is ImageJ's default depth of 1 unit (usually 1 µm), not a measured step; hover over the line for the reason. Type the real Z if your slices are a different distance apart, or click in the Z box and press Enter to confirm it. |
| detected from file (Z assumed = XY) | The header recorded a pixel size but no slice spacing, so Z was set equal to the pixel size. This is rarely right for a confocal stack: check your acquisition settings and type the real Z. If Z really is equal to the pixel size, click in the Z box and press Enter to confirm it: the value becomes yours and the "Z assumed" mark goes away in Steps 3 and 4. |
| detected from file · file now says … µm, ↺ to use it | These values were read from the file earlier and are the ones recorded for this source. The file now records something else (it was re-saved or corrected). Press ↺ to take the file's new values. |
| Z assumed · file now says … µm, ↺ to use it | The same, for a recorded value whose Z the file did not give. When the file's new value has no Z step of its own either, the line names only its pixel size, for example "file now says 0.3 × 0.3 µm (Z assumed)". |
| recorded for this source | The values were read from the file earlier, but the file cannot be read now (it was moved, or it no longer has spacing tags). They are still the ones your patches use. |
| Z assumed · recorded for this source | The same, for a recorded value whose Z the file did not give. |
| read from file: a pixel size over 50 µm, likely print resolution — not recorded; type the real values | The file's header gives a print resolution, not a voxel size. It is not put in the boxes and never recorded; type all three real values (pressing Enter on the boxes as they are keeps nothing). |
| recorded for this source · the file now says a pixel size over 50 µm, likely print resolution | The values recorded for this source are still used; the file now gives a print resolution instead of a voxel size, so ↺ does not offer it. |
| the file's Z step (… µm) is out of range — type the values | The file gives a usable pixel size but a Z step outside 0.001-1000 µm, so nothing was taken from the file. Type all three values. |
| your value is over 50 µm — likely a print resolution; type the real values | A value saved as typed by an earlier version of EpiCycle that looks like a print resolution. It is still yours and still used (Steps 3 and 4 included); type the real values to replace it. A value you type now is never questioned this way. |
| your value · ↺ to restore detected | You typed (or confirmed) these values, now or earlier. They are kept for this source and come back when you select it again. A value an earlier version of EpiCycle saved as typed that equals the file's header is marked as read from the file from then on (the console says "the voxel size saved as typed equals the file's header; now marked as read from the file"); when the file gives no Z step it then shows Z assumed in Steps 3 and 4 — press Enter in the Z box to confirm Z as yours. |
| edited by hand · ↺ to restore detected | You are changing the values right now. They are saved when you leave the box. |
| — unknown (1 µm assumed) | The file records no spacing. The viewer and the boxes work in pixels, and nothing real is recorded with your patches. Type the real values if you know them. |

## Changing it

Type a value in any of the three boxes. It is saved for this source as soon as
you leave the box, and it replaces whatever was there before, including a value
read from the file.

**↺** reads the file again and puts back its values, dropping the ones you
typed — or, when the image was opened from a prepared copy, puts back the
values that copy was made with. When the file cannot be read or no longer records a spacing, ↺ puts
back the values recorded for this source; a source that never had any goes back
to the unknown 1 µm.

Extracting a patch, or selecting a source, never changes a spacing the source
already has. They only record one when the source has none yet, and never a
pixel size over 50 µm read from a file; to change a recorded spacing, type the
new value here, or press ↺ to take the file's.

## Check it once per source

A wrong voxel size does not stop anything from running: every later
measurement is simply off by the same factor (a Z that is 2× too small makes
every cell look half as tall). Compare the numbers with your microscope's
settings the first time you load a file, before you extract.
