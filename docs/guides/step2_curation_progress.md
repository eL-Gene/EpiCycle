# View and Curation progress

Two blocks in the right-hand column of Curation: the **View** switch, which chooses what you work on, and **Curation progress**, a count of how far you have got.

---

## View

"Which bucket you curate and train. XY → lateral; XZ + YZ → pooled axial."

EpiCycle trains **two** Cellpose models per run:

* **Lateral** is trained on XY slices (looking down on the tissue).
* **Axial** is trained on XZ and YZ slices (side-on views, pooled together).

Step 3 combines the output of both models in 3D, so you will usually curate both. The two-segment switch **Lateral (XY) | Axial (XZ/YZ)** picks which one you work on now. It changes:

* which slices the Slices list shows,
* which counts the Patches list shows (the pictures stay the same),
* which bucket Train will train, and which staged slices the Training section lists.

You cannot switch while a round is training.

---

## Curation progress

"Slices in this run, all patches, by status."

A small table with one row per status (Predicted, Curating, Accepted, Finalized, Trained) and one column per bucket (Lateral (XY) and Axial (XZ/YZ)). The column of the view you are on is bold. The numbers cover every patch in this run, not only the loaded one, and they refresh by themselves every time you accept, finalize, train or discard.

Under the table a note explains the two words that are easy to mix up: "Trained = in a round model already (lateral rounds 0–2). Finalized = waiting for the next Train. Next round: lateral 3, axial 2."

Rounds are numbered as on disk, so the first round of a run is `round_00`. Counts are per run: switching runs shows the new run's numbers.
