<!-- copied from the Refine design document, section 11.1, 2026-10-13 -->
### 11.1 The description, verbatim

Visible line (the only sentence a first-time user is guaranteed to read, so it wraps rather than eliding):

> **Fixes the shape of the object you have selected — one object at a time. No other object ever loses a voxel, and nothing is final until you press Accept.**

Body, first paragraph — the one that decides which method to pick:

> Four of these never look at the image — **Smooth**, **Dilate/Erode**, **Fill Holes** and **Flood Fill** only change the shape. **Guided Grow** reads the reconstruction's probability map. **Active contour** and **Settle** read the channel you pick, unless their wall type is built from the shape or the probability map.

Then one line per method, in combo order, as drawn in the mockup — Settle's reads *"moves the outline onto the nearest wall"* (the channel is not always what it reads).

> **Changed 2026-09-24 (B3 audit; orchestrator decision, owner informed).**  The visible line said *"nothing is written until you press Accept"*, but a staged shape IS in the volume until Accept or Discard, so it now says *"nothing is final"*.  The first body paragraph said *"Three of these never look at the image … The other four read a channel"*, which was false for Flood Fill (reads no image) and Guided Grow (reads the probability map); it is the auditor's replacement above.  Code (`REFINE_ABOUT_LINE` / `REFINE_ABOUT_BODY`) and this section move in lockstep — `test_refine_help_parity.py` reads both quotes out of this file.

---

