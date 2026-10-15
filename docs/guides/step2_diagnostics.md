# Diagnostics

The last section of Step 2. It changes nothing; it only tells you what the computer and the training run are doing.

---

## System

"What this computer can train on."

One line, `GPU:` followed by the name of your graphics card (for example NVIDIA GeForce RTX 4060 Laptop GPU), or a note that no GPU was found. Training on a GPU is many times faster than on the processor, so if this line says there is none, Train will use the CPU and take far longer. This is the only place in Step 2 where the GPU is named.

## Log output

"Training events, checkpoint saves and pipeline messages."

A running log. Look here when something did not behave as expected:

* Each round's start, progress and end, with the loss values.
* A line when a model is saved (a checkpoint is a saved copy of the model's state).
* Warnings and errors, including a `[TRAIN FAILED]` line naming the view that failed, and notes such as a finalized slice that was left out because its image or mask was missing.

The log is also the first thing to copy when you ask for help.
