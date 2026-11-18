# Video-5 script: Rakesh (Member 5), innovation and final recommendation (about 2:00)

Open before recording: `results/cell_list_sketch.png`, `report.pdf` and `recommendation.pdf` from this folder, and `results/force_timing_loglog.png`.

---

### 0:00 to 0:30 · The idea
**Show:** `results/cell_list_sketch.png`.

**Say:**

> "I'm Rakesh. The baseline checks every pair, but each particle has only about 1.5 neighbours. So we cut the box into cells at least rc wide. Each particle only checks its own cell and the 8 around it. Here, particle 0 checks 7 of the 11 others instead of all of them, and 3 are inside the cutoff. With more particles, the number checked stays the same, so the cost grows like N instead of N squared. We compiled it with Numba."

### 0:30 to 1:05 · Correct and fast
**Show:** `force_timing_loglog.png`, then the timing table in `report.pdf`.

**Say:**

> "The forces agree with the loop version to 10 to the minus 14. The fitted slopes are 2.18 for NumPy and 1.10 for the cell list. It is 853 times faster at N equal to 1000, and it handles a million particles in 0.19 seconds, using 41 bytes per particle."

### 1:05 to 1:15 · Where it doesn't help
**Say:**

> "But compiling takes 0.9 seconds, and the 12-particle run takes only 0.06 seconds with NumPy, so for 12 particles it isn't worth it."

### 1:15 to 1:50 · Final recommendation
**Show:** the recommendation table in `recommendation.pdf`.

**Say:**

> "For the 12-particle system we recommend float64, because float32 gives no speedup; velocity Verlet with dt 0.0005, the largest step under 1 percent energy drift; the NumPy force; rc equal to 2, reporting the shifted energy; and gamma 5 for the fastest relaxation. For large N, the cell list."

### 1:50 to 2:00 · Closing
**Say:**

> "We changed the force algorithm and the timestep because of measured problems, and both helped. Thank you."
