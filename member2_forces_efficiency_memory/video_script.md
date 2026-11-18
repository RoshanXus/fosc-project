# Video-4 script: Chandar (Member 2), efficiency and memory (about 3:25)

Open before recording: `pairforce.c`, `report.pdf` from this folder, and from `results/`: `time_vs_N.png`, `speedup.png`, `bottleneck_breakdown.png`, `memory_scaling.png`.

---

### 0:00 to 0:10 · Intro
**Say:**
> "I'm Chandar. I measured how fast the force calculation is and how much memory it needs."

### 0:10 to 0:40 · Three versions of the same force
**Show:** `pairforce.c` for about 5 seconds, then the "Same forces?" table in section 3 of `report.pdf`.

**Say:**
> "We compared three versions of the same spring force: the Python double loop, the NumPy vectorized version, and a C loop that I wrote. The C loop is compiled at -O0 and -O2 and called from Python through ctypes, so all three get exactly the same positions. First, do they agree? C matches the Python loop exactly, a difference of 0.0. NumPy differs by about 10 to the minus 10, because of the 10 to the minus 12 the handout adds to avoid dividing by zero."

### 0:40 to 0:55 · How we measured
**Say:**
> "We kept the handout's density fixed and went from N equal to 50 up to 4000. Every timing is the median of 7 repeats, because the machine was shared and single timings jumped around a lot."

### 0:55 to 1:35 · Wall time and speedup
**Show:** `time_vs_N.png`, then the wall-time table, then the speedup table (or `speedup.png`).

**Say:**
> "On this log-log plot all the curves have a slope of about 2, so the cost goes like N squared: double the particles, four times the time.
> At N equal to 1000: the Python loop takes 60 milliseconds, NumPy 64, C at -O0 1.26, and C at -O2 0.375. So C -O2 is 161 times faster than Python, and the optimiser alone gives about 3 times over -O0, because it keeps values in registers instead of memory.
> And the surprise: NumPy is not faster than the Python loop at N equal to 1000. The ratio is 0.94."

### 1:35 to 2:15 · Where the time goes
**Show:** `bottleneck_breakdown.png`, right panel first, then left.

**Say:**
> "Why? In the right panel, 53 percent of the Python loop's time is actually NumPy, because it first calls the vectorized pairwise-distance function. At our density only 762 of half a million pairs are inside the cutoff, so the slow Python force code hardly ever runs.
> In the left panel, inside NumPy, the two sums over the last axis, which has length only 2, take two thirds of the time. When I stored x and y in separate arrays, with the same maths and the same memory, it got 2.8 to 5.4 times faster. So how the data is laid out matters, not just whether it's vectorized.
> C wins because it rejects far pairs after one comparison, keeps everything in registers, and never builds big temporary arrays."

### 2:15 to 3:10 · Memory and scalability
**Show:** the "Theoretical size of d" table, then `memory_scaling.png`, then the "Real peak" table and the "impractical" section.

**Say:**
> "Now memory. The displacement array is N squared times 2 times the item size: 1.6 gigabytes at N equal to 10,000 in float64. NumPy's nbytes matched this formula exactly every time.
> But one real force call peaks at about 4 times that, 65 times N squared bytes, because the displacements, the distances, the unit vectors, the force sizes and their product all exist at the same time.
> So when does this approach become impractical? At N equal to 5000, one call takes 2.2 seconds, so a 1000-step run takes about 48 minutes. At 10,000 it's 3.2 hours and 6.5 gigabytes. On our machine it becomes impractical at around N equal to 5000 because of time, and above about 16,000 it doesn't fit in memory at all."

### 3:10 to 3:25 · Hand-over
**Say:**
> "So a method that's fast for small N isn't necessarily one that scales. C fixes the memory problem, but it is still N squared. Rakesh will show how we got rid of the N squared."
