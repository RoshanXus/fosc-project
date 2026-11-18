# Video-4 script: Chandar (Member 2), efficiency and memory (about 2:00)

Open before recording: `report.pdf` from this folder, and from `results/`: `time_vs_N.png`, `bottleneck_breakdown.png`, `memory_scaling.png`.

---

### 0:00 to 0:20 · What we compared
**Say:**

> "I'm Chandar. I compared three versions of the same force: the Python loop, NumPy, and a C loop at -O0 and -O2, all on the same positions. C matches the Python loop exactly. Each timing is the median of 7 runs."

### 0:20 to 0:55 · Wall time and speedup
**Show:** `time_vs_N.png`, then the wall-time table in section 3 of `report.pdf`.

**Say:**

> "All curves have slope about 2, so the cost grows like N squared. At N equal to 1000, Python takes 60 milliseconds, NumPy 64, and C -O2 0.375. So C is 161 times faster than Python. Surprisingly, NumPy is not faster than the loop here: 0.94 times."

### 0:55 to 1:20 · Bottleneck
**Show:** `bottleneck_breakdown.png`.

**Say:**

> "Half of the Python loop's time is actually NumPy computing distances, and only 762 of half a million pairs interact, so the slow Python code rarely runs. Inside NumPy, sums over the length-2 axis take two thirds of the time. Splitting x and y into separate arrays made it up to 5.4 times faster."

### 1:20 to 2:00 · Memory
**Show:** `memory_scaling.png`.

**Say:**

> "The N by N by 2 array needs N squared times 2 times the item size, which is 1.6 gigabytes at N equal to 10,000. But a real force call peaks at 4 times that, because of temporary arrays. At N equal to 5000 a 1000-step run takes 48 minutes, so the vectorized method becomes impractical around there. Fast at small N doesn't mean it scales. Rakesh shows our fix."
