# Member 2: pair forces, efficiency (Task 6), memory (Task 7)

Run from inside this folder, in this order:

1. `python3 compile_c.py`: builds `pairforce.c` at -O0 and -O2, both as standalone programs (`pairforce_O0`, `pairforce_O2`) and as ctypes libraries (`libpairforce_O*.so`). Runs the standalone programs for N = 50 to 4000 and writes `results/c_standalone.csv`.
2. `python3 task6_timing.py` (about 1.5 min): checks that the forces agree (`force_check.csv`) and times the Python loop, NumPy, my x/y NumPy variant and C -O0/-O2 (7 interleaved repeats each, `timings.csv`). Also writes the bottleneck breakdowns (`breakdown_*.csv`), the ctypes overhead (`ctypes_overhead.csv`) and `machine_info.txt`.
3. `python3 task6_plots.py`: makes `time_vs_N.png`, `speedup.png` and `bottleneck_breakdown.png`, and writes `speedup.csv`, `loglog_slopes.csv` and `cost_per_pair.csv`.
4. `python3 task7_memory.py` (about 2.5 min, peak about 2 GB): compares the theoretical size of d with `.nbytes` (`memory_theory_vs_nbytes.csv`). Measures the peak memory and time of `tm.pair_forces` in a fresh process per N, using `peak_one_call.py` (`memory_peak_pair_forces.csv`). Writes the extrapolation to `memory_limits.txt` and the plot to `memory_scaling.png`.

`timings_run1.csv` and `task6_log_run1.txt` are a first full run of step 2, kept to show run-to-run variation. Timings were taken on a shared 4-core machine, so medians are reported.
