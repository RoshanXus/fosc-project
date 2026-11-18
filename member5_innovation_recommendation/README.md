# Member 5: innovation (cell-linked list + Numba) and final recommendation

Run from inside this folder (needs numpy, numba, matplotlib; imports ../member1_baseline_verification/toymodel.py):

- `cellforces.py`: the module, no output. Numba all-pairs force, cell-list force and cell-list potential energy, plus `make_cell_force(L)` to pass as `force_fn` to `tm.run`.
- `python3 check_correctness.py` (about 10 s): `results/force_check.csv`, `results/run12_check.csv`, `results/run12_compare.png`.
- `python3 benchmark_forces.py` (about 1 min): `results/force_timing.csv`, `force_timing_loglog.png`, `scaling_fits.txt`, `compile_time.txt`, `memory.csv`, `memory.png`, `memory_summary.txt`, `machine.txt`.
- `python3 benchmark_simulation.py` (about 5 min): `results/sim_timing.csv`, `sim_timing.png`, `run12_timing.csv`.
- `python3 dispatch_overhead.py`: `results/dispatch_overhead.csv` (why default arguments made Numba slow).
- `python3 summarize.py` (run after the benchmarks): `results/summary.txt` (compile pay-back, memory ratio, largest N per time budget).
- `smoothcut.py` + `python3 recommend_tests.py` (about 1.5 min), for Task 9: smooth cutoff and mirror walls vs the handout model, `results/recommend_checks.txt`, `dt_study_models*.csv`, `dt_study_models.png`, `large_N_check.csv`, `cell_precision_cost.csv`.
- `python3 cell_sketch.py`: `results/cell_list_sketch.png`, a picture of the cell list on the 12-particle system (used in the video).
- `results/force_timing_first_try_default_args.csv` is the output of the first version of `benchmark_forces.py`, before the default-argument fix. I kept it as a record; the current script no longer produces it.

`report.md` is the Task 8 report and `recommendation.md` is the Task 9 recommendation. Timings were taken on a shared 4-core machine, so they move by roughly 10-20% between runs. All reported times are medians.
