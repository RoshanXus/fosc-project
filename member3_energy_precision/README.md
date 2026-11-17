# Member 3: energy / timestep (Task 3) and float32 vs float64 (Task 5)

Run from inside this folder (imports `../member1_baseline_verification/toymodel.py`). All outputs go to `results/`.

1. `python3 energy_timestep.py` (about 40 s): K, U, E, E_shift for dt = 0.05 ... 0.0005, Euler and Verlet, seeds 0 to 4 -> `energy_dt_table.csv`, `energy_dt_seeds*.csv`, `energy_timeseries.csv`, `cutoff_jumps_dt0.01_verlet.txt`, `E_total_all_dt_*.png`, `KUE_panels.png`, `pairs_vs_energy_dt0.01.png`
2. `python3 integrator_control.py` (about 20 s, run after step 1, it reads `energy_dt_seeds_summary.csv`): control runs with no cutoff crossings, slope fits -> `control_eps_vs_dt.csv`, `control_slopes.txt`, `control_eps_vs_dt_loglog.png`, `control_energy_dt0.01.png`
3. `python3 error_sources.py` (about 5 s): energy error split by step type, wall clamp measurement -> `error_sources.csv/.txt/.png`
4. `python3 precision_accuracy.py` (about 30 s): float32 vs float64, eps_pos(t), sensitivity runs, statistics -> `precision_eps_pos*.csv/.png`, `precision_statistics.csv`, `precision_accuracy.txt`, `precision_energy.png`, `precision_control_*`
5. `python3 precision_cost.py` (about 3 min, uses up to about 0.6 GB): memory and median timings -> `precision_memory.csv`, `precision_timing.csv`, `precision_speedup.csv`, `precision_cost.png`

`common.py` holds the shared helpers (pair counting, shifted energy, eps_E_max, relaxed cluster).
