# Member 4: animation (Task 2) and dissipation (Task 4)

Run from inside this folder (each script writes to `results/`, needs ../member1_baseline_verification/toymodel.py):

- `python3 animate_default.py` (~1.5 min): default run animation (`animation_default.mp4/.gif`), dt = 0.1 animation, `snapshots_default.png`, diagnostics (`diagnostics_default.png`, `diagnostics_dt.csv/png`, `default_summary.txt`, `two_particle_capture.csv`)
- `python3 dissipation.py` (~1.5 min): runs gamma = 0 to 20 for t = 600, `relaxation_table.csv`, `energy_balance.csv`, `energy_balance_dt.csv`, time series per gamma, all Task 4 figures
- `python3 compare_animation.py`: `compare_gamma0_vs_gamma0.5.mp4`
- `python3 stepper_check.py`: `stepper_check.csv` (damped Verlet vs exact damped oscillator)
- `python3 wall_check.py`: `wall_handling_check.csv`, `wall_pair_test.csv` (clamp vs mirror walls)
- `damped.py` is the shared helper module (damped velocity Verlet, shifted energy, clusters); it is imported, not run.
