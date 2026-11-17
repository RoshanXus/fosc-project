# Member 1: baseline simulator and verification

- `toymodel.py`: the core simulator (positions, walls, pairwise distances, loop and vectorized
  spring forces, energies, Euler and Verlet steps, `run`). Other members import it; it writes nothing.
- `verify.py`: run with `python3 verify.py` from this folder (takes about 1 s). Prints all Task 1
  checks and writes to `results/`:
  - `verification_summary.txt` (everything printed), `checks.csv` (each check, value, target, PASS/FAIL)
  - `neighbours_200.csv`, `wall_tests.csv`, `box_check.csv`, `energy_cutoff_N12.csv`
  - `positions_initial_final.png` (initial and final positions with neighbour bonds, N = 12)
- Needs numpy and matplotlib only.
