# FSC Computational Thinking Project: Atomic Toy Model

## Details of members

1. Member 1 Name MSxxxxxxxx
2. Member 2 Name MSxxxxxxxx
3. Member 3 Name MSxxxxxxxx
4. Member 4 Name MSxxxxxxxx
5. Member 5 Name MSxxxxxxxx

## Sequence of videos

1. Video-1 Member1.mp4 (problem, model, baseline simulator, verification)
2. Video-2 Member2.mp4 (Python vs NumPy vs C timing, speedup, memory scaling)
3. Video-3 Member3.mp4 (K, U, E_total, timestep study, float32 vs float64)
4. Video-4 Member4.mp4 (animation, conservative vs dissipative dynamics, relaxation)
5. Video-5 Member5.mp4 (innovation: cell-linked list with Numba, final recommendation)

Talking points for each video are in `memberN_*/video_notes.md`.

## Roles and responsibilities

1. Member 1 - Model setup and verification: wrote the core simulator `toymodel.py` that everyone else imports (positions, velocities, wall reflection, pairwise distances, neighbour mask, loop and vectorized spring forces, Euler and Verlet steps, energies) and the Task 1 verification tests (47 numerical checks). Found the 10.8 energy jump at the cutoff. (Task 1)
2. Member 2 - Pair forces and efficiency: wrote `pairforce.c` (standalone and ctypes versions, -O0 and -O2), timed Python loop / NumPy / C for N = 50 to 4000 with medians and speedups, found the bottleneck, and measured the memory of the (N, N, 2) array and of a full force call. (Tasks 6 and 7)
3. Member 3 - Time integration and energy: timestep study for Euler and Verlet with K, U, E_total and the shifted energy, control runs to measure the integrator order, choice of dt, and the float32 vs float64 study. (Tasks 3 and 5)
4. Member 4 - Animation and dissipation: trajectory animations (MP4 and GIF) used as a diagnostic, viscous drag with a damped Verlet step, energy balance check, steady-state criterion and relaxation time vs gamma. Found the energy gain from the wall clamp. (Tasks 2 and 4)
5. Member 5 - Innovation and final recommendation: cell-linked neighbour list compiled with Numba, checked against the baseline forces and benchmarked up to N = 10^6, and the final engineering recommendation based on everyone's results. (Tasks 8 and 9)

Integration of the final report and README was done together by all members.

## Acknowledgment of AI usage

1. Which AI tools have been used and for what purpose

   Claude (Anthropic) - for code development and debugging, and for help with drafting and proofreading the report text. Places where we used it:
   - Member 1: checking the broadcasting expression for the (N, N, 2) displacement array and why float32 arrays were being turned into float64 under NumPy 2.
   - Member 2: the ctypes wrapper and compile commands for `pairforce.c`, and understanding why `ru_maxrss` showed no memory growth in a child process.
   - Member 3: setting up the log-log fit for the error vs dt plots and debugging the shifted-energy bookkeeping.
   - Member 4: the matplotlib FuncAnimation and ffmpeg writer setup, and checking the algebra of the implicit drag half-kick.
   - Member 5: Numba syntax for the cell-list loops and finding why the compiled functions were slow when called with default arguments.
   - All members: proofreading and shortening the report sections.

   All numbers, tables and figures in the reports come from our own runs of the scripts in this folder.

## Files shared

1. README (this file, `README.md` and `README.pdf`)
2. Video files (Video-1 to Video-5, listed above)
3. Source codes, testing codes, visualization codes and produced movies, one folder per member. Each folder has its own `README.md` with the run order:
   - `member1_baseline_verification/` - `toymodel.py` (shared simulator), `verify.py`
   - `member2_forces_efficiency_memory/` - `pairforce.c`, timing and memory scripts
   - `member3_energy_precision/` - timestep, control-run and precision scripts
   - `member4_animation_dissipation/` - animation, damped stepper, relaxation scripts; movies in `results/` (`animation_default.mp4/.gif`, `animation_dt0.1.mp4/.gif`, `compare_gamma0_vs_gamma0.5.mp4`)
   - `member5_innovation_recommendation/` - `cellforces.py` (Numba cell list), correctness and benchmark scripts, `recommendation.md`
4. Project document with results: `Project_Report.pdf` (all five sections with time snapshots, plots, tables, algorithms and implementation details), and each member's `report.md`
5. Prompts and recorded sessions with AI: `ai_usage/`

## How to run

Needs Python 3 with numpy, matplotlib and numba (`pip install -r requirements.txt`), gcc and ffmpeg. Run the scripts from inside each member folder, in the order given in that folder's `README.md`. Member 1's folder must stay next to the others because every other folder imports `../member1_baseline_verification/toymodel.py`. All outputs go to the folder's `results/`.
