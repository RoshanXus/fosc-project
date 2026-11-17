# Video notes: Member 5, innovation segment (about 2.5 min)

- (Show the handout `pair_forces` code line `d = pos[:, None, :] - pos[None, :, :]`.) The baseline builds an N by N by 2 array. At our density, each particle has only about 1.5 neighbours within rc = 2, so almost all of that work is wasted.
- (Simple sketch of a grid.) Idea: cut the box into cells of side at least rc. A particle only checks its own cell and the 8 around it, which is about 4 candidates whatever N is. Two details: the walls are hard, so edge cells have fewer neighbours, and particles sitting exactly on x = L have to be clamped into the last cell.
- We compiled it with Numba, and also compiled a plain all-pairs loop, so we can tell the gain from compiling apart from the gain from the algorithm.
- (Show `results/run12_compare.png`.) Correct first: the forces agree with the loop version to 3e-14. Against numpy they differ by 1e-10, which comes from the 1e-12 in the handout's unit vector. In the full N = 12 run the energy curves overlap and E_final = 303.9046 in all runs.
- (Show `results/force_timing_loglog.png`.) Time per force call vs N: slopes 2.18 for numpy, 2.00 for compiled all-pairs, 1.10 for the cell list. Speedup over numpy is 21x at N = 12, 853x at N = 1000 and 4171x at N = 5000. The cell list does one million particles in 0.19 s.
- (Show the table from the report, section 5.) Whole 200-step run: 513x faster at N = 1000 (12.4 s down to 24 ms).
- (Show `results/memory.png`.) Memory: numpy needs 65 N² bytes, 1.6 GB at N = 5000 and about 6.5 GB at N = 10000. The cell list needs 41 bytes per particle, 204 kB at N = 5000.
- Where it does not help: compiling costs 0.9 s. For the handout run (N = 12) the whole simulation takes 0.057 s with numpy, so Numba does not pay off there; it pays back after about 15 steps at N = 1000. Below N ≈ 40 the cell list is even slower than the compiled all-pairs loop.
- Also a mistake we made: with default arguments, Numba took 31 µs per call instead of 0.6 µs, so our first benchmark said the cell list was slower than numpy at N = 12.
- Answer: it changes the scaling from N² to about N in time and memory, and is worth it from about N = 40 to 100 upwards, not for N = 12.

# Final recommendation segment (about 1 min)

- (Show the recommendation table in `recommendation.md`.) For the handout system (N = 12) we recommend: float64, velocity Verlet, dt = 0.0005, the NumPy force, rc = 2 as given, and energies always reported as E_shift.
- Why dt = 0.0005: it is the largest dt that keeps E_shift drift under 1% for all 5 seeds (0.96%). The handout dt = 0.01 drifts by up to 32%. A run still takes only 1.4 s.
- Why Verlet: it is second order when the forces are smooth (slope 2.00 against 1.02 for Euler), and in our model it allows twice Euler's dt at the same cost.
- Why float64: float32 gives no speedup at N = 12 (0.98x) and only 0.99 to 1.12x with the cell list at large N, so there is nothing to gain.
- For large N: the Numba cell list (853x faster at N = 1000, 41 bytes per particle) and mirror walls.
- (Show `results/dt_study_models.png`.) We also tested a smooth cutoff with mirror walls. It allows a 4x larger dt at N = 12 and dt = 0.01 at N = 10000 (0.30% drift), but it changes the model, so it is an option, not the default.
- Drag: gamma = 0 for dynamics, gamma = 5 for the fastest relaxation (t_relax 6.3), gamma = 0.1 to reach the lowest-energy cluster.
- One-line answer: we changed the force algorithm and the timestep, both because of measured problems, and both helped, except that the innovation does not pay off for the 12-particle handout system.
