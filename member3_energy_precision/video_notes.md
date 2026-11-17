# Video notes, Member 3 (energy, timestep, precision), about 3 min

- (show `results/KUE_panels.png`, one row) K and U swap energy all the time, so K alone is not conserved. The quantity to check is E = K + U.
- (show `results/pairs_vs_energy_dt0.01.png`) But E jumps by 10.8 every time a pair crosses rc, because U at the cutoff is 0.5*15*1.2^2 = 10.8, not 0. The fitted jump size is 10.7959 per pair. Raw eps_E_max is 4 to 11 for every dt, so it cannot choose a timestep.
- The fix for the bookkeeping is the shifted energy E_shift = E - 10.8 * npairs. It is continuous at rc and is the energy the dynamics actually conserve.
- (show `results/E_total_all_dt_verlet.png`, bottom panel) E_shift stays flat for small dt and runs away at dt = 0.05.
- (show `results/control_eps_vs_dt_loglog.png`) Control run with no cutoff crossings and no walls: Verlet slope 2.00, Euler 1.02, Verlet 23 times better at dt = 0.01. In the real model both drop to slope of about 1. The force jumps 18 -> 0 at rc, and that kills Verlet's second order. A cluster hitting the wall (orange) is also first order, because of the clamp in walls_reflect.
- (show `results/error_sources.png`) Pair crossings carry 83 to 89 % of the error. Walls give about half of the final drift at dt = 0.001.
- (show the seed table from the report) Answer: target under 1 % drift of E_shift over t = 10 for all 5 seeds gives Verlet dt = 0.0005 (worst 0.96 %). Euler fails at the same dt. dt = 0.01 gives 19 %. The limit is the cutoff in the model: with a smooth cutoff Verlet would manage dt = 0.05.
- (show the memory and timing tables) Float32 halves every array, and the peak in pair_forces drops from 65 to 33 MB at N = 1000. Speed: 0.98x at N = 12 (no gain, it is all overhead), 1.2 to 1.8x for pair_forces at N >= 200. All medians, taken on a shared machine.
- (show `results/precision_eps_pos.png`) Float32 vs float64 grows to about 0.5, but so do float64 runs whose starts differ by 1e-7, and even by 1e-12. That is chaos, not a float32 problem.
- Statistics such as mean K and pair count from float32 fall inside the float64 spread. Conclusion: float32 is accurate enough and saves memory, but at N = 12 it gives no speedup, so we keep float64 there.
