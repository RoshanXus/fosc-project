# Task 8: time whole simulations (200 Verlet steps vs N, and the N = 12 handout run) with NumPy vs Numba forces
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import toymodel as tm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cellforces as cf

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

DENSITY = 0.12
NSTEPS = 200
REPEATS = 5


def simulate(N, L, force_fn):
    pos, vel, mass = tm.init_system(N, L, seed=0)
    F = None
    t0 = time.perf_counter()
    for s in range(NSTEPS):
        pos, vel, F = tm.step_verlet(pos, vel, mass, tm.DT, L, F, force_fn)
    return time.perf_counter() - t0, pos


def median_run(N, L, force_fn):
    times = []
    for r in range(REPEATS):
        t, pos = simulate(N, L, force_fn)
        times.append(t)
    return np.median(times), pos


# compile once before timing
cf.make_cell_force(10.0)(np.random.rand(20, 2) * 10)
cf.make_allpairs_force()(np.random.rand(20, 2) * 10)

plan = [(12, True, True), (100, True, True), (500, True, True), (1000, True, True),
        (5000, False, True), (10000, False, True), (20000, False, False),
        (50000, False, False), (100000, False, False), (200000, False, False)]
rows = []
for N, do_np, do_all in plan:
    L = np.sqrt(N / DENSITY)
    t_cell, pos_cell = median_run(N, L, cf.make_cell_force(L))
    t_np = t_all = diff = np.nan
    if do_np:
        t_np, pos_np = median_run(N, L, tm.pair_forces)
        diff = np.max(np.abs(pos_np - pos_cell))
    if do_all:
        t_all, _ = median_run(N, L, cf.make_allpairs_force())
    rows.append([N, L, t_np, t_all, t_cell, t_np / t_cell, t_all / t_cell, NSTEPS / t_cell, diff])
    print("N=%6d  numpy %.4f s  all-pairs %.4f s  cell %.4f s  speedup %.1f  max|pos diff| %.2e" % (
        N, t_np, t_all, t_cell, t_np / t_cell, diff))

rows = np.array(rows)
np.savetxt("results/sim_timing.csv", rows, delimiter=",", fmt="%.6g",
           header="N,L,t_numpy_200steps_s,t_numba_allpairs_200steps_s,t_cell_200steps_s,speedup_numpy_over_cell,speedup_allpairs_over_cell,cell_steps_per_s,max_pos_diff_numpy_vs_cell_after_200_steps",
           comments="")

# (b) the handout run, N = 12, 1000 steps, energies saved every step
pos, vel, mass = tm.init_system(12, 10.0, seed=0)
variants = {
    "numpy force + numpy energy (handout)": dict(force_fn=tm.pair_forces),
    "cell force + numpy energy": dict(force_fn=cf.make_cell_force(10.0)),
    "cell force + cell energy": dict(force_fn=cf.make_cell_force(10.0), pot_fn=cf.make_cell_energy(10.0)),
    "cell force + cell energy, save_every=10": dict(force_fn=cf.make_cell_force(10.0), pot_fn=cf.make_cell_energy(10.0), save_every=10),
}
cf.make_cell_energy(10.0)(pos)
with open("results/run12_timing.csv", "w") as f:
    f.write("variant (N=12 1000 Verlet steps via tm.run),median_time_s,E_final\n")
    for name, kw in variants.items():
        times = []
        for r in range(7):
            t0 = time.perf_counter()
            out = tm.run(pos, vel, mass, tm.DT, tm.NSTEPS, 10.0, **kw)
            times.append(time.perf_counter() - t0)
        f.write("%s,%.4f,%.4f\n" % (name, np.median(times), out["E"][-1]))
print(open("results/run12_timing.csv").read())

# figure
C_NP, C_ALL, C_CELL = "#2a78d6", "#eb6834", "#1baf7a"
fig, ax = plt.subplots(figsize=(6.5, 4.5))
for col, c, name in [(2, C_NP, "numpy (tm.pair_forces)"), (3, C_ALL, "numba all-pairs"), (4, C_CELL, "numba cell list")]:
    m = np.isfinite(rows[:, col])
    ax.loglog(rows[m, 0], rows[m, col], "o-", color=c, lw=2, ms=6, label=name)
ax.set_xlabel("N (density 0.12)")
ax.set_ylabel("median wall time of 200 Verlet steps [s]")
ax.set_title("whole simulation, 200 velocity-Verlet steps")
ax.legend(frameon=False)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("results/sim_timing.png", dpi=130)
print("done")
