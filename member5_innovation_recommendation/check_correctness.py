# Task 8: does the cell list give the same forces and trajectory as toymodel?
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import toymodel as tm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cellforces as cf

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

DENSITY = 0.12   # 12 particles per 10 x 10 box, as in the handout


def box_for(N):
    return np.sqrt(N / DENSITY)


def random_pos(N, L, seed):
    rng = np.random.default_rng(seed)
    return rng.random((N, 2)) * L


# ---- test cases -------------------------------------------------------------
cases = []

pos, vel, mass = tm.init_system(12, 10.0, seed=0)
cases.append(("handout N=12", pos, 10.0, tm.RC))

for N in [100, 1000, 5000]:
    L = box_for(N)
    cases.append(("random N=%d" % N, random_pos(N, L, N), L, tm.RC))

# particles exactly on the walls and corners (what walls_reflect produces)
N = 200
L = box_for(N)
p = random_pos(N, L, 7)
p[0:10, 0] = L
p[10:20, 1] = L
p[20:30, 0] = 0.0
p[30:40, 1] = 0.0
p[40] = [L, L]
p[41] = [0.0, 0.0]
p[42] = [L, 0.0]
p[43] = [0.0, L]
p[44] = [L - 1.0, L]       # close partner of the corner particle
cases.append(("walls N=200", p, L, tm.RC))

# lattice with spacing rc = cell side, so pairs sit at exactly rc on cell edges; plus a duplicate (r = 0)
g = np.arange(0.0, 10.0 + 1e-9, 2.0)
X, Y = np.meshgrid(g, g)
p = np.column_stack([X.ravel(), Y.ravel()])
p = np.vstack([p, [[5.0, 5.0], [5.0, 5.0], [7.3, 3.1]]])
cases.append(("lattice r=rc N=%d" % len(p), p, 10.0, tm.RC))

# cutoff that does not divide L, so the cells are bigger than rc
N = 1000
L = box_for(N)
cases.append(("rc=3.3 N=1000", random_pos(N, L, 3), L, 3.3))

rows = []
for name, p, L, rc in cases:
    F_np = tm.pair_forces(p, rc=rc)
    F_cell = cf.pair_forces_cell(p, L, tm.K, tm.R0, rc)
    F_all = cf.pair_forces_allpairs(p, tm.K, tm.R0, rc)
    d, dist, nb = tm.pairwise_distances(p, rc)
    npairs = int(np.triu(nb, 1).sum())
    if len(p) <= 1000:
        F_loop = tm.pair_forces_loop(p, rc=rc)
        loop_diff = np.max(np.abs(F_cell - F_loop))
    else:
        loop_diff = np.nan
    U_np = tm.potential_energy(p, rc=rc)
    U_cell = cf.potential_energy_cell(p, L, tm.K, tm.R0, rc)
    rows.append([name, len(p), L, rc, npairs, np.max(np.abs(F_np)),
                 np.max(np.abs(F_cell - F_np)), np.max(np.abs(F_all - F_np)), loop_diff,
                 np.max(np.abs(F_cell.sum(axis=0))), abs(U_cell - U_np) / U_np])
    del d, dist, nb

# float32 input: dtype must be kept
N = 1000
L = box_for(N)
p32 = random_pos(N, L, 5).astype(np.float32)
F32 = cf.pair_forces_cell(p32, np.float32(L), tm.K, tm.R0, tm.RC)
F64 = tm.pair_forces(p32.astype(np.float64))
print("float32 in -> dtype out:", F32.dtype, " max diff to float64 numpy:", np.max(np.abs(F32 - F64)))

header = "case,N,L,rc,pairs_within_rc,max_abs_F,maxdiff_cell_vs_numpy,maxdiff_numba_allpairs_vs_numpy,maxdiff_cell_vs_loop,max_abs_sum_F_cell,relerr_U_cell"
with open("results/force_check.csv", "w") as f:
    f.write(header + "\n")
    for r in rows:
        f.write("%s,%d,%.4f,%.2f,%d,%.4f,%.3e,%.3e,%.3e,%.3e,%.3e\n" % tuple(r))
    f.write("float32 N=1000 (dtype %s),1000,%.4f,2.00,,,%.3e,,,,\n" % (F32.dtype, L, np.max(np.abs(F32 - F64))))
print(open("results/force_check.csv").read())

# ---- full 1000-step N = 12 run with each force function ----------------------
pos, vel, mass = tm.init_system(12, 10.0, seed=0)
force_fns = {
    "numpy (tm.pair_forces)": tm.pair_forces,
    "cell list (Numba)": cf.make_cell_force(10.0),
    "all-pairs (Numba)": cf.make_allpairs_force(),
    "python loop (tm.pair_forces_loop)": tm.pair_forces_loop,
}
runs = {name: tm.run(pos, vel, mass, tm.DT, tm.NSTEPS, 10.0, force_fn=fn) for name, fn in force_fns.items()}
ref = runs["numpy (tm.pair_forces)"]

with open("results/run12_check.csv", "w") as f:
    f.write("force_fn,max_pos_diff_final,max_pos_diff_any_step,step_where_diff_first_exceeds_1e-6,max_E_diff,E_final\n")
    for name, r in runs.items():
        dpos = np.max(np.abs(r["pos"] - ref["pos"]), axis=(1, 2))
        big = np.nonzero(dpos > 1e-6)[0]
        first = str(big[0]) if len(big) else "never"
        f.write("%s,%.3e,%.3e,%s,%.3e,%.4f\n" % (name, dpos[-1], dpos.max(), first,
                                                 np.max(np.abs(r["E"] - ref["E"])), r["E"][-1]))
    # the loop divides by the exact distance, numpy by (dist + 1e-12); compare cell list with the loop too
    a, b = runs["cell list (Numba)"], runs["python loop (tm.pair_forces_loop)"]
    dpos = np.max(np.abs(a["pos"] - b["pos"]), axis=(1, 2))
    f.write("cell list vs python loop (not vs numpy),%.3e,%.3e,never,%.3e,%.4f\n" % (
        dpos[-1], dpos.max(), np.max(np.abs(a["E"] - b["E"])), a["E"][-1]))
print(open("results/run12_check.csv").read())

# figure: energy curves on top of each other, and how far trajectories drift apart
colors = {"cell list (Numba)": "#eb6834", "all-pairs (Numba)": "#1baf7a",
          "python loop (tm.pair_forces_loop)": "#6250d6"}
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
t = ref["t"]
ax[0].plot(t, ref["E"], color="#2a78d6", lw=3, label="E, numpy")
ax[0].plot(t, runs["cell list (Numba)"]["E"], color="#eb6834", lw=1.2, ls="--", label="E, cell list")
ax[0].plot(t, ref["K"], color="#2a78d6", lw=3, alpha=0.4, label="K, numpy")
ax[0].plot(t, runs["cell list (Numba)"]["K"], color="#eb6834", lw=1.2, ls=":", label="K, cell list")
ax[0].set_xlabel("t")
ax[0].set_ylabel("energy")
ax[0].set_title("N = 12, 1000 Verlet steps: energies")
ax[0].legend(frameon=False)
widths = {"cell list (Numba)": 4, "all-pairs (Numba)": 2.2, "python loop (tm.pair_forces_loop)": 1}
for name in colors:
    dpos = np.max(np.abs(runs[name]["pos"] - ref["pos"]), axis=(1, 2))
    ax[1].semilogy(t[1:], np.maximum(dpos[1:], 1e-17), color=colors[name], lw=widths[name], label=name + " vs numpy")
dpos = np.max(np.abs(runs["cell list (Numba)"]["pos"] - runs["python loop (tm.pair_forces_loop)"]["pos"]), axis=(1, 2))
ax[1].semilogy(t[1:], np.maximum(dpos[1:], 1e-17), color="#52514e", lw=1.2, ls="--", label="cell list vs python loop")
ax[1].set_xlabel("t")
ax[1].set_ylabel("max |pos difference|")
ax[1].set_title("how far the trajectories drift apart")
ax[1].legend(frameon=False, fontsize=8)
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig("results/run12_compare.png", dpi=130)
print("saved results/run12_compare.png")
