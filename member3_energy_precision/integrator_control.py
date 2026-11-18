# Task 3 controls: a cluster that never crosses rc (A: integrator error only, B: plus a wall hit)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

DTS = [0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
T_TOTAL = 10.0

xA, vA, mass, fres = relaxed_cluster()
xB = xA.copy()
xB[:, 0] -= xA[:, 0].mean() - 1.5        # cluster centre at x = 1.5
vB = vA.copy()
vB[:, 0] -= 1.0                          # whole cluster moves towards the left wall
_, dA, _ = tm.pairwise_distances(xA, tm.RC)
iu = np.triu_indices(len(xA), 1)
info = [f"relaxed cluster: max |F| after relaxation = {fres:.2e}",
        f"pair distances in the cluster: min {dA[iu].min():.3f}, max {dA[iu].max():.3f} (rc = {tm.RC})"]

rows = []
for name, x0, v0 in [("A_cluster_no_wall", xA, vA), ("B_cluster_wall", xB, vB)]:
    for meth in ["euler", "verlet"]:
        for dt in DTS:
            nsteps = int(round(T_TOTAL / dt))
            r = tm.run(x0, v0, mass, dt, nsteps, tm.L, method=meth)
            npairs = pair_counts(r["pos"])
            on_wall = np.any((r["pos"] == 0.0) | (r["pos"] == tm.L), axis=(1, 2))
            rmin = min_pair_dist(r["pos"]).min()
            rows.append([name, meth, dt, nsteps, r["E"][0], eps_max(r["E"]), npairs.min(), npairs.max(),
                         np.count_nonzero(on_wall), rmin])
            print(f"{name:18s} {meth:6s} dt={dt:<7} eps_E={eps_max(r['E']):.3e}  pairs {npairs.min()}-{npairs.max()}"
                  f"  wall frames {np.count_nonzero(on_wall)}  rmin {rmin:.3f}")
np.savetxt(os.path.join(RES, "control_eps_vs_dt.csv"), np.array(rows, dtype=object), fmt="%s",
           delimiter=",", header="case,method,dt,nsteps,E0,eps_E_max,npairs_min,npairs_max,wall_contact_frames,rmin",
           comments="")

# default model (seed 0..4, shifted energy), from energy_timestep.py
summ = np.genfromtxt(os.path.join(RES, "energy_dt_seeds_summary.csv"), delimiter=",", names=True, dtype=None,
                     encoding=None)

R = np.array(rows, dtype=object)
fig, ax = plt.subplots(figsize=(7.5, 5.5))
slopes = []
styles = {"euler": "o", "verlet": "s"}
cols = {"A_cluster_no_wall": "tab:blue", "B_cluster_wall": "tab:orange", "default_Eshift": "tab:red"}
for case in ["A_cluster_no_wall", "B_cluster_wall", "default_Eshift"]:
    for meth in ["euler", "verlet"]:
        if case == "default_Eshift":
            sel = summ["method"] == meth
            dts, eps = summ["dt"][sel].astype(float), summ["median_eps_Eshift"][sel].astype(float)
            label = f"default start, E_shift (median of 5 seeds), {meth}"
        else:
            sel = (R[:, 0] == case) & (R[:, 1] == meth)
            dts, eps = R[sel, 2].astype(float), R[sel, 5].astype(float)
            label = f"{case}, {meth}"
        # fit on dt <= 0.02 (the largest dt values are outside the asymptotic range)
        m = dts <= 0.02
        p = np.polyfit(np.log(dts[m]), np.log(eps[m]), 1)
        slopes.append([case, meth, p[0], np.exp(p[1])])
        ax.loglog(dts, eps, marker=styles[meth], color=cols[case], ls="-" if meth == "verlet" else "--",
                  label=f"{label}: slope {p[0]:.2f}")
ax.set_xlabel("dt")
ax.set_ylabel("eps_E_max over t = 10")
ax.set_title("Energy error vs timestep (fit on dt <= 0.02)")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(RES, "control_eps_vs_dt_loglog.png"), dpi=130)
plt.close(fig)

with open(os.path.join(RES, "control_slopes.txt"), "w") as f:
    f.write("\n".join(info) + "\n")
    f.write("fitted eps = C * dt^p on dt <= 0.02\n")
    for case, meth, p, C in slopes:
        f.write(f"{case:20s} {meth:6s} p = {p:.3f}   C = {C:.3g}\n")
print(open(os.path.join(RES, "control_slopes.txt")).read())

# E(t) of control A at dt = 0.01 for both methods (shows the bounded oscillation)
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
for j, (x0, v0, title) in enumerate([(xA, vA, "control A: cluster, no wall"), (xB, vB, "control B: cluster hits wall")]):
    for meth in ["euler", "verlet"]:
        r = tm.run(x0, v0, mass, 0.01, 1000, tm.L, method=meth)
        ax[j].plot(r["t"], (r["E"] - r["E"][0]) / abs(r["E"][0]), lw=1, label=meth)
    ax[j].set_title(f"{title}, dt = 0.01")
    ax[j].set_xlabel("t")
    ax[j].set_ylabel("(E(t) - E(0)) / |E(0)|")
    ax[j].grid(alpha=0.3)
    ax[j].legend()
fig.tight_layout()
fig.savefig(os.path.join(RES, "control_energy_dt0.01.png"), dpi=130)
plt.close(fig)
