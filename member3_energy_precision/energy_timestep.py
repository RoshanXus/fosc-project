# Task 3: K, U, E vs dt for Euler and Verlet, same seed-0 start, t = 10 for every dt
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

T_TOTAL = 10.0
DTS = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
METHODS = ["euler", "verlet"]
COLORS = plt.cm.viridis(np.linspace(0, 0.9, len(DTS)))

pos0, vel0, mass = tm.init_system()
runs = {}
rows = []
for meth in METHODS:
    for dt in DTS:
        nsteps = int(round(T_TOTAL / dt))
        t0 = time.perf_counter()
        r = tm.run(pos0, vel0, mass, dt, nsteps, tm.L, method=meth)
        wall = time.perf_counter() - t0
        Es, npairs = shifted_energy(r)
        r["Es"], r["npairs"] = Es, npairs
        runs[(meth, dt)] = r
        n_changes = np.count_nonzero(np.diff(npairs))
        rows.append([meth, dt, nsteps, r["E"][0], r["E"][-1], eps_max(r["E"]), Es[0], Es[-1],
                     eps_max(Es), npairs[0], npairs[-1], npairs.max(), n_changes, r["K"][-1], wall])
        print(f"{meth:6s} dt={dt:<7} eps_E={eps_max(r['E']):8.3f}  eps_Eshift={eps_max(Es):.4f}  "
              f"pairs {npairs[0]}->{npairs[-1]}  K_end={r['K'][-1]:.1f}  ({wall:.2f} s)")

hdr = ("method,dt,nsteps,E0,E_end,eps_E_max,Eshift0,Eshift_end,eps_Eshift_max,"
       "npairs0,npairs_end,npairs_max,pair_count_changes,K_end,wall_s")
np.savetxt(os.path.join(RES, "energy_dt_table.csv"), np.array(rows, dtype=object),
           fmt="%s", delimiter=",", header=hdr, comments="")

# time series (every step for dt >= 0.01, thinned to ~1000 points otherwise)
with open(os.path.join(RES, "energy_timeseries.csv"), "w") as f:
    f.write("method,dt,t,K,U,E,npairs,E_shift\n")
    for (meth, dt), r in runs.items():
        stride = max(1, (len(r["t"]) - 1) // 1000)
        for i in range(0, len(r["t"]), stride):
            f.write(f"{meth},{dt},{r['t'][i]:.5f},{r['K'][i]:.6f},{r['U'][i]:.6f},{r['E'][i]:.6f},"
                    f"{r['npairs'][i]},{r['Es'][i]:.6f}\n")

# E_total(t) for all dt, one figure per method (raw E on top, shifted E below)
for meth in METHODS:
    fig, ax = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
    for c, dt in zip(COLORS, DTS):
        r = runs[(meth, dt)]
        ax[0].plot(r["t"], r["E"], color=c, lw=1, label=f"dt={dt}")
        ax[1].plot(r["t"], r["Es"], color=c, lw=1, label=f"dt={dt}")
    ax[0].set_ylabel("E_total = K + U")
    ax[0].set_title(f"{meth}: total energy, same start, t = 10")
    ax[1].set_ylabel("E_shift = E - 10.8 * npairs")
    ax[1].set_xlabel("t")
    ax[0].legend(fontsize=8, ncol=2)
    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RES, f"E_total_all_dt_{meth}.png"), dpi=130)
    plt.close(fig)

# K, U, E panels: one row per dt, columns = Euler / Verlet
fig, ax = plt.subplots(len(DTS), 2, figsize=(11, 2.1 * len(DTS)), sharex=True)
for j, meth in enumerate(METHODS):
    for i, dt in enumerate(DTS):
        r = runs[(meth, dt)]
        a = ax[i, j]
        a.plot(r["t"], r["K"], lw=1, label="K")
        a.plot(r["t"], r["U"], lw=1, label="U")
        a.plot(r["t"], r["E"], lw=1.3, color="k", label="E = K + U")
        a.plot(r["t"], r["Es"], lw=1.3, color="tab:red", label="E_shift")
        a.set_title(f"{meth}, dt={dt}  (eps_E={eps_max(r['E']):.2f}, eps_shift={eps_max(r['Es']):.3f})",
                    fontsize=9)
        a.grid(alpha=0.3)
ax[0, 0].legend(fontsize=7, ncol=4, loc="upper left")
ax[-1, 0].set_xlabel("t")
ax[-1, 1].set_xlabel("t")
fig.tight_layout()
fig.savefig(os.path.join(RES, "KUE_panels.png"), dpi=120)
plt.close(fig)

# cutoff jumps: pair count next to E(t) for the handout case dt = 0.01, Verlet
r = runs[("verlet", 0.01)]
fig, ax = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
ax[0].plot(r["t"], r["E"], "k", lw=1, label="E_total")
ax[0].plot(r["t"], r["E"][0] + U_CUT * (r["npairs"] - r["npairs"][0]), "tab:orange", lw=1, ls="--",
           label="E(0) + 10.8 * (npairs - npairs(0))")
ax[0].set_ylabel("energy")
ax[0].legend(fontsize=8)
ax[1].step(r["t"], r["npairs"], where="post", color="tab:blue")
ax[1].set_ylabel("pairs with r <= rc")
ax[2].plot(r["t"], r["Es"], "tab:red", lw=1)
ax[2].set_ylabel("E_shift")
ax[2].set_xlabel("t")
ax[0].set_title("Verlet, dt = 0.01: every jump of E is a pair crossing rc")
for a in ax:
    a.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(RES, "pairs_vs_energy_dt0.01.png"), dpi=130)
plt.close(fig)

# zoom on a few steps to show that jumps are multiples of 10.8
dE = np.diff(r["E"])
dn = np.diff(r["npairs"])
with open(os.path.join(RES, "cutoff_jumps_dt0.01_verlet.txt"), "w") as f:
    f.write(f"U at the cutoff 0.5*k*(rc-r0)^2 = {U_CUT:.4f}\n")
    f.write(f"steps with a pair-count change: {np.count_nonzero(dn)} of {len(dn)}\n")
    on = dn != 0
    f.write(f"mean |dE| on steps where npairs changes: {np.abs(dE[on]).mean():.4f}\n")
    f.write(f"mean |dE| on other steps: {np.abs(dE[~on]).mean():.4f}\n")
    fit = np.sum(dE[on] * dn[on]) / np.sum(dn[on] ** 2)
    f.write(f"least-squares slope of dE vs d(npairs) on those steps: {fit:.4f} (expect ~10.8)\n")
    f.write(f"max |dE_shift| per step: {np.abs(np.diff(r['Es'])).max():.4f}\n")

# same study for 5 different seeds (shifted energy only), to see how much eps depends on the trajectory
seed_rows = []
for seed in range(5):
    p, v, m = tm.init_system(seed=seed)
    for meth in METHODS:
        for dt in DTS:
            r = tm.run(p, v, m, dt, int(round(T_TOTAL / dt)), tm.L, method=meth)
            Es, npairs = shifted_energy(r)
            seed_rows.append([seed, meth, dt, eps_max(r["E"]), eps_max(Es), Es[0],
                              np.count_nonzero(np.diff(npairs))])
np.savetxt(os.path.join(RES, "energy_dt_seeds.csv"), np.array(seed_rows, dtype=object), fmt="%s",
           delimiter=",", header="seed,method,dt,eps_E_max,eps_Eshift_max,Eshift0,pair_count_changes",
           comments="")
sr = np.array(seed_rows, dtype=object)
print("\nshifted-energy eps over 5 seeds (median / max):")
with open(os.path.join(RES, "energy_dt_seeds_summary.csv"), "w") as f:
    f.write("method,dt,median_eps_E,median_eps_Eshift,max_eps_Eshift\n")
    for meth in METHODS:
        for dt in DTS:
            sel = (sr[:, 1] == meth) & (sr[:, 2] == dt)
            e = sr[sel, 4].astype(float)
            eraw = sr[sel, 3].astype(float)
            f.write(f"{meth},{dt},{np.median(eraw):.4f},{np.median(e):.5f},{e.max():.5f}\n")
            print(f"  {meth:6s} dt={dt:<7} raw median {np.median(eraw):.3f}  shift median {np.median(e):.4f}  max {e.max():.4f}")
