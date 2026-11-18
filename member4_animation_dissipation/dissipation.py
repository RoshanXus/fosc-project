# Task 4: viscous drag F = -gamma v, energy balance and relaxation criterion
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # results/ is relative to this folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import toymodel as tm
import damped as dm

os.makedirs("results", exist_ok=True)
L = tm.L
DT = 0.01
T_END = 600.0                 # 60000 steps for every gamma
GAMMAS = [0.0, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948", "#52514e"]

# steady-state criterion (see report): both must hold from t_relax to the end of the run
V_TOL = 1e-3                  # mean particle speed
E_TOL = 1e-3                  # |E(t + DeltaT) - E(t)| / |E(0)|, window labelled by its start
DELTA_T = 10.0                # time units (1000 steps)
MIN_WINDOW = 100.0            # the run must continue at least this long after t_relax


def first_time_always(ok, t):
    # earliest time after which ok stays True until the end; None if ok[-1] is False
    bad = np.nonzero(~ok)[0]
    if len(bad) == 0:
        return t[0]
    if bad[-1] == len(ok) - 1:
        return None
    return t[bad[-1] + 1]


def softest_mode(pos, h=1e-6):
    # softest nonzero vibration omega from a finite-difference Hessian H = -dF/dx (mass 1)
    n = pos.size
    H = np.zeros((n, n))
    for a in range(n):
        dp = np.zeros(n)
        dp[a] = h
        Fp = tm.pair_forces(pos + dp.reshape(pos.shape)).ravel()
        Fm = tm.pair_forces(pos - dp.reshape(pos.shape)).ravel()
        H[:, a] = -(Fp - Fm) / (2 * h)
    w2 = np.linalg.eigvalsh(0.5 * (H + H.T))
    return np.sqrt(w2[w2 > 1e-3].min()), w2.min()


p0, v0, m = tm.init_system(tm.N, L, 0)
nsteps = int(round(T_END / DT))
lag = int(round(DELTA_T / DT))
runs = {}
table = []
balance = []
for g in GAMMAS:
    o = dm.simulate(p0, v0, m, DT, nsteps, L, gamma=g, pos_every=2)
    runs[g] = o
    t = o["t"]
    # drag work, trapezoidal rule on sum |v|^2 (mass = 1)
    w_step = g * DT * 0.5 * (o["v2"][1:] + o["v2"][:-1])
    W = np.concatenate([[0.0], np.cumsum(w_step)])
    o["W"] = W
    # per-step mismatch of the shifted energy balance: dE_sh + W_step should be 0
    mis = np.diff(o["Esh"]) + w_step
    changed = o["pairkey"][1:] != o["pairkey"][:-1]
    balance.append([g, o["Esh"][0] - o["Esh"][-1], W[-1], mis.sum(), mis.sum() / W[-1] if g > 0 else np.nan,
                    int(changed.sum()), mis[changed].sum(), mis[~changed].sum(), np.abs(mis[~changed]).max(),
                    o["E"][0] - o["E"][-1]])

    if g == 0.0:
        rmin2 = np.minimum(o["rmin"][1:], o["rmin"][:-1])
        wall = (o["nwall"][1:] > 0) | (o["nwall"][:-1] > 0)
        cls = np.where(changed, 0, np.where(rmin2 < 0.1, 1, np.where(wall, 2, 3)))
        with open("results/energy_error_breakdown_gamma0.csv", "w") as f:
            f.write("step_class,n_steps,sum_dEsh,sum_abs_dEsh\n")
            for c, name in enumerate(["pair set changed", "pair closer than 0.1 (no pair change)",
                                      "wall contact (none of the above)", "other steps"]):
                f.write(f"{name},{(cls == c).sum()},{mis[cls == c].sum():.5g},{np.abs(mis[cls == c]).sum():.5g}\n")

    E = o["E"]
    # window labelled by its start so t_relax is not pushed back by DeltaT; the last DeltaT stays 0
    dE = np.zeros(len(E))
    dE[:-lag] = np.abs(E[lag:] - E[:-lag]) / abs(E[0])
    ok_v = o["vmean"] < V_TOL
    ok_e = dE < E_TOL
    t_v = first_time_always(ok_v, t)
    t_e = first_time_always(ok_e, t)
    if t_e is not None and t_e >= t[-1] - DELTA_T:
        t_e = None                      # only the untested last DeltaT is "ok", so it never settled
    t_r = first_time_always(ok_v & ok_e, t)
    if t_r is not None and t[-1] - t_r < MIN_WINDOW:
        t_r = None
    o["t_relax"] = t_r
    chg = np.nonzero(changed)[0]
    t_last = t[chg[-1] + 1] if len(chg) else 0.0
    # fit ln K = a - rate t from the last pair change down to K = 1e-14, where round-off takes over
    after = t >= t_last
    low = after & (o["K"] <= 1e-14)
    sel = after & (t < (t[low][0] if low.any() else np.inf))
    if g > 0 and sel.sum() > 100:
        rate = -np.polyfit(t[sel], np.log(o["K"][sel]), 1)[0]
        i0 = np.searchsorted(t, t_last)
        # K ~ exp(-gamma t), so mean speed ~ exp(-gamma t / 2)
        t_pred = t_last + 2.0 / g * np.log(o["vmean"][i0] / V_TOL)
    else:
        rate, t_pred = np.nan, np.nan
    # K decay rate of the damped softest mode (under- or over-damped)
    w_min, w2_lowest = softest_mode(o["pos_final"]) if g > 0 else (np.nan, np.nan)
    rate_mode = g if g < 2 * w_min else g - np.sqrt(g * g - 4 * w_min ** 2)
    if g > 0:
        i0 = np.searchsorted(t, t_last)
        t_pred_mode = t_last + 2.0 / rate_mode * np.log(o["vmean"][i0] / V_TOL)
    else:
        t_pred_mode = np.nan
    cl = dm.clusters(o["pos_final"])
    table.append([g, t_r if t_r is not None else np.nan, t_v if t_v is not None else np.nan,
                  t_e if t_e is not None else np.nan, t_last, rate, rate / g if g > 0 else np.nan, t_pred,
                  w_min, w2_lowest, rate_mode, t_pred_mode,
                  o["K"].max(), E[-1], o["U"][-1], o["Esh"][-1], o["K"][-1], o["vmean"][-1],
                  o["npairs"][-1], " ".join(str(c) for c in cl)])
    print(f"gamma={g}: t_relax={t_r} t_speed={t_v} t_energy={t_e} t_lastpair={t_last:.2f} rate={rate:.4g} "
          f"E_end={E[-1]:.4f} U_end={o['U'][-1]:.4f} pairs={o['npairs'][-1]} clusters={cl} "
          f"balance mismatch={mis.sum():.4g} of W={W[-1]:.4g}")
    np.savetxt(f"results/dissipation_timeseries_gamma{g}.csv",
               np.column_stack([t, o["K"], o["U"], E, o["Esh"], o["npairs"], o["vmean"], W, dE])[::10],
               delimiter=",", header="t,K,U,E,E_shifted,npairs,mean_speed,drag_work,dE_next10_over_E0",
               comments="", fmt="%.8g")
    np.savetxt(f"results/final_positions_gamma{g}.csv", o["pos_final"], delimiter=",", header="x,y",
               comments="", fmt="%.6f")

hdr = ("gamma,t_relax,t_speed_only,t_energy_only,t_last_pair_change,K_decay_rate,rate_over_gamma,"
       "t_pred_exp_estimate,omega_min,lowest_eigenvalue,K_rate_from_mode,t_pred_mode_estimate,K_max,E_final,U_final,E_shifted_final,K_final,mean_speed_final,npairs_final,clusters")
with open("results/relaxation_table.csv", "w") as f:
    f.write(hdr + "\n")
    for r in table:
        f.write(",".join(f"{x:.6g}" if not isinstance(x, str) else x for x in r) + "\n")
with open("results/energy_balance.csv", "w") as f:
    f.write("gamma,drop_E_shifted,drag_work_W,total_mismatch,mismatch_over_W,steps_pair_set_changed,"
            "mismatch_on_change_steps,mismatch_other_steps,max_abs_mismatch_other_step,drop_E_unshifted\n")
    for r in balance:
        f.write(",".join(f"{x:.6g}" for x in r) + "\n")

# ---- energy balance at smaller dt (gamma = 0.1, t = 0..100) to see if the mismatch is a timestep error ----
with open("results/energy_balance_dt.csv", "w") as f:
    f.write("gamma,dt,t_end,drop_E_shifted,drag_work_W,mismatch,mismatch_over_W,steps_pair_set_changed,"
            "sum_abs_step_mismatch\n")
    for dt in [0.02, 0.01, 0.005, 0.0025]:
        n = int(round(100 / dt))
        ob = dm.simulate(p0, v0, m, dt, n, L, gamma=0.1, pos_every=n)
        w = 0.1 * dt * 0.5 * (ob["v2"][1:] + ob["v2"][:-1])
        dEs = ob["Esh"][0] - ob["Esh"][-1]
        nch = int((ob["pairkey"][1:] != ob["pairkey"][:-1]).sum())
        sabs = np.abs(np.diff(ob["Esh"]) + w).sum()
        f.write(f"0.1,{dt},100,{dEs:.6g},{w.sum():.6g},{w.sum() - dEs:.6g},{(w.sum() - dEs) / w.sum():.4g},{nch},"
                f"{sabs:.5g}\n")
print(open("results/energy_balance_dt.csv").read())

# ---- figures: K, U, E per gamma ----
fig, axs = plt.subplots(3, 3, figsize=(13, 10))
for ax, g in zip(axs.flat, GAMMAS):
    o = runs[g]
    tr = o["t_relax"]
    tmax = T_END if (g == 0 or tr is None) else min(T_END, max(10.0, 1.5 * tr))
    k = o["t"] <= tmax
    ax.plot(o["t"][k], o["K"][k], color="#2a78d6", lw=1.2, label="K")
    ax.plot(o["t"][k], o["U"][k], color="#eb6834", lw=1.2, label="U")
    ax.plot(o["t"][k], o["E"][k], color="#1a1a19", lw=1.2, label="E = K + U")
    if tr is not None:
        ax.axvline(tr, color="#8a8a86", ls="--", lw=1.0)
        ax.text(tr, ax.get_ylim()[1] * 0.92, f" t_relax = {tr:.1f}", fontsize=8, color="#52514e")
    ax.set_title(f"gamma = {g}", fontsize=10)
    ax.set_xlabel("time")
    ax.grid(alpha=0.25)
axs[0, 0].legend(frameon=False, fontsize=9)
fig.suptitle("K(t), U(t) and E(t) for each gamma (same start, dt = 0.01); dashed line = t_relax", fontsize=11)
fig.tight_layout()
fig.savefig("results/energies_per_gamma.png", dpi=130)
plt.close(fig)

# K on a log scale with the exp(-gamma t) estimate
fig, axs = plt.subplots(1, 2, figsize=(12, 4.6))
for g, c in zip(GAMMAS[1:], COLORS[1:]):
    o = runs[g]
    axs[0 if g <= 1.0 else 1].semilogy(o["t"], np.maximum(o["K"], 1e-30), color=c, lw=1.2, label=f"gamma = {g}")
for g, c in zip(GAMMAS[1:5], COLORS[1:5]):
    imax = np.argmax(runs[g]["K"])
    tt = np.linspace(runs[g]["t"][imax], T_END, 200)
    axs[0].semilogy(tt, runs[g]["K"][imax] * np.exp(-g * (tt - tt[0])), color=c, lw=0.9, ls="--")
axs[0].set_xlim(0, 450)
axs[1].set_xlim(0, 60)
for ax in axs:
    ax.set_ylim(1e-20, 1e3)
    ax.set_xlabel("time")
    ax.set_ylabel("K(t)")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
axs[0].set_title("weak damping: dashed = K_max * exp(-gamma (t - t_max))", fontsize=10)
axs[1].set_title("strong damping", fontsize=10)
fig.tight_layout()
fig.savefig("results/kinetic_decay_log.png", dpi=130)
plt.close(fig)

# energy balance: drop of shifted energy vs drag work
fig, ax = plt.subplots(figsize=(8, 4.8))
for g, c in zip(GAMMAS[1:], COLORS[1:]):
    o = runs[g]
    k = o["t"] <= 200
    ax.plot(o["t"][k], o["Esh"][0] - o["Esh"][k], color=c, lw=1.5, label=f"gamma = {g}")
    ax.plot(o["t"][k], o["W"][k], color="#1a1a19", lw=0.8, ls="--")
ax.set_xlabel("time")
ax.set_ylabel("energy")
ax.set_title("E_sh(0) - E_sh(t) (colour) and drag work W(t) = int gamma sum|v|^2 dt (black dashed)", fontsize=10)
ax.grid(alpha=0.25)
ax.legend(frameon=False, fontsize=8, ncol=2)
fig.tight_layout()
fig.savefig("results/energy_balance.png", dpi=130)
plt.close(fig)

# relaxation time vs gamma
tab = np.array([[r[0], r[1], r[7], r[11], r[5], r[10]] for r in table[1:]], dtype=float)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
ax = axs[1]
ax.loglog(tab[:, 0], tab[:, 4], "o", color="#2a78d6", ms=9, label="fitted decay rate of K")
ax.loglog(tab[:, 0], tab[:, 5], "x", color="#eb6834", ms=9, mew=2, label="damped softest mode prediction")
gg = np.logspace(-1.5, 1.5, 100)
ax.loglog(gg, gg, color="#8a8a86", lw=0.8, ls="--", label="rate = gamma")
ax.set_xlabel("gamma")
ax.set_ylabel("decay rate of K in the final cooling")
ax.grid(alpha=0.25, which="both")
ax.legend(frameon=False, fontsize=9)
ax = axs[0]
ax.loglog(tab[:, 0], tab[:, 1], "o-", color="#2a78d6", lw=1.5, ms=8, label="measured t_relax")
ax.loglog(tab[:, 0], tab[:, 2], "s--", color="#eb6834", lw=1.2, ms=7, label="estimate K ~ exp(-gamma t)")
ax.loglog(tab[:, 0], tab[:, 3], "^:", color="#1baf7a", lw=1.2, ms=7, label="estimate with softest mode")
ax.set_xlabel("gamma")
ax.set_ylabel("relaxation time")
ax.grid(alpha=0.25, which="both")
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("results/relaxation_time_vs_gamma.png", dpi=130)
plt.close(fig)

# final configurations
fig, axs = plt.subplots(2, 4, figsize=(13, 6.8))
for ax, g in zip(axs.flat, GAMMAS[1:]):
    pos = runs[g]["pos_final"]
    _, _, nb = tm.pairwise_distances(pos, tm.RC)
    I, J = np.nonzero(np.triu(nb, 1))
    for a, b in zip(I, J):
        ax.plot(pos[[a, b], 0], pos[[a, b], 1], color="#8a8a86", lw=0.7, zorder=1)
    ax.scatter(p0[:, 0], p0[:, 1], s=25, facecolor="none", edgecolor="#eb6834", lw=1.0, zorder=2)
    ax.scatter(pos[:, 0], pos[:, 1], s=45, color="#2a78d6", edgecolor="white", zorder=3)
    ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, lw=1.5, color="k"))
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, L + 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"gamma = {g}: pairs {runs[g]['npairs'][-1]}, U = {runs[g]['U'][-1]:.2f}", fontsize=9)
fig.suptitle("Final configurations at t = 600 (blue, grey lines = pairs in rc); orange rings = start positions",
             fontsize=11)
fig.tight_layout()
fig.savefig("results/final_states.png", dpi=130)
plt.close(fig)

# conservative vs dissipative side by side: paths for t = 0..30 and the energies
fig, axs = plt.subplots(2, 2, figsize=(11, 9.5))
for col, g in enumerate([0.0, 0.5]):
    o = runs[g]
    P = o["pos"][:1501]                     # pos saved every 2 steps -> t = 0..30
    ax = axs[0, col]
    for p in range(P.shape[1]):
        ax.plot(P[:, p, 0], P[:, p, 1], lw=0.8, color="#2a78d6", alpha=0.6)
    ax.scatter(P[0, :, 0], P[0, :, 1], s=25, facecolor="none", edgecolor="#eb6834", label="t = 0")
    ax.scatter(P[-1, :, 0], P[-1, :, 1], s=40, color="#1a1a19", label="t = 30")
    ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, lw=1.5, color="k"))
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, L + 0.5)
    ax.set_aspect("equal")
    ax.set_title(f"gamma = {g}: particle paths, t = 0 to 30", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2)
    ax = axs[1, col]
    k = o["t"] <= 30
    ax.plot(o["t"][k], o["K"][k], color="#2a78d6", lw=1.2, label="K")
    ax.plot(o["t"][k], o["U"][k], color="#eb6834", lw=1.2, label="U")
    ax.plot(o["t"][k], o["E"][k], color="#1a1a19", lw=1.2, label="E")
    ax.plot(o["t"][k], o["Esh"][k], color="#1baf7a", lw=1.2, label="E_shifted")
    ax.set_xlabel("time")
    ax.set_ylim(-250, 650)
    ax.set_title(f"gamma = {g}: energies", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig("results/compare_conservative_dissipative.png", dpi=130)
plt.close(fig)
