# Task 5, accuracy part: float32 vs float64 (Verlet).
# Same seed for both; the float32 start is the float64 start rounded to float32.
# To see if trajectory differences are a float32 problem or just sensitivity of the dynamics,
# we also run float64 from (a) the rounded start and (b) starts with random 1e-7 relative kicks.
# Two settings: the handout dt = 0.01 and a more accurate dt = 0.001, both run to t = 40
# (frames saved every 0.01 time units).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

T_END = 40.0
SETTINGS = [(0.01, 1), (0.001, 10)]      # (dt, save_every)
WINDOWS = [10.0, 40.0]

p64, v64, m64 = tm.init_system(dtype=np.float64)
p32, v32, m32 = tm.init_system(dtype=np.float32)
lines = [f"float32 start == float64 start rounded: {np.array_equal(p32, p64.astype(np.float32))}",
         f"||p32 - p64|| / ||p64|| at t=0: {np.linalg.norm(p32 - p64) / np.linalg.norm(p64):.3e}",
         f"||v32 - v64|| / ||v64|| at t=0: {np.linalg.norm(v32 - v64) / np.linalg.norm(v64):.3e}"]


def eps_pos(ra, rb):
    a = ra["pos"].astype(np.float64)
    b = rb["pos"].astype(np.float64)
    return np.linalg.norm((a - b).reshape(len(a), -1), axis=1) / np.linalg.norm(b.reshape(len(b), -1), axis=1)


stat_rows = []
all_curves = {}
energy_runs = {}
for dt, every in SETTINGS:
    nsteps = int(round(T_END / dt))
    rng = np.random.default_rng(1)

    def go(p, v, m):
        r = tm.run(p, v, m, dt, nsteps, tm.L, method="verlet", save_every=every)
        r["Es"], r["npairs"] = shifted_energy(r)
        return r

    runs = {"f64": go(p64, v64, m64), "f32": go(p32, v32, m32),
            "f64_rounded_start": go(p32.astype(np.float64), v32.astype(np.float64), m64)}
    for k in range(5):
        runs[f"f64_kick1e-7_{k}"] = go(p64 * (1 + 1e-7 * rng.uniform(-1, 1, p64.shape)), v64.copy(), m64)
    runs["f64_kick1e-12"] = go(p64 * (1 + 1e-12 * rng.uniform(-1, 1, p64.shape)), v64.copy(), m64)
    energy_runs[dt] = runs
    t = runs["f64"]["t"]

    curves = {"f32 vs f64": eps_pos(runs["f32"], runs["f64"]),
              "f64 rounded start vs f64": eps_pos(runs["f64_rounded_start"], runs["f64"]),
              "f32 vs f64 rounded start (arithmetic only)": eps_pos(runs["f32"], runs["f64_rounded_start"]),
              "f64 1e-12 kick vs f64": eps_pos(runs["f64_kick1e-12"], runs["f64"])}
    for k in range(5):
        curves[f"f64 1e-7 kick #{k} vs f64"] = eps_pos(runs[f"f64_kick1e-7_{k}"], runs["f64"])
    all_curves[dt] = (t, curves)

    with open(os.path.join(RES, f"precision_eps_pos_dt{dt}.csv"), "w") as f:
        names = list(curves)
        f.write("t," + ",".join(names) + "\n")
        for i in range(len(t)):
            f.write(f"{t[i]:.2f}," + ",".join(f"{curves[n][i]:.4e}" for n in names) + "\n")

    lines.append(f"\n===== dt = {dt}, Verlet, run to t = {T_END:g} =====")
    lines.append("curve: eps_pos at t=1, 5, 10, 20, 40 | first t with eps > 1e-3, > 1e-1 | growth rate (1/time)")
    for n, e in curves.items():
        at = [e[np.argmin(np.abs(t - x))] for x in [1, 5, 10, 20, 40]]
        t3 = t[np.argmax(e > 1e-3)] if np.any(e > 1e-3) else np.nan
        t1 = t[np.argmax(e > 1e-1)] if np.any(e > 1e-1) else np.nan
        m = (t >= 1) & (t <= (t3 if t3 == t3 else t[-1]))     # fit log(eps) before it saturates
        lam = np.polyfit(t[m], np.log(e[m]), 1)[0]
        lines.append(f"{n:44s} " + " ".join(f"{a:.2e}" for a in at) + f" | {t3:5.2f} {t1:5.2f} | {lam:.2f}")

    # quantities that do not depend on the exact trajectory
    labels = ["E_shift(end)", "eps_Eshift_max", "eps_E_max", "npairs(end)", "mean npairs 2nd half", "mean K 2nd half"]
    for t_end in WINDOWS:
        n_end = np.argmin(np.abs(t - t_end)) + 1
        half = (t >= t_end / 2) & (t <= t_end + 1e-9)
        lines.append(f"\nwindow t <= {t_end:g}: run, " + ", ".join(labels))
        for name, r in runs.items():
            row = [dt, t_end, name, r["Es"][n_end - 1], eps_max(r["Es"][:n_end]), eps_max(r["E"][:n_end]),
                   r["npairs"][n_end - 1], r["npairs"][half].mean(), r["K"][half].astype(np.float64).mean()]
            stat_rows.append(row)
            lines.append(f"{name:20s} {row[3]:9.4f} {row[4]:.4f} {row[5]:7.3f} {row[6]:4d} {row[7]:7.2f} {row[8]:8.2f}")
        ens = np.array([r[3:] for r in stat_rows if "kick1e-7" in r[2] and r[0] == dt and r[1] == t_end], dtype=float)
        lines.append("  spread of the five 1e-7-kicked float64 runs (min .. max):")
        for j, lab in enumerate(labels):
            lines.append(f"    {lab:22s} {ens[:, j].min():9.4f} .. {ens[:, j].max():9.4f}")

np.savetxt(os.path.join(RES, "precision_statistics.csv"), np.array(stat_rows, dtype=object), fmt="%s",
           delimiter=",", header="dt,t_end,run,Eshift_end,eps_Eshift_max,eps_E_max,npairs_end,"
                                 "mean_npairs_2nd_half,mean_K_2nd_half", comments="")

# float32 energy round-off on its own: control cluster (no rc crossing, no wall), Verlet, t = 10
xA, vA, mA, _ = relaxed_cluster()
lines.append("\ncontrol cluster (no rc crossing, no walls), Verlet, t = 10: dt, eps_E float64, eps_E float32")
ctrl = []
for dt in [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]:
    ns = int(round(10 / dt))
    e = []
    for dtype in [np.float64, np.float32]:
        r = tm.run(xA.astype(dtype), vA.astype(dtype), mA.astype(dtype), dt, ns, tm.L, method="verlet")
        e.append(eps_max(r["E"]))
    ctrl.append([dt, e[0], e[1]])
    lines.append(f"  {dt:<7} {e[0]:.3e} {e[1]:.3e}")
np.savetxt(os.path.join(RES, "precision_control_eps_vs_dt.csv"), np.array(ctrl), delimiter=",",
           header="dt,eps_E_float64,eps_E_float32", comments="", fmt="%.6e")

with open(os.path.join(RES, "precision_accuracy.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n".join(lines))

# figure 1: eps_pos(t), one panel per dt
fig, axs = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
for ax, (dt, _) in zip(axs, SETTINGS):
    t, curves = all_curves[dt]
    for k in range(5):
        ax.semilogy(t, curves[f"f64 1e-7 kick #{k} vs f64"], color="0.65", lw=0.8,
                    label="float64 with random 1e-7 kick vs float64 (5 runs)" if k == 0 else None)
    ax.semilogy(t, curves["f32 vs f64"], color="tab:red", lw=1.6, label="float32 vs float64")
    ax.semilogy(t, curves["f64 rounded start vs f64"], color="tab:blue", lw=1.2,
                label="float64 from the rounded start vs float64")
    ax.semilogy(t, curves["f64 1e-12 kick vs f64"], color="tab:purple", lw=1, ls=":",
                label="float64 with 1e-12 kick vs float64")
    ax.axvline(10, color="k", lw=0.8, ls=":")
    ax.set_xlabel("t")
    ax.set_title(f"Verlet, dt = {dt}")
    ax.grid(alpha=0.3, which="both")
axs[0].set_ylabel("eps_pos = ||x_a - x_b|| / ||x_b||")
axs[0].legend(fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig(os.path.join(RES, "precision_eps_pos.png"), dpi=130)
plt.close(fig)

# figure 2: energies in both precisions (dt = 0.001, where the integrator error is small)
runs = energy_runs[0.001]
t = runs["f64"]["t"]
fig, ax = plt.subplots(2, 1, figsize=(8, 6.5), sharex=True)
for name, c in [("f64", "k"), ("f32", "tab:red")]:
    ax[0].plot(t, runs[name]["E"], color=c, lw=1, label=name)
    ax[1].plot(t, runs[name]["Es"], color=c, lw=1, label=name)
for k in range(5):
    ax[1].plot(t, runs[f"f64_kick1e-7_{k}"]["Es"], color="0.7", lw=0.6, label="f64, 1e-7 kicks" if k == 0 else None)
ax[0].set_ylabel("E_total")
ax[1].set_ylabel("E_shift")
ax[1].set_xlabel("t")
ax[0].set_title("Energy in float32 and float64 (Verlet, dt = 0.001)")
for a in ax:
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "precision_energy.png"), dpi=130)
plt.close(fig)

# figure 3: control cluster energy error vs dt in both precisions
c = np.array(ctrl)
fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(c[:, 0], c[:, 1], "s-", label="float64")
ax.loglog(c[:, 0], c[:, 2], "o--", label="float32")
ax.set_xlabel("dt")
ax.set_ylabel("eps_E_max (control cluster, t = 10)")
ax.set_title("Verlet energy error: float32 hits a round-off floor")
ax.grid(alpha=0.3, which="both")
ax.legend()
fig.tight_layout()
fig.savefig(os.path.join(RES, "precision_control_floor.png"), dpi=130)
plt.close(fig)
