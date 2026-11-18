# Task 2: animation of the default system and diagnostics from its trajectory
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # results/ is relative to this folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
import toymodel as tm
import damped as dm

os.makedirs("results", exist_ok=True)
L = tm.L
NSTEPS = 2000          # t = 0..20
TRAIL_TIME = 0.3       # trail length in time units


def draw_frame(ax, P, i, t_arr, E, npairs, every, title=None, fs=9):
    # draws frame i of positions P (frames, N, 2) on ax (used for snapshots and animation)
    ax.clear()
    ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, lw=2.0, color="k"))
    pos = P[i]
    _, _, nb = tm.pairwise_distances(pos, tm.RC)
    I, J = np.nonzero(np.triu(nb, 1))
    for a, b in zip(I, J):
        ax.plot([pos[a, 0], pos[b, 0]], [pos[a, 1], pos[b, 1]], color="#8a8a86", lw=0.8, zorder=1)
    i0 = max(0, i - max(1, int(round(TRAIL_TIME / ((t_arr[1] - t_arr[0]) * every)))))
    for p in range(pos.shape[0]):
        ax.plot(P[i0:i + 1, p, 0], P[i0:i + 1, p, 1], color="#2a78d6", lw=1.0, alpha=0.35, zorder=2)
    ax.scatter(pos[:, 0], pos[:, 1], s=70, color="#2a78d6", edgecolor="white", lw=1.0, zorder=3)
    s = i * every
    ax.text(0.0, 1.015, f"t = {t_arr[s]:5.2f}  E_total = {E[s]:6.1f}  pairs = {npairs[s]}",
            transform=ax.transAxes, fontsize=fs, family="monospace")
    if title:
        ax.set_title(title, fontsize=10, pad=18)
    ax.set_xlim(-0.5, L + 0.5)
    ax.set_ylim(-0.5, L + 0.5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 5, 10])
    ax.set_yticks([0, 5, 10])


def make_animation(o, every, fname, fps, dpi, label):
    P = o["pos"][::every]
    fig, ax = plt.subplots(figsize=(5.2, 5.4))
    fig.subplots_adjust(left=0.08, right=0.97, bottom=0.06, top=0.88)

    def update(i):
        draw_frame(ax, P, i, o["t"], o["E"], o["npairs"], every, title=label)
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(P), blit=False)
    if fname.endswith(".mp4"):
        ani.save(fname, writer=animation.FFMpegWriter(fps=fps, bitrate=1200), dpi=dpi)
    else:
        ani.save(fname, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    print("saved", fname, f"{os.path.getsize(fname) / 1e6:.2f} MB, {len(P)} frames")


def longest_run(mask):
    # longest run of consecutive True values in a 1D bool array
    best = cur = 0
    for x in mask:
        cur = cur + 1 if x else 0
        best = max(best, cur)
    return best


def diagnostics(o, dt, name):
    P = o["pos"]
    V = o["vel"]
    lo, hi = P.min(), P.max()
    eps = 1e-12
    on_wall = (P[:, :, 0] < eps) | (P[:, :, 0] > L - eps) | (P[:, :, 1] < eps) | (P[:, :, 1] > L - eps)
    runs = [longest_run(on_wall[:, p]) for p in range(P.shape[1])]
    # displacement per step vs the speed: |dx| should be about |v| dt
    disp = np.sqrt(np.sum((P[1:] - P[:-1]) ** 2, axis=2))
    sp = np.sqrt(np.sum(V ** 2, axis=2))
    vref = np.maximum(sp[1:], sp[:-1]) * dt
    ratio = disp / np.maximum(vref, 1e-12)
    big = np.array([dm.clusters(p)[0] for p in P])
    Esh = o["Esh"]
    row = {
        "case": name, "dt": dt, "nsteps": len(P) - 1, "t_end": o["t"][-1],
        "min_coord": lo, "max_coord": hi,
        "wall_frames_total": int(on_wall.sum()), "longest_wall_run_steps": max(runs),
        "max_disp_per_step": disp.max(), "max_speed_x_dt": (sp.max() * dt), "max_disp_over_v_dt": ratio.max(),
        "max_speed": sp.max(), "rmin": o["rmin"].min(),
        "E0": o["E"][0], "E_end": o["E"][-1], "Esh0": Esh[0], "Esh_end": Esh[-1],
        "max_abs_Esh_drift": np.abs(Esh - Esh[0]).max(),
        "max_rel_Esh_drift": np.abs(Esh - Esh[0]).max() / abs(Esh[0]),
        "npairs0": o["npairs"][0], "npairs_end": o["npairs"][-1], "npairs_max": o["npairs"].max(),
        "largest_cluster0": big[0], "largest_cluster_end": big[-1],
        "K0": o["K"][0], "K_end": o["K"][-1],
    }
    return row, big, on_wall, runs


p0, v0, m = tm.init_system(tm.N, L, 0)

# ---- default run: N = 12, dt = 0.01, 2000 velocity Verlet steps ----
o = dm.simulate(p0, v0, m, 0.01, NSTEPS, L, gamma=0.0)
r = tm.run(p0, v0, m, 0.01, NSTEPS, L)
print("max |pos(my stepper) - pos(tm.run)| =", np.abs(o["pos"] - r["pos"]).max())

row, big, on_wall, runs = diagnostics(o, 0.01, "default dt=0.01")
np.savetxt("results/default_timeseries.csv",
           np.column_stack([o["t"], o["K"], o["U"], o["E"], o["Esh"], o["npairs"], big, o["rmin"], o["vmax"],
                            on_wall.sum(axis=1)]),
           delimiter=",", header="t,K,U,E,E_shifted,npairs,largest_cluster,rmin,vmax,n_on_wall", comments="",
           fmt="%.6g")
np.savetxt("results/default_wall_runs.csv", np.column_stack([np.arange(len(runs)), runs]), delimiter=",",
           header="particle,longest_consecutive_steps_on_wall", comments="", fmt="%d")

# ---- same initial condition with larger dt, same physical time t = 20 ----
rows = [row]
bad = {}
for dt in [0.005, 0.02, 0.05, 0.1]:
    ob = dm.simulate(p0, v0, m, dt, int(round(20 / dt)), L, gamma=0.0)
    rb, bigb, _, _ = diagnostics(ob, dt, f"dt={dt}")
    rows.append(rb)
    bad[dt] = (ob, bigb)
keys = list(rows[0].keys())
with open("results/diagnostics_dt.csv", "w") as f:
    f.write(",".join(keys) + "\n")
    for rr in rows:
        f.write(",".join(str(rr[k]) if isinstance(rr[k], str) else f"{rr[k]:.6g}" for k in keys) + "\n")
for rr in rows:
    print(rr)

# ---- can a lone pair (r = 3, offset b) capture each other without a third particle? ----
with open("results/two_particle_capture.csv", "w") as f:
    f.write("b,dt,rmin,time_inside_rc,final_r,x0_start,x0_end,Esh_start,Esh_end\n")
    for b in [0.0, 0.5]:
        for dt in [0.001, 0.0001]:
            p2 = np.array([[3.5, 5.0 - b / 2], [6.5, 5.0 + b / 2]])
            v2 = np.array([[0.1, 0.0], [-0.1, 0.0]])
            o2 = dm.simulate(p2, v2, np.ones(2), dt, int(round(8 / dt)), L, gamma=0.0)
            d2 = np.sqrt(np.sum((o2["pos"][:, 0] - o2["pos"][:, 1]) ** 2, axis=1))
            f.write(f"{b},{dt},{d2.min():.5g},{(d2 <= tm.RC).sum() * dt:.4f},{d2[-1]:.4f},3.5,"
                    f"{o2['pos'][-1, 0, 0]:.4f},{o2['Esh'][0]:.5g},{o2['Esh'][-1]:.5g}\n")
print(open("results/two_particle_capture.csv").read())

# ---- figures ----
fig, axs = plt.subplots(2, 3, figsize=(11, 7.6))
for ax, tt in zip(axs.flat, [0, 2, 4, 8, 14, 20]):
    i = int(round(tt / 0.01))
    draw_frame(ax, o["pos"], i, o["t"], o["E"], o["npairs"], 1, fs=8)
fig.suptitle("Default run (N = 12, dt = 0.01, velocity Verlet): snapshots, grey lines = pairs inside rc = 2",
             fontsize=11)
fig.tight_layout()
fig.savefig("results/snapshots_default.png", dpi=130)
plt.close(fig)

fig, axs = plt.subplots(4, 1, figsize=(8, 9.5), sharex=True)
axs[0].plot(o["t"], o["npairs"], color="#2a78d6", lw=1.5, label="pairs inside rc")
axs[0].plot(o["t"], big, color="#eb6834", lw=1.5, label="largest cluster (particles)")
axs[0].set_ylabel("count")
axs[0].legend(frameon=False)
axs[1].plot(o["t"], o["K"], color="#2a78d6", lw=1.5, label="K")
axs[1].plot(o["t"], o["E"], color="#eb6834", lw=1.5, label="E_total = K + U (handout U)")
axs[1].plot(o["t"], o["Esh"], color="#1baf7a", lw=1.5, label="E_shifted = E_total - 10.8 * pairs")
axs[1].set_ylabel("energy")
axs[1].legend(frameon=False, fontsize=9)
axs[2].plot(o["t"], o["rmin"], color="#2a78d6", lw=1.2)
axs[2].axhline(tm.R0, color="#8a8a86", lw=0.8, ls="--")
axs[2].text(0.2, tm.R0 + 0.05, "r0 = 0.8", color="#52514e", fontsize=9)
axs[2].set_ylabel("closest pair\ndistance")
axs[3].plot(o["t"], np.cumsum(on_wall.sum(axis=1)), color="#2a78d6", lw=1.5)
axs[3].set_ylabel("cumulative\nwall contacts")
axs[3].set_xlabel("time")
for ax in axs:
    ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig("results/diagnostics_default.png", dpi=130)
plt.close(fig)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
for dt, c in zip([0.005, 0.01, 0.02, 0.05, 0.1], ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]):
    oo = o if dt == 0.01 else bad[dt][0]
    axs[0].plot(oo["t"], oo["Esh"] - oo["Esh"][0], color=c, lw=1.3, label=f"dt = {dt}")
    axs[1].plot(oo["t"], oo["npairs"], color=c, lw=1.3, label=f"dt = {dt}")
axs[0].set_ylabel("E_shifted(t) - E_shifted(0)")
axs[1].set_ylabel("pairs inside rc")
for ax in axs:
    ax.set_xlabel("time")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
fig.savefig("results/diagnostics_dt.png", dpi=130)
plt.close(fig)

# ---- animations ----
make_animation(o, 4, "results/animation_default.mp4", 25, 110, "N = 12, dt = 0.01, velocity Verlet")
make_animation(o, 8, "results/animation_default.gif", 12, 70, "N = 12, dt = 0.01, velocity Verlet")
ob = bad[0.1][0]
make_animation(ob, 1, "results/animation_dt0.1.mp4", 10, 110, "same start, dt = 0.1 (too large)")
make_animation(ob, 1, "results/animation_dt0.1.gif", 10, 70, "same start, dt = 0.1 (too large)")

# ---- a few extra numbers from the default run quoted in the report ----
imax = np.argmax(o["npairs"])
with open("results/default_summary.txt", "w") as f:
    for s in [0, 1000, 2000]:
        f.write(f"t = {o['t'][s]:.0f}: K = {o['K'][s]:.4g}, U = {o['U'][s]:.4g}, E = {o['E'][s]:.4g}, "
                f"E_shifted = {o['Esh'][s]:.4g}, pairs = {o['npairs'][s]}, largest cluster = {big[s]}\n")
    f.write(f"max pair count {o['npairs'][imax]} at t = {o['t'][imax]:.2f} (largest cluster {big[imax]})\n")
    f.write(f"fraction of steps with t >= 10 where the largest cluster has >= 10 particles: "
            f"{np.mean(big[1000:] >= 10):.3f}\n")
    f.write(f"fraction of steps with some pair closer than 0.1: {np.mean(o['rmin'] < 0.1):.3f}, "
            f"closest approach {o['rmin'].min():.4g}\n")
    f.write(f"max K = {o['K'].max():.4g}, max speed = {o['vmax'].max():.4g}\n")
    f.write(f"total wall contacts (particle-steps on a wall) = {int(on_wall.sum())}\n")
print(open("results/default_summary.txt").read())
