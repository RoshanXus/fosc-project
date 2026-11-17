# compare_animation.py - side-by-side animation: conservative (gamma = 0) vs dissipative (gamma = 0.5)
import sys, os
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
NSTEPS = 4000          # t = 0..40
EVERY = 8              # draw every 8th step -> 501 frames
TRAIL = 4              # trail length in drawn frames

p0, v0, m = tm.init_system(tm.N, L, 0)
runs = [dm.simulate(p0, v0, m, 0.01, NSTEPS, L, gamma=g, pos_every=EVERY) for g in [0.0, 0.5]]

fig, axs = plt.subplots(1, 2, figsize=(10, 5.4))
fig.subplots_adjust(left=0.05, right=0.98, bottom=0.06, top=0.86, wspace=0.12)


def update(i):
    s = i * EVERY
    for ax, o, g in zip(axs, runs, [0.0, 0.5]):
        ax.clear()
        ax.add_patch(plt.Rectangle((0, 0), L, L, fill=False, lw=2.0, color="k"))
        pos = o["pos"][i]
        _, _, nb = tm.pairwise_distances(pos, tm.RC)
        I, J = np.nonzero(np.triu(nb, 1))
        for a, b in zip(I, J):
            ax.plot(pos[[a, b], 0], pos[[a, b], 1], color="#8a8a86", lw=0.8, zorder=1)
        i0 = max(0, i - TRAIL)
        for p in range(pos.shape[0]):
            ax.plot(o["pos"][i0:i + 1, p, 0], o["pos"][i0:i + 1, p, 1], color="#2a78d6", lw=1.0, alpha=0.35)
        ax.scatter(pos[:, 0], pos[:, 1], s=70, color="#2a78d6", edgecolor="white", lw=1.0, zorder=3)
        ax.set_title(f"gamma = {g}", fontsize=11, pad=34)
        ax.text(0.0, 1.075, f"t = {o['t'][s]:5.2f}   pairs = {o['npairs'][s]}", transform=ax.transAxes,
                fontsize=9, family="monospace")
        ax.text(0.0, 1.015, f"K = {o['K'][s]:7.2f}   E_shifted = {o['Esh'][s]:8.2f}", transform=ax.transAxes,
                fontsize=9, family="monospace")
        ax.set_xlim(-0.5, L + 0.5)
        ax.set_ylim(-0.5, L + 0.5)
        ax.set_aspect("equal")
        ax.set_xticks([0, 5, 10])
        ax.set_yticks([0, 5, 10])
    return []


ani = animation.FuncAnimation(fig, update, frames=NSTEPS // EVERY + 1, blit=False)
ani.save("results/compare_gamma0_vs_gamma0.5.mp4", writer=animation.FFMpegWriter(fps=25, bitrate=1500), dpi=100)
plt.close(fig)
print("saved", f"{os.path.getsize('results/compare_gamma0_vs_gamma0.5.mp4') / 1e6:.2f} MB")
