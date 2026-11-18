# Picture of the cell list on the handout system: one particle only checks its own cell and the 8 around it
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join("..", "member1_baseline_verification"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import toymodel as tm

pos, _, _ = tm.init_system()
L, rc = tm.L, tm.RC
ncell = int(L // rc)
side = L / ncell
cell = np.minimum((pos // side).astype(int), ncell - 1)

# pick the particle with the most candidates in its 3x3 block that is not on the edge of the grid
inner = [i for i in range(len(pos)) if 0 < cell[i, 0] < ncell - 1 and 0 < cell[i, 1] < ncell - 1]
near = lambda i: np.all(np.abs(cell - cell[i]) <= 1, axis=1)
i0 = max(inner, key=lambda i: near(i).sum())
cand = near(i0)
cand[i0] = False
dist = np.linalg.norm(pos - pos[i0], axis=1)
partner = cand & (dist <= rc)

fig, ax = plt.subplots(figsize=(6.4, 6.4))
cx, cy = cell[i0]
for a in range(ncell):
    for b in range(ncell):
        if abs(a - cx) <= 1 and abs(b - cy) <= 1:
            ax.add_patch(plt.Rectangle((a * side, b * side), side, side, color="#cfe3f7" if (a, b) != (cx, cy) else "#8fbfea", zorder=0))
for k in range(ncell + 1):
    ax.axhline(k * side, color="0.6", lw=0.8, zorder=1)
    ax.axvline(k * side, color="0.6", lw=0.8, zorder=1)
ax.add_patch(plt.Circle(pos[i0], rc, fill=False, ls="--", color="#d1495b", lw=1.5, zorder=2))
for j in np.nonzero(partner)[0]:
    ax.plot(*zip(pos[i0], pos[j]), color="#d1495b", lw=2, zorder=3)
ax.scatter(*pos[~cand].T, s=60, color="0.65", zorder=4, label="skipped (other cells)")
ax.scatter(*pos[cand & ~partner].T, s=60, color="#2e6fbf", zorder=4, label="checked, farther than rc")
ax.scatter(*pos[partner].T, s=60, color="#d1495b", zorder=4, label="checked, inside rc (force)")
ax.scatter(*pos[i0], s=140, marker="*", color="k", zorder=5, label=f"particle {i0}")
ax.set_xlim(0, L); ax.set_ylim(0, L); ax.set_aspect("equal")
ax.set_xticks(np.arange(0, L + 0.1, side)); ax.set_yticks(np.arange(0, L + 0.1, side))
ax.set_title(f"Cell list: {ncell}x{ncell} cells of side {side:g} >= rc = {rc:g}\n"
             f"particle {i0} checks {cand.sum()} of {len(pos) - 1} others, {partner.sum()} inside rc")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2, fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig("results/cell_list_sketch.png", dpi=150)
print(f"particle {i0}: checks {cand.sum()} of {len(pos) - 1}, partners {partner.sum()}")
