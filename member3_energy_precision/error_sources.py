# Task 3: where does the shifted-energy error come from in the default run?
# Every step's change of E_shift is put in one class (checked in this order):
#   1. the set of interacting pairs changed (a pair crossed rc, force jumps 18 -> 0)
#   2. a particle touched a wall (walls_reflect clamps it to x = 0 or L)
#   3. a very close pass, r_min < 0.1 (the force direction flips quickly near r = 0)
#   4. none of these (smooth step)
# The wall clamp is also measured directly: U_shift at the clamped position minus U_shift at the
# mirror-image position (where an exact reflection would have put the particle).
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *


def u_shift(pos):
    _, dist, nb = tm.pairwise_distances(pos, tm.RC)
    up = np.triu(nb, 1)
    return np.sum(0.5 * tm.K * (dist[up] - tm.R0) ** 2 - U_CUT)


def mirror(p, L):
    q = np.abs(p)                 # x < 0  -> -x
    return np.where(q > L, 2 * L - q, q)


def run_with_clamp_log(pos, vel, mass, dt, nsteps, method):
    # same update as tm.step_euler / tm.step_verlet, but keeps the position before the clamp
    m = mass[:, None]
    F = tm.pair_forces(pos)
    frames = [pos.copy()]
    Kin = [tm.kinetic_energy(vel, mass)]
    clamp_dU = np.zeros(nsteps + 1)
    for s in range(1, nsteps + 1):
        if method == "euler":
            F = tm.pair_forces(pos)
            vel = vel + F / m * dt
            raw = pos + vel * dt
        else:
            vel = vel + 0.5 * F / m * dt
            raw = pos + vel * dt
        pos = raw.copy()
        tm.walls_reflect(pos, vel, tm.L)
        if np.any(pos != raw):
            clamp_dU[s] = u_shift(pos) - u_shift(mirror(raw, tm.L))
        if method == "verlet":
            F = tm.pair_forces(pos)
            vel = vel + 0.5 * F / m * dt
        frames.append(pos.copy())
        Kin.append(tm.kinetic_energy(vel, mass))
    return np.array(frames), np.array(Kin), clamp_dU


pos0, vel0, mass = tm.init_system()
lines = []
table = []
for meth in ["euler", "verlet"]:
    for dt in [0.01, 0.001]:
        nsteps = int(round(10.0 / dt))
        frames, Kin, clamp_dU = run_with_clamp_log(pos0, vel0, mass, dt, nsteps, meth)
        ref = tm.run(pos0, vel0, mass, dt, nsteps, tm.L, method=meth)
        same = np.max(np.abs(frames - ref["pos"]))
        Es = Kin + np.array([u_shift(p) for p in frames])
        dEs = np.diff(Es)
        nbs = []
        rmin = np.zeros(len(frames))
        for f, p in enumerate(frames):
            _, dist, nb = tm.pairwise_distances(p, tm.RC)
            nbs.append(np.triu(nb, 1))
            rmin[f] = dist[np.triu_indices(len(p), 1)].min()
        nbs = np.array(nbs)
        pair_change = np.any(nbs[1:] != nbs[:-1], axis=(1, 2))
        wall = np.any((frames[1:] == 0.0) | (frames[1:] == tm.L), axis=(1, 2))
        close = np.minimum(rmin[1:], rmin[:-1]) < 0.1
        cls = np.full(nsteps, "smooth", dtype=object)
        cls[close] = "close_pass_r<0.1"
        cls[wall] = "wall_contact"
        cls[pair_change] = "pair_crosses_rc"
        lines.append(f"{meth} dt={dt}: max |pos - tm.run pos| = {same:.1e} (local stepper check)")
        lines.append(f"  E_shift(0) = {Es[0]:.4f}, E_shift(end) = {Es[-1]:.4f}, eps_Eshift_max = {eps_max(Es):.4f}")
        lines.append(f"  wall clamps: {np.count_nonzero(clamp_dU)} steps with a clamp that changes U; "
                     f"sum of clamp dU = {clamp_dU.sum():.4f}, sum |clamp dU| = {np.abs(clamp_dU).sum():.4f}")
        for c in ["pair_crosses_rc", "wall_contact", "close_pass_r<0.1", "smooth"]:
            sel = cls == c
            table.append([meth, dt, c, np.count_nonzero(sel), dEs[sel].sum(), np.abs(dEs[sel]).sum(),
                          np.abs(dEs[sel]).max() if sel.any() else 0.0])
            lines.append(f"  {c:18s} steps {np.count_nonzero(sel):6d}  net dE {dEs[sel].sum():8.3f}"
                         f"  sum|dE| {np.abs(dEs[sel]).sum():8.3f}  max|dE| {table[-1][-1]:.4f}")
        print("\n".join(lines[-7:]))

np.savetxt(os.path.join(RES, "error_sources.csv"), np.array(table, dtype=object), fmt="%s", delimiter=",",
           header="method,dt,step_class,n_steps,net_dEshift,sum_abs_dEshift,max_abs_dEshift", comments="")
with open(os.path.join(RES, "error_sources.txt"), "w") as f:
    f.write("\n".join(lines) + "\n")

# bar chart of sum |dE_shift| per class
T = np.array(table, dtype=object)
classes = ["pair_crosses_rc", "wall_contact", "close_pass_r<0.1", "smooth"]
fig, ax = plt.subplots(figsize=(8, 4))
w = 0.2
for j, (meth, dt) in enumerate([("euler", 0.01), ("verlet", 0.01), ("euler", 0.001), ("verlet", 0.001)]):
    vals = [T[(T[:, 0] == meth) & (T[:, 1] == dt) & (T[:, 2] == c), 5][0] for c in classes]
    ax.bar(np.arange(4) + (j - 1.5) * w, vals, w, label=f"{meth}, dt={dt}")
ax.set_xticks(np.arange(4))
ax.set_xticklabels(["pair crosses rc", "wall contact", "close pass r<0.1", "smooth step"])
ax.set_ylabel("sum over steps of |dE_shift|")
ax.set_yscale("log")
ax.set_title("Where the shifted-energy error is made (default start, t = 10)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(RES, "error_sources.png"), dpi=130)
plt.close(fig)
