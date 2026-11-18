# Member 4 helpers: velocity Verlet with linear drag, pair bookkeeping and cluster finding on top of toymodel.py
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import numpy as np
import toymodel as tm

U_RC = 0.5 * tm.K * (tm.RC - tm.R0) ** 2   # pair energy at the cutoff = 10.8


def step_damped(pos, vel, mass, dt, L, gamma, F_prev=None, walls=tm.walls_reflect):
    # drag explicit in the first half kick, implicit in the second (trapezoidal); gamma = 0 gives tm.step_verlet
    m = np.asarray(mass, dtype=pos.dtype)[:, None]
    if F_prev is None:
        F_prev = tm.pair_forces(pos)
    a = 0.5 * dt * gamma / m
    vel_half = vel * (1.0 - a) + 0.5 * dt * F_prev / m
    pos_new = pos + vel_half * dt
    walls(pos_new, vel_half, L)
    F_new = tm.pair_forces(pos_new)
    vel_new = (vel_half + 0.5 * dt * F_new / m) / (1.0 + a)
    return pos_new, vel_new, F_new


def walls_mirror(pos, vel, L):
    # alternative to the clamp: a particle delta past the wall goes back delta inside
    for d in range(2):
        lo = pos[:, d] < 0
        pos[lo, d] = -pos[lo, d]
        vel[lo, d] *= -1
        hi = pos[:, d] > L
        pos[hi, d] = 2 * L - pos[hi, d]
        vel[hi, d] *= -1
    return pos, vel


def pair_info(pos, rc=tm.RC):
    # number of pairs inside rc and the shifted potential (continuous at rc)
    _, dist, nb = tm.pairwise_distances(pos, rc)
    up = np.triu(nb, 1)
    r = dist[up]
    U = 0.5 * tm.K * np.sum((r - tm.R0) ** 2)
    npairs = len(r)
    key = hash(np.packbits(up).tobytes())     # changes when the set of pairs inside rc changes
    return npairs, U, U - U_RC * npairs, dist[np.triu_indices(len(pos), 1)].min(), key


def clusters(pos, rc=tm.RC):
    # connected components of the neighbour graph, returns sizes sorted large to small
    _, _, nb = tm.pairwise_distances(pos, rc)
    n = len(pos)
    label = -np.ones(n, dtype=int)
    c = 0
    for i in range(n):
        if label[i] >= 0:
            continue
        stack = [i]
        label[i] = c
        while stack:
            j = stack.pop()
            for k in np.nonzero(nb[j])[0]:
                if label[k] < 0:
                    label[k] = c
                    stack.append(k)
        c += 1
    sizes = np.bincount(label)
    return sorted(sizes.tolist(), reverse=True)


def simulate(pos, vel, mass, dt, nsteps, L, gamma=0.0, pos_every=1, walls=tm.walls_reflect):
    # per-step diagnostics every step, positions only every pos_every steps
    pos = pos.copy()
    vel = vel.copy()
    out = {k: np.zeros(nsteps + 1) for k in ["t", "K", "U", "Ush", "v2", "vmean", "vmax", "rmin"]}
    out["npairs"] = np.zeros(nsteps + 1, dtype=int)
    out["pairkey"] = np.zeros(nsteps + 1, dtype=np.int64)
    out["nwall"] = np.zeros(nsteps + 1, dtype=int)
    out["pos"] = np.zeros((nsteps // pos_every + 1, len(pos), 2))
    out["vel"] = np.zeros((nsteps // pos_every + 1, len(pos), 2))

    def save(s, pos, vel):
        npairs, U, Ush, rmin, key = pair_info(pos)
        sp = np.sqrt(np.sum(vel ** 2, axis=1))
        out["t"][s] = s * dt
        out["K"][s] = tm.kinetic_energy(vel, mass)
        out["U"][s] = U
        out["Ush"][s] = Ush
        out["npairs"][s] = npairs
        out["pairkey"][s] = key
        out["v2"][s] = np.sum(vel ** 2)
        out["vmean"][s] = sp.mean()
        out["vmax"][s] = sp.max()
        out["rmin"][s] = rmin
        out["nwall"][s] = np.sum(np.any((pos <= 0.0) | (pos >= L), axis=1))   # walls_reflect puts them exactly on 0 or L
        if s % pos_every == 0:
            out["pos"][s // pos_every] = pos
            out["vel"][s // pos_every] = vel

    save(0, pos, vel)
    F = None
    for s in range(1, nsteps + 1):
        pos, vel, F = step_damped(pos, vel, mass, dt, L, gamma, F, walls)
        save(s, pos, vel)
    out["E"] = out["K"] + out["U"]
    out["Esh"] = out["K"] + out["Ush"]
    out["pos_final"] = pos
    out["vel_final"] = vel
    return out
