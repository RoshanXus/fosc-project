# Core of the FSC 2D atomic toy model: springs between pairs within rc, hard reflecting walls
import numpy as np

N = 12
L = 10.0
DT = 0.01
NSTEPS = 1000   # number of steps (T in the handout)
K = 15.0        # spring stiffness k
R0 = 0.8        # preferred spacing r0
RC = 2.0        # neighbour cutoff rc


def init_system(N=12, L=10.0, seed=0, dtype=np.float64):
    np.random.seed(seed)
    pos = (np.random.rand(N, 2) * L).astype(dtype)
    vel = np.random.uniform(-1.0, 1.0, (N, 2)).astype(dtype)
    mass = np.ones(N, dtype=dtype)
    return pos, vel, mass


def walls_reflect(pos, vel, L):
    # works in place on pos and vel (same as the handout)
    left_hits = pos[:, 0] < 0
    vel[left_hits, 0] *= -1
    pos[left_hits, 0] = 0.0

    right_hits = pos[:, 0] > L
    vel[right_hits, 0] *= -1
    pos[right_hits, 0] = L

    bottom_hits = pos[:, 1] < 0
    vel[bottom_hits, 1] *= -1
    pos[bottom_hits, 1] = 0.0

    top_hits = pos[:, 1] > L
    vel[top_hits, 1] *= -1
    pos[top_hits, 1] = L

    return pos, vel


def pairwise_distances(pos, rc):
    d = pos[:, None, :] - pos[None, :, :]          # (N, N, 2), d[i, j] = r_i - r_j
    dist = np.sqrt(np.sum(d * d, axis=2))          # (N, N)
    neighbors = (dist > 0) & (dist <= rc)          # diagonal (dist = 0) is never a neighbour
    return d, dist, neighbors


def pair_forces_loop(pos, k=K, r0=R0, rc=RC):
    k, r0 = float(k), float(r0)    # numpy float64 scalars would upcast float32 arrays
    F = np.zeros_like(pos)
    d, dist, neighbors = pairwise_distances(pos, rc)
    n = pos.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if neighbors[i, j]:
                fmag = -k * (dist[i, j] - r0)
                f = fmag * d[i, j] / dist[i, j]    # d[i, j]/dist[i, j] points from j to i
                F[i] += f
                F[j] -= f
    return F


def pair_forces(pos, k=K, r0=R0, rc=RC):
    k, r0 = float(k), float(r0)
    d, dist, neighbors = pairwise_distances(pos, rc)
    u = d / (dist[:, :, None] + 1e-12)              # unit vectors (1e-12 avoids 0/0 on the diagonal)
    fmag = -k * (dist - r0) * neighbors             # zero outside the cutoff
    F = np.sum(fmag[:, :, None] * u, axis=1)
    return F


def potential_energy(pos, k=K, r0=R0, rc=RC):
    k, r0 = float(k), float(r0)
    d, dist, neighbors = pairwise_distances(pos, rc)
    upper = np.triu(neighbors, 1)                   # count each pair i < j once
    return 0.5 * k * np.sum((dist[upper] - r0) ** 2, dtype=pos.dtype)


def kinetic_energy(vel, mass):
    m = np.asarray(mass, dtype=vel.dtype)
    return 0.5 * np.sum(m[:, None] * vel ** 2, dtype=vel.dtype)


def step_euler(pos, vel, mass, dt, L, force_fn=pair_forces):
    # handout step_once: semi-implicit (symplectic) Euler
    dt = float(dt)
    m = np.asarray(mass, dtype=pos.dtype)[:, None]
    F = force_fn(pos)
    vel = vel + (F / m) * dt
    pos = pos + vel * dt
    walls_reflect(pos, vel, L)
    return pos, vel


def step_verlet(pos, vel, mass, dt, L, F_prev=None, force_fn=pair_forces):
    # handout step_smooth: velocity Verlet
    dt = float(dt)
    m = np.asarray(mass, dtype=pos.dtype)[:, None]
    if F_prev is None:
        F_prev = force_fn(pos)
    vel_half = vel + 0.5 * (F_prev / m) * dt
    pos_new = pos + vel_half * dt
    walls_reflect(pos_new, vel_half, L)
    F_new = force_fn(pos_new)
    vel_new = vel_half + 0.5 * (F_new / m) * dt
    return pos_new, vel_new, F_new


def run(pos, vel, mass, dt, nsteps, L, method="verlet", force_fn=pair_forces, save_every=1,
        pot_fn=potential_energy):
    # if force_fn uses a non-default k, r0 or rc, pass a matching pot_fn as well
    pos = pos.copy()
    vel = vel.copy()
    mass = np.asarray(mass, dtype=pos.dtype)
    dt = float(dt)
    nframes = nsteps // save_every + 1
    t = np.zeros(nframes)
    frames = np.zeros((nframes, pos.shape[0], 2), dtype=pos.dtype)
    Kin = np.zeros(nframes, dtype=pos.dtype)
    Pot = np.zeros(nframes, dtype=pos.dtype)

    def save(idx, s, pos, vel):
        t[idx] = s * dt
        frames[idx] = pos
        Kin[idx] = kinetic_energy(vel, mass)
        Pot[idx] = pot_fn(pos)

    save(0, 0, pos, vel)
    F = None
    for s in range(1, nsteps + 1):
        if method == "euler":
            pos, vel = step_euler(pos, vel, mass, dt, L, force_fn)
        elif method == "verlet":
            pos, vel, F = step_verlet(pos, vel, mass, dt, L, F, force_fn)
        else:
            raise ValueError("method must be 'euler' or 'verlet'")
        if s % save_every == 0:
            save(s // save_every, s, pos, vel)

    return {"t": t, "pos": frames, "K": Kin, "U": Pot, "E": Kin + Pot, "vel_final": vel}
