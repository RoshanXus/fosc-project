# smoothcut.py - cell-list force/energy with an optional smooth cutoff (for the Task 9 tests)
#
# Shifted pair potential  phi(r) = 0.5 k (r - r0)^2 - 0.5 k (rc - r0)^2   (0 at r = rc)
# hard   : U = phi(r) for r <= rc, force -k (r - r0) up to rc, then jumps to 0 (handout forces)
# smooth : U = phi(r) S(r), S = 1 for r <= rs, S goes 1 -> 0 between rs and rc with the quintic
#          1 - 10x^3 + 15x^4 - 6x^5 (x = (r - rs)/(rc - rs)), so U and F are both 0 at rc.
#          For r <= rs the force is exactly the handout force.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import toymodel as tm
import numpy as np
from numba import njit
from cellforces import build_cells, cell_of


@njit
def pair_term(r, k, r0, rc, rs, smooth):
    # returns (U_ij, -dU/dr) for one pair at distance r (0 < r <= rc)
    phi = 0.5 * k * (r - r0) ** 2 - 0.5 * k * (rc - r0) ** 2
    dphi = k * (r - r0)
    if smooth and r > rs:
        x = (r - rs) / (rc - rs)
        S = 1.0 - 10.0 * x ** 3 + 15.0 * x ** 4 - 6.0 * x ** 5
        dS = (-30.0 * x ** 2 + 60.0 * x ** 3 - 30.0 * x ** 4) / (rc - rs)
        return phi * S, -(dphi * S + phi * dS)
    return phi, -dphi


@njit
def forces_energy(pos, L, k, r0, rc, rs, smooth):
    # forces and the shifted (or smooth) potential energy in one pass over the cell list
    n = pos.shape[0]
    head, nxt, ncell = build_cells(pos, L, rc)
    side = L / ncell
    F = np.zeros_like(pos)
    U = 0.0
    npairs = 0
    for i in range(n):
        cx, cy = cell_of(pos[i, 0], pos[i, 1], side, ncell)
        for ny in range(max(cy - 1, 0), min(cy + 2, ncell)):
            for nx in range(max(cx - 1, 0), min(cx + 2, ncell)):
                j = head[ny * ncell + nx]
                while j != -1:
                    if j > i:
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        r = np.sqrt(dx * dx + dy * dy)
                        if r > 0.0 and r <= rc:
                            u, f = pair_term(r, k, r0, rc, rs, smooth)
                            U += u
                            npairs += 1
                            F[i, 0] += f * dx / r
                            F[i, 1] += f * dy / r
                            F[j, 0] -= f * dx / r
                            F[j, 1] -= f * dy / r
                    j = nxt[j]
    return F, U, npairs


def walls_mirror(pos, vel, L):
    # exact reflection: a particle that went delta past the wall is put delta inside
    for a in range(2):
        low = pos[:, a] < 0
        pos[low, a] = -pos[low, a]
        vel[low, a] *= -1
        high = pos[:, a] > L
        pos[high, a] = 2 * L - pos[high, a]
        vel[high, a] *= -1
    return pos, vel


def run_verlet(pos, vel, dt, nsteps, L, smooth, mirror, k=15.0, r0=0.8, rc=2.0, rs=1.5):
    # velocity Verlet (mass 1), same order of operations as tm.step_verlet
    # returns t, E_shift(t) (K + shifted/smooth U), npairs(t), final pos
    pos = pos.copy()
    vel = vel.copy()
    E = np.zeros(nsteps + 1)
    npairs = np.zeros(nsteps + 1, dtype=np.int64)
    F, U, npairs[0] = forces_energy(pos, L, k, r0, rc, rs, smooth)
    E[0] = 0.5 * np.sum(vel ** 2) + U
    for s in range(1, nsteps + 1):
        vel = vel + 0.5 * F * dt
        pos = pos + vel * dt
        if mirror:
            walls_mirror(pos, vel, L)
        else:
            tm.walls_reflect(pos, vel, L)      # handout rule: clamp onto the wall
        F, U, npairs[s] = forces_energy(pos, L, k, r0, rc, rs, smooth)
        vel = vel + 0.5 * F * dt
        E[s] = 0.5 * np.sum(vel ** 2) + U
    return np.arange(nsteps + 1) * dt, E, npairs, pos

