# Task 8: Numba all-pairs and cell-list forces; no default args, they add ~27 us per call
import numpy as np
from numba import njit

K, R0, RC = 15.0, 0.8, 2.0


@njit
def pair_forces_allpairs(pos, k, r0, rc):
    n = pos.shape[0]
    F = np.zeros_like(pos)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            r = np.sqrt(dx * dx + dy * dy)
            if r > 0.0 and r <= rc:
                fmag = -k * (r - r0) / r
                F[i, 0] += fmag * dx
                F[i, 1] += fmag * dy
                F[j, 0] -= fmag * dx
                F[j, 1] -= fmag * dy
    return F


@njit
def cell_of(x, y, side, ncell):
    cx = int(x / side)
    cy = int(y / side)
    # the wall clamp puts particles exactly on x = L, which would give cx = ncell
    cx = min(max(cx, 0), ncell - 1)
    cy = min(max(cy, 0), ncell - 1)
    return cx, cy


@njit
def build_cells(pos, L, rc):
    # cell side >= rc, so every partner is in the same cell or one of the 8 around it
    ncell = max(1, int(L / rc))
    side = L / ncell
    head = -np.ones(ncell * ncell, dtype=np.int64)   # first particle in each cell
    nxt = -np.ones(pos.shape[0], dtype=np.int64)     # next particle in the same cell
    for i in range(pos.shape[0]):
        cx, cy = cell_of(pos[i, 0], pos[i, 1], side, ncell)
        c = cy * ncell + cx
        nxt[i] = head[c]
        head[c] = i
    return head, nxt, ncell


@njit
def pair_forces_cell(pos, L, k, r0, rc):
    n = pos.shape[0]
    head, nxt, ncell = build_cells(pos, L, rc)
    side = L / ncell
    F = np.zeros_like(pos)
    for i in range(n):
        cx, cy = cell_of(pos[i, 0], pos[i, 1], side, ncell)
        for ny in range(max(cy - 1, 0), min(cy + 2, ncell)):
            for nx in range(max(cx - 1, 0), min(cx + 2, ncell)):
                j = head[ny * ncell + nx]
                while j != -1:
                    if j > i:      # each pair once (Newton's third law)
                        dx = pos[i, 0] - pos[j, 0]
                        dy = pos[i, 1] - pos[j, 1]
                        r = np.sqrt(dx * dx + dy * dy)
                        if r > 0.0 and r <= rc:
                            fmag = -k * (r - r0) / r
                            F[i, 0] += fmag * dx
                            F[i, 1] += fmag * dy
                            F[j, 0] -= fmag * dx
                            F[j, 1] -= fmag * dy
                    j = nxt[j]
    return F


@njit
def potential_energy_cell(pos, L, k, r0, rc):
    n = pos.shape[0]
    head, nxt, ncell = build_cells(pos, L, rc)
    side = L / ncell
    U = 0.0
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
                            U += 0.5 * k * (r - r0) ** 2
                    j = nxt[j]
    return U


def make_allpairs_force(k=K, r0=R0, rc=RC):
    def force_fn(pos):
        return pair_forces_allpairs(pos, k, r0, rc)
    return force_fn


def make_cell_force(L, k=K, r0=R0, rc=RC):
    # tm.run / tm.step_verlet call force_fn(pos) only, so fix L, k, r0, rc here
    def force_fn(pos):
        return pair_forces_cell(pos, L, k, r0, rc)
    return force_fn


def make_cell_energy(L, k=K, r0=R0, rc=RC):
    def pot_fn(pos):
        return potential_energy_cell(pos, L, k, r0, rc)
    return pot_fn
