# small helpers shared by my scripts (energy bookkeeping, pair counting)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import numpy as np
import toymodel as tm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
os.makedirs(RES, exist_ok=True)

U_CUT = 0.5 * tm.K * (tm.RC - tm.R0) ** 2     # 10.8, value of one pair's U at r = rc


def pair_counts(frames, rc=tm.RC):
    # number of interacting pairs (i<j, 0 < r <= rc) in every saved frame
    n = np.zeros(len(frames), dtype=int)
    for f, p in enumerate(frames):
        _, _, nb = tm.pairwise_distances(p.astype(np.float64), rc)
        n[f] = np.count_nonzero(np.triu(nb, 1))
    return n


def min_pair_dist(frames):
    out = np.zeros(len(frames))
    for f, p in enumerate(frames):
        _, dist, _ = tm.pairwise_distances(p.astype(np.float64), tm.RC)
        out[f] = dist[np.triu_indices(len(p), 1)].min()
    return out


def shifted_energy(res, rc=tm.RC, u_cut=U_CUT):
    # E_shift = K + sum_pairs [0.5k(r-r0)^2 - 0.5k(rc-r0)^2] = E - u_cut * npairs
    n = pair_counts(res["pos"], rc)
    return res["E"].astype(np.float64) - u_cut * n, n


def eps_max(E):
    E = np.asarray(E, dtype=np.float64)
    return np.max(np.abs(E - E[0])) / abs(E[0])


def relaxed_cluster():
    # squeeze the seed-0 positions into a 2x2 region and let them slide downhill (all pairs
    # interacting) until the forces vanish; gives a compact cluster with max pair distance ~1.1,
    # so with rc = 2 every pair interacts and no pair ever crosses the cutoff
    from functools import partial
    pos, vel, mass = tm.init_system()
    x = 5.0 + 0.2 * (pos - 5.0)
    f_all = partial(tm.pair_forces, rc=1000.0)
    for _ in range(20000):
        x = x + 0.002 * f_all(x)
    return x, vel, mass, np.abs(f_all(x)).max()
