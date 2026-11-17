# Task 5, cost part: memory and execution time of float32 vs float64.
# Timings are medians of repeated runs (the machine is shared with other group members).
import time
import tracemalloc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

DTYPES = [np.float64, np.float32]


def box_for(n):
    return 10.0 * np.sqrt(n / 12.0)      # same density as the N = 12, L = 10 system


def median_time(fn, repeats):
    fn()                                  # warm-up
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return np.median(ts), np.min(ts), ts


# ---------- memory ----------
mem_rows = []
for n in [12, 200, 1000, 2000, 3000]:
    for dtype in DTYPES:
        pos, vel, mass = tm.init_system(N=n, L=box_for(n), dtype=dtype)
        d, dist, nb = tm.pairwise_distances(pos, tm.RC)
        tracemalloc.start()
        tracemalloc.reset_peak()
        F = tm.pair_forces(pos)
        peak = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        theory = n * n * 2 * np.dtype(dtype).itemsize
        mem_rows.append([n, np.dtype(dtype).name, pos.nbytes, vel.nbytes, d.nbytes, theory, dist.nbytes,
                         nb.nbytes, peak])
        del d, dist, nb, F
hdr = "N,dtype,pos_bytes,vel_bytes,d_NN2_bytes,d_theory_N2x2xitemsize,dist_bytes,neighbors_bytes,pair_forces_peak_bytes"
np.savetxt(os.path.join(RES, "precision_memory.csv"), np.array(mem_rows, dtype=object), fmt="%s", delimiter=",",
           header=hdr, comments="")
print("memory (MB): N, dtype, pos, d(N,N,2), theory, dist, neighbors(bool), peak in pair_forces")
for r in mem_rows:
    print(f"  {r[0]:5d} {r[1]:8s} {r[2]/1e6:9.6f} {r[4]/1e6:9.3f} {r[5]/1e6:9.3f} {r[6]/1e6:8.3f} {r[7]/1e6:8.3f} {r[8]/1e6:9.3f}")
M = np.array(mem_rows, dtype=object)
print("ratio float64/float32 of peak:", [f"{M[i, 8] / M[i + 1, 8]:.2f}" for i in range(0, len(M), 2)])

# ---------- timing ----------
time_rows = []
# (a) the full handout run, N = 12, 1000 Verlet steps (includes energy bookkeeping every step)
for dtype in DTYPES:
    p, v, m = tm.init_system(dtype=dtype)
    med, mn, ts = median_time(lambda: tm.run(p, v, m, 0.01, 1000, tm.L, method="verlet"), 7)
    time_rows.append(["full_run_1000_steps", 12, np.dtype(dtype).name, med, mn, len(ts)])

# (b) one pair_forces call at larger N
for n in [200, 500, 1000, 2000, 3000]:
    reps = 9 if n <= 1000 else 7
    for dtype in DTYPES:
        p, v, m = tm.init_system(N=n, L=box_for(n), dtype=dtype)
        med, mn, ts = median_time(lambda: tm.pair_forces(p), reps)
        time_rows.append(["pair_forces", n, np.dtype(dtype).name, med, mn, len(ts)])

# (c) a short run (20 Verlet steps, energies every step) at larger N
for n in [500, 1000, 2000]:
    for dtype in DTYPES:
        p, v, m = tm.init_system(N=n, L=box_for(n), dtype=dtype)
        med, mn, ts = median_time(lambda: tm.run(p, v, m, 0.001, 20, box_for(n), method="verlet"), 5)
        time_rows.append(["short_run_20_steps", n, np.dtype(dtype).name, med, mn, len(ts)])

np.savetxt(os.path.join(RES, "precision_timing.csv"), np.array(time_rows, dtype=object), fmt="%s", delimiter=",",
           header="test,N,dtype,median_s,min_s,repeats", comments="")
T = np.array(time_rows, dtype=object)
print("\ntiming: test, N, median float64 (s), median float32 (s), speedup t64/t32")
speed = []
for i in range(0, len(T), 2):
    s = T[i, 3] / T[i + 1, 3]
    speed.append([T[i, 0], T[i, 1], T[i, 3], T[i + 1, 3], s])
    print(f"  {T[i, 0]:20s} {T[i, 1]:5d} {T[i, 3]:.5f} {T[i + 1, 3]:.5f} {s:.2f}")
np.savetxt(os.path.join(RES, "precision_speedup.csv"), np.array(speed, dtype=object), fmt="%s", delimiter=",",
           header="test,N,median_float64_s,median_float32_s,speedup_t64_over_t32", comments="")

# figure: pair_forces time vs N and memory peak vs N
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for dtype, mk in [("float64", "s-"), ("float32", "o--")]:
    sel = (T[:, 0] == "pair_forces") & (T[:, 2] == dtype)
    ax[0].loglog(T[sel, 1].astype(float), T[sel, 3].astype(float), mk, label=dtype)
    sel = M[:, 1] == dtype
    ax[1].loglog(M[sel, 0].astype(float), M[sel, 8].astype(float) / 1e6, mk, label=f"{dtype}, measured peak in pair_forces")
    ax[1].loglog(M[sel, 0].astype(float), M[sel, 5].astype(float) / 1e6, ":", color=ax[1].lines[-1].get_color(),
                 label=f"{dtype}, one (N,N,2) array: N^2*2*itemsize")
ax[0].set_xlabel("N")
ax[0].set_ylabel("median time of one pair_forces call (s)")
ax[0].set_title("pair_forces time")
ax[1].set_xlabel("N")
ax[1].set_ylabel("memory (MB)")
ax[1].set_title("pair_forces memory")
for a in ax:
    a.grid(alpha=0.3, which="both")
    a.legend(fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(RES, "precision_cost.png"), dpi=130)
plt.close(fig)
