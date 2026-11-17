# benchmark_forces.py - time and memory of one force evaluation vs N at fixed density
# numpy = tm.pair_forces (handout), numba all-pairs, numba cell list
import sys, os, time, tracemalloc, subprocess
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import toymodel as tm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cellforces as cf

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)

DENSITY = 0.12
REPEATS = 7
NS = [12, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000,
      100000, 200000, 500000, 1000000]
NMAX_NUMPY = 5000       # one call already needs ~1.6 GB and ~3 s here
NMAX_ALLPAIRS = 50000


def median_time(fn):
    # call fn enough times that one repeat takes >= 20 ms, return median time per call
    t0 = time.perf_counter()
    fn()
    t1 = time.perf_counter() - t0
    number = max(1, int(0.02 / max(t1, 1e-9)))
    times = []
    for r in range(REPEATS):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - t0) / number)
    return np.median(times), number


# ---- compile time, each in a fresh python so numba starts cold -----------------
compile_code = """
import time, numpy as np, cellforces as cf
p = np.random.default_rng(0).random((100, 2)) * 28.9
t0 = time.perf_counter(); cf.{fn}; t1 = time.perf_counter()
cf.{fn}; t2 = time.perf_counter()
print(t1 - t0, t2 - t1)
"""
with open("results/compile_time.txt", "w") as f:
    f.write("first call (includes JIT compile) vs second call, fresh python process each, N = 100\n")
    for label, call in [("cell list", "pair_forces_cell(p, 28.9, 15.0, 0.8, 2.0)"), ("all-pairs", "pair_forces_allpairs(p, 15.0, 0.8, 2.0)")]:
        firsts = []
        for rep in range(3):
            out = subprocess.run([sys.executable, "-c", compile_code.format(fn=call)],
                                 capture_output=True, text=True, check=True).stdout.split()
            firsts.append(float(out[0]))
            second = float(out[1])
        line = "%s: first call median of 3 = %.3f s (runs: %s), second call = %.2e s" % (
            label, np.median(firsts), ", ".join("%.3f" % x for x in firsts), second)
        print(line)
        f.write(line + "\n")

# record the machine (shared with other people running benchmarks at the same time)
with open("results/machine.txt", "w") as f:
    f.write(subprocess.run("lscpu | grep -E 'Model name|^CPU\\(s\\)|L1d|L2|L3'; uptime", shell=True,
                           capture_output=True, text=True).stdout)

# warm up (compile) before any timing
p = np.random.default_rng(0).random((50, 2)) * 20.0
cell_fn = cf.make_cell_force(20.0)
cell_fn(p)
all_fn = cf.make_allpairs_force()
all_fn(p)

# ---- timing -------------------------------------------------------------------
rows = []
for N in NS:
    L = np.sqrt(N / DENSITY)
    pos = np.random.default_rng(N).random((N, 2)) * L
    t_np = t_all = np.nan
    if N <= NMAX_NUMPY:
        t_np, _ = median_time(lambda: tm.pair_forces(pos))
    if N <= NMAX_ALLPAIRS:
        t_all, _ = median_time(lambda: all_fn(pos))
    cell_fn = cf.make_cell_force(L)
    t_cell, num = median_time(lambda: cell_fn(pos))
    head, nxt, ncell = cf.build_cells(pos, L, tm.RC)
    rows.append([N, L, ncell, t_np, t_all, t_cell, t_np / t_cell, t_all / t_cell, t_cell / N * 1e9])
    print("N=%7d  numpy %.3e  allpairs %.3e  cell %.3e  speedup np/cell %.1f  all/cell %.1f" % (
        N, t_np, t_all, t_cell, t_np / t_cell, t_all / t_cell))

rows = np.array(rows)
np.savetxt("results/force_timing.csv", rows, delimiter=",", fmt="%.6g",
           header="N,L,cells_per_side,t_numpy_s,t_numba_allpairs_s,t_numba_cell_s,speedup_numpy_over_cell,speedup_allpairs_over_cell,cell_ns_per_particle",
           comments="")

N_arr, t_np, t_all, t_cell = rows[:, 0], rows[:, 3], rows[:, 4], rows[:, 5]


def slope(N, t, nmin):
    m = (N >= nmin) & np.isfinite(t)
    return np.polyfit(np.log(N[m]), np.log(t[m]), 1)


def crossing(N, ta, tb):
    # first N where tb < ta, interpolated on log-log between the two grid points
    m = np.isfinite(ta) & np.isfinite(tb)
    N, ta, tb = N[m], ta[m], tb[m]
    g = np.log(ta / tb)        # > 0 when b is faster
    if g[-1] <= 0:
        return np.nan, "never"
    slower = np.nonzero(g <= 0)[0]
    if len(slower) == 0:
        return N[0], "already faster at the smallest N"
    i = slower[-1] + 1          # from here on b stays faster
    x = np.log(N[i - 1]) + (0 - g[i - 1]) * (np.log(N[i]) - np.log(N[i - 1])) / (g[i] - g[i - 1])
    return np.exp(x), "between N=%d and N=%d" % (N[i - 1], N[i])


fits = {"numpy": (slope(N_arr, t_np, 500), 500),
        "numba all-pairs": (slope(N_arr, t_all, 2000), 2000),
        "numba cell list": (slope(N_arr, t_cell, 5000), 5000)}
with open("results/scaling_fits.txt", "w") as f:
    for name, (c, nmin) in fits.items():
        line = "%s: t ~ N^%.2f (fit for N >= %d)" % (name, c[0], nmin)
        print(line)
        f.write(line + "\n")
    be1, where1 = crossing(N_arr, t_np, t_cell)
    be2, where2 = crossing(N_arr, t_all, t_cell)
    be3, where3 = crossing(N_arr, t_np, t_all)
    for line in ["break-even cell list vs numpy: N ~ %.0f (%s)" % (be1, where1),
                 "break-even cell list vs numba all-pairs: N ~ %.0f (%s)" % (be2, where2),
                 "break-even numba all-pairs vs numpy: N ~ %.0f (%s)" % (be3, where3)]:
        print(line)
        f.write(line + "\n")

# ---- figure: time vs N ---------------------------------------------------------
C_NP, C_ALL, C_CELL = "#2a78d6", "#eb6834", "#1baf7a"
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
for t, c, name in [(t_np, C_NP, "numpy"), (t_all, C_ALL, "numba all-pairs"), (t_cell, C_CELL, "numba cell list")]:
    m = np.isfinite(t)
    coef, nmin = fits[name]
    ax[0].loglog(N_arr[m], t[m], "o-", color=c, lw=2, ms=6,
                 label="%s (slope %.2f for N >= %d)" % (name, coef[0], nmin))
    nf = N_arr[m][N_arr[m] >= nmin]
    ax[0].loglog(nf, np.exp(np.polyval(coef, np.log(nf))), "--", color=c, lw=1, alpha=0.8)
ax[0].set_xlabel("N (density fixed at 0.12, L = sqrt(N/0.12))")
ax[0].set_ylabel("median time per force call [s]")
ax[0].set_title("one force evaluation")
ax[0].legend(frameon=False, fontsize=8.5)

m = np.isfinite(t_np)
ax[1].loglog(N_arr[m], t_np[m] / t_cell[m], "o-", color=C_NP, lw=2, ms=6, label="t_numpy / t_cell")
m = np.isfinite(t_all)
ax[1].loglog(N_arr[m], t_all[m] / t_cell[m], "s-", color=C_ALL, lw=2, ms=6, label="t_numba_allpairs / t_cell")
ax[1].axhline(1.0, color="#52514e", lw=1)
ax[1].set_xlabel("N")
ax[1].set_ylabel("speedup of the cell list")
ax[1].set_title("speedup (> 1 means the cell list is faster)")
ax[1].legend(frameon=False, fontsize=9)
for a in ax:
    a.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("results/force_timing_loglog.png", dpi=130)

# ---- memory: peak of one tm.pair_forces call vs the cell-list arrays ------------
mem_rows = []
for N in NS:
    L = np.sqrt(N / DENSITY)
    pos = np.random.default_rng(N).random((N, 2)) * L
    peak_np = np.nan
    if N <= NMAX_NUMPY:
        tracemalloc.start()
        tm.pair_forces(pos)
        peak_np = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
    tracemalloc.start()
    cf.pair_forces_cell(pos, L, tm.K, tm.R0, tm.RC)
    peak_cell = tracemalloc.get_traced_memory()[1]
    tracemalloc.stop()
    head, nxt, ncell = cf.build_cells(pos, L, tm.RC)
    arrays_cell = head.nbytes + nxt.nbytes + pos.nbytes   # F has the same size as pos
    mem_rows.append([N, 16.0 * N * N, peak_np, peak_cell, arrays_cell])
mem = np.array(mem_rows)
np.savetxt("results/memory.csv", mem, delimiter=",", fmt="%.6g",
           header="N,bytes_d_array_N*N*2*8,peak_bytes_tm_pair_forces_tracemalloc,peak_bytes_cell_tracemalloc,bytes_head+next+F",
           comments="")
m = np.isfinite(mem[:, 2])
per_n2 = np.median(mem[m, 2][mem[m, 0] >= 500] / mem[m, 0][mem[m, 0] >= 500] ** 2)
per_n = np.median(mem[:, 3][mem[:, 0] >= 10000] / mem[:, 0][mem[:, 0] >= 10000])
with open("results/memory_summary.txt", "w") as f:
    f.write("tm.pair_forces peak ~ %.1f bytes * N^2 (N >= 500)\n" % per_n2)
    f.write("cell list peak     ~ %.1f bytes * N   (N >= 10000)\n" % per_n)
    for N in [5000, 7000, 10000, 100000]:
        f.write("N = %6d: numpy predicted %.3g GB, cell list predicted %.3g MB\n" % (N, per_n2 * N ** 2 / 1e9, per_n * N / 1e6))
print(open("results/memory_summary.txt").read())

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.loglog(mem[m, 0], mem[m, 2], "o-", color=C_NP, lw=2, ms=6, label="tm.pair_forces, measured peak")
ax.loglog(mem[:, 0], mem[:, 1], ":", color=C_NP, lw=1.5, label="one (N,N,2) float64 array, 16 N^2")
ax.loglog(mem[:, 0], mem[:, 3], "o-", color=C_CELL, lw=2, ms=6, label="cell list, measured peak")
ax.axhline(3e9, color="#52514e", lw=1, ls="--")
ax.text(14, 3.6e9, "3 GB", color="#52514e", fontsize=9)
ax.set_xlabel("N")
ax.set_ylabel("bytes (tracemalloc peak of one force call)")
ax.set_title("memory of one force evaluation")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("results/memory.png", dpi=130)
print("done")
