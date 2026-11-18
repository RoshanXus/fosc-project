# Task 7: memory of the (N,N,2) array vs N, and the real peak of one vectorized force call
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import toymodel as tm

import subprocess
import time
import tracemalloc
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker

here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(here, "results"), exist_ok=True)
res = lambda name: os.path.join(here, "results", name)
GB = 1e9
BUDGET = 5 * GB          # what we agreed as the limit for one run on the shared machine


def mem_available():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            return int(line.split()[1]) * 1024
    return None


def mem_total():
    for line in open("/proc/meminfo"):
        if line.startswith("MemTotal"):
            return int(line.split()[1]) * 1024


# ---- Part A: theory N^2 * 2 * itemsize vs .nbytes of d from tm.pairwise_distances ----
rows = []
for N in [100, 1000, 5000, 10000]:
    for dtype in [np.float32, np.float64]:
        item = np.dtype(dtype).itemsize
        theory = N * N * 2 * item
        # its peak is ~2.5 x d (d plus d*d), so skip N=10000 float64 which would pass 2.5 GB
        if 2.5 * theory < 2.6 * GB:
            np.random.seed(0)
            pos = (np.random.rand(N, 2) * np.sqrt(N / 0.12)).astype(dtype)
            tracemalloc.start()
            d, dist, nb = tm.pairwise_distances(pos, tm.RC)
            peak = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            nbytes, pd_peak = d.nbytes, peak
            del d, dist, nb
        else:
            nbytes, pd_peak = "", ""
        rows.append([N, np.dtype(dtype).name, item, theory, nbytes, pd_peak])
        print(rows[-1])

with open(res("memory_theory_vs_nbytes.csv"), "w") as fh:
    fh.write("N,dtype,itemsize,theory_bytes,d_nbytes,pairwise_distances_peak_bytes\n")
    for r in rows:
        fh.write(",".join(str(x) for x in r) + "\n")

# ---- Part B: peak memory and time of one full tm.pair_forces call (fresh process each) ----
cases = [(N, "float64") for N in [250, 500, 1000, 2000, 3000, 4000, 5000]]
cases += [(N, "float32") for N in [250, 500, 1000, 2000, 3000, 4000, 5000, 6000, 7000]]
avail_start = mem_available()
out_rows = []
for N, dt in cases:
    out = subprocess.run([sys.executable, os.path.join(here, "peak_one_call.py"), str(N), dt],
                         capture_output=True, text=True, check=True).stdout.split()
    N_, dt_, peak, rss, tmed, tmin, tmax, tpot = out
    d_bytes = int(N_) ** 2 * 2 * np.dtype(dt_).itemsize
    out_rows.append([int(N_), dt_, d_bytes, int(peak), int(rss), int(peak) / d_bytes,
                     float(tmed), float(tmin), float(tmax), float(tpot)])
    print(out_rows[-1])

with open(res("memory_peak_pair_forces.csv"), "w") as fh:
    fh.write("N,dtype,d_bytes,tracemalloc_peak_bytes,VmHWM_growth_bytes,peak_over_d,"
             "t_call_median_s,t_call_min_s,t_call_max_s,t_potential_energy_median_s\n")
    for r in out_rows:
        fh.write(",".join(str(x) for x in r) + "\n")

# ---- Part C: extrapolate to find where it stops being practical ----
avail = mem_available()
lines = [f"MemTotal {mem_total() / GB:.2f} GB, MemAvailable at start of part B {avail_start / GB:.2f} GB, "
         f"at end {avail / GB:.2f} GB (other members were running too)"]
for dt in ["float64", "float32"]:
    r = np.array([x for x in out_rows if x[1] == dt and x[0] >= 2000], dtype=object)
    Ns = r[:, 0].astype(float)
    a_mem = np.mean(r[:, 3].astype(float) / Ns**2)       # peak = a_mem * N^2
    a_t = np.mean((r[:, 6].astype(float)) / Ns**2)        # t_call = a_t * N^2
    a_tp = np.mean((r[:, 9].astype(float)) / Ns**2)
    lines.append(f"\n{dt}: peak ~ {a_mem:.2f} * N^2 bytes, t_force ~ {a_t:.3e} * N^2 s, "
                 f"t_potential ~ {a_tp:.3e} * N^2 s  (fit on N >= 2000)")
    for name, limit in [("5 GB budget", BUDGET), ("MemAvailable now", avail), ("MemTotal", mem_total())]:
        lines.append(f"  peak reaches {name} ({limit / GB:.1f} GB) at N ~ {np.sqrt(limit / a_mem):.0f}")
    for N in [1000, 2000, 5000, 10000, 20000]:
        t1000 = 1000 * (a_t + a_tp) * N**2
        lines.append(f"  N={N:6d}: peak {a_mem * N**2 / GB:7.2f} GB, one force call {a_t * N**2:8.3f} s, "
                     f"1000-step tm.run (force + potential each step) ~ {t1000 / 60:8.1f} min")
# check of the "1000 x (force + potential)" estimate with a real tm.run of 30 steps at N=2000
N, nsteps = 2000, 30
np.random.seed(0)
Lbox = np.sqrt(N / 0.12)
pos = np.random.rand(N, 2) * Lbox
vel = np.random.uniform(-1, 1, (N, 2))
t0 = time.perf_counter()
tm.run(pos, vel, np.ones(N), tm.DT, nsteps, Lbox, method="verlet")
t_step = (time.perf_counter() - t0) / nsteps
pred = [x for x in out_rows if x[0] == N and x[1] == "float64"][0]
lines.append(f"\ncheck: tm.run verlet N={N}, {nsteps} steps: {t_step:.3f} s per step measured, "
             f"force + potential from the table = {pred[6] + pred[9]:.3f} s")
txt = "\n".join(lines)
print(txt)
open(res("memory_limits.txt"), "w").write(txt + "\n")

# ---- figure ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
Nline = np.logspace(2, 4.3, 50)
col = {"float64": "#2a78d6", "float32": "#eb6834"}
for dt in ["float64", "float32"]:
    item = np.dtype(dt).itemsize
    r = [x for x in out_rows if x[1] == dt]
    ax1.plot(Nline, Nline**2 * 2 * item / GB, "--", color=col[dt], lw=1.5, label=f"d only, N²·2·{item} B ({dt})")
    ax1.plot([x[0] for x in r], [x[3] / GB for x in r], "o-", color=col[dt], lw=2, ms=6,
             label=f"measured peak of pair_forces ({dt})")
    ax1.plot([x[0] for x in r], [x[4] / GB for x in r], "+", color="k", ms=8,
             label="peak RSS growth (VmHWM)" if dt == "float32" else None)
    ax2.errorbar([x[0] for x in r], [x[6] for x in r],
                 yerr=[[x[6] - x[7] for x in r], [x[8] - x[6] for x in r]],
                 fmt="o-", color=col[dt], lw=2, ms=6, capsize=3, label=dt)
nb = [x for x in rows if x[4] != ""]
ax1.plot([x[0] for x in nb], [x[4] / GB for x in nb], "kx", ms=9, mew=2, label="d.nbytes from pairwise_distances")
ax1.axhline(BUDGET / GB, color="0.4", lw=1, ls=":")
ax1.text(110, BUDGET / GB * 1.12, "5 GB budget per run", color="0.3", fontsize=8)
ax1.axhline(mem_total() / GB, color="0.4", lw=1, ls="-.")
ax1.text(110, mem_total() / GB * 1.12, f"total RAM {mem_total() / GB:.1f} GB", color="0.3", fontsize=8)
ax1.set_xscale("log"); ax1.set_yscale("log")
ax1.set_xlabel("N"); ax1.set_ylabel("memory (GB)")
ax1.set_title("Memory of the vectorized force calculation")
ax1.legend(fontsize=7.5, loc="lower right")
ax1.grid(True, which="major", alpha=0.3)
ax2.set_xscale("log"); ax2.set_yscale("log")
ax2.set_xticks([250, 500, 1000, 2000, 4000, 7000])
ax2.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
ax2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax2.set_xlabel("N"); ax2.set_ylabel("time per tm.pair_forces call (s)")
ax2.set_title("Time per call (median of 5, bars = min/max)")
ax2.legend(fontsize=8)
ax2.grid(True, which="major", alpha=0.3)
fig.tight_layout()
fig.savefig(res("memory_scaling.png"), dpi=130)
print("saved results/memory_scaling.png")
