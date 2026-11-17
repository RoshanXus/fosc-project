# summarize.py - derived numbers used in the report (reads the csv files, run the benchmarks first)
import os, re
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))

ft = np.genfromtxt("results/force_timing.csv", delimiter=",", names=True)
st = np.genfromtxt("results/sim_timing.csv", delimiter=",", names=True)
txt = open("results/compile_time.txt").read()
t_compile = float(re.search(r"cell list: first call median of 3 = ([0-9.]+)", txt).group(1))

out = []
out.append("cell-list compile time (first call, fresh process): %.3f s" % t_compile)
for N in [12, 100, 1000]:
    r = st[st["N"] == N][0]
    save_per_step = (r["t_numpy_200steps_s"] - r["t_cell_200steps_s"]) / 200
    out.append("N = %4d: Verlet step numpy %.3e s, cell %.3e s, saving %.3e s/step -> compile paid back after %.0f steps"
               % (N, r["t_numpy_200steps_s"] / 200, r["t_cell_200steps_s"] / 200, save_per_step, t_compile / save_per_step))

r12 = open("results/run12_timing.csv").read().splitlines()
t_hand = float(r12[1].split(",")[1])
t_cc = float(r12[3].split(",")[1])
out.append("handout run N=12, 1000 steps: numpy %.4f s, cell force + cell energy %.4f s (ratio %.2f); "
           "with the %.2f s force compile the Numba version needs at least %.2f s in a fresh python" % (t_hand, t_cc, t_hand / t_cc, t_compile, t_cc + t_compile))

# largest N each method can do in a given time per force call (log-log interpolation)
for budget in [0.01, 0.1, 1.0]:
    line = "largest N with one force call <= %.2f s:" % budget
    for col, name in [("t_numpy_s", "numpy"), ("t_numba_allpairs_s", "all-pairs"), ("t_numba_cell_s", "cell")]:
        m = np.isfinite(ft[col])
        n, t = ft["N"][m], ft[col][m]
        if t[-1] <= budget:
            line += "  %s > %d (largest measured)" % (name, n[-1])
        else:
            line += "  %s ~ %.0f" % (name, np.exp(np.interp(np.log(budget), np.log(t), np.log(n))))
    out.append(line)

mem = np.genfromtxt("results/memory.csv", delimiter=",", names=True)
r = mem[mem["N"] == 5000][0]
out.append("memory at N = 5000: tm.pair_forces peak %.4g bytes, cell list peak %.4g bytes, ratio %.0f"
           % (r["peak_bytes_tm_pair_forces_tracemalloc"], r["peak_bytes_cell_tracemalloc"],
              r["peak_bytes_tm_pair_forces_tracemalloc"] / r["peak_bytes_cell_tracemalloc"]))

with open("results/summary.txt", "w") as f:
    f.write("\n".join(out) + "\n")
print("\n".join(out))
