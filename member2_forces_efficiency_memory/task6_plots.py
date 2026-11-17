# Task 6: figures, speedup table and log-log slopes from the csv files of task6_timing.py
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
res = lambda name: os.path.join(here, "results", name)


def read_csv(name):
    lines = open(res(name)).read().strip().splitlines()
    head = lines[0].split(",")
    return [dict(zip(head, l.split(","))) for l in lines[1:]]


rows = read_csv("timings.csv")
T = {}   # T[method][N] = (median, min, max)
for r in rows:
    T.setdefault(r["method"], {})[int(r["N"])] = (float(r["median_s"]), float(r["min_s"]), float(r["max_s"]))

style = {   # label, colour, marker
    "python_loop": ("Python double loop (tm.pair_forces_loop)", "#2a78d6", "o"),
    "numpy": ("NumPy vectorized (tm.pair_forces)", "#eb6834", "s"),
    "c_O0": ("C, gcc -O0 (via ctypes)", "#1baf7a", "^"),
    "c_O2": ("C, gcc -O2 (via ctypes)", "#008300", "v"),
    "numpy_xy": ("NumPy, x/y split (my variant)", "#e87ba4", "D"),
}
methods = ["python_loop", "numpy", "c_O0", "c_O2", "numpy_xy"]

# ---- log-log slopes ----
slope_lines = ["method,slope_N>=200,slope_all_N,N_used"]
slopes = {}
for m in methods:
    Ns = np.array(sorted(T[m]))
    t = np.array([T[m][n][0] for n in Ns])
    sel = Ns >= 200
    s200 = np.polyfit(np.log(Ns[sel]), np.log(t[sel]), 1)[0]
    sall = np.polyfit(np.log(Ns), np.log(t), 1)[0]
    slopes[m] = s200
    slope_lines.append(f"{m},{s200:.3f},{sall:.3f},{' '.join(str(n) for n in Ns)}")
open(res("loglog_slopes.csv"), "w").write("\n".join(slope_lines) + "\n")
print("\n".join(slope_lines))

# ---- figure 1: wall time vs N ----
fig, ax = plt.subplots(figsize=(7.5, 5))
for m in methods:
    lab, c, mk = style[m]
    Ns = sorted(T[m])
    med = np.array([T[m][n][0] for n in Ns])
    lo = med - np.array([T[m][n][1] for n in Ns])
    hi = np.array([T[m][n][2] for n in Ns]) - med
    ls = "--" if m == "numpy_xy" else "-"
    ax.errorbar(Ns, med, yerr=[lo, hi], fmt=mk + ls, color=c, lw=2, ms=7, capsize=3,
                label=f"{lab}, slope {slopes[m]:.2f}")
Nref = np.array([200, 4000])
ax.plot(Nref, 6e-4 * (Nref / 1000.0) ** 2, ":", color="0.4", lw=1.5, label="N² reference")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("number of particles N")
ax.set_ylabel("wall time per force call (s)")
ax.set_title("Pair-force wall time vs N (median of 7, bars = min/max)")
ax.grid(True, which="major", alpha=0.3)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
fig.savefig(res("time_vs_N.png"), dpi=130)

# ---- speedup table + figure 2 ----
NS_PY = sorted(T["python_loop"])
sp_lines = ["N,t_python_s,speedup_numpy,speedup_numpy_xy,speedup_c_O0,speedup_c_O2"]
SP = {m: [] for m in ["numpy", "numpy_xy", "c_O0", "c_O2"]}
for n in NS_PY:
    tp = T["python_loop"][n][0]
    for m in SP:
        SP[m].append(tp / T[m][n][0])
    sp_lines.append(f"{n},{tp:.4e}," + ",".join(f"{SP[m][-1]:.2f}" for m in SP))
open(res("speedup.csv"), "w").write("\n".join(sp_lines) + "\n")
print("\n".join(sp_lines))

fig, ax = plt.subplots(figsize=(7.5, 4.6))
w = 0.2
x = np.arange(len(NS_PY))
for i, m in enumerate(["numpy", "numpy_xy", "c_O0", "c_O2"]):
    lab, c, mk = style[m]
    bars = ax.bar(x + (i - 1.5) * w, SP[m], width=w * 0.92, color=c, label=lab)
    for b, v in zip(bars, SP[m]):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.08, f"{v:.0f}" if v >= 10 else f"{v:.1f}",
                ha="center", va="bottom", fontsize=7)
ax.axhline(1, color="0.3", lw=1)
ax.set_yscale("log")
ax.set_xticks(x)
ax.set_xticklabels([f"N = {n}" for n in NS_PY])
ax.set_ylabel("speedup = t_Python / t_method")
ax.set_ylim(0.5, 1000)
ax.set_title("Speedup relative to the Python double loop")
ax.legend(fontsize=8, loc="upper left")
ax.grid(True, axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(res("speedup.png"), dpi=130)

# ---- figure 3: where the time goes ----
bn = read_csv("breakdown_numpy.csv")
stages = ["d", "dist", "neighbors", "u", "fmag", "F_sum"]
stage_lab = {"d": "d (N,N,2)", "dist": "dist = sqrt(sum(d*d))", "neighbors": "neighbour mask",
             "u": "u = d/dist", "fmag": "fmag", "F_sum": "F = sum(fmag*u)"}
stage_col = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
NS_ALL = sorted({int(r["N"]) for r in bn})
B = {(int(r["N"]), r["stage"]): float(r["median_s"]) for r in bn}

py = read_csv("breakdown_python_loop.csv")
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
for k, n in enumerate(NS_ALL):
    tot = sum(B[(n, s)] for s in stages)
    left = 0
    for s, c in zip(stages, stage_col):
        frac = 100 * B[(n, s)] / tot
        a1.barh(k, frac, left=left, color=c, edgecolor="white", linewidth=2,
                label=stage_lab[s] if k == 0 else None)
        if frac > 7:
            a1.text(left + frac / 2, k, f"{frac:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        left += frac
a1.set_yticks(range(len(NS_ALL)))
a1.set_yticklabels([f"N={n}" for n in NS_ALL])
a1.set_xlabel("share of NumPy force time (%)")
a1.set_title("NumPy tm.pair_forces: time per line")
a1.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3)
a1.set_xlim(0, 100)

parts_col = ["#eb6834", "#2a78d6", "#eda100", "#1baf7a"]
parts_lab = ["pairwise_distances (NumPy)", "bare Python double loop", "neighbors[i, j] lookups",
             "force update for interacting pairs"]
for k, r in enumerate(py):
    tot = float(r["t_pair_forces_loop_s"])
    t_pd = float(r["t_pairwise_distances_s"])
    t_bare = float(r["t_bare_double_loop_s"])
    t_chk = float(r["t_loop_with_neighbour_check_s"])
    parts = [t_pd, t_bare, t_chk - t_bare, max(tot - t_pd - t_chk, 0)]
    left = 0
    for p, c, lab in zip(parts, parts_col, parts_lab):
        frac = 100 * p / tot
        a2.barh(k, frac, left=left, color=c, edgecolor="white", linewidth=2, label=lab if k == 0 else None)
        if frac > 7:
            a2.text(left + frac / 2, k, f"{frac:.0f}%", ha="center", va="center", fontsize=7.5, color="white")
        left += frac
a2.set_yticks(range(len(py)))
a2.set_yticklabels([f"N={r['N']}" for r in py])
a2.set_xlabel("share of tm.pair_forces_loop time (%)")
a2.set_title("Python loop: where the time goes")
a2.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2)
a2.set_xlim(0, 100)
fig.tight_layout()
fig.savefig(res("bottleneck_breakdown.png"), dpi=130)
print("saved time_vs_N.png, speedup.png, bottleneck_breakdown.png")

# ---- cost per pair: t / (N(N-1)/2) for every method, in ns ----
lines = ["N," + ",".join(f"ns_per_pair_{m}" for m in methods)]
for n in sorted(T["numpy"]):
    npairs = n * (n - 1) / 2
    lines.append(f"{n}," + ",".join(f"{T[m][n][0] / npairs * 1e9:.2f}" if n in T[m] else "" for m in methods))
open(res("cost_per_pair.csv"), "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
