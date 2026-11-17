# Task 6: time the pair-force calculation with a Python loop, NumPy and C (-O0, -O2).
# Run compile_c.py first (it builds libpairforce_O0.so / libpairforce_O2.so).
# Writes the raw timings and checks to results/*.csv; task6_plots.py makes the figures.
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import toymodel as tm

import ctypes
import platform
import subprocess
import time
import numpy as np

here = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(here, "results"), exist_ok=True)
res = lambda name: os.path.join(here, "results", name)

K, R0, RC = tm.K, tm.R0, tm.RC
DENSITY = 12 / 10.0**2          # handout: 12 particles in a 10 x 10 box
NS = [50, 200, 500, 1000, 2000, 4000]
NS_PYTHON = [50, 200, 500, 1000]
REPEATS = 7


def make_positions(N, seed=0):
    L = np.sqrt(N / DENSITY)
    np.random.seed(seed)
    return np.random.rand(N, 2) * L, L


# --- C kernel through ctypes ---
libs = {}
for opt in ["O0", "O2"]:
    lib = ctypes.CDLL(os.path.join(here, f"libpairforce_{opt}.so"))
    lib.pair_forces_c.restype = None
    lib.pair_forces_c.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_double,
                                  ctypes.c_double, ctypes.c_double, ctypes.c_void_p]
    libs[opt] = lib


def c_forces(pos, opt):
    # full wrapper: convert to a contiguous float64 array, allocate F, call C
    p = np.ascontiguousarray(pos, dtype=np.float64)
    F = np.empty_like(p)
    libs[opt].pair_forces_c(p.shape[0], p.ctypes.data, K, R0, RC, F.ctypes.data)
    return F


def time_it(fn, repeats=REPEATS, min_time=0.05):
    # like timeit: if one call is very short, each repeat runs it `number` times
    fn()                                    # warm-up
    number = 1
    while True:
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        if time.perf_counter() - t0 >= min_time or number >= 100000:
            break
        number *= 2
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        times.append((time.perf_counter() - t0) / number)
    return np.array(times), number


def time_interleaved(methods, repeats=REPEATS):
    # repeats are interleaved (method A, B, C, A, B, C, ...) so that a burst of load
    # from other users on the shared machine hits all methods, not just one
    numbers = {name: time_it(fn, repeats=1)[1] for name, fn in methods.items()}
    times = {name: [] for name in methods}
    for _ in range(repeats):
        for name, fn in methods.items():
            n = numbers[name]
            t0 = time.perf_counter()
            for _ in range(n):
                fn()
            times[name].append((time.perf_counter() - t0) / n)
    return {name: (np.array(times[name]), numbers[name]) for name in methods}


def pair_forces_xy(pos, k=K, r0=R0, rc=RC):
    # my variant for the bottleneck test: same maths as tm.pair_forces, but x and y are
    # kept in separate (N, N) arrays instead of one (N, N, 2) array
    x, y = pos[:, 0], pos[:, 1]
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist = np.sqrt(dx * dx + dy * dy)
    nb = (dist > 0) & (dist <= rc)
    s = -k * (dist - r0) * nb / (dist + 1e-12)
    return np.stack([np.sum(s * dx, axis=1), np.sum(s * dy, axis=1)], axis=1)


def write_csv(name, header, rows):
    with open(res(name), "w") as fh:
        fh.write(header + "\n")
        for r in rows:
            fh.write(",".join(str(x) for x in r) + "\n")


load_before = os.getloadavg()
timing_rows, check_rows, ctypes_rows, py_rows, np_rows = [], [], [], [], []

for N in NS:
    pos, L = make_positions(N)
    npairs_total = N * (N - 1) // 2
    d, dist, nb = tm.pairwise_distances(pos, RC)
    n_interacting = int(np.sum(np.triu(nb, 1)))

    # ---- same forces? ----
    F_np = tm.pair_forces(pos)
    F_c0 = c_forces(pos, "O0")
    F_c2 = c_forces(pos, "O2")
    F_xy = pair_forces_xy(pos)
    F_loop = tm.pair_forces_loop(pos) if N in NS_PYTHON else None
    row = [N, round(L, 4), n_interacting, round(2 * n_interacting / N, 3),
           np.abs(F_np).max(),
           np.abs(F_np - F_loop).max() if F_loop is not None else "",
           np.abs(F_c0 - F_loop).max() if F_loop is not None else "",
           np.abs(F_c2 - F_loop).max() if F_loop is not None else "",
           np.abs(F_c2 - F_np).max(), np.abs(F_c0 - F_c2).max(), np.abs(F_xy - F_np).max(),
           np.linalg.norm(F_np.sum(axis=0)), np.linalg.norm(F_c2.sum(axis=0))]
    check_rows.append(row)

    # ---- timings ----
    pos_ptr = pos.ctypes.data
    F_buf = np.empty_like(pos)
    F_ptr = F_buf.ctypes.data
    methods = {
        "numpy": lambda: tm.pair_forces(pos),
        "numpy_xy": lambda: pair_forces_xy(pos),
        "c_O0": lambda: libs["O0"].pair_forces_c(N, pos_ptr, K, R0, RC, F_ptr),
        "c_O2": lambda: libs["O2"].pair_forces_c(N, pos_ptr, K, R0, RC, F_ptr),
        "c_O2_wrapper": lambda: c_forces(pos, "O2"),
    }
    if N in NS_PYTHON:
        methods = {"python_loop": lambda: tm.pair_forces_loop(pos), **methods}
    times = time_interleaved(methods)
    med = {}
    for name, (t, number) in times.items():
        med[name] = np.median(t)
        timing_rows.append([name, N, np.median(t), t.min(), t.max(), REPEATS, number])
        print(f"N={N:5d} {name:13s} median {np.median(t):.3e} s  (min {t.min():.3e}, max {t.max():.3e}, number {number})")
    ctypes_rows.append([N, med["c_O2"], med["c_O2_wrapper"], med["c_O2_wrapper"] - med["c_O2"]])

    # ---- NumPy bottleneck: time each line of tm.pair_forces separately ----
    # d: pos[:,None,:]-pos[None,:,:]   dist: sqrt(sum(d*d, axis=2))   neighbors: (dist>0)&(dist<=rc)
    # u: d/(dist[:,:,None]+1e-12)      fmag: -k*(dist-r0)*neighbors   F_sum: sum(fmag[:,:,None]*u, axis=1)
    u = d / (dist[:, :, None] + 1e-12)
    fmag = -K * (dist - R0) * nb
    stages = {
        "d": lambda: pos[:, None, :] - pos[None, :, :],
        "dist": lambda: np.sqrt(np.sum(d * d, axis=2)),
        "neighbors": lambda: (dist > 0) & (dist <= RC),
        "u": lambda: d / (dist[:, :, None] + 1e-12),
        "fmag": lambda: -K * (dist - R0) * nb,
        "F_sum": lambda: np.sum(fmag[:, :, None] * u, axis=1),
    }
    # same first two steps with x and y in separate (N, N) arrays (as in pair_forces_xy)
    x, y = pos[:, 0], pos[:, 1]
    dx, dy = x[:, None] - x[None, :], y[:, None] - y[None, :]
    stages["xy_dx_dy"] = lambda: (x[:, None] - x[None, :], y[:, None] - y[None, :])
    stages["xy_dist"] = lambda: np.sqrt(dx * dx + dy * dy)
    for name, fn in stages.items():
        t, number = time_it(fn, repeats=5)
        np_rows.append([N, name, np.median(t)])

    # ---- Python loop bottleneck ----
    if N in NS_PYTHON:
        def bare_loop():
            for i in range(N):
                for j in range(i + 1, N):
                    pass

        def check_loop():
            for i in range(N):
                for j in range(i + 1, N):
                    if nb[i, j]:
                        pass

        # all four timed in the same interleaved run so the shares add up
        bd = time_interleaved({"full": lambda: tm.pair_forces_loop(pos),
                               "pd": lambda: tm.pairwise_distances(pos, RC),
                               "bare": bare_loop, "check": check_loop})
        t_full, t_pd, t_bare, t_check = [np.median(bd[key][0]) for key in ["full", "pd", "bare", "check"]]
        py_rows.append([N, npairs_total, n_interacting, t_full, t_pd, t_bare, t_check,
                        (t_full - t_pd) / npairs_total * 1e9])

load_after = os.getloadavg()

write_csv("timings.csv", "method,N,median_s,min_s,max_s,repeats,calls_per_repeat", timing_rows)
write_csv("force_check.csv",
          "N,L,interacting_pairs,mean_neighbours,max_abs_F,maxdiff_numpy_vs_loop,maxdiff_cO0_vs_loop,"
          "maxdiff_cO2_vs_loop,maxdiff_cO2_vs_numpy,maxdiff_cO0_vs_cO2,maxdiff_numpyxy_vs_numpy,norm_sumF_numpy,norm_sumF_c", check_rows)
write_csv("ctypes_overhead.csv", "N,c_O2_kernel_call_s,c_O2_with_wrapper_s,wrapper_overhead_s", ctypes_rows)
write_csv("breakdown_python_loop.csv",
          "N,pair_iterations,interacting_pairs,t_pair_forces_loop_s,t_pairwise_distances_s,"
          "t_bare_double_loop_s,t_loop_with_neighbour_check_s,ns_per_pair_iteration", py_rows)
write_csv("breakdown_numpy.csv", "N,stage,median_s", np_rows)

gcc = subprocess.run(["gcc", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
cpu = [l for l in open("/proc/cpuinfo") if l.startswith("model name")][0].split(":")[1].strip()
with open(res("machine_info.txt"), "w") as fh:
    fh.write(f"cpu: {cpu}, {os.cpu_count()} cores (shared with other group members)\n")
    fh.write(f"python {platform.python_version()}, numpy {np.__version__}, {gcc}\n")
    fh.write(f"load average before timing: {load_before}\n")
    fh.write(f"load average after timing:  {load_after}\n")
print(open(res("machine_info.txt")).read())
