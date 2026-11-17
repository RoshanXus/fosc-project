# dispatch_overhead.py - why the first benchmark gave ~30 us per call even for N = 12
# Same compiled loop, once with default arguments left out by the caller, once with all
# arguments passed explicitly.
import sys, os, timeit
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import toymodel as tm
import numpy as np
from numba import njit
import cellforces as cf

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)


@njit
def allpairs_with_defaults(pos, k=15.0, r0=0.8, rc=2.0):
    return cf.pair_forces_allpairs(pos, k, r0, rc)


pos, vel, mass = tm.init_system(12, 10.0, seed=0)
cell_fn = cf.make_cell_force(10.0)
calls = {
    "tm.pair_forces(pos)  [numpy]": lambda: tm.pair_forces(pos),
    "njit with default args, called as f(pos)": lambda: allpairs_with_defaults(pos),
    "njit all-pairs, all args explicit": lambda: cf.pair_forces_allpairs(pos, 15.0, 0.8, 2.0),
    "njit cell list, all args explicit": lambda: cf.pair_forces_cell(pos, 10.0, 15.0, 0.8, 2.0),
    "cell list through make_cell_force(L) wrapper": lambda: cell_fn(pos),
}
with open("results/dispatch_overhead.csv", "w") as f:
    f.write("call (N = 12),median_time_per_call_us\n")
    for name, fn in calls.items():
        fn()   # compile
        t = np.median(timeit.repeat(fn, number=5000, repeat=7)) / 5000
        f.write("%s,%.3f\n" % (name, t * 1e6))
print(open("results/dispatch_overhead.csv").read())
