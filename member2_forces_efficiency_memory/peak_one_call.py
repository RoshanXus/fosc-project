# Helper for task7_memory.py: measures ONE configuration in a fresh process so that the
# memory numbers are not mixed up with earlier runs.
# usage: python3 peak_one_call.py N dtype     (dtype = float32 or float64)
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import toymodel as tm

import time
import tracemalloc
import numpy as np

N = int(sys.argv[1])
dtype = np.dtype(sys.argv[2])
L = np.sqrt(N / 0.12)                     # handout density
np.random.seed(0)
pos = (np.random.rand(N, 2) * L).astype(dtype)


def vm_hwm():
    # peak resident memory of this process in bytes. (ru_maxrss from getrusage did not work:
    # it is inherited from the parent process, which had already used ~2 GB in part A)
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) * 1024

# warm-up call on a tiny system so imports/caches are not counted
tm.pair_forces(pos[:10])
rss_before = vm_hwm()

tracemalloc.start()
F = tm.pair_forces(pos)
current, peak = tracemalloc.get_traced_memory()
tracemalloc.stop()
rss_after = vm_hwm()
assert F.dtype == dtype

# time per call without tracemalloc running (it slows allocations down)
repeats = 5
t_force, t_pot = [], []
for _ in range(repeats):
    t0 = time.perf_counter()
    tm.pair_forces(pos)
    t_force.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    tm.potential_energy(pos)
    t_pot.append(time.perf_counter() - t0)

print(N, dtype.name, peak, rss_after - rss_before,
      np.median(t_force), min(t_force), max(t_force), np.median(t_pot))
