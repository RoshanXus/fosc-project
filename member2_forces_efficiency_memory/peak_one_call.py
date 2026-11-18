# Task 7 helper: one N/dtype per fresh process so peaks do not mix (args: N float32|float64)
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
    # peak RSS in bytes; not ru_maxrss, which is inherited from the ~2 GB parent process
    for line in open("/proc/self/status"):
        if line.startswith("VmHWM"):
            return int(line.split()[1]) * 1024

# warm-up call on a tiny system so imports/caches are not counted
tm.pair_forces(pos[:10])
rss_before = vm_hwm()

tracemalloc.start()
F = tm.pair_forces(pos)
_, peak = tracemalloc.get_traced_memory()
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
