# wall_check.py - how much of the long-run energy drift at gamma = 0 comes from the wall handling?
# compares toymodel.walls_reflect (clamp to the wall, as in the handout) with a mirror reflection
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import numpy as np
import toymodel as tm
import damped as dm

os.makedirs("results", exist_ok=True)
p0, v0, m = tm.init_system(tm.N, tm.L, 0)
with open("results/wall_handling_check.csv", "w") as f:
    f.write("walls,dt,Esh_0,Esh_t100,Esh_t300,Esh_t600,max_abs_drift,steps_pair_set_changed,"
            "dEsh_on_pair_change_steps\n")
    for name, wf in [("clamp (toymodel)", tm.walls_reflect), ("mirror", dm.walls_mirror)]:
        for dt in [0.01, 0.005]:
            n = int(round(600 / dt))
            o = dm.simulate(p0, v0, m, dt, n, tm.L, gamma=0.0, pos_every=n, walls=wf)
            E = o["Esh"]
            ch = o["pairkey"][1:] != o["pairkey"][:-1]
            f.write(f"{name},{dt},{E[0]:.5g},{E[int(round(100 / dt))]:.5g},{E[int(round(300 / dt))]:.5g},"
                    f"{E[-1]:.5g},{np.abs(E - E[0]).max():.5g},{ch.sum()},{np.diff(E)[ch].sum():.5g}\n")
print(open("results/wall_handling_check.csv").read())

# controlled test: 200 random bonded pairs (r between 0.5 and 1.9) thrown at the left wall,
# speed 2-10, dt = 0.01, 0.5 time units (one wall hit each). Free flight + one spring, so the
# only source of energy error besides normal Verlet error is the wall.
rng = np.random.default_rng(1)
res = {"clamp (toymodel)": [], "mirror": []}
for trial in range(200):
    r = rng.uniform(0.5, 1.9)
    ang = rng.uniform(0, 2 * np.pi)
    c = np.array([rng.uniform(0.6, 1.5), 5.0])
    pos = np.array([c + 0.5 * r * np.array([np.cos(ang), np.sin(ang)]), c - 0.5 * r * np.array([np.cos(ang), np.sin(ang)])])
    pos[:, 0] = np.maximum(pos[:, 0], 0.05)
    v = -rng.uniform(2, 10)
    vel = np.array([[v, 0.0], [v, 0.0]]) + rng.normal(0, 0.5, (2, 2))
    for name, wf in [("clamp (toymodel)", tm.walls_reflect), ("mirror", dm.walls_mirror)]:
        o = dm.simulate(pos, vel, np.ones(2), 0.01, 50, tm.L, gamma=0.0, pos_every=50, walls=wf)
        if (o["npairs"] == 1).all():        # keep only cases where the pair stayed bonded (no cutoff crossing)
            res[name].append(o["Esh"][-1] - o["Esh"][0])
with open("results/wall_pair_test.csv", "w") as f:
    f.write("walls,n_cases,mean_dEsh,mean_abs_dEsh,max_abs_dEsh\n")
    for name, d in res.items():
        d = np.array(d)
        f.write(f"{name},{len(d)},{d.mean():.4g},{np.abs(d).mean():.4g},{np.abs(d).max():.4g}\n")
print(open("results/wall_pair_test.csv").read())
