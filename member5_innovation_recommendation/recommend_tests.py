# Task 9: checks of smoothcut.py, E_shift drift vs dt for {hard, smooth} x {clamp, mirror}, large-N run and cost
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import toymodel as tm
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cellforces as cf
import smoothcut as sc

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("results", exist_ok=True)
K, R0, RC, RS = 15.0, 0.8, 2.0, 1.5
out = open("results/recommend_checks.txt", "w")


def say(line):
    print(line)
    out.write(line + "\n")


# ---- 1. checks -----------------------------------------------------------------
N = 1000
L = np.sqrt(N / 0.12)
p = np.random.default_rng(3).random((N, 2)) * L
F_h, U_h, np_h = sc.forces_energy(p, L, K, R0, RC, RS, False)
nb = tm.pairwise_distances(p, RC)[2]
say("hard version vs tm.pair_forces, N=1000: max |dF| = %.2e" % np.max(np.abs(F_h - tm.pair_forces(p))))
say("hard version U_shift vs tm.potential_energy - 10.8*npairs: diff = %.2e (npairs %d vs %d)" % (
    U_h - (tm.potential_energy(p) - 10.8 * np.triu(nb, 1).sum()), np_h, np.triu(nb, 1).sum()))
F_s, U_s, _ = sc.forces_energy(p, L, K, R0, RC, RS, True)
h = 1e-6
fd = np.zeros_like(p)
for i in range(0, N, 50):                 # finite differences on every 50th particle
    for a in range(2):
        pp = p.copy(); pp[i, a] += h
        pm = p.copy(); pm[i, a] -= h
        fd[i, a] = -(sc.forces_energy(pp, L, K, R0, RC, RS, True)[1] - sc.forces_energy(pm, L, K, R0, RC, RS, True)[1]) / (2 * h)
say("smooth version: max |F - (-dU/dx finite diff)| on 20 particles = %.2e" % np.max(np.abs(F_s[::50] - fd[::50])))
for r in [RS - 1e-9, RS + 1e-9, RC - 1e-9]:
    u, f = sc.pair_term(r, K, R0, RC, RS, True)
    say("smooth pair term at r = %.9f: U = %.6f, F = %.6f" % (r, u, f))
pos, vel, mass = tm.init_system(12, 10.0, seed=0)
ref = tm.run(pos, vel, mass, 0.01, 1000, 10.0)
t, E, npairs, pfin = sc.run_verlet(pos, vel, 0.01, 1000, 10.0, False, False)
say("hard+clamp stepper vs tm.run, N=12 dt=0.01: max |pos diff| final = %.2e, E_shift drift = %.4f" % (
    np.max(np.abs(pfin - ref["pos"][-1])), np.max(np.abs(E - E[0])) / abs(E[0])))

# ---- 2. drift vs dt -------------------------------------------------------------
models = {"hard cutoff + clamp (handout)": (False, False), "hard cutoff + mirror": (False, True),
          "smooth cutoff + clamp": (True, False), "smooth cutoff + mirror": (True, True)}
dts = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
seeds = range(5)
rows = []
for name, (smooth, mirror) in models.items():
    for dt in dts:
        nsteps = int(round(10.0 / dt))
        for seed in seeds:
            pos, vel, mass = tm.init_system(12, 10.0, seed=seed)
            t, E, npairs, pfin = sc.run_verlet(pos, vel, dt, nsteps, 10.0, smooth, mirror)
            eps = np.max(np.abs(E - E[0])) / abs(E[0])
            rows.append((name, dt, seed, eps, E[0], npairs[0], npairs[-1], np.sum(npairs[1:] != npairs[:-1])))
with open("results/dt_study_models.csv", "w") as f:
    f.write("model,dt,seed,eps_Eshift_max,Eshift0,npairs0,npairs_final,steps_with_pair_change\n")
    for r in rows:
        f.write("%s,%g,%d,%.5g,%.4f,%d,%d,%d\n" % r)

summary = []
for name in models:
    for dt in dts:
        e = [r[3] for r in rows if r[0] == name and r[1] == dt]
        npf = [r[6] for r in rows if r[0] == name and r[1] == dt]
        summary.append((name, dt, np.median(e), np.max(e), np.median(npf)))
with open("results/dt_study_models_summary.csv", "w") as f:
    f.write("model,dt,median_eps_5_seeds,max_eps_5_seeds,median_npairs_final\n")
    for r in summary:
        f.write("%s,%g,%.4g,%.4g,%g\n" % r)
for name in models:
    s = [r for r in summary if r[0] == name]
    for target in [0.01, 0.05]:
        ok = [r[1] for r in s if r[3] < target]
        say("%s: largest dt with worst-seed drift < %g%%: %s" % (name, target * 100, max(ok) if ok else "none"))
    d = np.array([r[1] for r in s if r[1] <= 0.01])
    e = np.array([r[3] for r in s if r[1] <= 0.01])
    say("%s: log-log slope of worst-seed drift for dt <= 0.01: %.2f" % (name, np.polyfit(np.log(d), np.log(e), 1)[0]))

fig, ax = plt.subplots(figsize=(6.8, 4.6))
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6"]
for c, name in zip(colors, models):
    s = [r for r in summary if r[0] == name]
    ax.loglog([r[1] for r in s], [r[3] for r in s], "o-", color=c, lw=2, ms=6, label=name)
ax.axhline(0.01, color="#52514e", lw=1, ls="--")
ax.text(0.025, 0.0075, "1% target", color="#52514e", fontsize=9)
ax.set_xlabel("dt")
ax.set_ylabel("worst of 5 seeds: max |E_shift - E_shift(0)| / |E_shift(0)|")
ax.set_title("N = 12, Verlet, t = 10")
ax.legend(frameon=False, fontsize=9)
ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig("results/dt_study_models.png", dpi=130)

# ---- 3. large N at fixed density, cell list -------------------------------------
with open("results/large_N_check.csv", "w") as f:
    f.write("N,model,dt,steps,eps_Eshift_max,npairs0,npairs_final,median_time_per_step_s\n")
    for N in [1000, 10000]:
        L = np.sqrt(N / 0.12)
        pos, vel, mass = tm.init_system(N, L, seed=0)
        for name, dt in [("hard cutoff + clamp (handout)", 0.01), ("hard cutoff + clamp (handout)", 0.0005),
                         ("smooth cutoff + mirror", 0.01), ("smooth cutoff + mirror", 0.02)]:
            smooth, mirror = models[name]
            nsteps = int(round(10.0 / dt))
            times = []
            for rep in range(3 if nsteps * N <= 2e7 else 1):
                t0 = time.perf_counter()
                t, E, npairs, pfin = sc.run_verlet(pos, vel, dt, nsteps, L, smooth, mirror)
                times.append((time.perf_counter() - t0) / nsteps)
            eps = np.max(np.abs(E - E[0])) / abs(E[0])
            f.write("%d,%s,%g,%d,%.4g,%d,%d,%.3e\n" % (N, name, dt, nsteps, eps, npairs[0], npairs[-1], np.median(times)))
print(open("results/large_N_check.csv").read())

# ---- 4. cost: float32 vs float64, smooth vs hard ----------------------------------
def med(fn, number):
    fn()
    ts = []
    for r in range(7):
        t0 = time.perf_counter()
        for _ in range(number):
            fn()
        ts.append((time.perf_counter() - t0) / number)
    return np.median(ts)


with open("results/cell_precision_cost.csv", "w") as f:
    f.write("N,t_cell_float64_s,t_cell_float32_s,t64_over_t32,t_smooth_force_energy_s,t_hard_force_energy_s\n")
    for N, number in [(12, 20000), (1000, 200), (100000, 5), (1000000, 1)]:
        L = np.sqrt(N / 0.12)
        p64 = np.random.default_rng(N).random((N, 2)) * L
        p32 = p64.astype(np.float32)
        t64 = med(lambda: cf.pair_forces_cell(p64, L, K, R0, RC), number)
        t32 = med(lambda: cf.pair_forces_cell(p32, np.float32(L), np.float32(K), np.float32(R0), np.float32(RC)), number)
        ts = med(lambda: sc.forces_energy(p64, L, K, R0, RC, RS, True), number)
        th = med(lambda: sc.forces_energy(p64, L, K, R0, RC, RS, False), number)
        f.write("%d,%.4g,%.4g,%.3f,%.4g,%.4g\n" % (N, t64, t32, t64 / t32, ts, th))
print(open("results/cell_precision_cost.csv").read())
out.close()
