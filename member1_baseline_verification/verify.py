# Task 1: verification checks for toymodel.py, written to results/
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # results/ is relative to this folder
import ctypes
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import toymodel as tm

os.makedirs("results", exist_ok=True)
lines = []
checks = []   # rows for results/checks.csv


def say(*args):
    s = " ".join(str(a) for a in args)
    print(s)
    lines.append(s)


def check(name, value, target, ok):
    checks.append((name, value, target, "PASS" if ok else "FAIL"))
    say(f"  {name}: {value}   (target {target})  -> {'PASS' if ok else 'FAIL'}")


# ---------------------------------------------------------------- 1. memory layout
say("=== 1. dtype, itemsize and nbytes (handout section 1) ===")
pos, vel, mass = tm.init_system(tm.N, tm.L, seed=0)
pos32 = pos.astype(np.float32)
say("pos.shape", pos.shape, "vel.shape", vel.shape, "mass.shape", mass.shape)
say(f"float64 pos: dtype={pos.dtype}, itemsize={pos.itemsize}, nbytes={pos.nbytes}")
say(f"float32 pos: dtype={pos32.dtype}, itemsize={pos32.itemsize}, nbytes={pos32.nbytes}")
say(f"C on this machine (ctypes): sizeof(double)={ctypes.sizeof(ctypes.c_double)}, "
    f"sizeof(float)={ctypes.sizeof(ctypes.c_float)}")
say(f"nbytes ratio float64/float32 = {pos.nbytes / pos32.nbytes}")
check("itemsize float64 == sizeof(double)", pos.itemsize, ctypes.sizeof(ctypes.c_double),
      pos.itemsize == ctypes.sizeof(ctypes.c_double))
check("itemsize float32 == sizeof(float)", pos32.itemsize, ctypes.sizeof(ctypes.c_float),
      pos32.itemsize == ctypes.sizeof(ctypes.c_float))

# ---------------------------------------------------------------- 2. distance matrix
say("\n=== 2. distance matrix and neighbour mask, N=12 initial positions ===")
d, dist, nb = tm.pairwise_distances(pos, tm.RC)
say("shapes d, dist, neighbors:", d.shape, dist.shape, nb.shape)
check("max |dist - dist.T|", np.abs(dist - dist.T).max(), 0.0, np.abs(dist - dist.T).max() == 0)
check("max |d + d.transpose(1,0,2)|", np.abs(d + d.transpose(1, 0, 2)).max(), 0.0,
      np.abs(d + d.transpose(1, 0, 2)).max() == 0)
check("max |diag(dist)|", np.abs(np.diag(dist)).max(), 0.0, np.abs(np.diag(dist)).max() == 0)
check("number of True on diag(neighbors)", int(np.diag(nb).sum()), 0, not np.diag(nb).any())
check("neighbors symmetric (count of nb != nb.T)", int((nb != nb.T).sum()), 0, (nb == nb.T).all())
# compare with a slow double loop
dist_loop = np.zeros((tm.N, tm.N))
for i in range(tm.N):
    for j in range(tm.N):
        dist_loop[i, j] = np.sqrt((pos[i, 0] - pos[j, 0]) ** 2 + (pos[i, 1] - pos[j, 1]) ** 2)
check("max |dist - dist_loop|", np.abs(dist - dist_loop).max(), "< 1e-12", np.abs(dist - dist_loop).max() < 1e-12)
say("neighbour count per particle (rc=2):", np.sum(nb, axis=1).tolist())
say("number of neighbour pairs (i<j):", int(np.triu(nb, 1).sum()))

# ---------------------------------------------------------------- 3. hand-made 4-particle example
say("\n=== 3. hand-made 4-particle example ===")
p4 = np.array([[1.0, 1.0],    # A
               [3.0, 1.0],    # B
               [1.0, 3.5],    # C
               [4.0, 3.0]])   # D
# by hand: AB = 2 (exactly rc), AC = 2.5, BD = 2.236, CD = 3.041, BC = 3.202, AD = 3.606
hand = {2.0: [1, 1, 0, 0],   # only AB (AB = 2.0 is included because the test is dist <= rc)
        3.0: [2, 2, 1, 1]}   # AB, AC, BD  (CD = 3.041 is just outside)
d4, dist4, nb4 = tm.pairwise_distances(p4, 2.0)
say("distance matrix:")
say(np.array2string(dist4, precision=4))
check("4p max |dist - dist.T|", np.abs(dist4 - dist4.T).max(), 0.0, np.abs(dist4 - dist4.T).max() == 0)
check("4p max |diag(dist)|", np.abs(np.diag(dist4)).max(), 0.0, np.abs(np.diag(dist4)).max() == 0)
for rc in (2.0, 3.0):
    _, _, nb4 = tm.pairwise_distances(p4, rc)
    counts = np.sum(nb4, axis=1).tolist()
    check(f"4p neighbour counts rc={rc}", counts, hand[rc], counts == hand[rc])
    check(f"4p self-neighbours rc={rc}", int(np.diag(nb4).sum()), 0, not np.diag(nb4).any())
# force by hand for rc = 2: only pair AB, r = 2, |F| = 15*(2-0.8) = 18, attractive
F4 = tm.pair_forces(p4)
F4_hand = np.array([[18.0, 0.0], [-18.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
say("vectorized forces on 4 particles (rc=2):")
say(np.array2string(F4, precision=6))
check("4p max |F - F_hand|", np.abs(F4 - F4_hand).max(), "< 1e-9", np.abs(F4 - F4_hand).max() < 1e-9)
check("4p max |F_loop - F_hand|", np.abs(tm.pair_forces_loop(p4) - F4_hand).max(), "~0",
      np.abs(tm.pair_forces_loop(p4) - F4_hand).max() < 1e-9)
U4 = tm.potential_energy(p4)
check("4p potential energy (hand: 0.5*15*1.2^2 = 10.8)", U4, 10.8, abs(U4 - 10.8) < 1e-12)

# ---------------------------------------------------------------- 4. 200 particles in a 50x50 box
say("\n=== 4. 200 random particles in a 50x50 box ===")
np.random.seed(1)
pos_test = np.random.rand(200, 2) * 50      # separate array, does not touch pos
rows200 = []
for rc in (2.0, 3.0, 5.0):
    _, dist2, nb2 = tm.pairwise_distances(pos_test, rc)
    avg = nb2.sum(axis=1).mean()
    est = 199 / 50.0 ** 2 * np.pi * rc ** 2       # density of the other particles times circle area
    Fv = tm.pair_forces(pos_test, rc=rc)
    Fl = tm.pair_forces_loop(pos_test, rc=rc)
    diff = np.abs(Fv - Fl).max()
    rows200.append((rc, avg, est, avg / est, int(np.diag(nb2).sum()),
                    np.abs(dist2 - dist2.T).max(), diff, np.linalg.norm(Fv.sum(axis=0))))
    say(f"  rc={rc}: avg neighbours {avg:.3f}, estimate rho*pi*rc^2 = {est:.3f}, ratio {avg/est:.3f}, "
        f"self-neighbours {int(np.diag(nb2).sum())}, max|F_vec-F_loop| = {diff:.3e}, "
        f"||sum F|| = {np.linalg.norm(Fv.sum(axis=0)):.3e}")
np.savetxt("results/neighbours_200.csv", np.array(rows200), delimiter=",", fmt="%.6g",
           header="rc,avg_neighbours,estimate_rho_pi_rc2,ratio,self_neighbours,max_asym_dist,"
                  "max_abs_Fvec_minus_Floop,norm_sum_F", comments="")

# ---------------------------------------------------------------- 5. loop vs vectorized force
say("\n=== 5. loop vs vectorized pair force, N=12 initial positions ===")
F_vec = tm.pair_forces(pos)
F_loop = tm.pair_forces_loop(pos)
check("max |F_vec - F_loop|", np.abs(F_vec - F_loop).max(), "< 1e-9", np.abs(F_vec - F_loop).max() < 1e-9)
say("  largest single force component |F|max =", np.abs(F_vec).max())
say("  ||sum_i F_i|| vec =", np.linalg.norm(F_vec.sum(axis=0)), " loop =", np.linalg.norm(F_loop.sum(axis=0)))
say("  forces on first 3 particles:")
say(np.array2string(F_vec[:3], precision=5))

# ---------------------------------------------------------------- 6. wall tests
say("\n=== 6. wall reflection tests ===")
L = tm.L
p1 = np.array([[L + 0.1, 5.0]])
v1 = np.array([[1.0, 0.3]])
tm.walls_reflect(p1, v1, L)
say(f"  handout test: start [L+0.1, 5], v=[1, 0.3] -> pos {p1[0].tolist()}, vel {v1[0].tolist()}")
check("handout wall test x == L and vx < 0", (float(p1[0, 0]), float(v1[0, 0])), (L, "<0"), p1[0, 0] == L and v1[0, 0] < 0)
cases = [("left",   [-0.1, 5.0],     [-0.7, 0.2]),
         ("right",  [L + 0.1, 5.0],  [0.7, 0.2]),
         ("bottom", [5.0, -0.1],     [0.2, -0.7]),
         ("top",    [5.0, L + 0.1],  [0.2, 0.7]),
         ("corner", [-0.2, L + 0.3], [-0.5, 0.9]),
         ("inside", [3.0, 4.0],      [0.5, -0.5])]
expect = {"left": ([0, 5], [0.7, 0.2]), "right": ([L, 5], [-0.7, 0.2]),
          "bottom": ([5, 0], [0.2, 0.7]), "top": ([5, L], [0.2, -0.7]),
          "corner": ([0, L], [0.5, -0.9]), "inside": ([3, 4], [0.5, -0.5])}
pw = np.array([c[1] for c in cases])
vw = np.array([c[2] for c in cases])
pw0, vw0 = pw.copy(), vw.copy()
tm.walls_reflect(pw, vw, L)
wall_rows = []
for k, (name, _, _) in enumerate(cases):
    ok = np.array_equal(pw[k], expect[name][0]) and np.array_equal(vw[k], expect[name][1])
    wall_rows.append(f"{name},{pw0[k,0]},{pw0[k,1]},{vw0[k,0]},{vw0[k,1]},{pw[k,0]},{pw[k,1]},{vw[k,0]},{vw[k,1]},{ok}")
    check(f"wall {name}: pos {pw0[k].tolist()} -> {pw[k].tolist()}, vel {vw0[k].tolist()} -> {vw[k].tolist()}",
          ok, True, ok)
with open("results/wall_tests.csv", "w") as f:
    f.write("case,x0,y0,vx0,vy0,x1,y1,vx1,vy1,matches_hand\n" + "\n".join(wall_rows) + "\n")

# ---------------------------------------------------------------- 7. full run checks
say("\n=== 7. full run: N=12, L=10, dt=0.01, 1000 steps, Verlet ===")


def full_run(force_fn, method="verlet", dtype=np.float64):
    p, v, m = tm.init_system(tm.N, tm.L, seed=0, dtype=dtype)
    F = None
    sumF, hits, frames = [], np.zeros(4, dtype=int), [p.copy()]
    for s in range(tm.NSTEPS):
        if method == "verlet":
            if F is None:
                F = force_fn(p)
                sumF.append(np.linalg.norm(F.sum(axis=0)))
            p, v, F = tm.step_verlet(p, v, m, tm.DT, tm.L, F, force_fn)
            sumF.append(np.linalg.norm(F.sum(axis=0)))
        else:
            sumF.append(np.linalg.norm(force_fn(p).sum(axis=0)))
            p, v = tm.step_euler(p, v, m, tm.DT, tm.L, force_fn)
        # a clamped particle sits exactly on the wall after the step, so this counts wall hits
        hits += [(p[:, 0] == 0).sum(), (p[:, 0] == tm.L).sum(), (p[:, 1] == 0).sum(), (p[:, 1] == tm.L).sum()]
        frames.append(p.copy())
    return np.array(frames), np.array(sumF), hits


frames_vec, sumF_vec, hits_vec = full_run(tm.pair_forces)
frames_loop, sumF_loop, hits_loop = full_run(tm.pair_forces_loop)
frames_eu, sumF_eu, hits_eu = full_run(tm.pair_forces, "euler")
frames_32, sumF_32, hits_32 = full_run(tm.pair_forces, "verlet", np.float32)

# typical force size, to judge what "approximately zero" means
Fmax_run = max(np.abs(tm.pair_forces(f)).max() for f in frames_vec)
check("max over run ||sum F|| (vectorized, float64)", f"{sumF_vec.max():.3e}", "~1e-14", sumF_vec.max() < 1e-10)
check("max over run ||sum F|| (loop, float64)", f"{sumF_loop.max():.3e}", "~1e-14", sumF_loop.max() < 1e-10)
check("max over run ||sum F|| (vectorized, float32)", f"{sumF_32.max():.3e}", "~1e-5", sumF_32.max() < 1e-3)
check("max over run ||sum F|| (Euler, float64)", f"{sumF_eu.max():.3e}", "~1e-14", sumF_eu.max() < 1e-10)
say(f"  for scale: largest force component seen in the run = {Fmax_run:.3f}")
say(f"  mean ||sum F|| over run: vec {sumF_vec.mean():.3e}, loop {sumF_loop.mean():.3e}, float32 {sumF_32.mean():.3e}")
check("max |pos_vec - pos_loop| over whole run", f"{np.abs(frames_vec - frames_loop).max():.3e}", "small",
      np.abs(frames_vec - frames_loop).max() < 1e-6)

walls = ["left x=0", "right x=L", "bottom y=0", "top y=L"]
box_rows = []
for name, fr, hits in [("verlet_f64", frames_vec, hits_vec), ("verlet_loop_f64", frames_loop, hits_loop),
                       ("euler_f64", frames_eu, hits_eu), ("verlet_f32", frames_32, hits_32)]:
    xmin, xmax = fr[:, :, 0].min(), fr[:, :, 0].max()
    ymin, ymax = fr[:, :, 1].min(), fr[:, :, 1].max()
    outside = int(((fr < 0) | (fr > tm.L)).sum())
    box_rows.append(f"{name},{xmin},{xmax},{ymin},{ymax},{outside},{hits[0]},{hits[1]},{hits[2]},{hits[3]},{hits.sum()}")
    check(f"{name}: x in [{xmin:.4g}, {xmax:.4g}], y in [{ymin:.4g}, {ymax:.4g}], coords outside box",
          outside, 0, outside == 0)
    say("     wall hits: " + ", ".join(f"{w}: {h}" for w, h in zip(walls, hits)) + f", total {hits.sum()}")
with open("results/box_check.csv", "w") as f:
    f.write("run,xmin,xmax,ymin,ymax,coords_outside,hits_left,hits_right,hits_bottom,hits_top,hits_total\n")
    f.write("\n".join(box_rows) + "\n")

# symmetry and self-neighbour checks on every frame of the run
asym = max(np.abs(tm.pairwise_distances(f, tm.RC)[1] - tm.pairwise_distances(f, tm.RC)[1].T).max() for f in frames_vec)
diag = max(np.abs(np.diag(tm.pairwise_distances(f, tm.RC)[1])).max() for f in frames_vec)
selfn = sum(int(np.diag(tm.pairwise_distances(f, tm.RC)[2]).sum()) for f in frames_vec)
check("all 1001 frames: max |dist - dist.T|", asym, 0.0, asym == 0)
check("all 1001 frames: max |diag(dist)|", diag, 0.0, diag == 0)
check("all 1001 frames: total self-neighbour entries", selfn, 0, selfn == 0)

# ---------------------------------------------------------------- 8. API / dtype checks
say("\n=== 8. API checks: inputs not modified, float32 stays float32 ===")
p, v, m = tm.init_system(dtype=np.float32)
p0, v0 = p.copy(), v.copy()
res = tm.run(p, v, m, np.float64(0.01), 100, tm.L)
check("run does not modify input pos/vel", bool(np.array_equal(p, p0) and np.array_equal(v, v0)), True,
      np.array_equal(p, p0) and np.array_equal(v, v0))
dts = {"pairwise dist": tm.pairwise_distances(p, tm.RC)[1].dtype, "pair_forces": tm.pair_forces(p).dtype,
       "pair_forces_loop": tm.pair_forces_loop(p).dtype, "potential_energy": tm.potential_energy(p).dtype,
       "kinetic_energy": tm.kinetic_energy(v, m).dtype, "step_euler pos": tm.step_euler(p, v, m, 0.01, tm.L)[0].dtype,
       "step_verlet pos": tm.step_verlet(p, v, m, 0.01, tm.L)[0].dtype, "run pos": res["pos"].dtype,
       "run E": res["E"].dtype}
for kname, dt_ in dts.items():
    check(f"float32 in -> {kname} dtype", str(dt_), "float32", dt_ == np.float32)

# ---------------------------------------------------------------- 9. energy note (cutoff)
say("\n=== 9. energy bookkeeping and the cutoff ===")
res = tm.run(*tm.init_system(), tm.DT, tm.NSTEPS, tm.L)
npairs = np.array([np.triu(tm.pairwise_distances(f, tm.RC)[2], 1).sum() for f in res["pos"]])
Ucut = 0.5 * tm.K * (tm.RC - tm.R0) ** 2
Eshift = res["E"] - Ucut * npairs          # same as using U_ij - U(rc), which is continuous at rc
jumps = np.diff(npairs)
say(f"  U at the cutoff: 0.5*k*(rc-r0)^2 = {Ucut:.4f}")
say(f"  pair count: start {npairs[0]}, end {npairs[-1]}, min {npairs.min()}, max {npairs.max()}, "
    f"steps where it changed {int((jumps != 0).sum())}")
say(f"  E(0) = {res['E'][0]:.4f}, E(end) = {res['E'][-1]:.4f}, max |E-E0|/|E0| = "
    f"{np.abs((res['E'] - res['E'][0]) / res['E'][0]).max():.4f}")
say(f"  K(0) = {res['K'][0]:.4f}, K(end) = {res['K'][-1]:.4f}; U(0) = {res['U'][0]:.4f}, U(end) = {res['U'][-1]:.4f}")
dE = np.diff(res["E"])
say(f"  largest single-step change of E: {np.abs(dE).max():.4f}; on steps with no pair change: "
    f"{np.abs(dE[jumps == 0]).max():.4f}")
say(f"  shifted energy E - {Ucut:.1f}*npairs: start {Eshift[0]:.4f}, end {Eshift[-1]:.4f}, "
    f"max |dev| {np.abs(Eshift - Eshift[0]).max():.4f} (relative {np.abs((Eshift - Eshift[0]) / Eshift[0]).max():.4f})")
iu = np.triu_indices(tm.N, 1)
rmin = np.array([tm.pairwise_distances(f, tm.RC)[1][iu].min() for f in res["pos"]])
quiet = np.nonzero(jumps == 0)[0]
worst = quiet[np.argmax(np.abs(dE[quiet]))]
say(f"  closest pair distance over the run: {rmin.min():.4f} at t = {res['t'][np.argmin(rmin)]:.2f} "
    f"(barrier at r=0 is only 0.5*k*r0^2 = {0.5 * tm.K * tm.R0 ** 2:.1f})")
say(f"  the biggest no-pair-change jump of E is at t = {res['t'][worst + 1]:.2f}, where the closest pair is "
    f"at r = {rmin[worst + 1]:.4f}")
np.savetxt("results/energy_cutoff_N12.csv",
           np.column_stack([res["t"], res["K"], res["U"], res["E"], npairs, Eshift, rmin]),
           delimiter=",", fmt="%.8g", header="t,K,U,E,npairs,E_shifted,rmin", comments="")

# ---------------------------------------------------------------- figure
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
for ax, idx, title in [(axes[0], 0, "initial (t = 0)"), (axes[1], -1, f"final (t = {tm.NSTEPS * tm.DT:.0f})")]:
    P = frames_vec[idx]
    _, _, nbf = tm.pairwise_distances(P, tm.RC)
    if idx == -1:
        for i in range(tm.N):
            ax.plot(frames_vec[:, i, 0], frames_vec[:, i, 1], color="0.85", lw=0.6, zorder=0)
    for i, j in zip(*np.nonzero(np.triu(nbf, 1))):
        ax.plot(P[[i, j], 0], P[[i, j], 1], color="tab:orange", lw=1.5, zorder=1)
    ax.scatter(P[:, 0], P[:, 1], s=60, color="tab:blue", edgecolor="white", zorder=2)
    ax.plot([0, tm.L, tm.L, 0, 0], [0, 0, tm.L, tm.L, 0], color="k", lw=1.5)
    ax.set_xlim(-0.5, tm.L + 0.5)
    ax.set_ylim(-0.5, tm.L + 0.5)
    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(f"{title}: {int(np.triu(nbf, 1).sum())} bonds (r <= rc = {tm.RC})")
axes[1].text(0.1, 0.1, "grey: paths over the run", fontsize=8, color="0.4")
fig.suptitle("N = 12, L = 10, dt = 0.01, 1000 Verlet steps (orange = neighbour pairs)")
fig.tight_layout()
fig.savefig("results/positions_initial_final.png", dpi=130)

# ---------------------------------------------------------------- save
with open("results/checks.csv", "w") as f:
    f.write("check;value;target;result\n")
    for row in checks:
        f.write(";".join(str(x).replace("\n", " ") for x in row) + "\n")
nfail = sum(r[3] == "FAIL" for r in checks)
say(f"\n{len(checks)} checks, {nfail} failed")
with open("results/verification_summary.txt", "w") as f:
    f.write("\n".join(lines) + "\n")
