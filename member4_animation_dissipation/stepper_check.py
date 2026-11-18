# Damped Verlet step vs the exact solution of x'' = -2k x - gamma x' for one pair (x = r - r0)
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))  # results/ is relative to this folder
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "member1_baseline_verification"))
import numpy as np
import toymodel as tm
import damped as dm

os.makedirs("results", exist_ok=True)
g, T = 0.5, 5.0
w0 = np.sqrt(2 * tm.K)
wd = np.sqrt(w0 ** 2 - g ** 2 / 4)
x0 = 1.2 - tm.R0
x_exact = x0 * np.exp(-g * T / 2) * (np.cos(wd * T) + g / (2 * wd) * np.sin(wd * T))
rows = []
for dt in [0.04, 0.02, 0.01, 0.005, 0.0025]:
    pos = np.array([[4.4, 5.0], [5.6, 5.0]])
    vel = np.zeros((2, 2))
    m = np.ones(2)
    F = None
    for s in range(int(round(T / dt))):
        pos, vel, F = dm.step_damped(pos, vel, m, dt, tm.L, g, F)
    x = np.linalg.norm(pos[0] - pos[1]) - tm.R0
    rows.append((dt, x, abs(x - x_exact)))
with open("results/stepper_check.csv", "w") as f:
    f.write(f"# exact x(T=5) = {x_exact:.10f}\ndt,x_num,abs_error,error_ratio_to_previous\n")
    for i, (dt, x, e) in enumerate(rows):
        ratio = rows[i - 1][2] / e if i > 0 else float("nan")
        f.write(f"{dt},{x:.10f},{e:.4e},{ratio:.3f}\n")
print(open("results/stepper_check.csv").read())
