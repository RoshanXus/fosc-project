# Task 6: build pairforce.c at -O0/-O2 (program + ctypes library) and time the programs
import os
import subprocess

here = os.path.dirname(os.path.abspath(__file__))
os.chdir(here)
os.makedirs("results", exist_ok=True)

for opt in ["O0", "O2"]:
    cmds = [
        ["gcc", f"-{opt}", "pairforce.c", "-o", f"pairforce_{opt}", "-lm"],
        ["gcc", f"-{opt}", "-shared", "-fPIC", "-DNO_MAIN", "pairforce.c", "-o", f"libpairforce_{opt}.so", "-lm"],
    ]
    for c in cmds:
        print(" ".join(c))
        subprocess.run(c, check=True)

Ns = [50, 200, 500, 1000, 2000, 4000]
rows = ["opt,N,median_s,min_s,max_s,inner_calls"]
for opt in ["O0", "O2"]:
    for N in Ns:
        out = subprocess.run([f"./pairforce_{opt}", str(N)], capture_output=True, text=True, check=True).stdout
        print(opt, out.strip())
        f = dict(p.split("=") for p in out.split() if "=" in p)
        rows.append(f"{opt},{N},{f['median']},{f['min']},{f['max']},{f['inner']}")

with open("results/c_standalone.csv", "w") as fh:
    fh.write("\n".join(rows) + "\n")
print("wrote results/c_standalone.csv")
