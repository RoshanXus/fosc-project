# Video-2 script: Gowtham (Member 3), energy, timestep and precision (about 2:00)

Open before recording: `report.pdf` from this folder, and from `results/`: `pairs_vs_energy_dt0.01.png`, `control_eps_vs_dt_loglog.png`, `precision_eps_pos.png`.

---

### 0:00 to 0:10 · Intro
**Say:**

> "I'm Gowtham. I studied energy conservation, the timestep, and float32 against float64."

### 0:10 to 0:45 · Why the raw energy jumps
**Show:** `pairs_vs_energy_dt0.01.png`.

**Say:**

> "K and U swap energy all the time, so we check the total, K plus U. In our model it jumps: the dashed line, E plus 10.8 times the change in pair count, sits right on top of it. So the raw energy error is 400 to 1000 percent for every timestep. Instead we track the shifted energy, E minus 10.8 per pair, which is what the forces actually conserve."

### 0:45 to 1:15 · Integrator against model
**Show:** `control_eps_vs_dt_loglog.png`.

**Say:**

> "To test the integrator alone, I used a cluster where no pair crosses the cutoff. There Verlet has slope 2.00 and Euler 1.02, as theory says, and Verlet is 23 times more accurate at dt 0.01. In the real model both drop to first order, because the force jumps from 18 to 0 at the cutoff."

### 1:15 to 1:35 · The timestep
**Show:** the 5-seed table in section 3.2 of `report.pdf`.

**Say:**

> "My target was less than 1 percent drift for five different starts. The answer is Verlet with dt 0.0005, with a worst case of 0.96 percent. The handout's dt of 0.01 drifts by up to 32 percent."

### 1:35 to 2:00 · float32 vs float64
**Show:** `precision_eps_pos.png`.

**Say:**

> "Float32 halves the memory, but at 12 particles it isn't faster. Its trajectory drifts away from float64, but so do float64 runs that start 10 to the minus 12 apart, so that is chaos, not precision. Float32 is accurate enough, but we keep float64. Next, Asad."
