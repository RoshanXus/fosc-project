# Video-3 script: Asad (Member 4), animation and dissipation (about 2:00)

Open before recording: `report.pdf` from this folder, and from `results/`: `animation_default.mp4`, `diagnostics_default.png`, `compare_gamma0_vs_gamma0.5.mp4`, `energy_balance.png`.

---

### 0:00 to 0:25 · The animation
**Show:** play `animation_default.mp4`.

**Say:**

> "I'm Asad. This is the default run: 12 particles, 2000 Verlet steps, with pairs inside the cutoff drawn as grey lines. We used it as a diagnostic: nothing escapes the box, no particle sticks to a wall for more than one step, and there are no sudden jumps."

### 0:25 to 0:45 · Clustering
**Show:** `diagnostics_default.png`.

**Say:**

> "What it does show is strong clustering: pairs go from 8 to 40, and the kinetic energy from 3.3 to 280. Each pair inside the cutoff is a well 10.8 deep, so particles fall in and heat up."

### 0:45 to 1:10 · Adding drag
**Show:** play `compare_gamma0_vs_gamma0.5.mp4`.

**Say:**

> "We added drag, minus gamma times v, using a Verlet step with an implicit drag half-kick. It is still second order. On the left, gamma is 0 and the particles keep moving. On the right, gamma is 0.5 and they settle."

### 1:10 to 1:35 · Proof and criterion
**Show:** `energy_balance.png`.

**Say:**

> "The energy lost matches the work done by the drag to within 1 percent for gamma of 0.1 and above. We call the system steady when the mean speed is below 10 to the minus 3 and the energy changes by less than 10 to the minus 3 over 10 time units, until the end of the run."

### 1:35 to 2:00 · Effect of gamma
**Show:** the relaxation table in section 3.5 of `report.pdf`.

**Say:**

> "Up to gamma 2, the relaxation time goes like 1 over gamma: 349, 192, 31, 16, 8.3. Above about 4 the motion is over-damped and it slows again, so the fastest is near critical damping. Small gamma ends in one 11-particle cluster; large gamma freezes the starting group. Next, Chandar."
