# Video-2 script: Member 3, energy, timestep and precision (about 4:00)

Open before recording: `report.pdf` from this folder, and these figures from `results/`: `KUE_panels.png`, `pairs_vs_energy_dt0.01.png`, `E_total_all_dt_verlet.png`, `control_eps_vs_dt_loglog.png`, `error_sources.png`, `precision_cost.png`, `precision_eps_pos.png`.

Lines marked (optional) can be cut if the whole video runs over 20 minutes.

---

### 0:00 to 0:15 · Intro
**Show:** `report.pdf`, title.

**Say:**
> "I'm [Name 3]. I studied energy conservation, how large the timestep can be, and float32 against float64."

### 0:15 to 0:40 · K and U swap energy
**Show:** `KUE_panels.png`, point at one row.

**Say:**
> "Kinetic energy alone is not supposed to stay constant. In these panels, K and U swap energy back and forth all the time as the springs stretch and compress. For this kind of motion, the quantity that should stay constant is the total, E equals K plus U."

### 0:40 to 1:30 · The raw total energy cannot be conserved
**Show:** `pairs_vs_energy_dt0.01.png`, then Table 1 in `report.pdf` (section 3.1).

**Say:**
> "But in our model it isn't conserved, for any timestep. In the top panel, E is blue, and the dashed orange line is E at time zero plus 10.8 times the change in the number of pairs. They sit on top of each other. When I fit the energy jumps against the change in pair count, I get 10.796 per pair, against 10.8 expected.
> In the table, the assignment's maximum energy error is between 4 and 11, so 400 to 1000 percent, for every dt from 0.05 down to 0.0005 and for both methods. It doesn't go down with dt, so it can't be used to choose dt.
> So I used the shifted energy: E minus 10.8 times the number of pairs. The force is the slope of this shifted potential, so this is the energy the dynamics actually conserve, and it doesn't jump at the cutoff. In the bottom panel it stays between about minus 53 and minus 41."

### 1:30 to 2:20 · Separating the model from the integrator
**Show:** `E_total_all_dt_verlet.png` (bottom panel), then `control_eps_vs_dt_loglog.png`, then the slope table in section 3.2.

**Say:**
> "With the shifted energy, the error does fall as dt gets smaller, but only linearly, for both Euler and Verlet. That surprised me, because Verlet should be second order.
> So I made a control run: a relaxed cluster where all 66 pairs stay inside the cutoff and nothing touches a wall. Here Verlet has slope 2.00 and Euler 1.02, exactly what theory says, and at dt 0.01 Verlet is 23 times more accurate. When the same cluster hits a wall, both drop to first order.
> So the integrator itself is fine. What breaks second order is the force jumping from 18 to 0 at the cutoff, and the wall clamp moving particles."

### 2:20 to 2:35 · Where the error comes from (optional)
**Show:** `error_sources.png`.

**Say:**
> "Splitting the error by the type of step confirms this: steps where a pair crosses the cutoff carry 83 percent of the error at dt 0.01, and 89 percent at dt 0.001."

### 2:35 to 3:05 · The largest timestep we can defend
**Show:** the 5-seed table in section 3.2, then section 3.4.

**Say:**
> "So, what is the largest timestep I can defend? My target: the shifted energy may drift by less than 1 percent over t equals 10, for every one of five different random starts. 1 percent is about half an energy unit, under 5 percent of the depth of one bond.
> The answer is velocity Verlet with dt equal to 0.0005: the worst seed drifts 0.96 percent. Euler fails at the same dt, with 1.26 percent. The handout's dt of 0.01 drifts by up to 32 percent, so I can't defend it. And dt 0.0005 is still cheap here: a whole run takes about 1.4 seconds."

### 3:05 to 3:30 · float32 vs float64: memory and speed
**Show:** memory and timing tables in section 4, then `precision_cost.png`.

**Say:**
> "Now float32 against float64. Every array is exactly half the size in float32, and inside the force function the peak memory drops from 65 to 33 megabytes at N equal to 1000. But for speed, at N equal to 12 float32 is not faster at all, a ratio of 0.98, because the time is almost all Python overhead. It only helps for N of 1000 or more, by 1.2 to 1.8 times."

### 3:30 to 4:00 · float32 vs float64: accuracy, and conclusion
**Show:** `precision_eps_pos.png`, then the statistics table in section 4.4.

**Say:**
> "The float32 and float64 trajectories do drift apart. The position difference grows to about 0.5, which means the particles end up in completely different places. But look at the other curves: two float64 runs whose starts differ by 10 to the minus 7, or even 10 to the minus 12, end up just as far apart. So this is chaos, not a float32 problem. Averages like the mean kinetic energy from float32 fall inside the spread of the float64 runs.
> My conclusion: float32 is accurate enough for this problem, but at N equal to 12 it gives no speed benefit, so we keep float64 there. [Name 4] will now show the animation and what happens when we add friction."
