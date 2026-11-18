# Video-3 script: Asad (Member 4), animation and dissipation (about 4:15)

Open before recording: `report.pdf` from this folder, and from `results/`: `animation_default.mp4`, `diagnostics_default.png`, `animation_dt0.1.mp4`, `diagnostics_dt.png`, `compare_gamma0_vs_gamma0.5.mp4`, `energy_balance.png`, `relaxation_time_vs_gamma.png`, `final_states.png`.

Lines marked (optional) can be cut if the whole video runs over 20 minutes.

---

### 0:00 to 0:10 · Intro
**Say:**
> "I'm Asad. I made the animation, used it to look for problems, and added dissipation to the model."

### 0:10 to 0:40 · The animation
**Show:** play `animation_default.mp4` (about 20 seconds is enough).

**Say:**
> "This is the default run: 12 particles, 2000 Verlet steps. The grey lines are pairs inside the cutoff, and the numbers on top are the time, the total energy and the pair count. We didn't treat the animation as decoration. For everything we saw in it, we checked a number."

### 0:40 to 1:15 · What the animation tells us
**Show:** the check table in section 2 of `report.pdf`, then `diagnostics_default.png`.

**Say:**
> "No particle escapes: the coordinates always stay between 0 and 10. No particle gets stuck on a wall: the longest stay is one step. There are no unrealistic jumps: the largest step is 1.025 times speed times dt, which is what it should be.
> What the movie does show is strong clustering, which you can see in this plot. The number of pairs goes from 8 to 40, at one moment all 66 pairs are inside the cutoff, and the kinetic energy goes from 3.3 to 280."

### 1:15 to 1:45 · Why it clusters
**Show:** stay on `diagnostics_default.png`, or the "Why it clusters" paragraph in the report.

**Say:**
> "Why? As Roshan Raju showed, the pair energy at the cutoff is 10.8, not zero. Written as a shifted potential, every pair inside the cutoff sits in a well 10.8 deep, but the barrier against overlapping is only 4.8. So particles fall into each other's wells, overlap, and the released energy heats them up. One interesting detail: two particles on their own just pass through each other. It takes a third particle to carry the energy away and form a cluster."

### 1:45 to 2:10 · A bad run that looks fine
**Show:** play `animation_dt0.1.mp4` for about 10 seconds, then `diagnostics_dt.png`.

**Say:**
> "This is the same start with dt equal to 0.1. Honestly, the movie looks like a normal hot gas. But the shifted energy rises by up to 680 within t equals 20, and a single step can be 2.2 long, longer than the cutoff. So the animation on its own would have fooled us."

### 2:10 to 2:35 · Adding drag
**Show:** section 3.1 of the report (the three update lines).

**Say:**
> "For dissipation we added a drag force, minus gamma times v. Our step is velocity Verlet, with the drag in the second half-kick treated implicitly. Because drag is linear in v, that's just one division. We tested it against the exact solution of a damped spring: halving dt cuts the error by 4, so it is still second order."

### 2:35 to 2:55 · Conservative against dissipative
**Show:** play `compare_gamma0_vs_gamma0.5.mp4`.

**Say:**
> "Here is conservative against dissipative, side by side. On the left, gamma is 0 and the particles keep crossing the box forever. On the right, gamma is 0.5: they move a little and stop. They look still by about t equals 8, and pass our steady-state test at t equals 31."

### 2:55 to 3:15 · Proving the energy goes into the drag
**Show:** `energy_balance.png`, then the table in section 3.3.

**Say:**
> "To show the energy loss really comes from the drag, we compared the drop in shifted energy with the work done by the drag, the integral of gamma times v squared. They agree to within 1 percent for gamma of 0.1 and above, and 3.8 percent for gamma 0.05. Most of the mismatch is on steps where a pair crosses the cutoff."

### 3:15 to 3:35 · Our steady-state criterion
**Show:** section 3.4 of the report.

**Say:**
> "We didn't call the system steady just because it looks still. Our criterion has two parts: the mean speed is below 10 to the minus 3, and the energy changes by less than 10 to the minus 3 of E at time zero over 10 time units, and both must hold until the end of the run. We needed the speed condition because the energy test alone accepted states where particles were still visibly moving."

### 3:35 to 4:05 · How gamma changes relaxation and the final state
**Show:** the relaxation table in section 3.5, then `relaxation_time_vs_gamma.png`, then `final_states.png`.

**Say:**
> "Up to gamma equal to 2, the relaxation time goes like 1 over gamma: 349, 192, 31, 16 and 8.3. Kinetic energy decays exactly as e to the minus gamma t. Above about 4 the system is over-damped and relaxation slows down again: 6.3, 13.1, 16.7. So the fastest relaxation is close to critical damping.
> Gamma also changes where the system ends up. Small gamma ends as one 11-particle cluster with energy minus 563. Large gamma freezes the original group of 7 particles, at about minus 220. It's like slow annealing against a quench."

### 4:05 to 4:15 · Wall issue and hand-over (optional first sentence)
**Say:**
> "One more thing we found: the handout's wall clamp adds energy in long runs, about 430 of the 684 gained over t equals 600, and a mirror reflection cuts the total gain to 46. Next, Chandar on how fast the code is."
