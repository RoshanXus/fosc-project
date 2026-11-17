# Video-5 script: Member 5, innovation and final recommendation (about 4:30)

Open before recording: `../member1_baseline_verification/toymodel.py` (the `pair_forces` function), a simple grid sketch (drawn on paper or one slide), `report.pdf` and `recommendation.pdf` from this folder, and from `results/`: `run12_compare.png`, `force_timing_loglog.png`, `dt_study_models.png`.

Lines marked (optional) can be cut if the whole video runs over 20 minutes.

---

### 0:00 to 0:10 · Intro
**Say:**
> "I'm [Name 5]. I built our innovation, a cell-linked neighbour list, and put together our final recommendation."

### 0:10 to 0:35 · The problem with the baseline
**Show:** `toymodel.py`, the line that builds `d` in `pairwise_distances`.

**Say:**
> "The baseline force builds an N by N by 2 array, so it checks every pair. But at our density each particle has only about 1.5 neighbours inside the cutoff, so almost all of that work is wasted."

### 0:35 to 1:05 · The idea
**Show:** the grid sketch.

**Say:**
> "The idea is to cut the box into cells that are at least rc wide. Then any partner within rc must be in the particle's own cell or one of the 8 cells around it. That's about 4 candidates per particle, however big N is, so the cost grows like N instead of N squared. Two details mattered: the walls are hard, so edge cells just have fewer neighbours, and particles sitting exactly on x equal to L have to be put in the last cell. We compiled it with Numba, and we also compiled a plain all-pairs loop, so we could separate the gain from compiling from the gain from the algorithm."

### 1:05 to 1:30 · Correct first
**Show:** the correctness table in `report.pdf`, then `run12_compare.png`.

**Say:**
> "Correct first. Against the Python loop, the forces agree to 3 times 10 to the minus 14, including particles on the walls and pairs exactly at the cutoff. In the full 12-particle run, the energy curves lie on top of each other and the final energy is 303.9046 in every version."

### 1:30 to 2:05 · How much faster, and how much less memory
**Show:** `force_timing_loglog.png`, then the timing table in section 3 and the whole-simulation table in section 5.

**Say:**
> "The fitted slopes are 2.18 for NumPy, 2.00 for the compiled all-pairs loop and 1.10 for the cell list. Per force call, the cell list is 853 times faster than NumPy at N equal to 1000, and 4171 times faster at 5000. It handles a million particles in 0.19 seconds. For a whole 200-step simulation at N equal to 1000, it's 513 times faster: 12.4 seconds down to 24 milliseconds. Memory is 41 bytes per particle, against 65 times N squared bytes for NumPy."

### 2:05 to 2:35 · Where it doesn't help
**Show:** the "Where it does not help" section of `report.pdf`.

**Say:**
> "Where it doesn't help: compiling takes 0.9 seconds, and the whole 12-particle handout run takes only 0.057 seconds with NumPy. So for N equal to 12 it isn't worth it. Below about 40 particles the cell list is even slower than the compiled all-pairs loop.
> (optional) We also made a mistake along the way: with default arguments, Numba took 31 microseconds per call instead of 0.6, so our first benchmark said the cell list was slower than NumPy."

### 2:35 to 3:00 · One more test: a smooth cutoff
**Show:** `dt_study_models.png`.

**Say:**
> "For the recommendation I tested one more idea: a smooth cutoff, so the force goes to zero smoothly at rc, together with mirror walls. It allows a 4 times larger timestep at N equal to 12, and at N equal to 10,000, dt of 0.01 stays under 1 percent drift. We expected a factor of 100, so it helped less than we hoped, and it changes the model, so it's an option, not our default."

### 3:00 to 4:00 · Final recommendation
**Show:** the recommendation table in `recommendation.pdf`. Point at each row as you talk.

**Say:**
> "Our final recommendation, for the 12-particle handout system.
> Float64, because float32 gives no speedup here, a ratio of 0.98.
> Velocity Verlet, because it is second order when the forces are smooth, and in our model it allows twice Euler's timestep at the same cost.
> A timestep of 0.0005, the largest that keeps the shifted energy within 1 percent for all five random starts.
> The NumPy force, because it is fast enough for 12 particles.
> Keep rc equal to 2, but always report the shifted energy, not the raw total.
> For drag: gamma 0 for dynamics, gamma 5 for the fastest relaxation, and gamma 0.1 to reach the lowest-energy cluster.
> For large N: the Numba cell list, with mirror walls."

### 4:00 to 4:30 · What we changed, why, and did it help
**Show:** the "Final question" section of `recommendation.pdf`.

**Say:**
> "So what did we change, why, and did it help? We changed the force algorithm because NumPy's time and memory grow like N squared. It helped a lot above about 40 particles, but not for 12. We changed the timestep and the energy we check, because the handout's dt drifts by up to 32 percent. Now the drift is below 1 percent. Thank you for watching."
