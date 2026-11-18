# Video-1 script: Roshan Raju (Member 1), problem, model and verification (about 2:00)

Open before recording: a title slide (project name and the five names), `report.pdf` from this folder, `../member4_animation_dissipation/results/animation_default.mp4`.

---

### 0:00 to 0:20 · Intro
**Show:** title slide.

**Say:**

> "Hello, we are Roshan Raju, Chandar, Gowtham, Asad and Rakesh. We built a small simulator of particles that bounce around a box and pull on each other with springs, and tested whether it is accurate, efficient, and how to improve it. I'm Roshan. I wrote the base simulator and checked that it is correct."

### 0:20 to 0:55 · The model
**Show:** `report.pdf`, "Governing equations".

**Say:**

> "We have 12 particles in a 10 by 10 box with random starting positions and velocities. Two particles interact only if they are closer than rc equal to 2. Then a spring with stiffness 15 and rest length 0.8 pulls or pushes them, and the other particle gets the opposite force. Hard walls flip the velocity. We step in time with Euler or velocity Verlet, with dt equal to 0.01."

### 0:55 to 1:10 · One run
**Show:** play `animation_default.mp4` for about 10 seconds.

**Say:**

> "This is one run. Each step, NumPy broadcasting gives all pair distances at once, a mask picks the pairs inside the cutoff, and then we compute forces and move the particles."

### 1:10 to 1:50 · Verification
**Show:** the distance-matrix table, then the forces table and the walls table in `report.pdf`.

**Say:**

> "We checked correctness with numbers. The distance matrix is symmetric and its diagonal is zero, both exactly 0.0, and no particle is ever its own neighbour, on all 1001 frames. A 4-particle example I did by hand gave the same neighbour counts as the code. The sum of all forces stays below 3.3 times 10 to the minus 14, which is rounding error, so Newton's third law holds. And over full runs every particle stays inside the box."

### 1:50 to 2:00 · Hand-over
**Say:**

> "But the total energy went from 36.6 to 303.9, because the spring energy at the cutoff is 10.8, not zero. Gowtham explains what that means."
