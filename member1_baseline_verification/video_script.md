# Video-1 script: Roshan Raju (Member 1), problem, model and verification (about 3:45)

Open before recording: a title slide (project name and the five names), `report.pdf` from this folder, `toymodel.py`, `../member4_animation_dissipation/results/animation_default.mp4`, `results/positions_initial_final.png`.

Lines marked (optional) can be cut if the whole video runs over 20 minutes.

---

### 0:00 to 0:30 · Who we are and what we did
**Show:** title slide.

**Say:**
> "Hello, we are Roshan Raju, Chandar, Gowtham, Asad and Rakesh. Our project is the atomic toy model from Fundamentals of Scientific Computing. We built a small simulator of particles that move around a box and pull on each other with springs. Then we tested it for three things: is it accurate, is it efficient, and can we improve it. I'm Roshan Raju. I wrote the base simulator that everyone else used, and I checked that it is correct."

### 0:30 to 1:30 · The model
**Show:** `report.pdf`, sections "Model description and assumptions" and "Governing equations". Scroll slowly while talking.

**Say:**
> "The system is 12 particles in a 10 by 10 box, all with mass 1. The starting positions are random, and the velocity components are random between minus 1 and 1.
> Two particles only feel each other if they are closer than a cutoff, rc equal to 2. Inside the cutoff they are joined by a spring with stiffness k equal to 15 and rest length r0 equal to 0.8. If they are further apart than 0.8 the spring pulls them together; if they are closer, it pushes them apart. The force is minus k times r minus r0, along the line between the two particles, and the other particle gets the opposite force, by Newton's third law.
> The walls are hard. If a particle crosses a wall, we put it back on the wall and flip the velocity component that points into the wall.
> For time stepping we have two methods from the handout: the simple one, which is semi-implicit Euler, and the smoother one, velocity Verlet. The default timestep is 0.01 and we run 1000 steps."

### 1:30 to 2:10 · What one run looks like, and the algorithm
**Show:** play `animation_default.mp4` for about 15 seconds. Then switch to `toymodel.py`, the `pairwise_distances` function.

**Say:**
> "This is one run. The grey lines are pairs inside the cutoff.
> Every step, the code does four things. First, it gets every pairwise displacement vector at once with NumPy broadcasting: pos with a new axis minus pos with a new axis gives an N by N by 2 array, with no Python loop. Second, it builds the neighbour mask, which is true where the distance is above zero and at most rc. Third, it computes the spring forces using that mask. Fourth, it updates velocities and positions, and applies the walls. Asad will come back to this animation later and explain what it tells us."

### 2:10 to 3:15 · Verification with numbers
**Show:** `report.pdf`, section "Results". Show each table while you talk about it: distance matrix table, 4-particle table, forces table, walls table and the full-run box table.

**Say:**
> "Before trusting any result, we checked the code with numbers.
> First, the distance matrix. The largest difference between dist and its transpose is exactly 0.0, and the diagonal is exactly 0.0. It is exactly zero, not just small, because a minus b is exactly the negative of b minus a in floating point. No particle is ever its own neighbour, and that holds on all 1001 frames of the run.
> Second, a small example I could check by hand. Four particles, where one pair, A and B, sits exactly at distance 2, right on the cutoff. With rc equal to 2 the code gives neighbour counts 1, 1, 0, 0. With rc equal to 3 it gives 2, 2, 1, 1. Both match my hand calculation, and this also checks that we used less-than-or-equal at the cutoff. The force on A is 18 along x, which is 15 times the stretch of 1.2.
> Third, Newton's third law. Over the whole run, the norm of the sum of all forces is at most 3.3 times 10 to the minus 14, while single forces go up to 72. That is rounding error. In float32 it is 2 times 10 to the minus 5, the same thing at float32 precision. The loop version and the vectorized version of the force agree to about 10 to the minus 11.
> Fourth, the walls. The handout test, a particle at x equal to 10.1 moving right, ends exactly at x equal to 10, moving left. Over full runs, x and y always stay between 0 and 10, with zero coordinates outside the box."

### 3:15 to 3:25 · Memory check (optional)
**Show:** `report.pdf`, "Memory layout" table.

**Say:**
> "In float64 each number takes 8 bytes, the same as sizeof double in C, and in float32 it takes 4. So float32 exactly halves the memory."

### 3:25 to 3:45 · What we noticed, and hand-over
**Show:** `results/positions_initial_final.png`.

**Say:**
> "Here are the start and the end of the run: 8 bonds at the start, 33 at the end, with most particles in one cluster. While running this I noticed something odd: the total energy went from 36.6 to 303.9. The reason is in the model, not the code. At r equal to rc the spring energy is a half times 15 times 1.2 squared, which is 10.8, not zero. So every time a pair crosses the cutoff, the energy jumps by 10.8. Gowtham will show what this means for choosing the timestep."
