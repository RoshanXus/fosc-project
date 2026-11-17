# Video notes, Member 1 (model and verification, about 2-3 min)

- (Show the model equations from report.md.) 12 particles in a 10 x 10 box, springs with k = 15,
  r0 = 0.8, only for pairs closer than rc = 2. Hard walls flip the normal velocity. dt = 0.01, 1000 steps.
- Algorithm per step: distances by broadcasting into an (N, N, 2) array, neighbour mask
  0 < r <= rc, forces, Verlet update, wall reflection.
- (Show `positions_initial_final.png`.) Start: 8 bonds. End: 33 bonds, most particles in one cluster.
- (Show the distance table.) Symmetry error and diagonal are exactly 0.0, on all 1001 frames, and no
  particle is ever its own neighbour. Explain why it is exactly zero: a - b = -(b - a) in floating point.
- (Show the 4-particle table.) Hand counts [1,1,0,0] for rc = 2 and [2,2,1,1] for rc = 3 match;
  pair AB placed exactly at r = 2 checks the "<=". Hand force 18 matches.
- (Show the force table.) Max ||sum F|| over the run is 3.3e-14 in float64 (largest force 72.6),
  2.0e-5 in float32: rounding level. Loop and vectorized forces agree to 1e-11 (the +1e-12 term).
- (Show the wall table and box table.) Handout test [10.1, 5] ends at x = 10 with vx = -1. In the full
  runs x and y stay in [0, 10], 16 wall hits, 0 coordinates outside.
- Memory: float64 itemsize 8 = sizeof(double), float32 4 = sizeof(float), half the bytes.
- Hand over to energy: U at the cutoff is 10.8, not 0, so E jumps when pairs cross rc
  (E from 36.6 to 303.9 in the default run). Member 3 looks at this properly.
