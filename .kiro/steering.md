# Specification: Level 0 - Vector Algebra

## 1. Mathematical Definitions
We operate in a 2D Euclidean space. A vector is represented by a tuple of floats `(x, y)`.

## 2. Core Operations
Implement a `Vector2D` class with the following pure mathematical properties.
- Addition: A + B = (Ax + Bx, Ay + By)
- Subtraction: A - B = (Ax - Bx, Ay - By)
- Scalar Multiplication: cA = (c * Ax, c * Ay)
- Dot Product: A dot B = Ax * Bx + Ay * By
- Magnitude Squared: |A|^2 = A dot A
- Magnitude: |A| = sqrt(|A|^2)

## 3. Strict Constraints (ZERO Black Box)
- DO NOT import `numpy` or any other external mathematical libraries. 
- The class must be immutable. Operations like addition must return a NEW `Vector2D` instance, not modify the existing one.
- Generate a comprehensive Unit Test file to verify all algebraic operations with floating-point precision assertions.

## Specification: Level 1 - Particle and Dynamics Layer

### 1. Mathematical Definitions
- **Particle**: A physical entity defined by a `mass` (float > 0), a `position` (Vector2D), and a `velocity` (Vector2D). 
- **Force Law (Gravity)**: The instantaneous acceleration of particle `i` induced by particle `j` follows Newton's Law of Universal Gravitation: a_i = G * m_j * (r_j - r_i) / |r_j - r_i|^3.

### 2. Core Architecture
- Implement a `Particle` dataclass. Like Vector2D, it must be immutable. State updates must return a new `Particle` instance.
- Implement a `GravityForce` module or class. It must expose a pure function `compute_accelerations(particles: List[Particle], G: float) -> List[Vector2D]`. This function calculates the net acceleration for every particle in the system.

### 3. Strict Constraints & Invariants
- ZERO Black Box. You MUST use the previously built `Vector2D` class for all spatial calculations.
- DO NOT update particle positions inside the force computation. The function strictly returns derivatives (accelerations).
- Generate a comprehensive Unit Test file. Assert that the computed accelerations for a 2-body system exactly match manual floating-point calculations.
- Assert Newton's Third Law (Conservation of Momentum): The sum of (m * a) across all particles must equal the zero vector.

## Specification: Level 2 - Numerical Integrators

### 1. Mathematical Definitions
We require two distinct numerical integration algorithms to advance the system state from `t` to `t + dt`:
- **Explicit Euler**: First-order integration. Uses velocity at `t` to compute new position. Highly dissipative/unstable for long-term orbital mechanics.
  - v_new = v_old + a * dt
  - r_new = r_old + v_old * dt
- **Symplectic Euler (Semi-Implicit)**: First-order symplectic integration. Uses velocity at `t + dt` to compute new position. Maintains bounded energy oscillation for Hamiltonian systems.
  - v_new = v_old + a * dt
  - r_new = r_old + v_new * dt

### 2. Core Architecture
- Implement an abstract `Integrator` protocol or base class defining a method `step(particles: List[Particle], dt: float, compute_accelerations: Callable) -> List[Particle]`.
- Implement `ExplicitEulerIntegrator` and `SymplecticEulerIntegrator` adhering to this protocol.
- The `step` function MUST remain pure. It must return a completely new list of updated `Particle` instances based on the immutable principles from Level 0 and Level 1.

### 3. Strict Constraints & Testing
- ZERO Black Box. Everything builds upon `Vector2D` and `Particle`.
- Create a test scenario: A 2-body circular orbit simulation running for 1000 steps.
- **Crucial Assertion**: Write a test that mathematically proves the difference between the two integrators. Explicit Euler should cause the orbit to diverge (distance between bodies increases significantly), while Symplectic Euler should maintain a stable periodic orbit boundaries.
