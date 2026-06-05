"""Unit tests for :mod:`principia.integrators`.

Standard-library ``unittest`` only, to preserve the zero-dependency constraint.

The headline tests build a 2-body circular orbit and run it for 1000 steps to
prove the qualitative difference between the integrators:

* Explicit (forward) Euler injects energy -> the orbit spirals outward and the
  separation between the bodies diverges.
* Symplectic (semi-implicit) Euler keeps energy bounded -> the orbit stays on a
  stable, periodic band.
"""

import math
import unittest

from principia.gravity import compute_accelerations
from principia.integrators import (
    ExplicitEulerIntegrator,
    Integrator,
    SymplecticEulerIntegrator,
)
from principia.particle import Particle
from principia.vector2d import Vector2D


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def constant_acceleration(value):
    """Return an acceleration callable that ignores positions (for unit tests)."""
    return lambda particles: [value for _ in particles]


def make_circular_orbit(G, M, m, R):
    """Build a 2-body circular orbit with the centre of mass at rest.

    Returns ``(particles, period)`` where ``particles == [central, satellite]``.
    For a circular orbit the relative speed is ``sqrt(G (M + m) / R)``; momenta
    are balanced so the centre of mass does not drift.
    """
    v_rel = math.sqrt(G * (M + m) / R)
    total = M + m
    v_satellite = (M / total) * v_rel   # +y
    v_central = -(m / total) * v_rel     # -y (zero net momentum)
    central = Particle(M, Vector2D(0.0, 0.0), Vector2D(0.0, v_central))
    satellite = Particle(m, Vector2D(R, 0.0), Vector2D(0.0, v_satellite))
    period = 2.0 * math.pi * R / v_rel
    return [central, satellite], period


def separation(particles):
    return (particles[1].position - particles[0].position).magnitude()


def total_energy(particles, G):
    """Total mechanical energy: kinetic + gravitational potential."""
    kinetic = 0.0
    for p in particles:
        kinetic += 0.5 * p.mass * p.velocity.magnitude_squared()
    potential = 0.0
    n = len(particles)
    for i in range(n):
        for j in range(i + 1, n):
            d = (particles[j].position - particles[i].position).magnitude()
            potential += -G * particles[i].mass * particles[j].mass / d
    return kinetic + potential


def simulate(integrator, particles, dt, steps, G):
    """Run ``steps`` integration steps, returning per-step separations/energies."""
    accel_fn = lambda ps: compute_accelerations(ps, G)
    separations = [separation(particles)]
    energies = [total_energy(particles, G)]
    state = particles
    for _ in range(steps):
        state = integrator.step(state, dt, accel_fn)
        separations.append(separation(state))
        energies.append(total_energy(state, G))
    return state, separations, energies


def max_relative_energy_drift(energies):
    e0 = energies[0]
    return max(abs((e - e0) / e0) for e in energies)


# --------------------------------------------------------------------------- #
# Base-class / protocol behaviour
# --------------------------------------------------------------------------- #

class BaseClassTests(unittest.TestCase):
    def test_integrator_is_abstract(self):
        with self.assertRaises(TypeError):
            Integrator()  # type: ignore[abstract]

    def test_concrete_integrators_are_integrators(self):
        self.assertIsInstance(ExplicitEulerIntegrator(), Integrator)
        self.assertIsInstance(SymplecticEulerIntegrator(), Integrator)


# --------------------------------------------------------------------------- #
# Single-step correctness (isolated from gravity via constant acceleration)
# --------------------------------------------------------------------------- #

class SingleStepTests(unittest.TestCase):
    def setUp(self):
        # mass=2, position=(0,0), velocity=(1,0); constant accel=(0,-10), dt=0.5.
        self.particle = Particle(2.0, Vector2D(0.0, 0.0), Vector2D(1.0, 0.0))
        self.dt = 0.5
        self.accel = constant_acceleration(Vector2D(0.0, -10.0))

    def test_explicit_euler_step(self):
        (result,) = ExplicitEulerIntegrator().step(
            [self.particle], self.dt, self.accel
        )
        # v_new = (1,0) + (0,-10)*0.5 = (1,-5)
        self.assertEqual(result.velocity, Vector2D(1.0, -5.0))
        # r_new = (0,0) + v_OLD (1,0)*0.5 = (0.5, 0)
        self.assertEqual(result.position, Vector2D(0.5, 0.0))
        self.assertEqual(result.mass, 2.0)

    def test_symplectic_euler_step(self):
        (result,) = SymplecticEulerIntegrator().step(
            [self.particle], self.dt, self.accel
        )
        # v_new = (1,0) + (0,-10)*0.5 = (1,-5)
        self.assertEqual(result.velocity, Vector2D(1.0, -5.0))
        # r_new = (0,0) + v_NEW (1,-5)*0.5 = (0.5, -2.5)
        self.assertEqual(result.position, Vector2D(0.5, -2.5))
        self.assertEqual(result.mass, 2.0)

    def test_integrators_share_velocity_but_differ_in_position(self):
        (ex,) = ExplicitEulerIntegrator().step([self.particle], self.dt, self.accel)
        (sy,) = SymplecticEulerIntegrator().step([self.particle], self.dt, self.accel)
        # Same velocity update for both.
        self.assertEqual(ex.velocity, sy.velocity)
        # Different position update (old vs new velocity).
        self.assertNotEqual(ex.position, sy.position)


# --------------------------------------------------------------------------- #
# Purity: step must not mutate inputs and must return new instances
# --------------------------------------------------------------------------- #

class PurityTests(unittest.TestCase):
    def _check_purity(self, integrator):
        particles = [
            Particle(2.0, Vector2D(0.0, 0.0), Vector2D(1.0, 0.0)),
            Particle(3.0, Vector2D(5.0, 0.0), Vector2D(0.0, 2.0)),
        ]
        before = [(p.mass, p.position, p.velocity) for p in particles]
        result = integrator.step(particles, 0.1, constant_acceleration(Vector2D(0.0, -1.0)))

        # Inputs untouched.
        after = [(p.mass, p.position, p.velocity) for p in particles]
        self.assertEqual(before, after)
        # New list, new instances.
        self.assertIsNot(result, particles)
        for original, updated in zip(particles, result):
            self.assertIsNot(updated, original)
        self.assertEqual(len(result), len(particles))

    def test_explicit_euler_is_pure(self):
        self._check_purity(ExplicitEulerIntegrator())

    def test_symplectic_euler_is_pure(self):
        self._check_purity(SymplecticEulerIntegrator())


# --------------------------------------------------------------------------- #
# CRUCIAL: 2-body orbit -- Explicit diverges, Symplectic stays bounded
# --------------------------------------------------------------------------- #

class OrbitStabilityTests(unittest.TestCase):
    G = 1.0
    M = 1.0
    m = 1.0e-3
    R = 1.0
    STEPS = 1000

    def setUp(self):
        self.particles, self.period = make_circular_orbit(
            self.G, self.M, self.m, self.R
        )
        # ~10 orbits across 1000 steps.
        self.dt = self.period / 100.0

    def test_explicit_euler_orbit_diverges(self):
        _, seps, energies = simulate(
            ExplicitEulerIntegrator(), self.particles, self.dt, self.STEPS, self.G
        )
        # The separation must grow well beyond the initial circular radius.
        self.assertGreater(seps[-1], 1.5 * self.R)
        self.assertGreater(max(seps), 2.0 * self.R)
        # Forward Euler pumps energy into the system (energy rises toward 0).
        self.assertGreater(energies[-1], energies[0])
        self.assertGreater(max_relative_energy_drift(energies), 0.20)

    def test_symplectic_euler_orbit_is_bounded(self):
        _, seps, energies = simulate(
            SymplecticEulerIntegrator(), self.particles, self.dt, self.STEPS, self.G
        )
        # Separation stays in a tight band around the circular radius.
        self.assertLess(max(seps), 1.1 * self.R)
        self.assertGreater(min(seps), 0.9 * self.R)
        self.assertLess(abs(seps[-1] - self.R), 0.1 * self.R)
        # Energy oscillates but stays bounded (no secular drift).
        self.assertLess(max_relative_energy_drift(energies), 0.05)

    def test_explicit_diverges_relative_to_symplectic(self):
        _, ex_seps, ex_en = simulate(
            ExplicitEulerIntegrator(), self.particles, self.dt, self.STEPS, self.G
        )
        _, sy_seps, sy_en = simulate(
            SymplecticEulerIntegrator(), self.particles, self.dt, self.STEPS, self.G
        )
        # Direct contrast: explicit ends much farther out than symplectic ...
        self.assertGreater(ex_seps[-1], 1.5 * sy_seps[-1])
        # ... and its energy drift dwarfs the symplectic method's.
        self.assertGreater(
            max_relative_energy_drift(ex_en),
            10.0 * max_relative_energy_drift(sy_en),
        )


if __name__ == "__main__":
    unittest.main()
