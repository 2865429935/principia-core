"""Unit tests for the Level 3 velocity Verlet integrator.

Standard-library ``unittest`` only, to preserve the zero-dependency constraint.

The headline tests use the :class:`~principia.energy.EnergyMonitor` to compare
integrators on a highly eccentric 2-body orbit, and an independent convergence
study to prove velocity Verlet is second order, O(dt^2), versus the first-order
Symplectic Euler.
"""

import math
import unittest

from principia.energy import EnergyMonitor
from principia.gravity import compute_accelerations
from principia.integrators import (
    SymplecticEulerIntegrator,
    VelocityVerletIntegrator,
)
from principia.particle import Particle
from principia.vector2d import Vector2D


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

G = 1.0
M = 1.0
SAT_MASS = 1.0e-3
R_APO = 1.0


def constant_acceleration(value):
    return lambda particles: [value for _ in particles]


def make_eccentric_orbit(speed_factor):
    """Two-body orbit launched tangentially at apoapsis r=R_APO.

    ``speed_factor`` is the launch speed as a fraction of the circular speed;
    < 1 yields an eccentric orbit with periapsis closer in. Centre of mass is
    at rest. Returns ``(particles, period, eccentricity)``.
    """
    v_circ = math.sqrt(G * (M + SAT_MASS) / R_APO)
    v_rel = speed_factor * v_circ
    total = M + SAT_MASS
    v_sat = (M / total) * v_rel
    v_cen = -(SAT_MASS / total) * v_rel
    central = Particle(M, Vector2D(0.0, 0.0), Vector2D(0.0, v_cen))
    satellite = Particle(SAT_MASS, Vector2D(R_APO, 0.0), Vector2D(0.0, v_sat))
    eccentricity = 1.0 - (R_APO * v_rel * v_rel) / (G * total)
    semi_major = R_APO / (1.0 + eccentricity)
    period = 2.0 * math.pi * math.sqrt(semi_major ** 3 / (G * total))
    return [central, satellite], period, eccentricity


def max_relative_energy_drift(integrator, particles, dt, steps):
    """Run a simulation and return the worst |E - E0| / |E0| seen, via EnergyMonitor."""
    monitor = EnergyMonitor(G)
    accel_fn = lambda ps: compute_accelerations(ps, G)
    e0 = monitor.total(particles)
    worst = 0.0
    state = particles
    for _ in range(steps):
        state = integrator.step(state, dt, accel_fn)
        worst = max(worst, monitor.relative_drift(state, e0))
    return worst


def integrate(integrator, particles, dt, steps, accel_fn):
    state = particles
    for _ in range(steps):
        state = integrator.step(state, dt, accel_fn)
    return state


# --------------------------------------------------------------------------- #
# Single-step correctness (isolated via constant acceleration)
# --------------------------------------------------------------------------- #

class SingleStepTests(unittest.TestCase):
    def test_velocity_verlet_step(self):
        # mass=2, pos=(0,0), vel=(1,0); constant accel=(0,-10); dt=0.5.
        # a_old = (0,-10)
        # v_half = (1,0) + 0.5*(0,-10)*0.5 = (1,-2.5)
        # r_new  = (0,0) + (1,-2.5)*0.5     = (0.5,-1.25)
        # a_new  = (0,-10)  (constant)
        # v_new  = (1,-2.5) + 0.5*(0,-10)*0.5 = (1,-5)
        particle = Particle(2.0, Vector2D(0.0, 0.0), Vector2D(1.0, 0.0))
        (result,) = VelocityVerletIntegrator().step(
            [particle], 0.5, constant_acceleration(Vector2D(0.0, -10.0))
        )
        self.assertAlmostEqual(result.position.x, 0.5, places=12)
        self.assertAlmostEqual(result.position.y, -1.25, places=12)
        self.assertAlmostEqual(result.velocity.x, 1.0, places=12)
        self.assertAlmostEqual(result.velocity.y, -5.0, places=12)
        self.assertEqual(result.mass, 2.0)


# --------------------------------------------------------------------------- #
# Efficiency: exactly one force evaluation per step (plus one priming call)
# --------------------------------------------------------------------------- #

class ForceEvaluationCountTests(unittest.TestCase):
    def _counting_fn(self, counter):
        def fn(particles):
            counter[0] += 1
            return compute_accelerations(particles, G)
        return fn

    def test_one_evaluation_per_step_plus_priming(self):
        particles, _, _ = make_eccentric_orbit(0.6)
        counter = [0]
        fn = self._counting_fn(counter)
        integ = VelocityVerletIntegrator()

        steps = 50
        state = particles
        for _ in range(steps):
            state = integ.step(state, 0.001, fn)

        # N steps -> N + 1 force evaluations (one priming call on step 1).
        self.assertEqual(counter[0], steps + 1)

    def test_reset_forces_a_new_priming_evaluation(self):
        particles, _, _ = make_eccentric_orbit(0.6)
        counter = [0]
        fn = self._counting_fn(counter)
        integ = VelocityVerletIntegrator()

        s1 = integ.step(particles, 0.001, fn)   # 2 evals (prime + a_new)
        s2 = integ.step(s1, 0.001, fn)          # 1 eval (cache hit)
        self.assertEqual(counter[0], 3)

        integ.reset()
        _ = integ.step(s2, 0.001, fn)           # cache cleared -> 2 evals again
        self.assertEqual(counter[0], 5)

    def test_fresh_input_does_not_use_stale_cache(self):
        # Feeding a system the integrator did not just produce must recompute.
        particles, _, _ = make_eccentric_orbit(0.6)
        counter = [0]
        fn = self._counting_fn(counter)
        integ = VelocityVerletIntegrator()

        integ.step(particles, 0.001, fn)        # 2 evals
        integ.step(particles, 0.001, fn)        # same input object, not our
                                                # last output -> 2 evals again
        self.assertEqual(counter[0], 4)


# --------------------------------------------------------------------------- #
# Purity
# --------------------------------------------------------------------------- #

class PurityTests(unittest.TestCase):
    def test_step_does_not_mutate_inputs(self):
        particles = [
            Particle(2.0, Vector2D(0.0, 0.0), Vector2D(1.0, 0.0)),
            Particle(3.0, Vector2D(5.0, 0.0), Vector2D(0.0, 2.0)),
        ]
        before = [(p.mass, p.position, p.velocity) for p in particles]
        result = VelocityVerletIntegrator().step(
            particles, 0.1, constant_acceleration(Vector2D(0.0, -1.0))
        )
        after = [(p.mass, p.position, p.velocity) for p in particles]
        self.assertEqual(before, after)
        self.assertIsNot(result, particles)
        for original, updated in zip(particles, result):
            self.assertIsNot(updated, original)


# --------------------------------------------------------------------------- #
# Order of accuracy: Verlet is O(dt^2), Symplectic Euler is O(dt)
# --------------------------------------------------------------------------- #

class ConvergenceOrderTests(unittest.TestCase):
    """Halving dt should shrink the global error by ~4x for a 2nd-order method
    and by ~2x for a 1st-order method."""

    def _final_error_ratio(self, factory):
        particles, period, _ = make_eccentric_orbit(0.5)  # e = 0.75
        accel_fn = lambda ps: compute_accelerations(ps, G)
        sim_time = 0.5 * period  # half an orbit, through periapsis

        # High-accuracy reference (very fine velocity Verlet).
        ref_dt = period / 50000.0
        ref = integrate(
            VelocityVerletIntegrator(), particles, ref_dt,
            round(sim_time / ref_dt), accel_fn,
        )

        def err(dt):
            state = integrate(
                factory(), particles, dt, round(sim_time / dt), accel_fn
            )
            return (state[1].position - ref[1].position).magnitude()

        dt = period / 2000.0
        return err(dt) / err(dt / 2.0)

    def test_velocity_verlet_is_second_order(self):
        ratio = self._final_error_ratio(VelocityVerletIntegrator)
        # O(dt^2): error drops ~4x when dt is halved.
        self.assertGreater(ratio, 3.5)
        self.assertLess(ratio, 4.5)

    def test_symplectic_euler_is_first_order(self):
        ratio = self._final_error_ratio(SymplecticEulerIntegrator)
        # O(dt): error drops ~2x when dt is halved.
        self.assertGreater(ratio, 1.7)
        self.assertLess(ratio, 2.3)


# --------------------------------------------------------------------------- #
# CRUCIAL: energy conservation on a highly eccentric orbit, 2000 steps.
# Verlet's drift must be orders of magnitude below Symplectic Euler's.
# --------------------------------------------------------------------------- #

class EccentricOrbitEnergyTests(unittest.TestCase):
    STEPS = 2000

    def setUp(self):
        # speed_factor 0.6 -> eccentricity ~0.64; one full orbit over 2000 steps.
        self.particles, self.period, self.eccentricity = make_eccentric_orbit(0.6)
        self.dt = self.period / self.STEPS

    def test_orbit_is_highly_eccentric(self):
        # Sanity check the scenario really is eccentric.
        self.assertGreater(self.eccentricity, 0.6)

    def test_symplectic_euler_drift_is_significant(self):
        drift = max_relative_energy_drift(
            SymplecticEulerIntegrator(), self.particles, self.dt, self.STEPS
        )
        self.assertGreater(drift, 5.0e-3)

    def test_velocity_verlet_drift_is_tiny(self):
        drift = max_relative_energy_drift(
            VelocityVerletIntegrator(), self.particles, self.dt, self.STEPS
        )
        self.assertLess(drift, 1.0e-3)

    def test_verlet_is_orders_of_magnitude_better_than_symplectic_euler(self):
        se_drift = max_relative_energy_drift(
            SymplecticEulerIntegrator(), self.particles, self.dt, self.STEPS
        )
        vv_drift = max_relative_energy_drift(
            VelocityVerletIntegrator(), self.particles, self.dt, self.STEPS
        )
        # Observed ratio ~100x; assert a robust >30x (more than an order of
        # magnitude) to stay well clear of the boundary.
        self.assertGreater(se_drift, 30.0 * vv_drift)


if __name__ == "__main__":
    unittest.main()
