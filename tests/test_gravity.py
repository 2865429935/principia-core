"""Unit tests for :mod:`principia.gravity`.

Standard-library ``unittest`` only, to preserve the zero-dependency constraint.
Covers exact two-body accelerations, the purity of the force computation, edge
cases, and the headline invariant: Newton's Third Law expressed as
conservation of momentum (sum of m * a over the system is the zero vector).
"""

import math
import unittest

from principia.gravity import GravityForce, compute_accelerations
from principia.particle import Particle
from principia.vector2d import Vector2D


def total_momentum_rate(particles, accelerations):
    """Return sum_i (m_i * a_i) -- the net rate of change of momentum."""
    total = Vector2D(0.0, 0.0)
    for body, accel in zip(particles, accelerations):
        total = total + accel * body.mass
    return total


class TwoBodyExactTests(unittest.TestCase):
    """Accelerations must exactly match hand-computed values."""

    def test_axis_aligned_two_body(self):
        # G=1, m0=2, m1=3, separated by 5 along +y. |r|^3 = 125.
        # a0 = 1 * 3 * (0, 5) / 125 = (0, 0.12)
        # a1 = 1 * 2 * (0, -5) / 125 = (0, -0.08)
        p0 = Particle(2.0, Vector2D(0.0, 0.0), Vector2D(0.0, 0.0))
        p1 = Particle(3.0, Vector2D(0.0, 5.0), Vector2D(0.0, 0.0))

        a0, a1 = compute_accelerations([p0, p1], G=1.0)

        self.assertAlmostEqual(a0.x, 0.0, places=12)
        self.assertAlmostEqual(a0.y, 0.12, places=12)
        self.assertAlmostEqual(a1.x, 0.0, places=12)
        self.assertAlmostEqual(a1.y, -0.08, places=12)

    def test_diagonal_two_body(self):
        # offset (3,4) -> dist 5 -> |r|^3 = 125. G=1, m0=10, m1=20.
        # a0 = 20 * (3, 4) / 125 = (0.48, 0.64)
        # a1 = 10 * (-3, -4) / 125 = (-0.24, -0.32)
        p0 = Particle(10.0, Vector2D(1.0, 2.0), Vector2D(0.0, 0.0))
        p1 = Particle(20.0, Vector2D(4.0, 6.0), Vector2D(0.0, 0.0))

        a0, a1 = compute_accelerations([p0, p1], G=1.0)

        self.assertAlmostEqual(a0.x, 0.48, places=12)
        self.assertAlmostEqual(a0.y, 0.64, places=12)
        self.assertAlmostEqual(a1.x, -0.24, places=12)
        self.assertAlmostEqual(a1.y, -0.32, places=12)

    def test_acceleration_is_attractive(self):
        # a0 must point from p0 toward p1 (offset (3,4) -> positive components).
        p0 = Particle(10.0, Vector2D(1.0, 2.0), Vector2D(0.0, 0.0))
        p1 = Particle(20.0, Vector2D(4.0, 6.0), Vector2D(0.0, 0.0))
        a0, _ = compute_accelerations([p0, p1], G=1.0)
        self.assertGreater(a0.x, 0.0)
        self.assertGreater(a0.y, 0.0)

    def test_returns_vector2d_instances(self):
        p0 = Particle(1.0, Vector2D(0.0, 0.0), Vector2D(0.0, 0.0))
        p1 = Particle(1.0, Vector2D(1.0, 0.0), Vector2D(0.0, 0.0))
        result = compute_accelerations([p0, p1], G=1.0)
        self.assertTrue(all(isinstance(a, Vector2D) for a in result))


class NewtonsThirdLawTests(unittest.TestCase):
    """Conservation of momentum: sum of m * a must be the zero vector."""

    def assert_momentum_conserved(self, particles, G, places=10):
        accelerations = compute_accelerations(particles, G)
        net = total_momentum_rate(particles, accelerations)
        self.assertAlmostEqual(net.x, 0.0, places=places)
        self.assertAlmostEqual(net.y, 0.0, places=places)

    def test_two_body_equal_and_opposite(self):
        p0 = Particle(10.0, Vector2D(1.0, 2.0), Vector2D(0.0, 0.0))
        p1 = Particle(20.0, Vector2D(4.0, 6.0), Vector2D(0.0, 0.0))
        a0, a1 = compute_accelerations([p0, p1], G=1.0)
        # m0 * a0 == -(m1 * a1)
        f0 = a0 * p0.mass
        f1 = a1 * p1.mass
        self.assertAlmostEqual(f0.x, -f1.x, places=12)
        self.assertAlmostEqual(f0.y, -f1.y, places=12)

    def test_two_body_momentum_conserved(self):
        p0 = Particle(2.0, Vector2D(0.0, 0.0), Vector2D(0.0, 0.0))
        p1 = Particle(3.0, Vector2D(0.0, 5.0), Vector2D(0.0, 0.0))
        self.assert_momentum_conserved([p0, p1], G=1.0)

    def test_three_body_momentum_conserved(self):
        p0 = Particle(1.0, Vector2D(0.0, 0.0), Vector2D(0.0, 0.0))
        p1 = Particle(2.0, Vector2D(1.0, 0.0), Vector2D(0.0, 0.0))
        p2 = Particle(3.0, Vector2D(0.0, 1.0), Vector2D(0.0, 0.0))
        self.assert_momentum_conserved([p0, p1, p2], G=2.0)

    def test_three_body_momentum_conserved_realistic_G(self):
        p0 = Particle(5.97e5, Vector2D(0.0, 0.0), Vector2D(0.0, 0.0))
        p1 = Particle(7.34e3, Vector2D(3.84e2, 0.0), Vector2D(0.0, 0.0))
        p2 = Particle(1.0e3, Vector2D(-2.0e2, 1.5e2), Vector2D(0.0, 0.0))
        self.assert_momentum_conserved([p0, p1, p2], G=6.67430e-11, places=8)

    def test_asymmetric_four_body_momentum_conserved(self):
        particles = [
            Particle(1.5, Vector2D(-2.0, 3.0), Vector2D(0.0, 0.0)),
            Particle(4.0, Vector2D(5.0, -1.0), Vector2D(0.0, 0.0)),
            Particle(0.5, Vector2D(0.0, 0.0), Vector2D(0.0, 0.0)),
            Particle(9.0, Vector2D(-4.0, -6.0), Vector2D(0.0, 0.0)),
        ]
        self.assert_momentum_conserved(particles, G=3.0)


class PurityTests(unittest.TestCase):
    """The force computation must not mutate inputs or advance positions."""

    def test_inputs_are_not_modified(self):
        p0 = Particle(2.0, Vector2D(0.0, 0.0), Vector2D(1.0, 1.0))
        p1 = Particle(3.0, Vector2D(0.0, 5.0), Vector2D(-2.0, 0.0))
        particles = [p0, p1]

        before = [(p.mass, p.position, p.velocity) for p in particles]
        _ = compute_accelerations(particles, G=1.0)
        after = [(p.mass, p.position, p.velocity) for p in particles]

        self.assertEqual(before, after)
        # Positions specifically were not advanced.
        self.assertEqual(p0.position, Vector2D(0.0, 0.0))
        self.assertEqual(p1.position, Vector2D(0.0, 5.0))

    def test_returns_new_list(self):
        p0 = Particle(1.0, Vector2D(0.0, 0.0), Vector2D(0.0, 0.0))
        p1 = Particle(1.0, Vector2D(1.0, 0.0), Vector2D(0.0, 0.0))
        particles = [p0, p1]
        result = compute_accelerations(particles, G=1.0)
        self.assertIsNot(result, particles)
        self.assertEqual(len(result), len(particles))


class EdgeCaseTests(unittest.TestCase):
    def test_empty_system(self):
        self.assertEqual(compute_accelerations([], G=1.0), [])

    def test_single_particle_has_zero_acceleration(self):
        p = Particle(5.0, Vector2D(3.0, 4.0), Vector2D(0.0, 0.0))
        result = compute_accelerations([p], G=1.0)
        self.assertEqual(result, [Vector2D(0.0, 0.0)])

    def test_coincident_particles_raise(self):
        p0 = Particle(1.0, Vector2D(2.0, 2.0), Vector2D(0.0, 0.0))
        p1 = Particle(1.0, Vector2D(2.0, 2.0), Vector2D(0.0, 0.0))
        with self.assertRaises(ValueError):
            compute_accelerations([p0, p1], G=1.0)


class GravityForceClassTests(unittest.TestCase):
    def test_class_matches_function(self):
        p0 = Particle(2.0, Vector2D(0.0, 0.0), Vector2D(0.0, 0.0))
        p1 = Particle(3.0, Vector2D(0.0, 5.0), Vector2D(0.0, 0.0))
        particles = [p0, p1]

        via_func = compute_accelerations(particles, G=1.0)
        via_class = GravityForce(G=1.0).compute_accelerations(particles)

        self.assertEqual(via_func, via_class)


if __name__ == "__main__":
    unittest.main()
