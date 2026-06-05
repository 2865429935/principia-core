"""Unit tests for :class:`principia.particle.Particle`.

Standard-library ``unittest`` only, to preserve the zero-dependency constraint.
"""

import unittest
from dataclasses import FrozenInstanceError

from principia.particle import Particle
from principia.vector2d import Vector2D


class ConstructionTests(unittest.TestCase):
    def test_stores_state(self):
        pos = Vector2D(1.0, 2.0)
        vel = Vector2D(-3.0, 4.0)
        p = Particle(5.0, pos, vel)
        self.assertEqual(p.mass, 5.0)
        self.assertEqual(p.position, pos)
        self.assertEqual(p.velocity, vel)

    def test_mass_coerced_to_float(self):
        p = Particle(7, Vector2D(0.0, 0.0), Vector2D(0.0, 0.0))
        self.assertIsInstance(p.mass, float)
        self.assertEqual(p.mass, 7.0)

    def test_zero_mass_is_rejected(self):
        with self.assertRaises(ValueError):
            Particle(0.0, Vector2D(0.0, 0.0), Vector2D(0.0, 0.0))

    def test_negative_mass_is_rejected(self):
        with self.assertRaises(ValueError):
            Particle(-1.0, Vector2D(0.0, 0.0), Vector2D(0.0, 0.0))

    def test_non_vector_position_is_rejected(self):
        with self.assertRaises(TypeError):
            Particle(1.0, (0.0, 0.0), Vector2D(0.0, 0.0))  # type: ignore[arg-type]

    def test_non_vector_velocity_is_rejected(self):
        with self.assertRaises(TypeError):
            Particle(1.0, Vector2D(0.0, 0.0), (0.0, 0.0))  # type: ignore[arg-type]


class ImmutabilityTests(unittest.TestCase):
    def setUp(self):
        self.p = Particle(2.0, Vector2D(1.0, 1.0), Vector2D(0.0, 0.0))

    def test_cannot_reassign_mass(self):
        with self.assertRaises(FrozenInstanceError):
            self.p.mass = 9.0  # type: ignore[misc]

    def test_cannot_reassign_position(self):
        with self.assertRaises(FrozenInstanceError):
            self.p.position = Vector2D(9.0, 9.0)  # type: ignore[misc]

    def test_cannot_reassign_velocity(self):
        with self.assertRaises(FrozenInstanceError):
            self.p.velocity = Vector2D(9.0, 9.0)  # type: ignore[misc]


class StateUpdateTests(unittest.TestCase):
    def setUp(self):
        self.p = Particle(2.0, Vector2D(1.0, 1.0), Vector2D(3.0, 4.0))

    def test_with_position_returns_new_instance(self):
        new = self.p.with_position(Vector2D(5.0, 6.0))
        self.assertIsNot(new, self.p)
        self.assertEqual(new.position, Vector2D(5.0, 6.0))
        # Mass and velocity carried over.
        self.assertEqual(new.mass, self.p.mass)
        self.assertEqual(new.velocity, self.p.velocity)
        # Original untouched.
        self.assertEqual(self.p.position, Vector2D(1.0, 1.0))

    def test_with_velocity_returns_new_instance(self):
        new = self.p.with_velocity(Vector2D(7.0, 8.0))
        self.assertIsNot(new, self.p)
        self.assertEqual(new.velocity, Vector2D(7.0, 8.0))
        self.assertEqual(self.p.velocity, Vector2D(3.0, 4.0))

    def test_with_state_returns_new_instance(self):
        new = self.p.with_state(Vector2D(5.0, 6.0), Vector2D(7.0, 8.0))
        self.assertIsNot(new, self.p)
        self.assertEqual(new.position, Vector2D(5.0, 6.0))
        self.assertEqual(new.velocity, Vector2D(7.0, 8.0))
        self.assertEqual(new.mass, self.p.mass)


class DerivedQuantityTests(unittest.TestCase):
    def test_momentum(self):
        p = Particle(3.0, Vector2D(0.0, 0.0), Vector2D(2.0, -5.0))
        self.assertEqual(p.momentum(), Vector2D(6.0, -15.0))

    def test_equality_and_hashable(self):
        a = Particle(1.0, Vector2D(1.0, 2.0), Vector2D(3.0, 4.0))
        b = Particle(1.0, Vector2D(1.0, 2.0), Vector2D(3.0, 4.0))
        self.assertEqual(a, b)
        self.assertEqual(len({a, b}), 1)


if __name__ == "__main__":
    unittest.main()
