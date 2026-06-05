"""Unit tests for :mod:`principia.energy`.

Standard-library ``unittest`` only, to preserve the zero-dependency constraint.
"""

import unittest

from principia.energy import (
    EnergyMonitor,
    kinetic_energy,
    potential_energy,
    total_energy,
)
from principia.particle import Particle
from principia.vector2d import Vector2D


class EnergyFunctionTests(unittest.TestCase):
    def setUp(self):
        # p0: m=2 at (0,0) moving (3,4) -> KE = 0.5*2*25 = 25
        # p1: m=3 at (0,4) at rest      -> KE = 0
        # separation = 4 -> U = -G*2*3/4 = -1.5 (G=1)
        self.p0 = Particle(2.0, Vector2D(0.0, 0.0), Vector2D(3.0, 4.0))
        self.p1 = Particle(3.0, Vector2D(0.0, 4.0), Vector2D(0.0, 0.0))
        self.system = [self.p0, self.p1]

    def test_kinetic_energy(self):
        self.assertAlmostEqual(kinetic_energy(self.system), 25.0)

    def test_potential_energy(self):
        self.assertAlmostEqual(potential_energy(self.system, G=1.0), -1.5)

    def test_total_energy_is_sum(self):
        self.assertAlmostEqual(total_energy(self.system, G=1.0), 23.5)

    def test_potential_scales_with_G(self):
        self.assertAlmostEqual(potential_energy(self.system, G=2.0), -3.0)

    def test_coincident_particles_have_singular_potential(self):
        a = Particle(1.0, Vector2D(1.0, 1.0), Vector2D(0.0, 0.0))
        b = Particle(1.0, Vector2D(1.0, 1.0), Vector2D(0.0, 0.0))
        with self.assertRaises(ValueError):
            potential_energy([a, b], G=1.0)

    def test_single_particle_has_no_potential(self):
        self.assertEqual(potential_energy([self.p0], G=1.0), 0.0)


class EnergyMonitorTests(unittest.TestCase):
    def setUp(self):
        self.p0 = Particle(2.0, Vector2D(0.0, 0.0), Vector2D(3.0, 4.0))
        self.p1 = Particle(3.0, Vector2D(0.0, 4.0), Vector2D(0.0, 0.0))
        self.system = [self.p0, self.p1]
        self.monitor = EnergyMonitor(G=1.0)

    def test_matches_module_functions(self):
        self.assertAlmostEqual(self.monitor.kinetic(self.system), 25.0)
        self.assertAlmostEqual(self.monitor.potential(self.system), -1.5)
        self.assertAlmostEqual(self.monitor.total(self.system), 23.5)

    def test_relative_drift(self):
        # Current total is 23.5; against a reference of 23.0 the drift is
        # |23.5 - 23.0| / 23.0.
        self.assertAlmostEqual(
            self.monitor.relative_drift(self.system, 23.0),
            0.5 / 23.0,
        )

    def test_zero_drift_against_self(self):
        e = self.monitor.total(self.system)
        self.assertAlmostEqual(self.monitor.relative_drift(self.system, e), 0.0)


if __name__ == "__main__":
    unittest.main()
