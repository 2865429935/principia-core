"""Mechanical-energy (Hamiltonian) monitoring for an N-body system.

Level 3 -- Invariant Monitor of the ``principia-core`` engine.

The total mechanical energy of a self-gravitating system is the Hamiltonian

    E = T + U

where the kinetic energy is

    T = Sum_i  0.5 * m_i * |v_i|^2

and the gravitational potential energy is the pairwise sum

    U = Sum_{i<j}  -G * m_i * m_j / |r_i - r_j|

For an exact trajectory ``E`` is conserved. Tracking how much a numerical
integrator lets ``E`` drift is the standard way to judge its quality, which is
exactly what the Level 3 tests use to compare integrators.

Design constraints (see ``.kiro/steering.md``):

* ZERO Black Box -- builds only on :class:`~principia.particle.Particle` and
  :class:`~principia.vector2d.Vector2D`; no numpy or external libraries.
* The monitor is read-only: it never mutates the particles it inspects.
"""

from __future__ import annotations

from typing import Sequence

from principia.particle import Particle


def kinetic_energy(particles: Sequence[Particle]) -> float:
    """Return the total kinetic energy ``Sum 0.5 * m * |v|^2``."""
    total = 0.0
    for p in particles:
        total += 0.5 * p.mass * p.velocity.magnitude_squared()
    return total


def potential_energy(particles: Sequence[Particle], G: float) -> float:
    """Return the total gravitational potential energy.

    ``U = Sum_{i<j} -G * m_i * m_j / |r_i - r_j|``.

    Raises:
        ValueError: If two particles are coincident (the potential diverges).
    """
    total = 0.0
    n = len(particles)
    for i in range(n):
        for j in range(i + 1, n):
            separation = (particles[i].position - particles[j].position).magnitude()
            if separation == 0.0:
                raise ValueError(
                    f"particles {i} and {j} are coincident; "
                    "gravitational potential is singular"
                )
            total += -G * particles[i].mass * particles[j].mass / separation
    return total


def total_energy(particles: Sequence[Particle], G: float) -> float:
    """Return the total mechanical energy ``E = T + U``."""
    return kinetic_energy(particles) + potential_energy(particles, G)


class EnergyMonitor:
    """Computes the Hamiltonian of a system for a fixed gravitational constant.

    Convenience wrapper around the pure module-level functions for callers who
    want to bind ``G`` once and repeatedly sample a trajectory's energy.
    """

    def __init__(self, G: float) -> None:
        self.G = float(G)

    def kinetic(self, particles: Sequence[Particle]) -> float:
        """Return the kinetic energy ``T`` of ``particles``."""
        return kinetic_energy(particles)

    def potential(self, particles: Sequence[Particle]) -> float:
        """Return the potential energy ``U`` of ``particles`` for this ``G``."""
        return potential_energy(particles, self.G)

    def total(self, particles: Sequence[Particle]) -> float:
        """Return the total energy ``E = T + U`` of ``particles``."""
        return total_energy(particles, self.G)

    def relative_drift(
        self, particles: Sequence[Particle], reference_energy: float
    ) -> float:
        """Return ``|E(particles) - reference_energy| / |reference_energy|``."""
        return abs(
            (self.total(particles) - reference_energy) / reference_energy
        )
