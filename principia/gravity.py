"""Newtonian gravity force law.

Level 1 -- Particle and Dynamics Layer of the ``principia-core`` engine.

This module computes gravitational accelerations for an N-body system. The
instantaneous acceleration of particle ``i`` induced by particle ``j`` follows
Newton's Law of Universal Gravitation:

    a_i = G * m_j * (r_j - r_i) / |r_j - r_i|^3

The net acceleration on a particle is the sum of these pairwise contributions
over every other particle in the system.

Design constraints (see ``.kiro/steering.md``):

* ZERO Black Box -- all spatial math uses the existing
  :class:`~principia.vector2d.Vector2D` type; no numpy or external libraries.
* :func:`compute_accelerations` is a *pure* function: it reads particle state
  and returns accelerations only. It NEVER mutates the input particles or
  updates positions -- it strictly returns derivatives.
"""

from __future__ import annotations

import math
from typing import List, Sequence

from principia.particle import Particle
from principia.vector2d import Vector2D

_ZERO = Vector2D(0.0, 0.0)


def compute_accelerations(particles: Sequence[Particle], G: float) -> List[Vector2D]:
    """Compute the net gravitational acceleration for every particle.

    Args:
        particles: The bodies in the system. Treated as read-only.
        G: The gravitational constant for this simulation.

    Returns:
        A new list of :class:`Vector2D`, one acceleration per input particle,
        in the same order. A particle in isolation (or a single-particle
        system) receives the zero vector.

    Raises:
        ValueError: If two distinct particles share the same position, which
            makes the inverse-cube term singular.

    Notes:
        This is an O(n^2) all-pairs computation. It is pure: the input
        particles are never modified and no positions are advanced.
    """
    n = len(particles)
    accelerations: List[Vector2D] = [_ZERO] * n

    for i in range(n):
        body_i = particles[i]
        net_i = _ZERO
        for j in range(n):
            if i == j:
                continue
            body_j = particles[j]

            # Displacement from i toward j: (r_j - r_i).
            offset = body_j.position - body_i.position
            distance_squared = offset.magnitude_squared()
            if distance_squared == 0.0:
                raise ValueError(
                    f"particles {i} and {j} are coincident; "
                    "gravitational acceleration is singular"
                )
            distance = math.sqrt(distance_squared)
            distance_cubed = distance_squared * distance

            # a_i += G * m_j * (r_j - r_i) / |r_j - r_i|^3
            net_i = net_i + offset * (G * body_j.mass / distance_cubed)

        accelerations[i] = net_i

    return accelerations


class GravityForce:
    """Class wrapper around the Newtonian gravity acceleration law.

    Provided for callers who prefer an object that carries the gravitational
    constant ``G``. It simply delegates to the pure module-level
    :func:`compute_accelerations` and holds no mutable state.
    """

    def __init__(self, G: float) -> None:
        self.G = float(G)

    def compute_accelerations(self, particles: Sequence[Particle]) -> List[Vector2D]:
        """Return net accelerations for ``particles`` using this instance's ``G``."""
        return compute_accelerations(particles, self.G)
