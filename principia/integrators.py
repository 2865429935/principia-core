"""Numerical integrators that advance system state from ``t`` to ``t + dt``.

Level 2 -- Numerical Integrators of the ``principia-core`` engine.

Two first-order integrators are provided. Both evaluate the acceleration once,
at the current particle positions, and then update each particle. They differ
only in *which* velocity is used to advance the position:

* :class:`ExplicitEulerIntegrator` uses the **old** velocity (``v_old``). This
  is the classic forward Euler method. For oscillatory/orbital (Hamiltonian)
  systems it systematically injects energy, so orbits spiral outward and
  diverge over time.
* :class:`SymplecticEulerIntegrator` uses the **new** velocity (``v_new``).
  This semi-implicit method is symplectic: it does not conserve energy exactly
  but keeps it bounded, so orbits remain stable and periodic.

Design constraints (see ``.kiro/steering.md``):

* ZERO Black Box -- everything builds on :class:`~principia.vector2d.Vector2D`
  and :class:`~principia.particle.Particle`.
* :meth:`Integrator.step` is *pure*: it returns a brand new list of new
  ``Particle`` instances and never mutates its inputs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, List, Sequence

from principia.particle import Particle
from principia.vector2d import Vector2D

# An acceleration provider maps the current particles to their accelerations.
# Callers typically bind the gravitational constant, e.g.::
#
#     accel_fn = lambda ps: compute_accelerations(ps, G)
AccelerationFn = Callable[[Sequence[Particle]], List[Vector2D]]


class Integrator(ABC):
    """Abstract base class for a one-step numerical integrator."""

    @abstractmethod
    def step(
        self,
        particles: Sequence[Particle],
        dt: float,
        compute_accelerations: AccelerationFn,
    ) -> List[Particle]:
        """Advance ``particles`` by one time step ``dt``.

        Args:
            particles: The current system state. Treated as read-only.
            dt: The time step.
            compute_accelerations: A callable returning one acceleration
                (:class:`Vector2D`) per particle, evaluated at the current
                positions.

        Returns:
            A new list of new :class:`Particle` instances. The input particles
            are never mutated.
        """
        raise NotImplementedError


class ExplicitEulerIntegrator(Integrator):
    """Forward (explicit) Euler.

    ``v_new = v_old + a * dt``
    ``r_new = r_old + v_old * dt``  (position uses the *old* velocity)
    """

    def step(
        self,
        particles: Sequence[Particle],
        dt: float,
        compute_accelerations: AccelerationFn,
    ) -> List[Particle]:
        accelerations = compute_accelerations(particles)
        updated: List[Particle] = []
        for particle, acceleration in zip(particles, accelerations):
            new_velocity = particle.velocity + acceleration * dt
            # Explicit Euler advances position with the OLD velocity.
            new_position = particle.position + particle.velocity * dt
            updated.append(particle.with_state(new_position, new_velocity))
        return updated


class SymplecticEulerIntegrator(Integrator):
    """Semi-implicit (symplectic) Euler.

    ``v_new = v_old + a * dt``
    ``r_new = r_old + v_new * dt``  (position uses the *new* velocity)
    """

    def step(
        self,
        particles: Sequence[Particle],
        dt: float,
        compute_accelerations: AccelerationFn,
    ) -> List[Particle]:
        accelerations = compute_accelerations(particles)
        updated: List[Particle] = []
        for particle, acceleration in zip(particles, accelerations):
            new_velocity = particle.velocity + acceleration * dt
            # Symplectic Euler advances position with the NEW velocity.
            new_position = particle.position + new_velocity * dt
            updated.append(particle.with_state(new_position, new_velocity))
        return updated
