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
from typing import Callable, List, Optional, Sequence

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



class VelocityVerletIntegrator(Integrator):
    """Second-order symplectic velocity Verlet (Strang / kick-drift-kick).

    The full step is the Strang splitting

        v_half = v_old  + 0.5 * a_old * dt        (half kick)
        r_new  = r_old  + v_half * dt             (drift)
        a_new  = a(r_new)                         (force eval)
        v_new  = v_half + 0.5 * a_new * dt        (half kick)

    This is time-reversible and globally second-order accurate, O(dt^2).

    Efficiency requirement (see ``.kiro/steering.md``): a full step needs the
    acceleration at the *old* positions (``a_old``) and at the *new* positions
    (``a_new``). The naive implementation would therefore call
    ``compute_accelerations`` twice per step. Instead, ``a_new`` is cached and
    reused as the next step's ``a_old`` -- so when the caller feeds each step's
    output straight back in (the normal simulation loop), the integrator
    performs exactly **one** force evaluation per step, plus a single priming
    evaluation on the very first step (N steps -> N + 1 evaluations).

    Caching is keyed on object identity of the returned particle list, so the
    fast path only triggers when the same trajectory is being advanced. Any
    other input (a fresh system, or :meth:`reset`) safely recomputes ``a_old``.

    The step remains pure with respect to its inputs: it never mutates the
    particles passed in and always returns a brand new list of new particles.
    """

    def __init__(self) -> None:
        self._cached_output: Optional[List[Particle]] = None
        self._cached_accelerations: Optional[List[Vector2D]] = None

    def reset(self) -> None:
        """Discard cached accelerations (use when starting a new trajectory)."""
        self._cached_output = None
        self._cached_accelerations = None

    def step(
        self,
        particles: Sequence[Particle],
        dt: float,
        compute_accelerations: AccelerationFn,
    ) -> List[Particle]:
        # Reuse a_new from the previous step iff the caller fed our output back.
        if (
            self._cached_accelerations is not None
            and particles is self._cached_output
        ):
            a_old = self._cached_accelerations
        else:
            a_old = compute_accelerations(particles)

        half_dt = 0.5 * dt

        # Half kick + drift -> particles at their new positions.
        half_velocities: List[Vector2D] = []
        moved: List[Particle] = []
        for particle, acceleration in zip(particles, a_old):
            v_half = particle.velocity + acceleration * half_dt
            new_position = particle.position + v_half * dt
            half_velocities.append(v_half)
            # Velocity here is a placeholder (gravity ignores it); it carries
            # the new position into the single force evaluation below.
            moved.append(particle.with_state(new_position, v_half))

        # The one force evaluation of the step, at the new positions.
        a_new = compute_accelerations(moved)

        # Second half kick to finalise the velocities.
        result: List[Particle] = []
        for moved_particle, v_half, acceleration in zip(
            moved, half_velocities, a_new
        ):
            v_new = v_half + acceleration * half_dt
            result.append(moved_particle.with_velocity(v_new))

        self._cached_output = result
        self._cached_accelerations = a_new
        return result
