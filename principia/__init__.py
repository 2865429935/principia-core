"""principia-core: a zero-dependency mathematical physics engine.

* Level 0 (State & Vector Layer): the immutable
  :class:`~principia.vector2d.Vector2D` type.
* Level 1 (Particle and Dynamics Layer): the immutable
  :class:`~principia.particle.Particle` and the Newtonian gravity force law in
  :mod:`principia.gravity`.
* Level 2 (Numerical Integrators): the :class:`~principia.integrators.Integrator`
  base class with :class:`~principia.integrators.ExplicitEulerIntegrator` and
  :class:`~principia.integrators.SymplecticEulerIntegrator`.
"""

from principia.gravity import GravityForce, compute_accelerations
from principia.integrators import (
    ExplicitEulerIntegrator,
    Integrator,
    SymplecticEulerIntegrator,
)
from principia.particle import Particle
from principia.vector2d import Vector2D

__all__ = [
    "Vector2D",
    "Particle",
    "GravityForce",
    "compute_accelerations",
    "Integrator",
    "ExplicitEulerIntegrator",
    "SymplecticEulerIntegrator",
]
