"""principia-core: a zero-dependency mathematical physics engine.

* Level 0 (State & Vector Layer): the immutable
  :class:`~principia.vector2d.Vector2D` type.
* Level 1 (Particle and Dynamics Layer): the immutable
  :class:`~principia.particle.Particle` and the Newtonian gravity force law in
  :mod:`principia.gravity`.
"""

from principia.gravity import GravityForce, compute_accelerations
from principia.particle import Particle
from principia.vector2d import Vector2D

__all__ = [
    "Vector2D",
    "Particle",
    "GravityForce",
    "compute_accelerations",
]
