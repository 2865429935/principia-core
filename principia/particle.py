"""Immutable point-mass particle.

Level 1 -- Particle and Dynamics Layer of the ``principia-core`` engine.

A :class:`Particle` bundles the dynamical state of a single point mass: its
``mass`` (a positive float), its ``position`` and its ``velocity`` (both
:class:`~principia.vector2d.Vector2D`). Like ``Vector2D`` it is immutable --
every state update returns a brand new ``Particle`` and never mutates an
existing one.

Design constraints (see ``.kiro/steering.md``):

* ZERO Black Box -- spatial state is expressed with the existing ``Vector2D``
  type; no numpy or other external mathematical libraries.
* Immutability -- updates produce new instances.
"""

from __future__ import annotations

from dataclasses import dataclass

from principia.vector2d import Vector2D


@dataclass(frozen=True)
class Particle:
    """An immutable 2D point mass.

    Attributes:
        mass: A strictly positive ``float``. Integer inputs are coerced.
        position: The particle's position as a :class:`Vector2D`.
        velocity: The particle's velocity as a :class:`Vector2D`.
    """

    mass: float
    position: Vector2D
    velocity: Vector2D

    def __post_init__(self) -> None:
        if not isinstance(self.position, Vector2D):
            raise TypeError(
                f"position must be a Vector2D, got {type(self.position).__name__}"
            )
        if not isinstance(self.velocity, Vector2D):
            raise TypeError(
                f"velocity must be a Vector2D, got {type(self.velocity).__name__}"
            )
        mass = float(self.mass)
        if mass <= 0.0:
            raise ValueError(f"mass must be strictly positive, got {mass}")
        # Frozen dataclass: assign the coerced float via object.__setattr__.
        object.__setattr__(self, "mass", mass)

    # -- Immutable state updates ---------------------------------------------

    def with_position(self, position: Vector2D) -> "Particle":
        """Return a copy of this particle at a new ``position``."""
        return Particle(self.mass, position, self.velocity)

    def with_velocity(self, velocity: Vector2D) -> "Particle":
        """Return a copy of this particle with a new ``velocity``."""
        return Particle(self.mass, self.position, velocity)

    def with_state(self, position: Vector2D, velocity: Vector2D) -> "Particle":
        """Return a copy of this particle with new ``position`` and ``velocity``."""
        return Particle(self.mass, position, velocity)

    # -- Derived quantities ---------------------------------------------------

    def momentum(self) -> Vector2D:
        """Return the linear momentum ``p = m * v``."""
        return self.velocity * self.mass
