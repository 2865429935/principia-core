"""Immutable 2D Euclidean vector.

Level 0 -- State & Vector Layer of the ``principia-core`` engine.

This module is intentionally zero-dependency. It relies only on the Python
standard library (``math`` and ``dataclasses``) and contains no external
mathematical "black boxes" such as numpy. Every algebraic operation defined
in the Level 0 specification is implemented here from first principles.

Design constraints (see ``.kiro/steering.md``):

* No numpy or other external mathematical libraries.
* The vector is **immutable**: every operation returns a brand new
  ``Vector2D`` instance and never mutates an operand.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Vector2D:
    """An immutable vector ``(x, y)`` in 2D Euclidean space.

    Components are stored as ``float`` values. The instance is frozen, so any
    attempt to rebind ``x`` or ``y`` raises ``dataclasses.FrozenInstanceError``.
    All arithmetic operations return a new :class:`Vector2D`, leaving both
    operands unchanged.
    """

    x: float
    y: float

    def __post_init__(self) -> None:
        # Coerce components to ``float`` so integer inputs behave consistently
        # in subsequent floating-point arithmetic. ``object.__setattr__`` is the
        # sanctioned way to assign fields on a frozen dataclass during init.
        object.__setattr__(self, "x", float(self.x))
        object.__setattr__(self, "y", float(self.y))

    # -- Additive group operations -------------------------------------------

    def __add__(self, other: "Vector2D") -> "Vector2D":
        """Vector addition: ``A + B = (Ax + Bx, Ay + By)``."""
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Vector2D") -> "Vector2D":
        """Vector subtraction: ``A - B = (Ax - Bx, Ay - By)``."""
        if not isinstance(other, Vector2D):
            return NotImplemented
        return Vector2D(self.x - other.x, self.y - other.y)

    def __neg__(self) -> "Vector2D":
        """Additive inverse: ``-A = (-Ax, -Ay)``."""
        return Vector2D(-self.x, -self.y)

    # -- Scalar multiplication ------------------------------------------------

    def scale(self, scalar: float) -> "Vector2D":
        """Scalar multiplication: ``cA = (c * Ax, c * Ay)``."""
        return Vector2D(self.x * scalar, self.y * scalar)

    def __mul__(self, scalar: float) -> "Vector2D":
        """Support ``A * c`` for a numeric scalar ``c``."""
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            return NotImplemented
        return self.scale(scalar)

    def __rmul__(self, scalar: float) -> "Vector2D":
        """Support ``c * A`` for a numeric scalar ``c``."""
        if isinstance(scalar, bool) or not isinstance(scalar, (int, float)):
            return NotImplemented
        return self.scale(scalar)

    # -- Inner product and norms ---------------------------------------------

    def dot(self, other: "Vector2D") -> float:
        """Dot product: ``A . B = Ax * Bx + Ay * By``."""
        if not isinstance(other, Vector2D):
            raise TypeError(
                f"dot product requires a Vector2D, got {type(other).__name__}"
            )
        return self.x * other.x + self.y * other.y

    def magnitude_squared(self) -> float:
        """Squared magnitude: ``|A|^2 = A . A``.

        Cheaper than :meth:`magnitude` (no square root) and the preferred
        quantity for comparisons since it avoids floating-point error from
        ``sqrt``.
        """
        return self.dot(self)

    def magnitude(self) -> float:
        """Euclidean magnitude: ``|A| = sqrt(|A|^2)``."""
        return math.sqrt(self.magnitude_squared())

    # -- Convenience ----------------------------------------------------------

    def as_tuple(self) -> tuple:
        """Return the ``(x, y)`` components as a plain tuple of floats."""
        return (self.x, self.y)
