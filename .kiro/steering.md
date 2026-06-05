# Specification: Level 0 - Vector Algebra

## 1. Mathematical Definitions
We operate in a 2D Euclidean space. A vector is represented by a tuple of floats `(x, y)`.

## 2. Core Operations
Implement a `Vector2D` class with the following pure mathematical properties.
- Addition: A + B = (Ax + Bx, Ay + By)
- Subtraction: A - B = (Ax - Bx, Ay - By)
- Scalar Multiplication: cA = (c * Ax, c * Ay)
- Dot Product: A dot B = Ax * Bx + Ay * By
- Magnitude Squared: |A|^2 = A dot A
- Magnitude: |A| = sqrt(|A|^2)

## 3. Strict Constraints (ZERO Black Box)
- DO NOT import `numpy` or any other external mathematical libraries. 
- The class must be immutable. Operations like addition must return a NEW `Vector2D` instance, not modify the existing one.
- Generate a comprehensive Unit Test file to verify all algebraic operations with floating-point precision assertions.
