"""Unit tests for :class:`principia.vector2d.Vector2D`.

These tests use only the standard-library ``unittest`` framework to honour the
project's zero-dependency constraint. Floating-point comparisons use
``assertAlmostEqual`` (7 decimal places by default) to guard against binary
representation error.
"""

import math
import unittest
from dataclasses import FrozenInstanceError

from principia.vector2d import Vector2D


class ConstructionTests(unittest.TestCase):
    def test_stores_components(self):
        v = Vector2D(1.5, -2.25)
        self.assertAlmostEqual(v.x, 1.5)
        self.assertAlmostEqual(v.y, -2.25)

    def test_integer_inputs_are_coerced_to_float(self):
        v = Vector2D(3, 4)
        self.assertIsInstance(v.x, float)
        self.assertIsInstance(v.y, float)
        self.assertEqual(v.x, 3.0)
        self.assertEqual(v.y, 4.0)

    def test_as_tuple(self):
        self.assertEqual(Vector2D(1, 2).as_tuple(), (1.0, 2.0))


class ImmutabilityTests(unittest.TestCase):
    def test_cannot_reassign_x(self):
        v = Vector2D(1.0, 2.0)
        with self.assertRaises(FrozenInstanceError):
            v.x = 9.0  # type: ignore[misc]

    def test_cannot_reassign_y(self):
        v = Vector2D(1.0, 2.0)
        with self.assertRaises(FrozenInstanceError):
            v.y = 9.0  # type: ignore[misc]

    def test_operations_do_not_mutate_operands(self):
        a = Vector2D(1.0, 2.0)
        b = Vector2D(3.0, 4.0)
        _ = a + b
        _ = a - b
        _ = a * 5
        _ = a.dot(b)
        # Operands remain exactly as constructed.
        self.assertEqual(a.as_tuple(), (1.0, 2.0))
        self.assertEqual(b.as_tuple(), (3.0, 4.0))

    def test_operations_return_new_instances(self):
        a = Vector2D(1.0, 2.0)
        b = Vector2D(3.0, 4.0)
        self.assertIsNot(a + b, a)
        self.assertIsNot(a * 1, a)


class AdditionTests(unittest.TestCase):
    def test_addition(self):
        result = Vector2D(1.0, 2.0) + Vector2D(3.0, 4.0)
        self.assertAlmostEqual(result.x, 4.0)
        self.assertAlmostEqual(result.y, 6.0)

    def test_addition_with_negative_and_float(self):
        result = Vector2D(-1.5, 2.5) + Vector2D(0.25, -0.5)
        self.assertAlmostEqual(result.x, -1.25)
        self.assertAlmostEqual(result.y, 2.0)

    def test_addition_is_commutative(self):
        a, b = Vector2D(1.1, -2.2), Vector2D(3.3, 4.4)
        self.assertEqual((a + b).as_tuple(), (b + a).as_tuple())

    def test_zero_vector_is_additive_identity(self):
        a = Vector2D(7.0, -8.0)
        zero = Vector2D(0.0, 0.0)
        self.assertEqual((a + zero).as_tuple(), a.as_tuple())

    def test_adding_non_vector_returns_notimplemented(self):
        with self.assertRaises(TypeError):
            _ = Vector2D(1.0, 2.0) + 5  # type: ignore[operator]


class SubtractionTests(unittest.TestCase):
    def test_subtraction(self):
        result = Vector2D(5.0, 7.0) - Vector2D(2.0, 3.0)
        self.assertAlmostEqual(result.x, 3.0)
        self.assertAlmostEqual(result.y, 4.0)

    def test_subtracting_self_yields_zero(self):
        a = Vector2D(3.14, -2.71)
        result = a - a
        self.assertAlmostEqual(result.x, 0.0)
        self.assertAlmostEqual(result.y, 0.0)

    def test_negation(self):
        result = -Vector2D(1.5, -2.5)
        self.assertAlmostEqual(result.x, -1.5)
        self.assertAlmostEqual(result.y, 2.5)


class ScalarMultiplicationTests(unittest.TestCase):
    def test_scale_method(self):
        result = Vector2D(2.0, -3.0).scale(2.5)
        self.assertAlmostEqual(result.x, 5.0)
        self.assertAlmostEqual(result.y, -7.5)

    def test_left_multiplication(self):
        result = Vector2D(2.0, -3.0) * 2.5
        self.assertAlmostEqual(result.x, 5.0)
        self.assertAlmostEqual(result.y, -7.5)

    def test_right_multiplication(self):
        result = 2.5 * Vector2D(2.0, -3.0)
        self.assertAlmostEqual(result.x, 5.0)
        self.assertAlmostEqual(result.y, -7.5)

    def test_left_and_right_multiplication_agree(self):
        v = Vector2D(1.25, -6.5)
        self.assertEqual((v * 3).as_tuple(), (3 * v).as_tuple())

    def test_scaling_by_zero(self):
        result = Vector2D(9.9, -9.9) * 0
        self.assertAlmostEqual(result.x, 0.0)
        self.assertAlmostEqual(result.y, 0.0)

    def test_scaling_by_negative(self):
        result = Vector2D(1.0, -2.0) * -3
        self.assertAlmostEqual(result.x, -3.0)
        self.assertAlmostEqual(result.y, 6.0)

    def test_multiplying_by_vector_is_not_supported(self):
        # Scalar multiplication only; vector * vector must not silently work.
        with self.assertRaises(TypeError):
            _ = Vector2D(1.0, 2.0) * Vector2D(3.0, 4.0)  # type: ignore[operator]

    def test_bool_is_not_treated_as_scalar(self):
        # bool is a subclass of int; ensure it is rejected to avoid surprises.
        with self.assertRaises(TypeError):
            _ = Vector2D(1.0, 2.0) * True  # type: ignore[operator]


class DotProductTests(unittest.TestCase):
    def test_dot_product(self):
        result = Vector2D(1.0, 2.0).dot(Vector2D(3.0, 4.0))
        self.assertAlmostEqual(result, 11.0)  # 1*3 + 2*4

    def test_orthogonal_vectors_have_zero_dot(self):
        result = Vector2D(1.0, 0.0).dot(Vector2D(0.0, 1.0))
        self.assertAlmostEqual(result, 0.0)

    def test_dot_is_commutative(self):
        a, b = Vector2D(1.3, -2.7), Vector2D(4.1, 5.9)
        self.assertAlmostEqual(a.dot(b), b.dot(a))

    def test_dot_with_floats(self):
        result = Vector2D(0.1, 0.2).dot(Vector2D(0.3, 0.4))
        self.assertAlmostEqual(result, 0.11)  # 0.03 + 0.08

    def test_dot_requires_vector(self):
        with self.assertRaises(TypeError):
            Vector2D(1.0, 2.0).dot(5)  # type: ignore[arg-type]


class MagnitudeTests(unittest.TestCase):
    def test_magnitude_squared(self):
        self.assertAlmostEqual(Vector2D(3.0, 4.0).magnitude_squared(), 25.0)

    def test_magnitude_squared_equals_dot_with_self(self):
        v = Vector2D(-2.5, 6.0)
        self.assertAlmostEqual(v.magnitude_squared(), v.dot(v))

    def test_magnitude_of_3_4_is_5(self):
        self.assertAlmostEqual(Vector2D(3.0, 4.0).magnitude(), 5.0)

    def test_magnitude_of_zero_vector(self):
        self.assertAlmostEqual(Vector2D(0.0, 0.0).magnitude(), 0.0)

    def test_magnitude_irrational(self):
        # |(1, 1)| = sqrt(2)
        self.assertAlmostEqual(Vector2D(1.0, 1.0).magnitude(), math.sqrt(2))

    def test_magnitude_is_nonnegative(self):
        self.assertGreaterEqual(Vector2D(-3.0, -4.0).magnitude(), 0.0)

    def test_magnitude_is_sqrt_of_magnitude_squared(self):
        v = Vector2D(7.2, -1.4)
        self.assertAlmostEqual(v.magnitude(), math.sqrt(v.magnitude_squared()))


class EqualityAndReprTests(unittest.TestCase):
    def test_value_equality(self):
        self.assertEqual(Vector2D(1, 2), Vector2D(1.0, 2.0))

    def test_inequality(self):
        self.assertNotEqual(Vector2D(1.0, 2.0), Vector2D(1.0, 2.5))

    def test_hashable(self):
        # Frozen dataclasses are hashable; usable in sets/dict keys.
        points = {Vector2D(1.0, 2.0), Vector2D(1.0, 2.0), Vector2D(3.0, 4.0)}
        self.assertEqual(len(points), 2)


class AlgebraicLawTests(unittest.TestCase):
    """Spot-check vector-space axioms with concrete operands."""

    def test_distributivity_of_scalar_over_addition(self):
        a, b, c = Vector2D(1.0, -2.0), Vector2D(3.5, 4.5), 2.0
        left = (a + b) * c
        right = (a * c) + (b * c)
        self.assertAlmostEqual(left.x, right.x)
        self.assertAlmostEqual(left.y, right.y)

    def test_dot_distributes_over_addition(self):
        a, b, c = Vector2D(1.0, 2.0), Vector2D(3.0, 4.0), Vector2D(-1.0, 5.0)
        self.assertAlmostEqual(a.dot(b + c), a.dot(b) + a.dot(c))


if __name__ == "__main__":
    unittest.main()
