import numpy as np
import pytest

from machine_learning_from_scratch.cost import compute_cost


def test_cost_is_zero_when_the_line_fits_perfectly():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([15.0, 18.0, 21.0])  # exactly 3 * x + 12

    assert compute_cost(x, y, w=3, b=12) == 0.0


def test_cost_matches_a_hand_calculation():
    x = np.array([1.0, 2.0, 3.0])
    y = np.array([1.0, 2.0, 3.0])

    # With w=0 and b=0 every prediction is 0, so the errors are -1, -2, -3.
    # Squared: 1 + 4 + 9 = 14. Divided by 2m = 6: 14 / 6.
    assert compute_cost(x, y, w=0, b=0) == pytest.approx(14 / 6)
