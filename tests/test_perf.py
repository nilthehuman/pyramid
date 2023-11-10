"""Performance tests for the validation of the main working hypothesis."""

import pytest
from time import time

from ..src.pyramid import cells_from_floats, ParadigmaticSystem


def test_is_monotonic_fast_enough_positive():
    matrix = [ [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9],
               [1.0, 0.9, 0.8, 0.7, 0.6],
               [0.5, 0.4, 0.3, 0.2, 0.1],
               [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9],
               [1.0, 0.9, 0.8, 0.7, 0.6],
               [0.5, 0.4, 0.3, 0.2, 0.1] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    start = time()
    assert para.can_be_made_monotonic()
    elapsed_time = time() - start
    assert elapsed_time < 2


def test_is_monotonic_fast_enough_negative():
    matrix = [ [0.0, 0.7, 0.2, 0.3, 0.1],
               [0.8, 0.6, 0.7, 0.5, 0.9],
               [1.0, 0.9, 0.8, 0.7, 0.6],
               [0.5, 0.4, 0.3, 0.2, 1.0],
               [0.0, 0.7, 0.2, 0.3, 0.1],
               [0.8, 0.6, 0.7, 0.5, 0.9],
               [1.0, 0.9, 0.8, 0.7, 0.6],
               [0.5, 0.4, 0.3, 0.2, 1.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    start = time()
    assert not para.can_be_made_monotonic()
    elapsed_time = time() - start
    assert elapsed_time < 2


def test_is_monotonic_strict_fast_enough_positive():
    matrix = [ [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9],
               [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9],
               [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    start = time()
    assert para.can_be_made_monotonic_strict()
    elapsed_time = time() - start
    assert elapsed_time < 2


@pytest.mark.xfail
def test_is_monotonic_strict_fast_enough_negative():
    matrix = [ [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9],
               [1.0, 0.9, 0.8, 0.7, 0.6],
               [0.5, 0.4, 0.3, 0.2, 0.1],
               [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    start = time()
    assert not para.can_be_made_monotonic_strict()
    elapsed_time = time() - start
    assert elapsed_time < 2

