"""Performance tests for the validation of the main working hypothesis."""

import pytest
from time import time

from ..src.pyramid import Paradigm


def test_is_pyramid_fast_enough_positive():
    matrix = [ [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9],
               [1.0, 0.9, 0.8, 0.7, 0.6],
               [0.5, 0.4, 0.3, 0.2, 0.1],
               [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9],
               [1.0, 0.9, 0.8, 0.7, 0.6],
               [0.5, 0.4, 0.3, 0.2, 0.1] ]
    state = Paradigm.State(matrix=matrix)
    para = Paradigm(state=state)
    start = time()
    assert para.can_be_made_pyramid()
    elapsed_time = time() - start
    assert elapsed_time < 2


def test_is_pyramid_fast_enough_negative():
    matrix = [ [0.0, 0.7, 0.2, 0.3, 0.1],
               [0.8, 0.6, 0.7, 0.5, 0.9],
               [1.0, 0.9, 0.8, 0.7, 0.6],
               [0.5, 0.4, 0.3, 0.2, 1.0],
               [0.0, 0.7, 0.2, 0.3, 0.1],
               [0.8, 0.6, 0.7, 0.5, 0.9],
               [1.0, 0.9, 0.8, 0.7, 0.6],
               [0.5, 0.4, 0.3, 0.2, 1.0] ]
    state = Paradigm.State(matrix=matrix)
    para = Paradigm(state=state)
    start = time()
    assert not para.can_be_made_pyramid()
    elapsed_time = time() - start
    assert elapsed_time < 2


def test_is_pyramid_strict_fast_enough_positive():
    matrix = [ [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9],
               [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9],
               [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9] ]
    state = Paradigm.State(matrix=matrix)
    para = Paradigm(state=state)
    start = time()
    assert para.can_be_made_pyramid_strict()
    elapsed_time = time() - start
    assert elapsed_time < 2


@pytest.mark.xfail
def test_is_pyramid_strict_fast_enough_negative():
    matrix = [ [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9],
               [1.0, 0.9, 0.8, 0.7, 0.6],
               [0.5, 0.4, 0.3, 0.2, 0.1],
               [0.0, 0.1, 0.2, 0.3, 0.4],
               [0.5, 0.6, 0.7, 0.8, 0.9] ]
    state = Paradigm.State(matrix=matrix)
    para = Paradigm(state=state)
    start = time()
    assert not para.can_be_made_pyramid_strict()
    elapsed_time = time() - start
    assert elapsed_time < 2

