"""Unit tests to check basic expected behaviors."""

from ..src.pyramid import Paradigm

def test_always_pass():
    assert True

def test_is_pyramid_both_true():
    matrix = [ [1, 1, 1, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0] ]
    para = Paradigm(matrix=matrix)
    assert para.is_pyramid()
    assert para.is_pyramid_strict()

def test_is_pyramid_both_false():
    matrix = [ [1, 1, 1, 1, 0],
               [0, 0, 1, 0, 1],
               [0, 0, 1, 0, 0],
               [1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0] ]
    para = Paradigm(matrix=matrix)
    assert not para.is_pyramid()
    assert not para.is_pyramid_strict()

def test_is_pyramid_only_strict_false():
    matrix = [ [0.6, 1, 1, 1, 0],
               [0.7, 1, 1, 1, 0],
               [1.0, 0, 0, 0, 0],
               [1.0, 0, 0, 0, 0],
               [0.0, 0, 0, 0, 0] ]
    para = Paradigm(matrix=matrix)
    assert para.is_pyramid()
    assert not para.is_pyramid_strict()

def test_nudge_once():
    matrix = [ [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0] ]
    para = Paradigm(matrix=matrix)
    para.nudge(1, 1, True)
    assert para[1][1] == 1

def test_nudge_twice():
    matrix = [ [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0] ]
    para = Paradigm(matrix=matrix)
    para.nudge(1, 1, True)
    para.experience += 1
    para.nudge(1, 1, False)
    assert para[1][1] == 0.5
