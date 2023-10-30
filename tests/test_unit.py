"""Unit tests to check basic expected behaviors."""

from copy import deepcopy

from ..src.pyramid import cells_from_floats, Paradigm


def test_always_pass():
    assert True


def test_is_closed_both_true():
    matrix = [ [1, 1, 1, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0] ]
    state = Paradigm.State(matrix=cells_from_floats(matrix))
    para = Paradigm(state=state)
    assert para.can_be_made_closed()
    assert para.can_be_made_closed_strict()


def test_is_closed_both_false():
    matrix = [ [1, 1, 1, 1, 0],
               [0, 0, 1, 0, 1],
               [0, 0, 1, 0, 0],
               [1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0] ]
    state = Paradigm.State(matrix=cells_from_floats(matrix))
    para = Paradigm(state=state)
    assert not para.can_be_made_closed()
    assert not para.can_be_made_closed_strict()


def test_is_closed_only_strict_false():
    matrix = [ [0.6, 1, 1, 1, 0],
               [0.7, 1, 1, 1, 0],
               [1.0, 0, 0, 0, 0],
               [1.0, 0, 0, 0, 0],
               [0.0, 0, 0, 0, 0] ]
    state = Paradigm.State(matrix=cells_from_floats(matrix))
    para = Paradigm(state=state)
    assert para.can_be_made_closed()
    assert not para.can_be_made_closed_strict()


def test_nudge_once():
    matrix = [ [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0] ]
    state = Paradigm.State(matrix=cells_from_floats(matrix))
    para = Paradigm(state=state)
    para.nudge(1, 1, True)
    assert para[1][1] == 1


def test_nudge_twice():
    matrix = [ [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0] ]
    state = Paradigm.State(matrix=cells_from_floats(matrix))
    para = Paradigm(state=state)
    para.nudge(1, 1, True)
    para.state().iteration += 1
    para.nudge(1, 1, False)
    assert para[1][1] == 0.5


def test_step_once():
    matrix = [ [0.001] ]
    state = Paradigm.State(matrix=cells_from_floats(matrix))
    para = Paradigm(state=state)
    para.step()
    assert para[0][0] < 0.001


def test_step_twice():
    matrix = [ [0.5] ]
    state = Paradigm.State(matrix=cells_from_floats(matrix))
    para = Paradigm(state=state)
    para.step()
    after_first_step = para[0][0]
    para.step()
    assert (after_first_step - 0.5) * (para[0][0] - 0.5) > 0


def test_step_once_and_undo():
    matrix = [ [0.5] ]
    state = Paradigm.State(matrix=cells_from_floats(matrix))
    para = Paradigm(state=state)
    para.track_history(True)
    para_orig = deepcopy(list(para))
    para.step()
    para.undo_step()
    assert para_orig == list(para)


def test_step_twice_and_undo():
    matrix = [ [0.5] ]
    state = Paradigm.State(matrix=cells_from_floats(matrix))
    para = Paradigm(state=state)
    para.track_history(True)
    para_orig = deepcopy(list(para))
    para.step()
    para.step()
    para.undo_step()
    para.undo_step()
    assert para_orig == list(para)


def test_step_once_rewind_forward():
    matrix = [ [1, 0, 1, 0, 1],
               [0, 1, 0, 1, 0],
               [1, 0, 1, 0, 1],
               [0, 1, 0, 1, 0],
               [1, 0, 1, 0, 1] ]
    state = Paradigm.State(matrix=cells_from_floats(matrix))
    para = Paradigm(state=state)
    para.track_history(True)
    para_orig = deepcopy(list(para))
    para.step()
    para_last = deepcopy(list(para))
    para.rewind_all()
    assert para_orig == list(para)
    para.rewind_all()
    assert para_orig == list(para)
    para.forward_all()
    assert para_last == list(para)
    para.rewind_all()
    assert para_orig == list(para)


def test_step_twice_undo_rewind_forward():
    matrix = [ [1, 0, 1, 0, 1],
               [0, 1, 0, 1, 0],
               [1, 0, 1, 0, 1],
               [0, 1, 0, 1, 0],
               [1, 0, 1, 0, 1] ]
    state = Paradigm.State(matrix=cells_from_floats(matrix))
    para = Paradigm(state=state)
    para.track_history(True)
    para_orig = deepcopy(list(para))
    para.step()
    para_middle = deepcopy(list(para))
    para.step()
    para_last = deepcopy(list(para))
    para.undo_step()
    assert para_middle == list(para)
    para.rewind_all()
    assert para_orig == list(para)
    para.undo_step()
    assert para_orig == list(para)
    para.rewind_all()
    assert para_orig == list(para)
    para.forward_all()
    assert para_last == list(para)
    para.undo_step()
    assert para_middle == list(para)
    para.undo_step()
    assert para_orig == list(para)
