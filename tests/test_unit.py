"""Unit tests to check basic expected behaviors."""

from copy import deepcopy
from filecmp import cmp
from os import remove

from ..src.pyramid import cells_from_floats, ParadigmaticSystem


def test_always_pass():
    assert True


def test_is_conjunctive_all_true():
    matrix = [ [0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    assert para.is_conjunctive_binary()
    assert para.is_conjunctive_tripartite()
    assert para.is_conjunctive_strict()


def test_is_conjunctive_only_binary_true_1():
    matrix = [ [0.0, 0.0, 0.0, 0.0, 0.0],
               [0.4, 0.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    assert para.is_conjunctive_binary()
    assert not para.is_conjunctive_tripartite()
    assert not para.is_conjunctive_strict()


def test_is_conjunctive_only_binary_true_2():
    matrix = [ [0.0, 0.0, 0.3, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.8, 0.2, 0.0],
               [1.0, 1.0, 0.6, 0.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    assert para.is_conjunctive_binary()
    assert not para.is_conjunctive_tripartite()
    assert not para.is_conjunctive_strict()


def test_is_conjunctive_all_false():
    matrix = [ [0.0, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.2, 1.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    assert not para.is_conjunctive_binary()
    assert not para.is_conjunctive_tripartite()
    assert not para.is_conjunctive_strict()


def test_is_monotonic_all_true():
    matrix = [ [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.6, 0.2, 0.0, 0.0],
               [1.0, 0.8, 0.8, 0.0, 0.0],
               [1.0, 1.0, 0.9, 0.5, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    assert para.is_monotonic_binary()
    assert para.is_monotonic_tripartite()
    assert para.is_monotonic_strict()


def test_is_monotonic_all_false():
    matrix = [ [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.6, 0.2, 0.0, 0.0],
               [0.3, 0.8, 0.8, 0.0, 0.0],
               [1.0, 1.0, 0.9, 0.5, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    assert not para.is_monotonic_binary()
    assert not para.is_monotonic_tripartite()
    assert not para.is_monotonic_strict()


def test_is_monotonic_only_strict_false():
    matrix = [ [0.1, 0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.0, 0.0, 0.0],
               [0.5, 0.6, 0.7, 0.0, 0.0],
               [1.0, 0.8, 0.8, 0.0, 0.0],
               [1.0, 0.9, 1.0, 0.5, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.tripartite_cutoff = 0.8
    assert para.is_monotonic_binary()
    assert para.is_monotonic_tripartite()
    assert not para.is_monotonic_strict()


def test_is_monotonic_tripartite_true():
    matrix = [ [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.6, 0.2, 0.0, 0.0],
               [1.0, 0.8, 0.8, 0.0, 0.0],
               [1.0, 1.0, 0.9, 0.5, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.tripartite_cutoff = 0.8
    assert para.is_monotonic_tripartite()


def test_is_monotonic_tripartite_false():
    matrix = [ [0.5, 0.0, 0.0, 0.0, 0.0],
               [0.4, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.6, 0.5, 0.0, 0.0],
               [0.9, 0.8, 0.1, 0.0, 0.0],
               [1.0, 1.0, 0.9, 0.5, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.tripartite_cutoff = 0.8
    assert not para.is_monotonic_tripartite()


def test_can_be_made_monotonic_all_true():
    matrix = [ [1, 1, 1, 1, 0],
               [0, 0, 1, 0, 0],
               [0, 0, 1, 0, 0],
               [1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    assert para.can_be_made_monotonic_binary()
    assert para.can_be_made_monotonic_tripartite()
    assert para.can_be_made_monotonic_strict()


def test_can_be_made_monotonic_all_false():
    matrix = [ [1, 1, 1, 1, 0],
               [0, 0, 1, 0, 1],
               [0, 0, 1, 0, 0],
               [1, 1, 1, 1, 0],
               [0, 0, 0, 0, 0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    assert not para.can_be_made_monotonic_binary()
    assert not para.can_be_made_monotonic_tripartite()
    assert not para.can_be_made_monotonic_strict()


def test_can_be_made_monotonic_only_binary_true():
    matrix = [ [0.6, 1, 1, 1, 0],
               [0.7, 1, 1, 1, 0],
               [1.0, 0, 0, 0, 0],
               [1.0, 0, 0, 0, 0],
               [0.0, 0, 0, 0, 0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    assert para.can_be_made_monotonic_binary()
    assert not para.can_be_made_monotonic_tripartite()
    assert not para.can_be_made_monotonic_strict()


def test_nudge_once_with_delta():
    matrix = [ [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.decaying_delta = False
    para.settings.delta = 0.1
    para.nudge(1, 1, True)
    assert para[1][1] == 0.1


def test_nudge_twice_with_delta():
    matrix = [ [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.decaying_delta = False
    para.settings.delta = 0.1
    para.nudge(1, 1, True)
    para.state().total_steps += 1
    para.nudge(1, 1, False)
    assert para[1][1] == 0


def test_nudge_once_with_kappa():
    matrix = [ [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.decaying_delta = True
    para.settings.kappa = 1
    para.nudge(1, 1, True)
    assert para[1][1] == 1/2


def test_nudge_twice_with_kappa():
    matrix = [ [0, 0, 0],
               [0, 0, 0],
               [0, 0, 0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.decaying_delta = True
    para.settings.kappa = 1
    para.nudge(1, 1, True)
    para.state().total_steps += 1
    para.nudge(1, 1, False)
    assert para[1][1] == round(1/2 - 1/3, 2)


def test_step_once():
    matrix = [ [0.001] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.step()
    assert para[0][0] < 0.001


def test_step_twice():
    matrix = [ [0.5] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.step()
    assert round(float(para[0][0]), 1) in (0.4, 0.6)
    para.step()
    assert round(float(para[0][0]), 1) in (0.3, 0.5, 0.7)


def test_step_once_and_undo():
    matrix = [ [0.5] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.track_history(True)
    para_orig = deepcopy(list(para))
    para.step()
    para.undo_step()
    assert para_orig == list(para)


def test_step_twice_and_undo():
    matrix = [ [0.5] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
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
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
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
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
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


def test_sim_result_all_conjunctive_strict():
    matrix = [ [1.0, 1.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.delta = 0
    para.settings.conjunctive_criterion = ParadigmaticSystem.is_conjunctive_strict
    para.simulate(max_steps=100)
    assert para.state().sim_result.conjunctive_states == 100
    assert para.state().sim_result.total_states == 100


def test_sim_result_none_conjunctive_strict():
    matrix = [ [0.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.2, 0.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 1.0, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.delta = 0
    para.settings.conjunctive_criterion = ParadigmaticSystem.is_conjunctive_strict
    para.simulate(max_steps=100)
    assert para.state().sim_result.conjunctive_states == 0
    assert para.state().sim_result.total_states == 100


def test_sim_result_all_monotonic_strict():
    matrix = [ [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.6, 0.2, 0.0, 0.0],
               [1.0, 0.8, 0.8, 0.0, 0.0],
               [1.0, 1.0, 0.9, 0.5, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.delta = 0
    para.settings.monotonic_criterion = ParadigmaticSystem.is_monotonic_strict
    para.simulate(max_steps=100)
    assert para.state().sim_result.monotonic_states == 100
    assert para.state().sim_result.total_states == 100


def test_sim_result_none_monotonic_strict():
    matrix = [ [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0, 0.0],
               [0.3, 0.7, 1.0, 0.0, 0.0],
               [1.0, 1.0, 0.5, 0.1, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.delta = 0
    para.settings.monotonic_criterion = ParadigmaticSystem.is_monotonic_strict
    para.simulate(max_steps=100)
    assert para.state().sim_result.monotonic_states == 0
    assert para.state().sim_result.total_states == 100


def test_sim_result_all_conjunctive_tripartite():
    matrix = [ [1.0, 1.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.delta = 0
    para.settings.conjunctive_criterion = ParadigmaticSystem.is_conjunctive_tripartite
    para.simulate(max_steps=100)
    assert para.state().sim_result.conjunctive_states == 100
    assert para.state().sim_result.total_states == 100


def test_sim_result_none_conjunctive_tripartite():
    matrix = [ [0.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.2, 0.0, 0.0],
               [1.0, 1.0, 1.0, 0.0, 0.0],
               [1.0, 1.0, 1.0, 1.0, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.delta = 0
    para.settings.conjunctive_criterion = ParadigmaticSystem.is_conjunctive_tripartite
    para.simulate(max_steps=100)
    assert para.state().sim_result.conjunctive_states == 0
    assert para.state().sim_result.total_states == 100


def test_sim_result_all_monotonic_tripartite():
    matrix = [ [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.6, 0.2, 0.0, 0.0],
               [1.0, 0.8, 0.8, 0.0, 0.0],
               [1.0, 1.0, 0.9, 0.5, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.delta = 0
    para.settings.monotonic_criterion = ParadigmaticSystem.is_monotonic_tripartite
    para.simulate(max_steps=100)
    assert para.state().sim_result.monotonic_states == 100
    assert para.state().sim_result.total_states == 100


def test_sim_result_none_monotonic_tripartite():
    matrix = [ [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 0.0, 0.0, 0.0, 0.0],
               [1.0, 1.0, 0.0, 0.0, 0.0],
               [0.3, 0.7, 1.0, 0.0, 0.0],
               [1.0, 1.0, 0.5, 0.1, 0.0] ]
    state = ParadigmaticSystem.State(matrix=cells_from_floats(matrix))
    para = ParadigmaticSystem(state=state)
    para.settings.delta = 0
    para.settings.monotonic_criterion = ParadigmaticSystem.is_monotonic_tripartite
    para.simulate(max_steps=100)
    assert para.state().sim_result.monotonic_states == 0
    assert para.state().sim_result.total_states == 100


def test_export_result():
    para = ParadigmaticSystem()
    output_filename = 'test_output.csv'
    expected_output_filename = 'tests/expected.csv'
    para.export_results(output_filename)
    assert cmp(output_filename, expected_output_filename, shallow=False)
    remove(output_filename)
