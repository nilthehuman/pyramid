"""The core class that implements most of the business logic: a generalized two-dimensional morphological paradigm."""

from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from itertools import permutations, product
from logging import warning
from math import ceil
from multiprocessing import Pool
from random import seed, random, randrange
from re import search

seed()


class Paradigm:
    """An m x n table of two orthogonal, multivalued morpho(phono)logical features that jointly determine
    the binary value of a third feature."""

    class EffectDir(Enum):
        INWARD  = 1
        OUTWARD = 2

    class SimStatus(Enum):
        STOPPED   = 1
        RUNNING   = 2
        CANCELLED = 3

    @dataclass
    class State:
        """Struct defining a discrete point along the simulation history. Does not include user settings."""
        row_labels: list[str] = field(default_factory=list)
        col_labels: list[str] = field(default_factory=list)
        matrix: list[list[float]] = field(default_factory=list)
        iteration: int = 0

    def __init__(self, state=None, history=None, history_index=None):
        # default settings
        self.effect_direction = Paradigm.EffectDir.INWARD
        self.effect_radius = 1
        # housekeeping variables
        self.para_state = deepcopy(state)
        self.history = deepcopy(history)
        self.history_index = history_index
        self.sim_status = Paradigm.SimStatus.STOPPED
        if history:
            assert state is None
            assert history_index is not None
            self.history_index = history_index
        elif state:
            assert history_index is None
            assert state.row_labels is None or len(state.row_labels) == len(set(state.row_labels))
            assert state.col_labels is None or len(state.col_labels) == len(set(state.col_labels))
            # load initial state
            if not state.matrix:
                for _ in state.row_labels:
                    self.para_state.matrix.append([None for _ in state.col_labels])
        else:
            self.para_state = Paradigm.State()

    def state(self):
        """The current matrix of bias values."""
        assert self.para_state is not None or self.history is not None
        if self.history:
            return self.history[self.history_index]
        return self.para_state

    def __repr__(self):
        return str(self.state().matrix)

    def __iter__(self):
        return self.state().matrix.__iter__()

    def __len__(self):
        return len(self.state().matrix)

    def __getitem__(self, index):
        return self.state().matrix[index]

    def clone(self):
        """Return a copy of this Paradigm object."""
        return deepcopy(self)

    def initialize(self, corner_rows, corner_cols):
        """Fill the bottom left corner of the given size with 1's, the rest of the table with 0's."""
        assert 0 < corner_rows < len(self)
        assert 0 < corner_cols < len(self[0])
        assert self[0][0] is None
        for row in range(len(self)):
            for col in range(len(self[0])):
                self[row][col] = 1 if len(self) - row <= corner_rows and col < corner_cols else 0

    def show_warning(_self, message):
        """Print a simple textual warning message."""
        warning(message)

    def track_history(self, on=True):
        """Enable or disable keeping a history of previous paradigm states."""
        if on:
            if not self.history:
                self.history = []
                self.history_index = -1
                self.store_snapshot()
                self.para_state = None
        else:
            self.para_state = self.state()
            self.history = None

    def with_history(func):
        """Decorator to call func only if history is being tracked."""
        def check_history(self):
            if self.history is not None:
                func(self)
            else:
                # do not warn for actions that never come directly from the user
                if func.__name__ not in ['store_snapshot', 'invalidate_future_history']:
                    self.show_warning("Cannot %s: history tracking is off." % func.__name__.replace('_', ' '))
        return check_history

    @with_history
    def store_snapshot(self):
        """Save a copy of the current state of the paradigm, to be restored later if needed."""
        self.history.append(deepcopy(self.state()))
        self.history_index += 1

    @with_history
    def invalidate_future_history(self):
        """Remove the forward-facing part of the history on account of the present state being changed."""
        del self.history[self.history_index + 1:]

    @with_history
    def undo_step(self):
        """Restore previous paradigm state from the history."""
        if 0 < self.history_index:
            self.history_index -= 1
        else:
            self.show_warning("Undo unavailable: already at oldest state in history.")

    @with_history
    def redo_step(self):
        """Restore next paradigm state from the history."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
        else:
            self.show_warning("Redo unavailable: already at newest state in history.")

    @with_history
    def rewind_all(self):
        """Undo all steps done so far and return to initial paradigm state."""
        self.history_index = 0

    @with_history
    def forward_all(self):
        """Redo all steps done so far and return to last paradigm state."""
        self.history_index = len(self.history) - 1

    def running(self):
        """Is the simulation currently in progress?"""
        return self.sim_status == Paradigm.SimStatus.RUNNING

    def cancel(self):
        """Cancel running simulation."""
        if self.running():
            self.sim_status = Paradigm.SimStatus.CANCELLED

    def pick_cell(self):
        """Select a uniformly random cell in the paradigm."""
        row = randrange(len(self))
        col = randrange(len(self[0]))
        return row, col

    def nudge(self, row, col, outcome):
        """Adjust the value of a single cell based on an outcome in a neighboring cell or cells."""
        delta = (1 if outcome else -1) / (self.state().iteration + 1)
        self[row][col] = min(max(self[row][col] + delta, 0), 1)

    def step(self):
        """Perform a single iteration of the stochastic simulation."""
        if self.history and self.history_index < len(self.history) - 1:
            self.redo_step()
            return
        self.store_snapshot()
        self.state().iteration += 1
        row, col = self.pick_cell()
        if self.effect_direction == Paradigm.EffectDir.INWARD:
            # picked cell looks around, sees which way the average leans
            # and is adjusted that way
            relevant_biases = [ self[row][col] ]
            for i in range(1, min(self.effect_radius + 1, max(len(self), len(self[0])))):
                for y, x in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    current_row = row + i * y
                    current_col = col + i * x
                    if 0 <= current_row < len(self) and 0 <= current_col < len(self[0]):
                        relevant_biases.append(self[current_row][current_col])
            outcome = 0.5 <= sum(relevant_biases) / len(relevant_biases)
            self.nudge(row, col, outcome)
        elif self.effect_direction == Paradigm.EffectDir.OUTWARD:
            # picked cell adjusts neighboring cells (probably) toward itself
            outcome = random() < self[row][col]
            self.nudge(row, col, outcome)
            for i in range(1, min(self.effect_radius + 1, max(len(self), len(self[0])))):
                for y, x in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    current_row = row + i * y
                    current_col = col + i * x
                    if 0 <= current_row < len(self) and 0 <= current_col < len(self[0]):
                        self.nudge(current_row, current_col, outcome)
        else:
            raise ValueError("The EffectDir enum has no such value")

    def simulate(self, max_iterations=None, batch_size=None):
        """Run a predefined number of iterations of the simulation or until cancelled by the user."""
        assert max_iterations or batch_size
        if self.sim_status == Paradigm.SimStatus.STOPPED:
            self.sim_status = Paradigm.SimStatus.RUNNING
        if max_iterations is None:
            max_iterations = int(1e9)  # math.inf is not applicable
        if batch_size is None:
            batch_size = int(1e9)  # math.inf is not applicable
        self.iterations = 0
        for _ in range(batch_size):
            if self.sim_status == Paradigm.SimStatus.CANCELLED or self.iterations >= max_iterations:
                self.sim_status = Paradigm.SimStatus.STOPPED
                break
            self.step()
            self.iterations += 1

    def is_pyramid(self):
        """Check if the central working hypothesis holds for the current state of the paradigm."""
        if not self or not self[0]:
            return False
        assert self[0][0] is not None
        para_truth = self
        if type(self[0][0]) is not bool:
            para_truth = self.clone()
            for row in range(len(self)):
                for col in range(len(self[0])):
                    para_truth[row][col] = 0.5 <= self[row][col]
        last_row_first_false = None
        for row in para_truth:
            first_false = None
            try:
                first_false = row.index(False)
            except ValueError:
                first_false = len(para_truth[0])
            if not all(not cell for cell in row[first_false + 1:]):
                # discontiguous row
                return False
            if last_row_first_false is not None and last_row_first_false > first_false:
                # row shorter than previous row
                return False
            last_row_first_false = first_false
        return True

    def is_pyramid_strict(self):
        """Check if the central working hypothesis holds for the current state of the paradigm,
        but make sure all cell values are ordered monotonously as well."""
        if not self or not self[0]:
            return False
        assert self[0][0] is not None
        for row in self:
            for cell, next_cell in zip(row, row[1:]):
                if cell < next_cell:
                    return False
        for row_i in range(len(self) - 1):
            for col_i in range(len(self[0])):
                if self[row_i][col_i] > self[row_i + 1][col_i]:
                    return False
        return True

    def can_be_made_pyramid(self):
        """Check if the paradigm can be rearranged to fit the central working hypothesis."""
        para_truth = self.clone()
        for row in range(len(self)):
            for col in range(len(self[0])):
                para_truth[row][col] = 0.5 <= self[row][col]
        # isolate trivial (i.e. full or empty) rows and columns
        full_rows  = set(filter(lambda row: all(para_truth[row]), range(len(para_truth))))
        empty_rows = set(filter(lambda row: all(map(lambda x: not x, para_truth[row])), range(len(para_truth))))
        full_cols  = set(filter(lambda col: all(row[col] for row in para_truth), range(len(para_truth[0]))))
        empty_cols = set(filter(lambda col: all(map(lambda x: not x, (row[col] for row in para_truth))), range(len(para_truth[0]))))
        if len(para_truth) - len(full_rows) - len(empty_rows) > 8 or len(para_truth) - len(full_cols) - len(empty_cols) > 8:
            # no way, don't even try, we might run out of memory
            raise ValueError('Input paradigm is too large, aborting calculation, sorry')
            return None
        # use brute force for now
        row_permutations = [tuple(empty_rows) + perm + tuple(full_rows) for perm in permutations(set(range(len(para_truth))) - set(full_rows) - set(empty_rows))]
        col_permutations = [tuple(full_cols) + perm + tuple(empty_cols) for perm in permutations(set(range(len(para_truth[0]))) - set(full_cols) - set(empty_cols))]
        all_permutations = product(row_permutations, col_permutations)
        next_para = self.clone()
        next_para.store_snapshot()
        for row_permutation, col_permutation in all_permutations:
            for row in range(len(para_truth)):
                for col in range(len(para_truth[0])):
                    permuted_row = row_permutation[row]
                    permuted_col = col_permutation[col]
                    if para_truth.state().row_labels:
                        next_para.state().row_labels[row] = para_truth.state().row_labels[permuted_row]
                    if para_truth.state().col_labels:
                        next_para.state().col_labels[col] = para_truth.state().col_labels[permuted_col]
                    next_para[row][col] = para_truth[permuted_row][permuted_col]
            if next_para.is_pyramid():
                return next_para
        return None  # no solution

    def can_be_made_pyramid_strict(self):
        """Check if the paradigm can be rearranged to fit the central working hypothesis,
        but make sure all cells are ordered monotonously as well."""
        if len(self) > 8 or len(self) > 8:
            # no way, don't even try, we might run out of memory
            raise ValueError('Input paradigm is too large, aborting calculation, sorry')
            return None
        # use brute force for now
        all_permutations = product(permutations(range(len(self))), permutations(range(len(self[0]))))
        next_para = self.clone()
        next_para.store_snapshot()
        for row_permutation, col_permutation in all_permutations:
            for row in range(len(self)):
                for col in range(len(self[0])):
                    permuted_row = row_permutation[row]
                    permuted_col = col_permutation[col]
                    if self.state().row_labels:
                        next_para.state().row_labels[row] = self.state().row_labels[permuted_row]
                    if self.state().col_labels:
                        next_para.state().col_labels[col] = self.state().col_labels[permuted_col]
                    next_para[row][col] = self[permuted_row][permuted_col]
            if next_para.is_pyramid_strict():
                return next_para
        return None

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

def default_criterion(para):
    return para.is_pyramid_strict() is not None

def subproc_simulate(item, reps=None, max_iterations=None, num_processes=None, criterion=None, silent=False):
    """Number crunching function executed by CPU-bound subprocesses, runs one simulation
    from the common starting paradigm and checks if it satisfies the criterion."""
    # the reps and num_processes arguments are only used in displaying progress
    assert max_iterations is not None
    assert num_processes is not None
    assert criterion is not None
    current_rep, para = item
    para.simulate(max_iterations)
    result = criterion(para)
    if num_processes and not silent:
        # is this the first process in the pool?
        if current_rep < reps // num_processes:
            # report progress live
            progress = (current_rep + 1) * num_processes
            if reps is not None:
                print("\rPerforming %d repeats with %d steps each: %d repeats done." % (reps, max_iterations, progress), end='')
            else:
                print("\rPerforming simulation with %d steps each: %d repeats done." % (max_iterations, progress), end='')
    return result

def repeat_simulation(para, reps, max_iterations, num_processes=4, criterion=default_criterion):
    """Run several simulations from the same starting state and aggregate the results."""
    para.track_history(False)
    with Pool(num_processes) as pool:
        results = pool.map(partial(subproc_simulate,
                                   reps=reps,
                                   max_iterations=max_iterations,
                                   num_processes=num_processes,
                                   criterion=criterion),
                           [(rep, para.clone()) for rep in range(reps)],
                           chunksize=reps // num_processes)
    reps_met_criterion = sum(results)
    return reps_met_criterion
