"""The core class that implements most of the business logic: a generalized two-dimensional morphological paradigm."""

from copy import deepcopy
from enum import Enum
from itertools import permutations, product
from logging import warning
from random import random, randrange, seed  # TODO: seed() the generator at the appropriate site

class Paradigm:
    """An m x n table of two orthogonal, multivalued morpho(phono)logical features that jointly determine the binary value of a third feature."""

    class EffectDir(Enum):
        INWARD  = 1
        OUTWARD = 2

    class SimStatus(Enum):
        STOPPED   = 1
        RUNNING   = 2
        CANCELLED = 3

    def __init__(self, row_labels=None, col_labels=None, matrix=None, history=None, iteration=0):
        """Overloaded constructor, works both with a matrix as argument or a pair of label lists."""
        # default settings
        self.effect_direction = Paradigm.EffectDir.INWARD
        self.effect_radius = 1
        # housekeeping variables
        self.iteration = iteration
        self.sim_status = Paradigm.SimStatus.STOPPED
        if history:
            self.para_state = None
            self.history = deepcopy(history)
        else:
            self.para_state = []
            self.history = None
            # load initial state
            if matrix:
                for row in matrix:
                    self.para_state.append(deepcopy(row))
            else:
                for _ in row_labels:
                    self.para_state.append([None for _ in col_labels])
        if row_labels:
            assert len(row_labels) == len(set(row_labels))
            self.row_labels = row_labels
        else:
            self.row_labels = []
        if col_labels:
            assert len(col_labels) == len(set(col_labels))
            self.col_labels = col_labels
        else:
            self.col_labels = []

    def state(self):
        """The current matrix of bias values."""
        assert self.para_state is not None or self.history is not None
        if self.history:
            return self.history[self.iteration]
        return self.para_state

    def __repr__(self):
        return str(self.state())

    def __iter__(self):
        return self.state().__iter__()

    def __len__(self):
        return len(self.state())

    def __getitem__(self, index):
        return self.state()[index]

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
                self.store_snapshot()
                self.iteration = 0
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
                self.show_warning("history tracking is off")
                pass
        return check_history

    @with_history
    def store_snapshot(self):
        """Save a copy of the current state of the paradigm, to be restored later if needed."""
        self.history.append(deepcopy(list(self)))
        self.iteration += 1

    @with_history
    def invalidate_future_history(self):
        """Remove the forward-facing part of the history on account of the present state being changed."""
        del self.history[self.iteration + 1:]

    @with_history
    def undo_step(self):
        """Restore previous paradigm state from the history."""
        if 0 < self.iteration:
            self.iteration -= 1

    @with_history
    def redo_step(self):
        """Restore next paradigm state from the history."""
        if self.iteration < len(self.history) - 1:
            self.iteration += 1

    @with_history
    def rewind_all(self):
        """Undo all steps done so far and return to initial paradigm state."""
        self.iteration = 0

    @with_history
    def forward_all(self):
        """Redo all steps done so far and return to last paradigm state."""
        self.iteration = len(self.history) - 1

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
        delta = (1 if outcome else -1) / (self.iteration + 1)
        self[row][col] = min(max(self[row][col] + delta, 0), 1)

    def step(self):
        """Perform a single iteration of the stochastic simulation."""
        if self.history and self.iteration < len(self.history) - 1:
            self.redo_step()
            return
        self.store_snapshot()
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
        """Check the central working hypothesis for the current state of the paradigm."""
        def check(paradigm):
            last_row_first_false = None
            for row in paradigm:
                first_false = None
                try:
                    first_false = row.index(False)
                except ValueError:
                    first_false = len(paradigm[0])
                if not all(not cell for cell in row[first_false + 1:]):
                    # discontiguous row
                    return False
                if last_row_first_false is not None and last_row_first_false > first_false:
                    # row shorter than previous row
                    return False
                last_row_first_false = first_false
            return True
        assert self[0][0] is not None
        if not self or not self[0]:
            return self
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
                    if para_truth.row_labels:
                        next_para.row_labels[row] = para_truth.row_labels[permuted_row]
                    if para_truth.col_labels:
                        next_para.col_labels[col] = para_truth.col_labels[permuted_col]
                    next_para[row][col] = para_truth[permuted_row][permuted_col]
            if check(next_para):
                return next_para
        return None  # no solution

    def is_pyramid_strict(self):
        """Check the central working hypothesis for the current state of the paradigm,
        but make sure all cells are ordered monotonously as well."""
        def check(paradigm):
            for row in paradigm:
                for cell, next_cell in zip(row, row[1:]):
                    if cell < next_cell:
                        return False
            for row_i in range(len(paradigm) - 1):
                for col_i in range(len(paradigm[0])):
                    if paradigm[row_i][col_i] > paradigm[row_i + 1][col_i]:
                        return False
            return True
        if not self or not self[0]:
            return self
        assert self[0][0] is not None
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
                    if self.row_labels:
                        next_para.row_labels[row] = self.row_labels[permuted_row]
                    if self.col_labels:
                        next_para.col_labels[col] = self.col_labels[permuted_col]
                    next_para[row][col] = self[permuted_row][permuted_col]
            if check(next_para):
                return next_para
        return None
