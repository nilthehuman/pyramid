"""The core class that implements most of the business logic: a generalized two-dimensional morphological paradigm."""

from copy import deepcopy
from itertools import permutations, product
from random import random, randrange, seed  # TODO: seed() the generator at the appropriate site

class Paradigm(list):
    """An m x n table of two orthogonal, multivalued morpho(phono)logical features that jointly determine the binary value of a third feature."""

    def __init__(self, row_labels=None, col_labels=None, para=None):
        """Overloaded constructor, works both with a paradigm as argument or a pair of label lists."""
        super().__init__([])
        self.affect_farther_cells = False
        self.experience = 0
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
        if para:
            for row in para:
                self.append(deepcopy(row))
        else:
            for _ in row_labels:
                self.append([None for _ in col_labels])

    def initialize(self, corner_rows, corner_cols):
        """Fill the top left corner of the given size with 1's, the rest of the table with 0's."""
        assert 0 < corner_rows < len(self)
        assert 0 < corner_cols < len(self[0])
        assert self[0][0] is None
        for row in range(len(self)):
            for col in range(len(self[0])):
                self[row][col] = 1 if row < corner_rows and col < corner_cols else 0

    def pick_cell(self):
        """Select a uniformly random cell in the paradigm."""
        row = randrange(len(self))
        col = randrange(len(self[0]))
        return row, col

    def nudge(self, row, col, outcome):
        """Adjust the value of a single cell based on an outcome in a neighboring cell."""
        delta = (1 if outcome else -1) / (self.experience + 1)
        self[row][col] = min(max(self[row][col] + delta, 0), 1)

    def step(self):
        """Perform a single iteration of the stochastic simulation."""
        assert len(self) > 0
        row, col = self.pick_cell()
        outcome = True if random() < self[row][col] else False
        self.nudge(row, col, outcome)
        for i in range(1, max(len(self), len(self[0]))):
            for y, x in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                current_row = row + i * y
                current_col = col + i * x
                if 0 <= current_row < len(self) and 0 <= current_col < len(self[0]):
                    self.nudge(current_row, current_col, outcome)
            if not self.affect_farther_cells:
                break
        self.experience += 1

    def simulate(self, iterations):
        pass  # TODO

    def is_pyramid(self):
        """Check the central working hypothesis for the current state of the paradigm."""
        def check(paradigm):
            last_row_first_false = None
            for row in paradigm:
                first_false = None
                try:
                    first_false = row.index(False)
                except ValueError:
                    first_false = len(self)
                if not all(not cell for cell in row[first_false + 1:]):
                    # discontiguous row
                    return False
                if last_row_first_false is not None and last_row_first_false < first_false:
                    # row longer than previous row
                    return False
                last_row_first_false = first_false
            return True
        assert self[0][0] is not None
        if not self or not self[0]:
            return self
        para_truth = deepcopy(self)
        for row in range(len(self)):
            for col in range(len(self[0])):
                para_truth[row][col] = False if self[row][col] < 0.5 else True
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
        row_permutations = [tuple(full_rows) + perm + tuple(empty_rows) for perm in permutations(set(range(len(para_truth))) - set(full_rows) - set(empty_rows))]
        col_permutations = [tuple(full_cols) + perm + tuple(empty_cols) for perm in permutations(set(range(len(para_truth[0]))) - set(full_cols) - set(empty_cols))]
        all_permutations = product(row_permutations, col_permutations)
        for row_permutation, col_permutation in all_permutations:
            next_para = deepcopy(para_truth)
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
                    if paradigm[row_i][col_i] < paradigm[row_i + 1][col_i]:
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
        for row_permutation, col_permutation in all_permutations:
            next_para = deepcopy(self)
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
