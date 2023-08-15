"""The core class that implements most of the business logic: a generalized two-dimensional morphological paradigm."""

from copy import deepcopy
from itertools import permutations, product
from random import random, randrange, seed  # TODO: seed() the generator at the appropriate site

class Paradigm(list):
    """An m x n table of two orthogonal, multivalued morpho(phono)logical features that jointly determine the binary value of a third feature."""

    def __init__(self, row_labels=None, col_labels=None, matrix=None):
        """Overloaded constructor, works both with a matrix as argument or a pair of label lists."""
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
        if matrix:
            for row in matrix:
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
        para_truth = deepcopy(self)
        for row in range(len(self)):
            for col in range(len(self[0])):
                para_truth[row][col] = False if self[row][col] < 0.5 else True
        # get rid of trivial (i.e. full or empty) rows and columns
        trivial_rows = []
        for row in range(len(para_truth)):
            if all(para_truth[row]) or all(not p for p in para_truth[row]):
                trivial_rows.append(row)
        trivial_rows.reverse()
        for row in trivial_rows:
            del para_truth[row]
        trivial_cols = []
        for col in range(len(para_truth[0])):
            if all(para_truth[row][col] for row in range(len(para_truth))) or all(not para_truth[row][col] for row in range(len(para_truth))):
                trivial_cols.append(col)
        trivial_cols.reverse()
        for col in trivial_cols:
            for row in range(len(para_truth)):
                del para_truth[row][col]
        if not para_truth or not para_truth[0]:
            return True
        # use brute force for now
        all_permutations = product(permutations(range(len(para_truth))), permutations(range(len(para_truth[0]))))
        for permutation in all_permutations:
            next_para = deepcopy(para_truth)
            for row in range(len(para_truth)):
                for col in range(len(para_truth[0])):
                    permuted_row = permutation[0][row]
                    permuted_col = permutation[1][col]
                    next_para[row][col] = para_truth[permuted_row][permuted_col]
            if check(next_para):
                return True
        return False

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
            return True
        # use brute force for now
        all_permutations = product(permutations(range(len(self))), permutations(range(len(self[0]))))
        for permutation in all_permutations:
            next_para = deepcopy(self)
            for row in range(len(self)):
                for col in range(len(self[0])):
                    permuted_row = permutation[0][row]
                    permuted_col = permutation[1][col]
                    next_para[row][col] = self[permuted_row][permuted_col]
            if check(next_para):
                return True
        return False
