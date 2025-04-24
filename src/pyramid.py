"""The core classes that implements most of the business logic:
a generalized two-dimensional morphological paradigmatic system."""

from copy import deepcopy
from csv import DictWriter
from dataclasses import asdict, dataclass, field, fields
from enum import Enum
from functools import partial
from itertools import permutations, product
from logging import warning
from math import ceil, inf, log
from multiprocessing import Pool
from os.path import isfile
from random import seed, random, randrange
from re import search


seed()


class Cell:
    """The struct that represents a single cell in the matrix holding a float value between 0 and 1
    and an experience count."""

    def __init__(self, val):
        self.value = val
        self.experience = 1

    def __repr__(self):
        return str(self.value)

    def __eq__(self, other):
        return self.value == other

    def __bool__(self):
        return bool(self.value)

    def __float__(self):
        return float(self.value)

    def __lt__(self, other):
        try:
            return self.value < other
        except TypeError:
            return self.value < other.value

    def __le__(self, other):
        try:
            return self.value <= other
        except TypeError:
            return self.value <= other.value

    def __gt__(self, other):
        try:
            return self.value > other
        except TypeError:
            return self.value > other.value

    def __ge__(self, other):
        try:
            return self.value >= other
        except TypeError:
            return self.value >= other.value

    def __iadd__(self, delta):
        self.value = round(self.value + delta, 2)
        self.value = min(max(self.value, 0), 1)
        return self

    def __add__(self, other):
        return self.value + other

    def __radd__(self, other):
        return self.value + other

    def __sub__(self, other):
        return self.value - other

    def __rsub__(self, other):
        return other - self.value

    def __mul__(self, other):
        return self.value * other


def cells_from_floats(float_matrix):
    """Convenience function to turn a 2D matrix of float values into a matrix of Cell objects."""
    assert isinstance(float_matrix, list)
    assert isinstance(float_matrix[0][0], (float, int))
    return list(map(lambda row: list(map(Cell, row)), float_matrix))


class ParadigmaticSystem:
    """An m x n table of stems and suffixes that jointly determine the binary value of a morphological feature."""

    @dataclass
    class Settings:
        """Struct to hold user settings i.e. the parameters of the stochastic simulation."""
        class EffectDir(Enum):
            INWARD  = 1
            OUTWARD = 2
        effect_direction: EffectDir = EffectDir.INWARD
        effect_radius: int = 1
        cells_own_weight: float = 1
        no_reordering: bool = False
        no_edges: bool = True  # add mock cells around the table so that all real cells have four neighbors
        decaying_delta: bool = False
        delta: float = 0.1
        kappa: float = 1
        tripartite_colors: bool = True
        tripartite_cutoff: float = 0.8
        max_steps: int = 1000
        # the properties of the ParadigmaticSystem to check after each step
        # TODO: add type hints
        conjunctive_criterion = None
        monotonic_criterion   = None

    class SimStatus(Enum):
        STOPPED = 1
        RUNNING = 2

    @dataclass
    class SimResult:
        """Struct written by the simulate method tallying the number of monotonic vs non-monotonic states."""
        current_state_conjunctive: bool = True
        current_state_monotonic:   bool = True
        current_state_changed:     bool = False
        conjunctive_states:  int = 0
        conjunctive_changes: int = 0
        monotonic_states:    int = 0
        monotonic_changes:   int = 0
        total_states:        int = 0
        total_changes:       int = 0

    @dataclass
    class State:
        """Struct defining a point between steps along the simulation timeline. Does not include user settings."""
        row_labels: list[str] = field(default_factory=list)
        col_labels: list[str] = field(default_factory=list)
        matrix: list[list[Cell]] = field(default_factory=list)
        last_pick: tuple[int, int] = field(default_factory=tuple)
        total_steps: int = 0
        # TODO: add type hint
        sim_result = None  # the tally of states during simulation satisfying the criteria

    def __init__(self, state=None, history=None, history_index=None):
        self.settings = ParadigmaticSystem.Settings()
        # housekeeping variables
        self.para_state = None
        self.history = None
        self.history_index = None
        self.set_state(state, history, history_index)
        self.sim_status = ParadigmaticSystem.SimStatus.STOPPED
        self.state().sim_result = ParadigmaticSystem.SimResult()

    def set_state(self, state=None, history=None, history_index=None):
        """The chief part of __init__: store state values in member variables."""
        if history:
            # replace all of current history
            assert state is None
            assert history_index is not None
            self.history = deepcopy(history)
            self.history_index = history_index
        elif state:
            # keep current history, replace only last state
            assert history_index is None
            assert state.row_labels is None or len(state.row_labels) == len(set(state.row_labels))
            assert state.col_labels is None or len(state.col_labels) == len(set(state.col_labels))
            self.para_state = deepcopy(state)
            # load initial state
            if not state.matrix:
                for _ in state.row_labels:
                    self.para_state.matrix.append([None for _ in state.col_labels])
        else:
            self.para_state = ParadigmaticSystem.State()

    # FIXME: this is probably unnecessary, safe to remove
    # def set_para(self, other):
    #     """Copy labels and matrix contents from the other object."""
    #     self.set_state(other.para_state, other.history, other.history_index)

    def state(self):
        """The current matrix of bias values."""
        assert self.para_state is not None or self.history is not None
        if self.history:
            return self.history[self.history_index]
        return self.para_state

    def prev_state(self):
        """The previous matrix of bias values."""
        assert self.history is not None
        try:
            return self.history[self.history_index - 1]
        except IndexError:
            return None

    def __repr__(self):
        def inner_str(row):
            return str(', '.join(map(lambda x: f"%.2f" % x, row)))
        return '\n'.join(map(inner_str, self.state().matrix))

    def __iter__(self):
        return self.state().matrix.__iter__()

    def __len__(self):
        return len(self.state().matrix)

    def __getitem__(self, index):
        return self.state().matrix[index]

    def clone(self):
        """Return a copy of this ParadigmaticSystem object without the history."""
        return ParadigmaticSystem(state=deepcopy(self.state()))

    def clone_binary(self):
        """Return a copy of this ParadigmaticSystem with each cell's preference
        rounded to False or True."""
        para_binary = self.clone()
        for row in range(len(self)):
            for col in range(len(self[0])):
                para_binary[row][col] = self[row][col] >= 0.5
        return para_binary

    def clone_quantized(self):
        """Return a copy of this ParadigmaticSystem mapped to the { A, AB, B } set."""
        para_quant = self.clone()
        for row in range(len(self)):
            for col in range(len(self[0])):
                para_quant[row][col] = self.quantize(row, col)
        return para_quant

    def initialize(self, corner_rows, corner_cols):
        """Fill the bottom left corner of the given size with 1's, the rest of the table with 0's."""
        assert 0 < corner_rows < len(self)
        assert 0 < corner_cols < len(self[0])
        assert self[0][0] is None
        for row in range(len(self)):
            for col in range(len(self[0])):
                self[row][col].value = 1 if len(self) - row <= corner_rows and col < corner_cols else 0

    def show_warning(_self, message):
        """Print a simple textual warning message."""
        warning(message)

    def track_history(self, on=True):
        """Enable or disable keeping a history of previous states of the paradigmatic system."""
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
        """Save a copy of the current state of the paradigmatic system, to be restored later if needed."""
        self.history.append(deepcopy(self.state()))
        self.history_index += 1

    @with_history
    def invalidate_future_history(self):
        """Remove the forward-facing part of the history on account of the present state being changed."""
        del self.history[self.history_index + 1:]

    @with_history
    def undo_step(self):
        """Restore previous paradigmatic system state from the history."""
        if 0 < self.history_index:
            self.history_index -= 1
        else:
            self.show_warning("Undo unavailable: already at oldest state in history.")

    @with_history
    def redo_step(self):
        """Restore next paradigmatic system state from the history."""
        if self.history_index < len(self.history) - 1:
            self.history_index += 1
        else:
            self.show_warning("Redo unavailable: already at newest state in history.")

    @with_history
    def rewind_all(self):
        """Undo all steps done so far and return to initial paradigmatic system state."""
        self.history_index = 0

    @with_history
    def forward_all(self):
        """Redo all steps done so far and return to last paradigmatic system state."""
        self.history_index = len(self.history) - 1

    @with_history
    def seek_prev_change(self):
        """Jump to the last state where a cell changed its color."""
        if self.history_index == 0:
            self.show_warning("Already at oldest state in history.")
            return
        while not self.state().sim_result.current_state_changed and self.history_index > 0:
            self.undo_step()
        if 0 < self.history_index:
            self.undo_step()

    @with_history
    def seek_next_change(self):
        """Jump to the next state where a cell changes its color."""
        if self.state().total_steps == self.settings.max_steps:
            self.show_warning("Maximum number of steps reached.")
            return
        self.step()
        while not self.state().sim_result.current_state_changed and self.history_index < self.settings.max_steps:
            self.step()

    @with_history
    def delete_rest_of_history(self):
        """Drop the remaining states in the history forward from the current state."""
        del self.history[self.history_index + 1:]

    def running(self):
        """Is the simulation currently in progress?"""
        return self.sim_status == ParadigmaticSystem.SimStatus.RUNNING

    def pick_cell(self):
        """Select a uniformly random cell in the paradigmatic system."""
        row = randrange(len(self))
        col = randrange(len(self[0]))
        return row, col

    def quantize(self, row, col):
        """Determine which morphological pattern a given cell follows, or if it vacillates."""
        bias = self[row][col]
        if self.settings.tripartite_colors:
            if bias < 1 - self.settings.tripartite_cutoff:
                return 'A'
            elif 1 - self.settings.tripartite_cutoff <= bias <= self.settings.tripartite_cutoff:
                return 'AB'
            elif self.settings.tripartite_cutoff < bias:
                return 'B'
            else:
                assert False
        else:
            return bias >= 0.5

    def nudge(self, row, col, outcome):
        """Adjust the value of a single cell based on an outcome in a neighboring cell or cells."""
        quant_before = self.quantize(row, col)
        if self.settings.decaying_delta:
            assert self.settings.kappa is not None
            delta = (1 if outcome else -1) / (self[row][col].experience * self.settings.kappa + 1)
        else:
            assert self.settings.delta is not None
            delta = (1 if outcome else -1) * self.settings.delta
        self[row][col] += delta
        self[row][col].experience += 1
        quant_after = self.quantize(row, col)
        return quant_before != quant_after

    def step(self):
        """Perform a single iteration of the stochastic simulation."""
        if self.state().total_steps == self.settings.max_steps:
            self.show_warning("Maximum number of steps reached.")
            return
        if self.history and self.history_index < len(self.history) - 1:
            self.redo_step()
            return
        self.store_snapshot()
        self.state().total_steps += 1
        row, col = self.pick_cell()
        self.state().last_pick = row, col
        if self.settings.effect_direction == ParadigmaticSystem.Settings.EffectDir.INWARD:
            # picked cell looks around, sees which way the average leans
            # and is adjusted (probably) that way
            relevant_biases = [ self[row][col] * self.settings.cells_own_weight ]
            for i in range(1, min(self.settings.effect_radius + 1, max(len(self), len(self[0])))):
                for y, x in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    current_row = row + i * y
                    current_col = col + i * x
                    if 0 <= current_row < len(self) and 0 <= current_col < len(self[0]):
                        relevant_biases.append(self[current_row][current_col])
            def avg(lst):
                return sum(lst) / len(lst)
            if self.settings.no_edges and 1 < len(relevant_biases):
                phony_cell_bias = avg(relevant_biases[1:])
                relevant_biases += [phony_cell_bias] * (5 - len(relevant_biases))
            outcome = random() < avg(relevant_biases)
            self.state().sim_result.current_state_changed = self.nudge(row, col, outcome)
        elif self.settings.effect_direction == ParadigmaticSystem.Settings.EffectDir.OUTWARD:
            # picked cell adjusts neighboring cells (probably) toward itself
            outcome = random() < self[row][col]
            self.state().sim_result.current_state_changed = self.nudge(row, col, outcome)
            for i in range(1, min(self.settings.effect_radius + 1, max(len(self), len(self[0])))):
                for y, x in [(-1, 0), (0, -1), (1, 0), (0, 1)]:
                    current_row = row + i * y
                    current_col = col + i * x
                    if 0 <= current_row < len(self) and 0 <= current_col < len(self[0]):
                        self.state().sim_result.current_state_changed |= self.nudge(current_row, current_col, outcome)
        else:
            raise ValueError("The EffectDir enum has no such value:", self.settings.effect_direction)
        self.eval_criteria()

    def eval_criteria(self):
        """Check the preselected conjunctivity and monotonicity criteria for the lastest state of the matrix."""
        if self.settings.conjunctive_criterion is not None:
            if not self.state().sim_result.current_state_changed:
                # nothing happened in the last step, so don't bother brute-forcing the criterion
                if self.history:
                    self.state().sim_result.current_state_conjunctive = bool(self.prev_state().sim_result.current_state_conjunctive)
                else:
                    self.state().sim_result.current_state_conjunctive = bool(self.state().sim_result.current_state_conjunctive)
            else:
                self.state().sim_result.current_state_conjunctive = self.settings.conjunctive_criterion(self)
            if self.state().sim_result.current_state_conjunctive:
                self.state().sim_result.conjunctive_states += 1
                if self.state().sim_result.current_state_changed:
                    self.state().sim_result.conjunctive_changes += 1
        reordered = False
        if self.settings.monotonic_criterion is not None:
            if not self.state().sim_result.current_state_changed:
                # nothing happened in the last step, so don't bother brute-forcing the criterion
                if self.history:
                    self.state().sim_result.current_state_monotonic = bool(self.prev_state().sim_result.current_state_monotonic)
                else:
                    self.state().sim_result.current_state_monotonic = bool(self.state().sim_result.current_state_monotonic)
            else:
                self.state().sim_result.current_state_monotonic = self.settings.monotonic_criterion(self)
            if type(self.state().sim_result.current_state_monotonic) is bool:
                # monotonic_criterion is an "is_..." kind of criterion,
                # so no rearranging required by the user
                if self.state().sim_result.current_state_monotonic:
                    self.state().sim_result.monotonic_states += 1
            else:
                # monotonic_criterion is a "can_be_made_..." kind of criterion,
                # so rearranged matrices are also checked
                assert type(self.state().sim_result.current_state_monotonic) is tuple or self.state().sim_result.current_state_monotonic is None
                if self.state().sim_result.current_state_monotonic is not None:
                    row_permutation, col_permutation = self.state().sim_result.current_state_monotonic
                    # change to the new arrangement and keep the monotonic property
                    if not self.settings.no_reordering:
                        reordered = (list(row_permutation) != list(range(len(row_permutation))) or
                                     list(col_permutation) != list(range(len(col_permutation))))
                    if reordered:
                        # store state both before and after reordering, easier to follow on the UI
                        self.state().sim_result.monotonic_states += 1
                        if self.state().sim_result.current_state_changed:
                            self.state().sim_result.monotonic_changes += 1
                        self.state().sim_result.total_states += 1
                        if self.state().sim_result.current_state_changed:
                            self.state().sim_result.total_changes += 1
                        self.store_snapshot()
                        self.permute(self, row_permutation, col_permutation)
                    if not reordered:
                        self.state().sim_result.monotonic_states += 1
                        if self.state().sim_result.current_state_changed:
                            self.state().sim_result.monotonic_changes += 1
        if not reordered:
            self.state().sim_result.total_states += 1
            if self.state().sim_result.current_state_changed:
                self.state().sim_result.total_changes += 1

    def simulate(self, max_steps=None, batch_size=None):
        """Run a predefined number of iterations of the simulation or until cancelled by the user."""
        assert self.state().sim_result is not None
        if self.sim_status == ParadigmaticSystem.SimStatus.STOPPED:
            self.sim_status = ParadigmaticSystem.SimStatus.RUNNING
        if max_steps:
            max_steps = min(max_steps, self.settings.max_steps)
        else:
            max_steps = self.settings.max_steps
        if batch_size is None:
            batch_size = int(1e9)  # math.inf is not applicable
        for _ in range(batch_size):
            if max_steps is not None and self.state().total_steps >= max_steps:
                self.sim_status = ParadigmaticSystem.SimStatus.STOPPED
                break
            self.step()

    def export_results(self, filename):
        """Write the current tally along with settings to a CSV file."""
        csv_fields = [f.name for f in fields(ParadigmaticSystem.Settings) + fields(ParadigmaticSystem.SimResult)] + ['conjunctive_log_odds', 'monotonic_log_odds', 'last_state_monotonic']
        header = ','.join(csv_fields)
        file_has_header = False
        if isfile(filename):
            with open(filename, 'r', encoding='utf-8') as csv_file:
                first_line = csv_file.readline().strip()
                if first_line == header:
                    file_has_header = True
                else:
                    # this does not seem to be our kind of file
                    raise FileExistsError("A file with the same name already exists " +
                                          "and it lacks the right CSV header")
        with open(filename, 'a', encoding='utf-8') as csv_file:
            csv_writer = DictWriter(csv_file, csv_fields)
            if not file_has_header:
                csv_writer.writeheader()
            data = asdict(self.settings) | asdict(self.state().sim_result)
            try:
                data['conjunctive_log_odds'] = log(float(data['conjunctive_states']) / (data['total_states'] - data['conjunctive_states']))
            except ValueError:
                data['conjunctive_log_odds'] = -inf
            except ZeroDivisionError:
                data['conjunctive_log_odds'] = inf
            try:
                data['monotonic_log_odds'] = log(float(data['monotonic_states']) / (data['total_states'] - data['monotonic_states']))
            except ValueError:
                data['monotonic_log_odds'] = -inf
            except ZeroDivisionError:
                data['monotonic_log_odds'] = inf
            data['last_state_monotonic'] = self.state().sim_result.current_state_monotonic
            csv_writer.writerow(data)

    def is_conjunctive_binary(self):
        """Check if the current state of the paradigmatic system shows a rectangular pattern."""
        assert self[0][0] is not None
        para_binary = self
        if type(self[0][0]) is not bool:
            para_binary = self.clone_binary()
        rectangle_first_false = None
        for row in para_binary:
            first_false = None
            try:
                first_false = row.index(False)
            except ValueError:
                first_false = len(para_binary[0])
            if not all(not cell for cell in row[first_false + 1:]):
                # discontiguous row
                return False
            if rectangle_first_false is not None and rectangle_first_false != first_false:
                # row out of sync with previous rows
                return False
            if first_false != 0:
                rectangle_first_false = first_false
        return True

    def is_monotonic_binary(self):
        """Check if the current state of the paradigmatic system is compactly arranged."""
        assert self[0][0] is not None
        para_binary = self
        if type(self[0][0]) is not bool:
            para_binary = self.clone_binary()
        last_row_first_false = None
        for row in para_binary:
            first_false = None
            try:
                first_false = row.index(False)
            except ValueError:
                first_false = len(para_binary[0])
            if not all(not cell for cell in row[first_false + 1:]):
                # discontiguous row
                return False
            if last_row_first_false is not None and last_row_first_false > first_false:
                # row shorter than previous row
                return False
            last_row_first_false = first_false
        return True

    def is_conjunctive_tripartite(self):
        """Check if the current state of the paradigmatic system shows a rectangular pattern,
        quantizing to three discrete values: A, A/B and B."""
        assert self[0][0] is not None
        def vacillates(bias):
            return 1 - self.settings.tripartite_cutoff <= bias <= self.settings.tripartite_cutoff
        if any(map(lambda row: any(vacillates(x) for x in row), self)):
            return False
        return self.is_conjunctive_binary()

    def is_monotonic_tripartite(self):
        """Check if the current state of the paradigmatic system is compactly arranged,
        quantizing to three discrete values: A, A/B and B."""
        assert self[0][0] is not None
        assert type(self[0][0].value) in (int, float)
        para_quant = self.clone_quantized()
        for row in para_quant:
            for cell, next_cell in zip(row, row[1:]):
                if cell < next_cell:
                    return False
        for row_i in range(len(self) - 1):
            for col_i in range(len(self[0])):
                if para_quant[row_i][col_i] > para_quant[row_i + 1][col_i]:
                    return False
        return True

    def is_conjunctive_strict(self):
        """Check if the current state of the paradigmatic system shows a rectangular pattern,
        while also ordered strictly monotonically as well."""
        assert self[0][0] is not None
        return self.is_conjunctive_tripartite() and self.is_monotonic_strict()

    def is_monotonic_strict(self):
        """Check if the current state of the paradigmatic system is compactly arranged,
        but make sure all cell values are ordered strictly monotonically as well."""
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

    def permute(self, original_para, row_permutation, col_permutation):
        """Destructively assign a permutation of the state of the original paradigm."""
        if self is original_para:
            # avoid changing matrix while iterating through it
            original_para = self.clone()
        for row in range(len(original_para)):
            for col in range(len(original_para[0])):
                permuted_row = row_permutation[row]
                permuted_col = col_permutation[col]
                if original_para.state().row_labels:
                    self.state().row_labels[row] = original_para.state().row_labels[permuted_row]
                if original_para.state().col_labels:
                    self.state().col_labels[col] = original_para.state().col_labels[permuted_col]
                self[row][col] = original_para[permuted_row][permuted_col]
        if len(original_para.state().last_pick):
            # point to the new cell
            new_row = row_permutation.index(original_para.state().last_pick[0])
            new_col = col_permutation.index(original_para.state().last_pick[1])
            self.state().last_pick = (new_row, new_col)

    def can_be_made_monotonic_binary(self):
        """Check if the paradigmatic system can be rearranged to be compact."""
        para_binary = self.clone_binary()
        # isolate trivial (i.e. full or empty) rows and columns
        full_rows  = set(filter(lambda row: all(para_binary[row]), range(len(para_binary))))
        empty_rows = set(filter(lambda row: all(map(lambda x: not x, para_binary[row])), range(len(para_binary))))
        full_cols  = set(filter(lambda col: all(row[col] for row in para_binary), range(len(para_binary[0]))))
        empty_cols = set(filter(lambda col: all(map(lambda x: not x, (row[col] for row in para_binary))), range(len(para_binary[0]))))
        if len(para_binary) - len(full_rows) - len(empty_rows) > 8 or len(para_binary) - len(full_cols) - len(empty_cols) > 8:
            # no way, don't even try, we might run out of memory
            raise ValueError('Input paradigmatic system is too large, aborting calculation, sorry')
            return None
        # use brute force for now
        row_permutations = [tuple(empty_rows) + perm + tuple(full_rows) for perm in permutations(set(range(len(para_binary))) - set(full_rows) - set(empty_rows))]
        col_permutations = [tuple(full_cols) + perm + tuple(empty_cols) for perm in permutations(set(range(len(para_binary[0]))) - set(full_cols) - set(empty_cols))]
        all_permutations = product(row_permutations, col_permutations)
        next_para = self.clone()
        next_para.store_snapshot()
        for row_permutation, col_permutation in all_permutations:
            next_para.permute(self, row_permutation, col_permutation)
            if next_para.is_monotonic_binary():
                return (row_permutation, col_permutation)
        return None  # no solution

    def can_be_made_monotonic_tripartite(self):
        """Check if the paradigmatic system can be rearranged to be compact,
        quantizing to three discrete values: A, A/B and B."""
        para_quant = self.clone_quantized()
        return para_quant.can_be_made_monotonic_strict()

    def can_be_made_monotonic_strict(self):
        """Check if the paradigmatic system can be rearranged to be compact,
        but make sure all cells are ordered strictly monotonically as well."""
        if len(self) > 8 or len(self) > 8:
            # no way, don't even try, we might run out of memory
            raise ValueError('Input paradigmatic system is too large, aborting calculation, sorry')
            return None
        # use brute force for now
        all_permutations = product(permutations(range(len(self))), permutations(range(len(self[0]))))
        next_para = self.clone()
        next_para.store_snapshot()
        for row_permutation, col_permutation in all_permutations:
            next_para.permute(self, row_permutation, col_permutation)
            if next_para.is_monotonic_strict():
                return (row_permutation, col_permutation)
        return None

#### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ####

def default_criterion(para):
    return para.is_monotonic_strict() is not None

def subproc_simulate(item, reps=None, max_steps=None, num_processes=None, criterion=None, silent=False):
    """Number crunching function executed by CPU-bound subprocesses, runs one simulation
    from the common starting paradigmatic system and checks if it satisfies the criterion."""
    # the reps and num_processes arguments are only used in displaying progress
    assert max_steps is not None
    assert num_processes is not None
    assert criterion is not None
    current_rep, para = item
    para.simulate(max_steps)
    result = criterion(para)
    if num_processes and not silent:
        # is this the first process in the pool?
        if current_rep < reps // num_processes:
            # report progress live
            progress = (current_rep + 1) * num_processes
            if reps is not None:
                print("\rPerforming %d repeats with %d steps each: %d repeats done." % (reps, max_steps, progress), end='')
            else:
                print("\rPerforming simulation with %d steps each: %d repeats done." % (max_steps, progress), end='')
    return result

def repeat_simulation(para, reps, max_steps, num_processes=4, criterion=default_criterion):
    """Run several simulations from the same starting state and aggregate the results."""
    para.track_history(False)
    with Pool(num_processes) as pool:
        results = pool.map(partial(subproc_simulate,
                                   reps=reps,
                                   max_steps=max_steps,
                                   num_processes=num_processes,
                                   criterion=criterion),
                           [(rep, para.clone()) for rep in range(reps)],
                           chunksize=reps // num_processes)
    reps_met_criterion = sum(results)
    return reps_met_criterion
