"""The program's graphical frontend, eventually."""

from copy import deepcopy

from kivy import require as kivy_require
kivy_require('2.1.0')
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.core.window import Window

from .pyramid import Cell, cells_from_floats, Paradigm


CURRENT_CELL_FRAME = None


class KeyboardHandler(Widget):
    """Listens for keypresses in the application's window and dispatches the appropriate calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.keyboard = Window.request_keyboard(lambda: True, self)
        self.keyboard.bind(on_key_down=self.on_keypressed)
        self.keyboard.bind(on_key_up=self.on_keyreleased)

    def on_keypressed(self, _keyboard, keycode, _text, modifiers):
        """Catch and handle user keypresses corresponding to app functions."""
        if App.get_running_app().root.help_window:
            if keycode[1] == 'escape':
                App.get_running_app().root.toggle_help_window()
            # block all other keypresses too
            return True
        if App.get_running_app().root.ids.grid.warning_label:
            if keycode[1] == 'escape':
                App.get_running_app().root.ids.grid.hide_warning()
            # block all other keypresses too
            return True
        if keycode[1] == 'right' and 'ctrl' not in modifiers:
            # run a single step of the simulation
            App.get_running_app().root.ids.grid.step()
            return True
        if keycode[1] == 'left' and 'ctrl' not in modifiers:
            # revert last step of the simulation
            App.get_running_app().root.ids.grid.undo_step()
            return True
        if keycode[1] == 'pagedown':
            # redo ten steps of the simulation
            for _ in range(0, 10):
                App.get_running_app().root.ids.grid.step()
            return True
        if keycode[1] == 'pageup':
            # undo ten steps of the simulation
            for _ in range(0, 10):
                App.get_running_app().root.ids.grid.undo_step()
            return True
        if keycode[1] == 'home' or keycode[1] == 'left' and 'ctrl' in modifiers:
            # reset simulation to initial state
            App.get_running_app().root.ids.grid.rewind_all()
            return True
        if keycode[1] == 'end' or keycode[1] == 'right' and 'ctrl' in modifiers:
            # reset simulation to latest state
            App.get_running_app().root.ids.grid.forward_all()
            return True
        if keycode[1] == 'spacebar':
            # run simulation until spacebar pressed again
            App.get_running_app().root.ids.grid.start_stop_simulation()
            return True
        if keycode[1] == 'shift' or keycode[1] == 'rshift':
            App.get_running_app().root.show_overlay_grid()
            return True
        if keycode[1] == 'enter' and 'shift' in modifiers:
            App.get_running_app().root.replace_para_with_overlay()
            return True
        return False

    def on_keyreleased(self, _keyboard, keycode):
        """Remove overlay paradigm once user releases shift key."""
        if keycode[1] == 'shift' or keycode[1] == 'rshift':
            App.get_running_app().root.hide_overlay_grid()
            return True
        return False


class PyramidWindow(AnchorLayout):

    def __init__(self, para=None, **kwargs):
        super().__init__(**kwargs)
        self.help_window = None
        self.overlay = None
        self.ids.grid.set_para(para)

    def toggle_help_window(self, *_):
        """Show or hide fullscreen Label with help text."""
        if not self.help_window:
            self.help_window = HelpWindow()
            self.add_widget(self.help_window)
        else:
            self.remove_widget(self.help_window)
            self.help_window = None

    def show_overlay_grid(self):
        """Show paradigm rearranged according to our working hypothesis."""
        if not self.overlay:
            para_rearranged = self.ids.grid.can_be_made_closed_strict()
            if para_rearranged:
                self.overlay = para_rearranged
                self.add_widget(self.overlay)
                self.overlay.update_all_cells()
            else:
                self.ids.grid.show_warning('No solution, sorry :(')

    def replace_para_with_overlay(self):
        """Overwrite the current state of the paradigm with the rearranged one
        and make it the new starting state."""
        if self.overlay:
            self.ids.grid.set_para(self.overlay)
            self.ids.grid.invalidate_future_history()
            self.ids.grid.update_all_cells()

    def hide_overlay_grid(self):
        """Hide the rearranged paradigm and show the original again."""
        if self.overlay:
            self.remove_widget(self.overlay)
            self.overlay = None
        else:
            self.ids.grid.hide_warning()


class HelpButton(Button):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(on_release=self.toggle_help_window)

    # you can't bind to PyramidWindow in __init__ because of Kivy's initialization order
    def toggle_help_window(self, *args):
        """Show or hide fullscreen Label with help text."""
        App.get_running_app().root.toggle_help_window(*args)
        return True


class HelpWindow(Label):

    def __init__(self, para=None, **kwargs):
        super().__init__(**kwargs)
        # block click events from Widgets below
        self.bind(on_touch_down=lambda *_: True)
        self.bind(on_touch_up=self.toggle_help_window)
        self.text = '''[size=20][b]Help[/b][/size]\n
            Each matrix cell shows the prevalence (the "bias") of a certain morphological phenomenon
            when the morphemes in its row and column are combined. Bias values range from 0 to 1.\n
            Click on any row or column label to edit the morpheme corresponding to that row or column.
            Click on any paradigm cell to change the value of its bias.\n
            Press [b]RightArrow (->)[/b] to perform one iteration of the simulation.
            Press [b]LeftArrow (<-)[/b] to undo one iteration of the simulation.\n
            Press [b]PageDown[/b] to perform ten iterations of the simulation.
            Press [b]PageUp[/b] to undo ten iterations of the simulation.\n
            Press [b]End[/b] or [b]Ctrl-RightArrow[/b] to skip to the last state of the simulation.
            Press [b]Home[/b] or [b]Ctrl-LeftArrow[/b] to skip to the initial state of the simulation.\n
            Press [b]Space[/b] to start or stop an open-ended simulation.\n
            Hold [b]Shift[/b] to see if the paradigm can be rearranged to fit the research project\'s
            working hypothesis.\n
            While holding [b]Shift[/b], press [b]Enter[/b] to keep the rearranged paradigm and replace
            the original paradigm with it.'''

    def toggle_help_window(self, *args):
        """Show or hide fullscreen Label with help text."""
        App.get_running_app().root.toggle_help_window(args)
        return True


class ParadigmGrid(Paradigm, GridLayout):

    def __init__(self, para=None, **kwargs):
        Paradigm.__init__(self)
        GridLayout.__init__(self, **kwargs)
        self.warning_label = None
        self.timed_callback = None
        if para:
            self.set_para(para)

    def set_para(self, para):
        """Create text fields and buttons if needed and load the contents of the new paradigm we have been handed."""
        if not para:
            return
        self.para_state = deepcopy(para.para_state)
        self.history = deepcopy(para.history)
        self.history_index = para.history_index
        self.row_text_inputs = []
        self.col_text_inputs = []
        if not self.children:
            self.add_widget(Widget())  # spacer in the top left corner
            for j, label in enumerate(para.state().col_labels):
                self.col_text_inputs.append(ParadigmText(col=j, text=label))
                self.add_widget(self.col_text_inputs[-1])
            for i, (label, row) in enumerate(zip(para.state().row_labels, para)):
                self.row_text_inputs.append(ParadigmText(row=i, text=label))
                self.add_widget(self.row_text_inputs[-1])
                for j, value in enumerate(row):
                    self.add_widget(ParadigmCell(i, j))
        self.update_all_cells()

    def clone(self):
        """Return a copy of this ParadigmGrid object."""
        new_para = Paradigm(state=self.para_state,
                            history=self.history,
                            history_index=self.history_index)
        clone = ParadigmGrid(para=new_para)
        return clone

    def get_cell(self, row, col):
        """Return the ParadigmCell widget corresponding to an underlying Paradigm cell."""
        return self.children[- (row + 1) * (len(self[0]) + 1) - (col + 1) - 1]

    def show_warning(self, message):
        """Display a warning popup on the screen."""
        if not self.warning_label:
            self.warning_label = WarningLabel()
            self.warning_label.text = message
            self.parent.add_widget(self.warning_label)

    def hide_warning(self):
        """Remove warning popup."""
        if self.warning_label:
            self.parent.remove_widget(self.warning_label)
            self.warning_label = None

    def show_current_cell_frame(self, on=True):
        """Enable/disable drawing a bright gold rectangle above the ParadigmCell that was chosen last."""
        global CURRENT_CELL_FRAME
        if on:
            if not CURRENT_CELL_FRAME:
                CURRENT_CELL_FRAME = CurrentCellFrame()
            pick = self.state().last_pick
            if pick:
                if CURRENT_CELL_FRAME.parent:
                    CURRENT_CELL_FRAME.parent.remove_widget(CURRENT_CELL_FRAME)
                current_cell = self.get_cell(*pick)
                current_cell.add_widget(CURRENT_CELL_FRAME)
            else:
                self.show_current_cell_frame(False)
        else:
            if CURRENT_CELL_FRAME:
                if CURRENT_CELL_FRAME.parent:
                    CURRENT_CELL_FRAME.parent.remove_widget(CURRENT_CELL_FRAME)
                CURRENT_CELL_FRAME = None

    def step(self):
        """Perform one iteration of the simulation (thin wrapper around Paradigm.step)."""
        super().step()
        self.update_all_cells()
        self.show_current_cell_frame(True)

    def undo_step(self):
        """Revert one iteration of the simulation (thin wrapper around Paradigm.undo_step)."""
        super().undo_step()
        self.update_all_cells()
        self.show_current_cell_frame(True)

    def rewind_all(self):
        """Revert simulation all the way to initial state."""
        if self.timed_callback:
            self.start_stop_simulation()
        super().rewind_all()
        self.update_all_cells()
        self.show_current_cell_frame(True)

    def forward_all(self):
        """Redo all iterations until the latest state."""
        if self.timed_callback:
            self.start_stop_simulation()
        super().forward_all()
        self.update_all_cells()
        self.show_current_cell_frame(True)

    def start_stop_simulation(self):
        """Keep running the simulation until the same method is called again."""
        if self.timed_callback:
            assert self.running()
            self.timed_callback.cancel()
            self.timed_callback = None
            self.cancel()
        else:
            self.run_batch(0)
            self.timed_callback = Clock.schedule_interval(self.run_batch, 0.1)

    def run_batch(self, _elapsed_time):
        """Callback to perform one batch of iterations of the simulation."""
        para_size = len(self) * len(self[0])
        self.simulate(batch_size=para_size)
        self.show_current_cell_frame(False)
        self.update_all_cells()

    def update_label(self, row=None, col=None, text=None):
        """Set the user's desired string as row or column label in the paradigm."""
        assert (row is None) != (col is None)
        if not text:
            # not a good idea
            # TODO: this seems like a dead branch?
            self.show_warning("Please don't leave row or column labels empty")
            if row:
                self.row_text_inputs[row].text = self.state().row_labels[row]
            else:
                self.col_text_inputs[col].text = self.state().col_labels[col]
            return
        if row is not None:
            self.state().row_labels[row] = text
        if col is not None:
            self.state().col_labels[col] = text
        assert len(self.state().row_labels) == len(set(self.state().row_labels))
        assert len(self.state().col_labels) == len(set(self.state().col_labels))

    def update_cell(self, row, col, new_bias):
        """Set the bias of a cell in the underlying Paradigm object to a new value."""
        self.invalidate_future_history()
        self.store_snapshot()
        self[row][col].value = new_bias
        # N.B. Kivy's add_widget function pushes widgets to the front of the child widget list
        self.get_cell(row, col).update()

    def update_all_cells(self):
        """Sync all visual grid cells with the cells of the underlying Paradigm object."""
        for child in self.children:
            if isinstance(child, ParadigmCell):
                child.update()
            else:
                # refresh all row and column labels as well
                try:
                    child.text = self.state().row_labels[child.row]
                except (AttributeError, TypeError):
                    try:
                        child.text = self.state().col_labels[child.col]
                    except (AttributeError, TypeError):
                        # this must be the blank spaceholder widget in the top left corner
                        assert type(child) == Widget


class ParadigmText(TextInput):

    def __init__(self, row=None, col=None, **kwargs):
        super().__init__(**kwargs)
        self.row = row
        self.col = col
        self.bind(focus=self.text_changed)

    def text_changed(self, instance, focused=None):
        """Set the user's desired string as row or column label in the paradigm."""
        assert self == instance
        if focused is False:
            self.parent.update_label(row=self.row, col=self.col, text=self.text)


class ParadigmCell(AnchorLayout, Button):

    def __init__(self, row, col, **kwargs):
        super().__init__(**kwargs)
        self.row = row
        self.col = col
        self.bind(on_release=self.edit_bias)

    def edit_bias(self, *_args):
        """Show a temporary TextInput box to allow the user to redefine the label of a column or a row."""
        textinput = CellEditText()
        self.add_widget(textinput)
        textinput.focus = True

    def update(self):
        """Sync this cell's content and color with the bias of the underlying Paradigm's cell."""
        bias = self.parent[self.row][self.col]
        if isinstance(bias, bool):
            self.text = str(bias)
        else:
            assert isinstance(bias, Cell)
            self.text  = "%0.2g" % round(bias.value, 2)
        lime       = Color(0.22, 0.8, 0.22)
        grapefruit = Color(0.9, 0.31, 0.3)
        self.background_color = [sum(x) for x in zip([bias * c for c in lime.rgb],
                                                     [(1-bias) * c for c in grapefruit.rgb])]


class CellEditText(TextInput):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(on_text_validate=self.text_validated)
        self.bind(focus=self.focus_changed)

    def text_validated(self, instance):
        """Set a new bias value in the cell once the user finished typing."""
        assert self == instance
        try:
            new_value = float(self.text)
            # clamp to [0, 1]
            new_value = max(0, min(1, new_value))
            self.parent.parent.update_cell(self.parent.row, self.parent.col, new_value)
        except ValueError:
            try:
                if "true" == self.text.lower():
                    new_value = True
                elif "false" == self.text.lower():
                    new_value = False
                self.parent.parent.update_cell(self.parent.row, self.parent.col, new_value)
            except NameError:
                self.parent.parent.show_warning("Matrix values are supposed to be numeric or Boolean.")
                pass

    def focus_changed(self, instance, focused=None):
        """Remove this TextInput box if the user has clicked elsewhere."""
        assert self == instance
        if focused is False:
            self.parent.remove_widget(self)


class CurrentCellFrame(Widget):
    pass


class WarningLabel(Label):

    def __init__(self, para=None, **kwargs):
        super().__init__(**kwargs)
        # block click events from Widgets below
        self.bind(on_touch_down=lambda *_: True)
        self.bind(on_touch_up=self.dismiss_warning)

    # you can't bind to PyramidWindow in __init__ because of Kivy's initialization order
    def dismiss_warning(self, *args):
        """Remove warning label from the screen."""
        self.parent.ids.grid.hide_warning()
        #App.get_running_app().root.toggle_help_window(*args)
        return True


class PyramidApp(App):
    def build(self):
        state = Paradigm.State( row_labels=['bordó', 'millió', 'szigorú', 'józan', 'új'],
                                col_labels=['-n', '-k', '-bb', '-t', '-nAk'],
                                matrix=cells_from_floats(
                                           [ [0, 0, 0, 0, 0],
                                             [1, 0, 0, 0, 0],
                                             [1, 1, 0, 0, 0],
                                             [1, 1, 1, 0, 0],
                                             [1, 1, 1, 1, 0]
                                           ]) )
        para = Paradigm(state)
        para.track_history(True)
        root = PyramidWindow(para)
        self.keyboardhandler = KeyboardHandler()
        return root


if __name__ == '__main__':
    PyramidApp().run()
