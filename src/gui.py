"""The program's graphical frontend, eventually."""

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

from .pyramid import Paradigm


# pylint: disable=no-member


class KeyboardHandler(Widget):
    """Listens for keypresses in the application's window and dispatches the appropriate calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.keyboard = Window.request_keyboard(lambda: True, self)
        self.keyboard.bind(on_key_down=self.on_keypressed)
        self.keyboard.bind(on_key_up=self.on_keyreleased)

    def on_keypressed(self, _keyboard, keycode, _text, modifiers):
        """Catch and handle user keypresses corresponding to app functions."""
        if keycode[1] == 'right' and 'ctrl' not in modifiers:
            # run a single step of the simulation
            App.get_running_app().root.ids.grid.step()
            return True
        if keycode[1] == 'left' and 'ctrl' not in modifiers:
            # revert last step of the simulation
            App.get_running_app().root.ids.grid.undo_step()
            return True
        if keycode[1] == '0' or keycode[1] == 'left' and 'ctrl' in modifiers:
            # reset simulation to initial state
            App.get_running_app().root.ids.grid.rewind_all()
            return True
        if keycode[1] == '9' or keycode[1] == 'right' and 'ctrl' in modifiers:
            # reset simulation to latest state
            App.get_running_app().root.ids.grid.forward_all()
            return True
        if keycode[1] == 'spacebar':
            # run simulation until spacebar pressed again
            App.get_running_app().root.ids.grid.start_stop_simulation()
            return True
        if keycode[1] == 'shift' or keycode[1] == 'rshift':
            App.get_running_app().root.toggle_overlay_grid()
            return True
        if keycode[1] == '?':
            return True
        if keycode[1] == 'escape':
            if App.get_running_app().root.help_window:
                App.get_running_app().root.toggle_help_window()
                return True
        return False

    def on_keyreleased(self, _keyboard, keycode):
        """Remove overlay paradigm once user releases shift key."""
        if keycode[1] == 'shift' or keycode[1] == 'rshift':
            App.get_running_app().root.toggle_overlay_grid()
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

    def toggle_overlay_grid(self):
        """Show or hide paradigm rearranged according to our working hypothesis."""
        if not self.overlay:
            para_rearranged = self.ids.grid.para.is_pyramid()
            if para_rearranged:
                self.overlay = ParadigmGrid(para_rearranged)
                self.add_widget(self.overlay)
            else:
                self.overlay = NoSolutionLabel()
                self.add_widget(self.overlay)

        else:
            self.remove_widget(self.overlay)
            self.overlay = None


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
        self.text = '''[size=20][b]Help[/b][/size]\n\n
            Each matrix cell shows the prevalence (the "bias") of a certain
            morphological phenomenon when the morphemes in its row and column
            are combined. Bias values range from 0 to 1.\n
            Click on any row or column label to edit the morpheme corresponding
            to that row or column.\n
            Click on any paradigm cell to change the value of its bias.\n
            Press [b]RightArrow (→)[/b] to perform one iteration of the
            simulation.\n
            Press [b]LeftArrow (←)[/b] to undo one iteration of the
            simulation.\n
            Hold [b]Shift[/b] to see if the paradigm can be rearranged to fit
            the research project\'s working hypothesis.'''

    def toggle_help_window(self, *args):
        """Show or hide fullscreen Label with help text."""
        App.get_running_app().root.toggle_help_window(args)
        return True


class ParadigmGrid(GridLayout):

    def __init__(self, para=None, **kwargs):
        super().__init__(**kwargs)
        self.timed_callback = None
        if para:
            self.set_para(para)

    def set_para(self, para):
        """Clear our child widgets and replace them with text fields and buttons corresponding to the new paradigm we have been handed."""
        self.clear_widgets()
        self.para = para
        if not para:
            return
        self.row_text_inputs = []
        self.col_text_inputs = []
        self.add_widget(Widget())  # spacer in the top left corner
        for j, label in enumerate(para.col_labels):
            self.col_text_inputs.append(ParadigmText(col=j, text=label))
            self.add_widget(self.col_text_inputs[-1])
        for i, (label, row) in enumerate(zip(para.row_labels, para)):
            self.row_text_inputs.append(ParadigmText(row=i, text=label))
            self.add_widget(self.row_text_inputs[-1])
            for j, value in enumerate(row):
                self.add_widget(ParadigmCell(i, j))
        self.update_all_cells()

    def step(self):
        """Perform one iteration of the simulation (thin wrapper around Paradigm.step)."""
        self.para.step()
        self.update_all_cells()

    def undo_step(self):
        """Revert one iteration of the simulation (thin wrapper around Paradigm.undo_step)."""
        self.para.undo_step()
        self.update_all_cells()

    def rewind_all(self):
        """Revert simulation all the way to initial state."""
        self.para.rewind_all()
        self.update_all_cells()

    def forward_all(self):
        """Redo all iterations until the latest state."""
        self.para.forward_all()
        self.update_all_cells()

    def start_stop_simulation(self):
        """Keep running the simulation until the same method is called again."""
        if self.timed_callback:
            assert self.para.running()
            self.timed_callback.cancel()
            self.timed_callback = None
            self.para.cancel()
        else:
            self.run_batch(0)
            self.timed_callback = Clock.schedule_interval(self.run_batch, 0.1)

    def run_batch(self, _elapsed_time):
        """Callback to perform one batch of iterations of the simulation."""
        para_size = len(self.para) * len(self.para[0])
        self.para.simulate(batch_size=para_size)
        self.update_all_cells()

    def update_label(self, row=None, col=None, text=None):
        """Set the user's desired string as row or column label in the paradigm."""
        assert (row is None) != (col is None)
        if not text:
            # not a good idea
            #warn("Please don't leave row or column labels empty")
            if row:
                self.row_text_inputs[row].text = self.para.row_labels[row]
            else:
                self.col_text_inputs[col].text = self.para.col_labels[col]
            return
        if row is not None:
            self.para.row_labels[row] = text
        if col is not None:
            self.para.col_labels[col] = text
        assert len(self.para.row_labels) == len(set(self.para.row_labels))
        assert len(self.para.col_labels) == len(set(self.para.col_labels))

    def update_cell(self, row, col, new_bias):
        """Set the bias of a cell in the underlying Paradigm object to a new value."""
        self.para[row][col] = new_bias
        # N.B. Kivy's add_widget function pushes widgets to the front of the child widget list
        self.children[- (row + 1) * (len(self.para[0]) + 1) - (col + 1) - 1].update()

    def update_all_cells(self):
        """Sync all visual grid cells with the cells of the underlying Paradigm object."""
        for child in self.children:
            if isinstance(child, ParadigmCell):
                child.update()


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
        bias = self.parent.para[self.row][self.col]
        if isinstance(bias, bool):
            self.text = str(bias)
        else:
            assert isinstance(bias, (int, float))
            self.text  = "%0.3g" % bias
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
            #warn("Matrix values are supposed to be numeric.")
            pass

    def focus_changed(self, instance, focused=None):
        """Remove this TextInput box if the user has clicked elsewhere."""
        assert self == instance
        if focused is False:
            self.parent.remove_widget(self)


class NoSolutionLabel(Label):
    pass


class PyramidApp(App):
    def build(self):
        para = Paradigm( row_labels=['ház', 'gáz', 'tűz', 'pénz'],
                         col_labels=['-k', '-t', '-m', '-d'],
                         matrix=[[1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1]] )
                         #matrix=[[0, 0, 0, 0], [1, 0, 1, 1], [1, 0, 1, 1], [1, 0, 1, 1]] )
        root = PyramidWindow(para)
        self.keyboardhandler = KeyboardHandler()
        return root


if __name__ == '__main__':
    PyramidApp().run()
