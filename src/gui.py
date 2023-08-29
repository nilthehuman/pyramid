"""The program's graphical frontend, eventually."""

from kivy import require as kivy_require
kivy_require('2.1.0')
from kivy.app import App
from kivy.graphics import Color
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.core.window import Window

from pyramid import Paradigm


# pylint: disable=no-member


class KeyboardHandler(Widget):
    """Listens for keypresses in the application's window and dispatches the appropriate calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.keyboard = Window.request_keyboard(lambda: True, self)
        self.keyboard.bind(on_key_down=self.on_keypressed)
        self.keyboard.bind(on_key_up=self.on_keyreleased)

    def on_keypressed(self, _keyboard, keycode, _text, modifiers):
        if keycode[1] == 'spacebar':
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
        if not self.help_window:
            self.help_window = HelpWindow()
            self.add_widget(self.help_window)
        else:
            self.remove_widget(self.help_window)
            self.help_window = None

    def toggle_overlay_grid(self):
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
            Click on any matrix cell to change the value of its bias.\n
            Hold [b]Shift[/b] to see if the matrix can be rearranged to fit the
            research project\'s working hypothesis.'''

    def toggle_help_window(self, *args):
        App.get_running_app().root.toggle_help_window(args)
        return True


class ParadigmGrid(GridLayout):

    def __init__(self, para=None, **kwargs):
        super().__init__(**kwargs)
        if para:
            self.set_para(para)

    def set_para(self, para):
        """Clear our child widgets and replace them with text fields and buttons corresponding to the new paradigm we have been handed."""
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
        self.para.step()
        self.update_all_cells()

    def update_label(self, row=None, col=None, text=None):
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
        self.para[row][col] = new_bias
        # N.B. Kivy's add_widget function pushes widgets to the front of the child widget list
        self.children[- (row + 1) * (len(self.para[0]) + 1) - (col + 1) - 1].update()

    def update_all_cells(self):
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
        textinput = CellEditText()
        self.add_widget(textinput)
        textinput.focus = True

    def update(self):
        bias = self.parent.para[self.row][self.col]
        self.text  = str(bias)
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
