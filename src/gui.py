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


class KeyboardHandler(Widget):
    """Listens for keypresses in the application's window and dispatches the appropriate calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.keyboard = Window.request_keyboard(lambda: True, self)
        self.keyboard.bind(on_key_down=self.on_keypressed)
        self.keyboard.bind(on_key_up=self.on_keyreleased)

    def on_keypressed(self, _keyboard, keycode, _text, modifiers):
        print("keycode:", keycode)
        if keycode[1] == 'spacebar':
            print("yay you pressed space!")
            return True
        if keycode[1] == 'shift' or keycode[1] == 'rshift':
            print("yay you pressed shift!")
            App.get_running_app().root.toggle_overlay_grid()
            return True
        if keycode[1] == '?':
            print("yay you pressed question mark!")
            return True
        if keycode[1] == 'escape':
            print("yay you pressed escape!")
            if App.get_running_app().root.help_window:
                App.get_running_app().root.toggle_help_window()
                return True
        return False

    def on_keyreleased(self, _keyboard, keycode):
        print("keycode:", keycode)
        if keycode[1] == 'shift' or keycode[1] == 'rshift':
            print("yay you released shift!")
            App.get_running_app().root.toggle_overlay_grid()
            return True
        return False


class PyramidWindow(AnchorLayout):

    def __init__(self, para=None, **kwargs):
        super().__init__(**kwargs)
        self.help_window = None
        self.overlay_grid = None
        self.ids.grid.set_para(para)

    def toggle_help_window(self, *_):
        if not self.help_window:
            self.help_window = HelpWindow()
            self.add_widget(self.help_window)
        else:
            self.remove_widget(self.help_window)
            self.help_window = None

    def toggle_overlay_grid(self):
        if not self.overlay_grid:
            para_rearranged = self.ids.grid.para.is_pyramid()
            if para_rearranged:
                self.overlay_grid = ParadigmGrid(para_rearranged)
                self.add_widget(self.overlay_grid)
        else:
            self.remove_widget(self.overlay_grid)
            self.overlay_grid = None


class HelpButton(Button):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(on_release=self.toggle_help_window)

    # you can't bind to PyramidWindow in __init__ because of Kivy's initialization order
    def toggle_help_window(self, *args):
        App.get_running_app().root.toggle_help_window(*args)


class HelpWindow(Label):
    pass


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
                self.children[0].update()

    def label_changed(self, row=None, col=None, text=None):
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
            print("setting", self.para.row_labels[row], "to", text)
            self.para.row_labels[row] = text
        if col is not None:
            self.para.col_labels[col] = text
        print(self.para.row_labels)
        assert len(self.para.row_labels) == len(set(self.para.row_labels))
        assert len(self.para.col_labels) == len(set(self.para.col_labels))


class ParadigmText(TextInput):

    def __init__(self, row=None, col=None, **kwargs):
        super().__init__(**kwargs)
        self.row = row
        self.col = col
        self.bind(focus=self.text_changed)

class ParadigmCell(Button):
    def text_changed(self, instance, focused=None):
        assert self == instance
        if focused is False:
            self.parent.label_changed(row=self.row, col=self.col, text=self.text)



    def __init__(self, row, col, **kwargs):
        super().__init__(**kwargs)
        self.row = row
        self.col = col

    def update(self):
        bias = self.parent.para[self.row][self.col]
        self.text  = str(bias)
        lime       = Color(0.22, 0.8, 0.22)
        grapefruit = Color(0.9, 0.31, 0.3)
        self.background_color = [sum(x) for x in zip([bias * c for c in lime.rgb],
                                                     [(1-bias) * c for c in grapefruit.rgb])]


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
