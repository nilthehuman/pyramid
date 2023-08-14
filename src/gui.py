"""The program's graphical frontend, eventually."""

from kivy import require as kivy_require
kivy_require('2.1.0')
from kivy.app import App
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
        if keycode[1] == 'shift':
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
        if keycode[1] == 'q':
            print("yay you pressed q!")
            App.get_running_app().stop()
            return True
        return False

    def on_keyreleased(self, _keyboard, keycode):
        print("keycode:", keycode)
        if keycode[1] == 'shift':
            print("yay you released shift!")
            App.get_running_app().root.toggle_overlay_grid()
            return True
        return False


class PyramidWindow(AnchorLayout):

    def __init__(self, para=None, **kwargs):
        super().__init__(**kwargs)
        self.help_window = None
        self.overlay_grid = None

    def toggle_help_window(self, *_):
        if not self.help_window:
            self.help_window = HelpWindow()
            self.add_widget(self.help_window)
        else:
            self.remove_widget(self.help_window)
            self.help_window = None

    def toggle_overlay_grid(self):
        if not self.overlay_grid:
            self.overlay_grid = ParadigmGrid()
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
        self.para = para
        self.add_widget(Widget())  # spacer in the top left corner
        for _ in range(4):
            self.add_widget(ParadigmText())
        for _ in range(4):
            self.add_widget(ParadigmText())
            for _ in range(4):
                self.add_widget(ParadigmCell())


class ParadigmText(TextInput):
    pass


class ParadigmCell(Button):
    pass


class PyramidApp(App):
    def build(self):
        root = PyramidWindow()
        self.keyboardhandler = KeyboardHandler()
        return root

if __name__ == '__main__':
    PyramidApp().run()
