"""The program's graphical frontend, eventually."""

from kivy import require as kivy_require
kivy_require('2.1.0')
from kivy.app import App
#from kivy.core.window import Window
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.gridlayout import GridLayout
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
            App.get_running_app().root.toggle_overlay_panel()
            return True
        if keycode[1] == '?':
            print("yay you pressed question mark!")
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
            App.get_running_app().root.toggle_overlay_panel()
            return True
        return False


class PyramidWindow(AnchorLayout):

    def __init__(self, para=None, **kwargs):
        super().__init__(**kwargs)
        self.overlay_panel = None

    def toggle_overlay_panel(self):
        if not self.overlay_panel:
            self.overlay_panel = ParadigmCell()
            self.add_widget(self.overlay_panel)
        else:
            self.remove_widget(self.overlay_panel)
            self.overlay_panel = None


class ParadigmPanel(AnchorLayout):
    pass


class ParadigmGrid(GridLayout):

    def __init__(self, para=None, **kwargs):
        super().__init__(**kwargs)
        self.para = para
        for _ in range(16):
            self.add_widget(ParadigmCell())


class ParadigmCell(Button):
    pass


class PyramidApp(App):
    def build(self):
        root = PyramidWindow()
        self.keyboardhandler = KeyboardHandler()
        return root

if __name__ == '__main__':
    PyramidApp().run()
