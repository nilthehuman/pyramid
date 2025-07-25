"""The application's graphical frontend."""

from copy import deepcopy
from textwrap import dedent

from kivy import require as kivy_require
kivy_require('2.1.0')
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics import Color
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Line
from kivy.uix.gridlayout import GridLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.widget import Widget
from kivy.core.window import Window

from .pyramid import Cell, cells_from_floats, ParadigmaticSystem


CURRENT_CELL_FRAME = None


class KeyboardHandler(Widget):
    """Listens for keypresses in the application's window and dispatches the appropriate calls."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.enabled = True
        self.keyboard = Window.request_keyboard(lambda: True, self)
        self.keyboard.bind(on_key_down=self.on_keypressed)
        self.keyboard.bind(on_key_up=self.on_keyreleased)

    def disable(self):
        """Temporarily pause the KeyboardHandler, do nothing until reenabled."""
        self.enabled = False

    def enable(self):
        """Reenable the KeyboardHandler, listen for keypresses again."""
        self.enabled = True

    def on_keypressed(self, _keyboard, keycode, _text, modifiers):
        """Catch and handle user keypresses corresponding to app functions."""
        if not self.enabled:
            # except we're still responsible for dismissing the help window
            if App.get_running_app().root.help_window:
                if keycode[1] == 'escape':
                    App.get_running_app().root.toggle_help_window()
                # block all other keypresses too
                return True
            return False
        # we should be disabled if the help window is being shown
        assert App.get_running_app().root.help_window is None
        if App.get_running_app().root.ids.grid.demo_callback is None:
            if keycode[1] == 'd':
                App.get_running_app().root.ids.grid.demo_setup(0)
                return True
            if keycode[1] == 'e':
                App.get_running_app().root.ids.grid.demo_setup_slow(0)
                return True
        else:
            if App.get_running_app().root.ids.grid.running():
                App.get_running_app().root.ids.grid.start_stop_simulation()
        if App.get_running_app().root.ids.grid.warning_label:
            if keycode[1] in ['escape', 'right', 'left', 'pageup', 'pagedown',
                              'home', 'end', 'spacebar', 'delete']:
                App.get_running_app().root.ids.grid.hide_warning()
            if keycode[1] == 'escape':
                return True
            # fallthrough on purpose
        if keycode[1] == 'right' and 'ctrl' not in modifiers:
            # run a single step of the simulation
            App.get_running_app().root.ids.grid.step()
            return True
        if keycode[1] == 'left' and 'ctrl' not in modifiers or keycode[1] == 'backspace':
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
        if keycode[1] == 'left' and 'ctrl' in modifiers:
            # seek to last color change
            App.get_running_app().root.ids.grid.seek_prev_change()
            return True
        if keycode[1] == 'right' and 'ctrl' in modifiers:
            # seek to next color change
            App.get_running_app().root.ids.grid.seek_next_change()
            return True
        if keycode[1] == 'home' or keycode[1] == 'left' and 'ctrl' in modifiers:
            # reset simulation to initial state
            App.get_running_app().root.ids.grid.rewind_all()
            return True
        if keycode[1] == 'end' or keycode[1] == 'right' and 'ctrl' in modifiers:
            # reset simulation to latest state
            App.get_running_app().root.ids.grid.forward_all()
            return True
        if keycode[1] == 'w':
            # write the current results to file (hard-wired filepath for now)
            App.get_running_app().root.ids.grid.export_results('results.csv')
            return True
        if keycode[1] == 's':
            # run the predefined number of steps (max_steps)
            App.get_running_app().root.ids.grid.simulate()
            return True
        if keycode[1] == 'spacebar':
            # run simulation until spacebar pressed again
            App.get_running_app().root.ids.grid.start_stop_simulation()
            return True
        if keycode[1] == 'delete':
            # forget rest of history from this point
            App.get_running_app().root.ids.grid.delete_rest_of_history()
            return True
        if keycode[1] == 'shift' or keycode[1] == 'rshift':
            App.get_running_app().root.show_overlay_grid()
            return True
        if keycode[1] == 'enter' and 'shift' in modifiers:
            App.get_running_app().root.replace_para_with_overlay()
            return True
        if keycode[1] == 'f':
            Window.fullscreen = not Window.fullscreen
            return True
        if keycode[1] == 'p':
            App.get_running_app().root.toggle_control_panel()
            return True
        return False

    def on_keyreleased(self, _keyboard, keycode):
        """Remove overlay ParadigmaticSystemGrid once user releases shift key."""
        if keycode[1] == 'shift' or keycode[1] == 'rshift':
            App.get_running_app().root.hide_overlay_grid()
            return True
        return False


class PyramidWindow(AnchorLayout):

    def __init__(self, para=None, **kwargs):
        super().__init__(**kwargs)
        self.help_window = None
        self.control_panel = None
        self.overlay = None
        self.ids.grid.set_para(para)
        self.update_information_labels()
        self.toggle_control_panel()

    def update_information_labels(self):
        """Update and refresh all texts on screen indicating the state of the simulation."""
        self.ids.settings_results_label.update()
        self.ids.conjunctive_label.update()
        self.ids.monotonic_label.update()

    def toggle_help_window(self, *args):
        """Show or hide fullscreen Label with help text."""
        if args and 1 < len(args[0]):
            event = args[0][1]
            if event.is_mouse_scrolling:
                # don't dismiss on mousewheel event
                return
        if not self.help_window:
            self.keyboardhandler.disable()
            self.help_window = HelpWindow()
            self.add_widget(self.help_window)
        else:
            self.remove_widget(self.help_window)
            self.help_window = None
            self.keyboardhandler.enable()

    def toggle_control_panel(self, *_args):
        """Show or hide ControlPanel widget at the bottom."""
        if not self.control_panel:
            self.control_panel = ControlPanel()
            self.ids.control_panel_anchor.add_widget(self.control_panel)
        else:
            self.ids.control_panel_anchor.remove_widget(self.control_panel)
            self.control_panel = None

    def show_overlay_grid(self):
        """Show paradigmatic system rearranged to be compact and monotonic."""
        if not self.overlay and not self.ids.grid.warning_label:
            self.ids.grid.show_info('Trying all permutations, hang tight...')
            self.ids.grid.warning_label.canvas.ask_update()  # not sure if this is necessary
            # we need to give the Kivy event loop a bit of a headstart to draw the next frame
            # before the number crunching starts and the UI hangs (possibly for several seconds)
            Clock.schedule_once(self.find_rearranged_para, 0.1)

    def find_rearranged_para(self, *_args):
        """Callback to actually crunch the numbers and come up with a compact paradigmatic system."""
        try:
            good_permutation = self.ids.grid.settings.monotonic_criterion(self.ids.grid)
            self.ids.grid.hide_info()
        except ValueError as error:
            self.ids.grid.hide_info()
            self.ids.grid.show_warning(str(error))
            return
        # FIXME: we're supposed to check if Shift is still being held at this point but I don't know how
        if good_permutation:
            para_rearranged = self.ids.grid.clone()
            para_rearranged.permute(self.ids.grid, *good_permutation)
            self.overlay = ParadigmaticSystemGrid(para=para_rearranged)
            self.ids.grid_anchor.add_widget(self.overlay)
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
            self.ids.grid_anchor.remove_widget(self.overlay)
            self.overlay = None
        else:
            self.ids.grid.hide_warning()


class ScaledFontBehavior:

    def __init__(self, **_kwargs):
        self.orig_font_size = self.font_size
        self.resize_text()
        Window.bind(size=self.resize_text)

    def resize_text(self, *_args):
        """Adjust font size proportionally to Window size."""
        self.font_size = int(self.orig_font_size * Window.height / 600)


class HelpButton(Button):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.bind(on_release=self.toggle_help_window)

    # you can't bind to PyramidWindow in __init__ because of Kivy's initialization order
    def toggle_help_window(self, *args):
        """Show or hide fullscreen Label with help text."""
        App.get_running_app().root.toggle_help_window(args)
        return True


class HelpWindow(ScaledFontBehavior, Label):

    def __init__(self, para=None, **kwargs):
        super(ScaledFontBehavior, self).__init__(**kwargs)
        super(HelpWindow, self).__init__()
        # block click events from Widgets below
        self.bind(on_touch_down=lambda *_: True)
        self.bind(on_touch_up=self.toggle_help_window)
        self.text = dedent('''\
            [size=32][b]Help[/b][/size]\n
            Each matrix cell shows the prevalence (the "bias") of a certain morphological phenomenon
            when the morphemes in its row and column are combined. Bias values range from 0 to 1.\n
            Click on any row or column label to edit the morpheme corresponding to that row or column.
            Click on any paradigm cell to change the value of its bias.\n
            Press [b]RightArrow (->)[/b] to perform one iteration of the simulation.
            Press [b]LeftArrow (<-)[/b] to undo one iteration of the simulation.\n
            Press [b]PageDown[/b] to perform ten iterations of the simulation.
            Press [b]PageUp[/b] to undo ten iterations of the simulation.\n
            Press [b]Ctrl-RightArrow[/b] to skip to the next cell color change.
            Press [b]Ctrl-LeftArrow[/b] to rewind to the last cell color change.\n
            Press [b]End[/b] to skip to the last state of the simulation.
            Press [b]Home[/b] to skip to the initial state of the simulation.\n
            Press the [b]S key[/b] to run the simulation to the predefined limit.
            Press [b]Space[/b] to start or stop an open-ended simulation.\n
            Press [b]Delete[/b] to clear history from the current state onward.\n
            Hold [b]Shift[/b] to see if the paradigm can be rearranged to be compact and monotonic.
            While holding [b]Shift[/b], press [b]Enter[/b] to keep the rearranged paradigm and replace
            the original paradigm with it.''')

    def toggle_help_window(self, *args):
        """Show or hide fullscreen Label with help text."""
        App.get_running_app().root.toggle_help_window(args)
        return True


class ControlPanel(BoxLayout):

    class ControlPanelButton(ScaledFontBehavior, Button):

        def __init__(self, **kwargs):
            super(ScaledFontBehavior, self).__init__(**kwargs)
            super(ControlPanel.ControlPanelButton, self).__init__()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rewind_all_button = self.ControlPanelButton(font_size=44, text='I<<')
        rewind_all_button.bind(on_release=lambda _: App.get_running_app().root.ids.grid.rewind_all())
        self.add_widget(rewind_all_button)
        seek_prev_button = self.ControlPanelButton(font_size=44, text='<<')
        seek_prev_button.bind(on_release=lambda _: App.get_running_app().root.ids.grid.seek_prev_change())
        self.add_widget(seek_prev_button)
        undo_step_button = self.ControlPanelButton(font_size=44, text='<')
        undo_step_button.bind(on_release=lambda _: App.get_running_app().root.ids.grid.undo_step())
        self.add_widget(undo_step_button)
        go_button = self.ControlPanelButton(font_size=32, text='Walk')
        go_button.bind(on_release=lambda _: App.get_running_app().root.ids.grid.demo_setup_slow(0))
        self.add_widget(go_button)
        go_faster_button = self.ControlPanelButton(font_size=32, text='Run')
        go_faster_button.bind(on_release=lambda _: App.get_running_app().root.ids.grid.demo_setup(0))
        self.add_widget(go_faster_button)
        do_step_button = self.ControlPanelButton(font_size=44, text='>')
        do_step_button.bind(on_release=lambda _: App.get_running_app().root.ids.grid.step())
        self.add_widget(do_step_button)
        seek_next_change = self.ControlPanelButton(font_size=44, text='>>')
        seek_next_change.bind(on_release=lambda _: App.get_running_app().root.ids.grid.seek_next_change())
        self.add_widget(seek_next_change)
        forward_all_button = self.ControlPanelButton(font_size=44, text='>>I')
        forward_all_button.bind(on_release=lambda _: App.get_running_app().root.ids.grid.simulate())
        self.add_widget(forward_all_button)


class ParadigmaticSystemGrid(ParadigmaticSystem, GridLayout):

    def __init__(self, para=None, **kwargs):
        ParadigmaticSystem.__init__(self)
        GridLayout.__init__(self, **kwargs)
        self.warning_label = None
        self.timed_callback = None
        self.demo_callback = None
        self.pegs = []
        # a convenient alias to maintain symmetry with show_info
        self.hide_info = self.hide_warning
        if para:
            self.set_para(para)

    def set_para(self, para):
        """Create text fields and buttons if needed and load the contents of the new paradigm we have been handed."""
        if not para:
            return
        assert isinstance(para, ParadigmaticSystem)
        self.para_state = deepcopy(para.para_state)
        self.history = deepcopy(para.history)
        self.history_index = para.history_index
        self.settings = deepcopy(para.settings)
        self.rows = len(para) + 1
        self.cols = len(para[0]) + 1
        self.row_text_inputs = []
        self.col_text_inputs = []
        if not self.children:
            self.add_widget(Widget())  # spacer in the top left corner
            for j, label in enumerate(para.state().col_labels):
                self.col_text_inputs.append(ParadigmaticSystemText(col=j, text=label))
                self.add_widget(self.col_text_inputs[-1])
            for i, (label, row) in enumerate(zip(para.state().row_labels, para)):
                self.row_text_inputs.append(ParadigmaticSystemText(row=i, text=label))
                self.add_widget(self.row_text_inputs[-1])
                for j, value in enumerate(row):
                    self.add_widget(ParadigmaticSystemCell(i, j))
        self.update_all_cells()

    def get_cell(self, row, col):
        """Return the ParadigmaticSystemCell widget corresponding to an underlying ParadigmaticSystem cell."""
        # N.B. Kivy's add_widget function pushes widgets to the front of the child widget list
        return self.children[- (row + 1) * (len(self[0]) + 1) - (col + 1) - 1]

    def show_warning(self, message):
        """Display a warning popup on the screen."""
        if not self.warning_label:
            self.warning_label = WarningLabel()
            self.warning_label.text = message
            self.parent.add_widget(self.warning_label)

    def show_info(self, message):
        """Display a popup on screen about some expected event."""
        # FIXME: this is probably bad UI design, but an info label always replaces a warning
        if self.warning_label:
            self.hide_info()
        self.warning_label = InfoLabel()
        self.warning_label.text = message
        self.parent.add_widget(self.warning_label)

    def hide_warning(self):
        """Remove warning popup."""
        if self.warning_label:
            self.parent.remove_widget(self.warning_label)
            self.warning_label = None

    def show_current_cell_frame(self, on=True, pick=None):
        """Enable/disable drawing a bright gold rectangle above the ParadigmaticSystemCell that was chosen last."""
        global CURRENT_CELL_FRAME
        if on:
            if not CURRENT_CELL_FRAME:
                CURRENT_CELL_FRAME = CurrentCellFrame()
            if not pick:
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

    def hide_pegs(self):
        """Remove demonstrative pegs from the screen."""
        for s in self.pegs:
            self.canvas.remove(s)
        self.pegs = []

    def show_pegs(self):
        """Display illustrative pegs for an easier visual explanation of the model's mechanics."""
        self.hide_pegs()
        if pick := self.state().last_pick:
            current_cell = self.get_cell(*pick)
            cell_size = current_cell.width
            pick_coordinates = (current_cell.x + cell_size * 0.5, current_cell.y + cell_size * 0.5)
            length = cell_size * 0.5
            gap = 50
            width = 3
            for drow, dcol in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                row = pick[0] + drow
                col = pick[1] + dcol
                try:
                    neighbor = self.get_cell(row, col)
                except IndexError:
                    continue
                if not isinstance(neighbor, ParadigmaticSystemCell):
                    continue
                if self[row][col] == 0.5:
                    continue
                elif self[row][col] < 0.5:
                    peg_color = Color(0.90, 0.31, 0.30)
                else:
                    peg_color = Color(0.22, 0.80, 0.22)
                border_color = Color(0.85, 0.9, 0)
                from_coordinates = (pick_coordinates[0] + dcol * (length + gap * 0.5),
                                    pick_coordinates[1] - drow * (length + gap * 0.5))
                to_coordinates = (pick_coordinates[0] + dcol * gap * 0.5,
                                  pick_coordinates[1] - drow * gap * 0.5)
                next_peg = Line(points=[from_coordinates[0], from_coordinates[1],
                                        to_coordinates[0], to_coordinates[1]],
                                width=width)
                next_border = Line(points=[from_coordinates[0], from_coordinates[1],
                                           to_coordinates[0], to_coordinates[1]],
                                   width=width+2)
                self.pegs.append(next_peg)
                self.pegs.append(next_border)
                if self.settings.effect_direction == ParadigmaticSystem.Settings.EffectDir.INWARD:
                    self.canvas.add(border_color)
                    self.canvas.add(next_border)
                    self.canvas.add(peg_color)
                    self.canvas.add(next_peg)
                else:
                    # TODO
                    pass

    def step(self):
        """Perform one iteration of the simulation (thin wrapper around ParadigmaticSystem.step)."""
        super().step()
        self.update_all_cells()
        self.show_pegs()

    def undo_step(self):
        """Revert one iteration of the simulation (thin wrapper around ParadigmaticSystem.undo_step)."""
        super().undo_step()
        self.update_all_cells()
        self.show_pegs()

    def rewind_all(self):
        """Revert simulation all the way to initial state."""
        # TODO: refactor this
        if self.demo_callback is not None:
            if self.running():
                self.start_stop_simulation()
            if self.demo_callback is not None:
                self.demo_callback.cancel()
            self.demo_callback = None
        if self.timed_callback:
            self.start_stop_simulation()
        super().rewind_all()
        self.update_all_cells()

    def forward_all(self):
        """Redo all iterations until the latest state."""
        # TODO: refactor this
        if self.demo_callback is not None:
            if self.running():
                self.start_stop_simulation()
            if self.demo_callback is not None:
                self.demo_callback.cancel()
            self.demo_callback = None
        if self.timed_callback:
            self.start_stop_simulation()
        super().forward_all()
        self.update_all_cells()

    def seek_prev_change(self):
        """Jump to the last state where a cell changed its color."""
        # TODO: refactor this
        if self.demo_callback is not None:
            if self.running:
                self.start_stop_simulation()
            self.demo_callback = None
        if self.timed_callback:
            self.start_stop_simulation()
        super().seek_prev_change()
        if 0 < self.history_index < len(self.history) - 1:
            frame_pick = self.history[self.history_index + 1].last_pick
            self.show_current_cell_frame(True, pick=frame_pick)

    def seek_next_change(self):
        """Jump to the next state where a cell changes its color."""
        # TODO: refactor this
        if self.demo_callback is not None:
            if self.running:
                self.start_stop_simulation()
            self.demo_callback = None
        if self.timed_callback:
            self.start_stop_simulation()
        if self.state().total_steps < self.settings.max_steps and self.history_index > len(self.history) - 10:
            # this might take a while, let the user know we didn't hang
            self.show_info("Seeking next change...")
            Clock.schedule_once(self.find_next_change, 0.1)
        else:
            super().seek_next_change()

    def find_next_change(self, *_args):
        """Actually do the number crunching part of seek_next_change."""
        super().seek_next_change()
        self.hide_info()

    def delete_rest_of_history(self):
        """Get rid of history items forward from current state."""
        if self.timed_callback:
            self.start_stop_simulation()
        super().delete_rest_of_history()
        self.show_info('Forward history cleared.')

    def start_stop_simulation(self):
        """Keep running the simulation until the same method is called again."""
        if self.timed_callback:
            assert self.running()
            self.timed_callback.cancel()
            self.timed_callback = None
        else:
            if self.state().total_steps >= self.settings.max_steps:
                self.show_warning("Maximum number of steps reached.")
            else:
                self.sim_status = ParadigmaticSystem.SimStatus.RUNNING
                self.run_batch(0)
                self.timed_callback = Clock.schedule_interval(self.run_batch, 0.1)

    def run_batch(self, _elapsed_time):
        """Callback to perform the next batch of iterations of an open-ended simulation."""
        if not self.running():
            self.timed_callback.cancel()
            self.timed_callback = None
            return
        #para_size = len(self) * len(self[0])
        super().simulate(batch_size=10)
        # FIXME: is this really supposed to happen here?
        self.show_current_cell_frame(False)

    def simulate(self):
        """Run the simulation until the maximum number of steps is reached."""
        if self.state().total_steps >= self.settings.max_steps:
            self.show_warning("Maximum number of steps reached.")
            return
        if self.history is not None:
            # skip to the last simulation state in existing history
            self.forward_all()
        if self.state().total_steps < self.settings.max_steps:
            self.show_info("Simulation running...")
            Clock.schedule_once(self.run_simulation, 0.1)

    def run_simulation(self, _elapsed_time):
        """Callback to run the simulation up to the predefined number of maximum steps."""
        super().simulate()
        self.hide_info()

    def demo_setup(self, _elapsed_time):
        # TODO: refactor this
        if self.demo_callback is not None:
            if self.running():
                self.start_stop_simulation()
            self.demo_callback.cancel()
            self.demo_callback = None
        else:
            self.demo_callback = Clock.schedule_once(self.demo_run, 0)

    def demo_reset(self, _elapsed_time):
        self.rewind_all()
        self.delete_rest_of_history()
        self.hide_info()
        self.demo_callback = None
        self.demo_setup(0)

    def demo_run(self, _elapsed_time):
        self.start_stop_simulation()
        self.demo_callback = Clock.schedule_once(self.demo_reset, 30)

    def demo_setup_slow(self, _elapsed_time):
        if self.timed_callback:
            assert self.running()
            self.timed_callback.cancel()
            self.timed_callback = None
        if self.demo_callback is not None:
            self.demo_callback.cancel()
            self.demo_callback = None
        else:
            self.demo_callback = Clock.schedule_once(self.demo_run_slow, 0)

    def demo_run_slow(self, _elapsed_time):
        self.step()
        if self.state().total_steps < 1000:
            self.demo_callback = Clock.schedule_once(self.demo_run_slow, 0.4)
        else:
            self.rewind_all()
            self.delete_rest_of_history()
            self.hide_info()
            self.demo_callback = Clock.schedule_once(self.demo_setup_slow, 3)

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
        """Set the bias of a cell in the underlying ParadigmaticSystem object to a new value."""
        if self[row][col].value == new_bias:
            return False
        self.invalidate_future_history()
        self.store_snapshot()
        self.state().total_steps += 1
        self.state().last_pick = row, col
        quant_before = self.quantize(row, col)
        self[row][col].value = new_bias
        quant_after = self.quantize(row, col)
        self.state().sim_result.current_state_changed = quant_before != quant_after
        self.get_cell(row, col).update()
        return True

    def update_all_cells(self):
        """Sync all visual grid cells with the cells of the underlying ParadigmaticSystem object."""
        for child in self.children:
            if isinstance(child, ParadigmaticSystemCell):
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
        self.show_current_cell_frame(True)
        if App.get_running_app().root:
            App.get_running_app().root.update_information_labels()


class ParadigmaticSystemText(TextInput):

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


class ParadigmaticSystemCell(ScaledFontBehavior, AnchorLayout, Button):

    def __init__(self, row, col, **kwargs):
        super(ScaledFontBehavior, self).__init__(**kwargs)
        super(ParadigmaticSystemCell, self).__init__()
        self.row = row
        self.col = col
        self.bind(on_release=self.edit_bias)

    def edit_bias(self, *_args):
        """Show a temporary TextInput box to allow the user to redefine the label of a column or a row."""
        textinput = CellEditText()
        self.add_widget(textinput)
        textinput.focus = True

    def update(self):
        """Sync this cell's content and color with the bias of the corresponding cell in the ParadigmaticSystem."""
        cell = self.parent[self.row][self.col]
        if isinstance(cell, bool):
            self.text = str(cell)
        else:
            assert isinstance(cell, Cell)
            try:
                self.text = f"%0.2g" % cell.value
            except TypeError:
                self.text = cell.value
        grapefruit = Color(0.90, 0.31, 0.30)
        grey       = Color(0.50, 0.50, 0.50)
        lime       = Color(0.22, 0.80, 0.22)
        if self.parent.settings.tripartite_colors:
            cutoff = self.parent.settings.tripartite_cutoff
            if cell < 1 - cutoff:
                self.background_color = grapefruit.rgb
            elif 1 - cutoff <= cell <= cutoff:
                self.background_color = grey.rgb
            elif cutoff < cell:
                self.background_color = lime.rgb
            else:
                assert False
        else:
            self.background_color = [sum(x) for x in zip([   cell  * c for c in lime.rgb],
                                                         [(1-cell) * c for c in grapefruit.rgb])]


class CellEditText(ScaledFontBehavior, TextInput):

    def __init__(self, **kwargs):
        super(ScaledFontBehavior, self).__init__(**kwargs)
        super(CellEditText, self).__init__()
        App.get_running_app().root.keyboardhandler.disable()
        self.bind(focus=self.focus_changed)

    def set_new_value(self, instance):
        """Set a new bias value in the cell once the user finished typing."""
        assert self == instance
        if not self.text:
            # user left the text field blank
            return
        updated = False
        try:
            new_value = float(self.text)
            # clamp to [0, 1]
            new_value = max(0, min(1, new_value))
            updated = self.parent.parent.update_cell(self.parent.row, self.parent.col, new_value)
        except ValueError:
            try:
                if "true" == self.text.lower():
                    new_value = True
                elif "false" == self.text.lower():
                    new_value = False
                updated = self.parent.parent.update_cell(self.parent.row, self.parent.col, new_value)
            except NameError:
                self.parent.parent.show_warning("Matrix values are supposed to be numeric or Boolean.")
                pass
        if updated:
            App.get_running_app().root.ids.grid.eval_criteria()
            App.get_running_app().root.ids.grid.update_all_cells()
            App.get_running_app().root.update_information_labels()

    def focus_changed(self, instance, focused=None):
        """Get rid of this TextInput box if the user has clicked elsewhere."""
        assert self == instance
        if focused is False:
            self.set_new_value(self)
            self.parent.remove_widget(self)
            App.get_running_app().root.keyboardhandler.enable()


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
        App.get_running_app().root.ids.grid.hide_warning()
        return True


class InfoLabel(WarningLabel):
    pass


class CriterionLabel(Label):

    # you can't bind to PyramidWindow in __init__ because of Kivy's initialization order
    def update(self):
        """Set label according to the current status of the criterion."""
        # this relative reference thing is getting a bit out of hand...
        results = self.parent.parent.parent.parent.ids.grid.state().sim_result
        if self.criterion_name == 'conjunctive':
            satisfied = results.current_state_conjunctive
        elif self.criterion_name == 'monotonic':
            satisfied = results.current_state_monotonic
        else:
            assert False
        if satisfied:
            self.text = self.criterion_name.upper()
            self.background_color = Color(0.4, 0.75, 0.15).rgba
        else:
            self.text = 'NOT ' + self.criterion_name.upper()
            self.background_color = Color(0.85, 0.25, 0.18).rgba


class SettingsAndResultsLabel(ScaledFontBehavior, Label):

    def __init__(self, **kwargs):
        super(ScaledFontBehavior, self).__init__(**kwargs)
        super(SettingsAndResultsLabel, self).__init__()

    def update(self):
        """Display latest information about the simulation's state and settings."""
        settings = self.parent.parent.ids.grid.settings
        results = self.parent.parent.ids.grid.state().sim_result
        text = (f'''[size=29][b]______ Settings ______[/b][/size]
            effect_radius = {settings.effect_radius}
            cells_own_weight = {settings.cells_own_weight}
            no_reordering = {settings.no_reordering}
            no_edges = {settings.no_edges}\n''' +
            (f'kappa = {settings.kappa}\n' if settings.decaying_delta else f'delta = {settings.delta}\n') +
            f'tripartite_colors = {settings.tripartite_colors}\n' +
            (f'tripartite_cutoff = {settings.tripartite_cutoff}\n' if settings.tripartite_colors else '') +
            f'''max_steps = {settings.max_steps}

            [size=29][b]______ Results ______[/b][/size]
            conjunctive_states = {results.conjunctive_states}
            monotonic_states = {results.monotonic_states}
            total_states = {results.total_states}
            conjunctive_changes = {results.conjunctive_changes}
            monotonic_changes = {results.monotonic_changes}
            total_changes = {results.total_changes}
            ''')
        self.text = '\n'.join(line.strip() for line in text.split('\n'))


class PyramidApp(App):
    def build(self):
        state_5x5 = ParadigmaticSystem.State( row_labels=['kapu', 'hiú', 'dal', 'jel', 'nap'],
                                              col_labels=['-k', '-d', '-t', '-rA', '-bA'],
                                              matrix=cells_from_floats(
                                                         [ [0, 0, 0, 0, 0],
                                                           [0, 0, 0, 0, 0],
                                                           [1, 1, 1, 0, 0],
                                                           [1, 1, 1, 0, 0],
                                                           [1, 1, 1, 0, 0]
                                                         ]) )
        state_10x10 = ParadigmaticSystem.State( row_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'],
                                                col_labels=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'],
                                                matrix=cells_from_floats(
                                                           [ [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
                                                             [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
                                                           ]) )
        para = ParadigmaticSystem(state_5x5)
        if para.settings.tripartite_colors:
            para.settings.conjunctive_criterion = ParadigmaticSystem.is_conjunctive_tripartite
            para.settings.monotonic_criterion = ParadigmaticSystem.can_be_made_monotonic_tripartite
        else:
            para.settings.conjunctive_criterion = ParadigmaticSystem.is_conjunctive_strict
            para.settings.monotonic_criterion = ParadigmaticSystem.can_be_made_monotonic_strict
        para.track_history(True)
        root = PyramidWindow(para)
        if para.settings.no_reordering:
            if para.settings.monotonic_criterion in [ParadigmaticSystem.can_be_made_monotonic_strict,
                                                     ParadigmaticSystem.can_be_made_monotonic_tripartite]:
                root.ids.grid.show_warning('Warning: monotonic criterion allows reordering')
        root.keyboardhandler = KeyboardHandler()
        return root


if __name__ == '__main__':
    PyramidApp().run()
