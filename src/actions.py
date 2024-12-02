import sys
from enum import Enum
import pynput.mouse
import pynput.keyboard
from pynput.keyboard import Key

mouse = pynput.mouse.Controller()
keyboard = pynput.keyboard.Controller()

mouse_sensitivity = 10
if "-s" in sys.argv:
    i = sys.argv.index("-s")
    if len(sys.argv) > i + 1:
        s = sys.argv[i + 1]
        if s.isdecimal():
            mouse_sensitivity = int(s)


class Action(Enum):
    MOUSE_UP = 1
    MOUSE_DOWN = 2
    MOUSE_LEFT = 3
    MOUSE_RIGHT = 4
    ARROW_UP = 10
    ARROW_DOWN = 11
    ARROW_LEFT = 12
    ARROW_RIGHT = 13
    PRESS_Q = 100
    PRESS_W = 101
    PRESS_E = 102
    PRESS_R = 103
    PRESS_A = 120
    PRESS_S = 121
    PRESS_D = 122
    PRESS_F = 123
    PRESS_Y = 140
    PRESS_X = 141
    PRESS_C = 142

    def trigger(self, value: float):
        value = abs(value)
        match self:
            case Action.MOUSE_UP:
                mouse.move(0, -value * mouse_sensitivity)
            case Action.MOUSE_DOWN:
                mouse.move(0, value * mouse_sensitivity)
            case Action.MOUSE_LEFT:
                mouse.move(-value * mouse_sensitivity, 0)
            case Action.MOUSE_RIGHT:
                mouse.move(value * mouse_sensitivity, 0)
            case Action.ARROW_UP:
                if value > 0:
                    keyboard.tap(Key.up)
            case Action.ARROW_DOWN:
                if value > 0:
                    keyboard.tap(Key.down)
            case Action.ARROW_LEFT:
                if value > 0:
                    keyboard.tap(Key.left)
            case Action.ARROW_RIGHT:
                if value > 0:
                    keyboard.tap(Key.right)
            case _:
                if self.name.startswith("PRESS_") and value > 0:
                    key = self.name.split('_', 1)[1].lower()
                    keyboard.tap(key)
                    return
