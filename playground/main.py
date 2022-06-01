import sys
import os.path

# # Linear sRGB monkey patch
# from playground.color import _srgb_to_linear
# import kivy.parser
# parse_color = kivy.parser.parse_color
# kivy.parser.parse_color = lambda text: _srgb_to_linear(parse_color(text))

# import kivy.graphics
# LinearColor = kivy.graphics.Color
# class Color(LinearColor):
#     def set_state(self, param, value):
#         if param == 'color':
#             value = list(_srgb_to_linear(value[:3])) + value[3]
#         super(Color, self).set_state(param, value)
# kivy.graphics.Color = Color

import kivy
#kivy.require('1.0.6')

from kivy.app import App
from kivy.config import Config
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.properties import StringProperty, NumericProperty, ListProperty, ObjectProperty
from kivy.clock import Clock
from kivy.resources import resource_add_path

# from ouruix import CodeEditor, OurSandbox, Scene, Playground, VarSlider
from uix.code_editor import CodeEditor
from uix.our_sandbox import OurSandbox
from uix.playground import Playground
from uix.var_slider import VarSlider
from uix.scene import Scene
from uix.action_step_slider import ActionStepSlider
# from uix. import 


class PlaygroundApp(App):
    def build(self):
        # Load Kivy theme from data/
        resource_add_path(os.path.join(os.path.dirname(sys.argv[0]), 'data'))

        LabelBase.register(
            DEFAULT_FONT,
            '../fonts/FiraSans-Regular.ttf',
            '../fonts/FiraSans-Italic.ttf',
            '../fonts/FiraSans-Bold.ttf',
            '../fonts/FiraSans-BoldItalic.ttf')

        LabelBase.register(
            'Fira Code',
            '../fonts/FiraCode-Regular.ttf')
            # fn_bold='fonts/FiraCode-Bold.ttf')

        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['source'] = sys.argv[1]
        return Playground(**kwargs)


if __name__ == '__main__':
    print('Kivy path:', kivy.__path__)
    Config.set('input', 'mouse', 'mouse,disable_multitouch')
    PlaygroundApp().run()
