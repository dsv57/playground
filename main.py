import sys
import kivy
kivy.require('1.0.6')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
from kivy.core.window import Window
from kivy.core.text import LabelBase, DEFAULT_FONT
from kivy.properties import StringProperty, NumericProperty, ListProperty, ObjectProperty
from kivy.clock import Clock

from ouruix import CodeEditor, Ball, OurSandbox, MapViewer, Playground, VarSlider

class PlaygroundApp(App):
    def build(self):
        LabelBase.register(
            DEFAULT_FONT,
            'fonts/FiraSans-Regular.ttf',
            'fonts/FiraSans-Italic.ttf',
            'fonts/FiraSans-Bold.ttf',
            'fonts/FiraSans-BoldItalic.ttf')

        LabelBase.register(
            'Fira Code',
            'fonts/FiraCode-Regular.ttf',
            fn_bold='fonts/FiraCode-Bold.ttf')

        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['source'] = sys.argv[1]
        return Playground(**kwargs)


if __name__ == '__main__':
    PlaygroundApp().run()
