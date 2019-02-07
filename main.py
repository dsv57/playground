import sys
import kivy
kivy.require('1.0.6')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.floatlayout import FloatLayout
#from kivy.uix.scatterlayout import ScatterLayout
#from kivy.uix.relativelayout import RelativeLayout
from kivy.core.window import Window
#from kivy.factory import Factory
#from kivy.graphics import Color, Line, Rectangle, Ellipse
#from kivy.graphics.transformation import Matrix
from kivy.properties import StringProperty, NumericProperty, ListProperty, ObjectProperty
from kivy.clock import Clock

#import traceback
#from traceback import TracebackException

from ouruix import CodeEditor, Ball, OurSandbox, MapViewer, Playground, VarSlider
#from turtle import Vec2D

#class CodeEditor(BoxLayout):
#    def change_scroll_y(self, ti, scrlv):
#        y_cursor = ti.cursor_pos[1]
#        y_bar = scrlv.scroll_y * (ti.height-scrlv.height)
#        if ti.height > scrlv.height:
#            if y_cursor >= y_bar + scrlv.height:
#                dy = y_cursor - (y_bar + scrlv.height)
#                scrlv.scroll_y = scrlv.scroll_y + scrlv.convert_distance_to_scroll(0, dy)[1] 
#            if y_cursor - ti.line_height <= y_bar:
#                dy = (y_cursor - ti.line_height) - y_bar
#                scrlv.scroll_y = scrlv.scroll_y + scrlv.convert_distance_to_scroll(0, dy)[1] 


class PlaygroundApp(App):
    def build(self):
        kwargs = {}
        if len(sys.argv) > 1:
            kwargs['source'] = sys.argv[1]
        return Playground(**kwargs)


if __name__ == '__main__':
    PlaygroundApp().run()
