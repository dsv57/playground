from kivy.uix.boxlayout import BoxLayout
from kivy.uix.actionbar import ActionItem
from kivy.properties import NumericProperty


class ActionStepSlider(BoxLayout, ActionItem):
    step = NumericProperty(0)
    max_step = NumericProperty(0)
