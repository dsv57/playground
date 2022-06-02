import re

from kivy.uix.gridlayout import GridLayout
from kivy.properties import StringProperty, NumericProperty, OptionProperty


class VarSlider(GridLayout):
    var_name = StringProperty("a")
    value = NumericProperty(0)
    min = NumericProperty(-10)
    max = NumericProperty(10)
    step = NumericProperty(0.01)
    type = OptionProperty("int", options=["float", "int"])

    _VALID_ID = re.compile(r"^[^\d\W]\w*")

    def _to_numtype(self, v):
        try:
            if self.type == "float":
                return round(float(v), 1)
            else:
                return int(v)
        except ValueError:
            return self.min

    def _str(self, v):
        if self.type == "float":
            return "{:.1f}".format(v)
        else:
            return str(v)

    def _filter_var(self, wid, substring, from_undo):
        if from_undo:
            cc = 0
            new_text = substring
        else:
            text = self.var_name
            cc, cr = wid.cursor
            new_text = text[:cc] + substring
        m = re.match(self._VALID_ID, new_text)
        if m:
            return m.group()[cc:]
