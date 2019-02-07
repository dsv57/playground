#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import random, randint, uniform, choice, seed
#from numpy import sin, cos, arctan2, sqrt, ceil, floor, degrees, radians, log, pi, exp
from math import sin, cos, atan2, sqrt, ceil, floor, degrees, radians, log, pi, exp
from copy import deepcopy
from sys import exc_info
import re

# Code analysis
# import sys  # for sys.path (autocomp)
import ast
import jedi
from sys import getsizeof
from collections import defaultdict

from kivy.uix.textinput import FL_IS_LINEBREAK

from kivy.uix.codeinput import CodeInput
from kivy.uix.widget import Widget
from kivy.uix.stencilview import StencilView
from kivy.uix.scatter import ScatterPlane
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Color, Line, Rectangle, Ellipse, Triangle, \
        PushMatrix, PopMatrix
from kivy.graphics.transformation import Matrix
from kivy.graphics.context_instructions import Rotate, Translate, Scale
from kivy.properties import StringProperty, NumericProperty, \
        ListProperty, ObjectProperty, BooleanProperty, \
        OptionProperty, DictProperty
from kivy.clock import Clock

from ourturtle import Turtle, Vec2d

try:
    import mycolors
except:
    pass

# Code An
_MAX_VAR_SIZE = 4096
jedi.settings.add_bracket_after_function = True
jedi.settings.case_insensitive_completion = False
jedi.preload_module('ourturtle')


def autocomp(src, namespace, lineno, column):
    return jedi.Interpreter(
        'from ourturtle import Turtle\n' + src, [namespace],
        line=lineno + 1 + 1,
        column=column).completions()


# Дамп _до_ каждой строчки
class Tracer2(ast.NodeTransformer):
    def __init__(self, **kvargs):
        self._lines_done = []
        super(Tracer2, self).__init__(**kvargs)

    def visit(self, node):
        def tracer_ast(lineno):
            return ast.Expr(
                value=ast.Call(
                    func=ast.Name(id='_dump_vars', ctx=ast.Load()),
                    args=[
                        ast.Call(
                            func=ast.Name(id='locals', ctx=ast.Load()),
                            args=[],
                            keywords=[]),
                        ast.Num(n.lineno)
                    ],
                    keywords=[]))

        if hasattr(node, 'body'):
            new_body = []
            prev_lineno = -1
            for i, n in enumerate(node.body):
                if n.lineno not in self._lines_done and n.lineno > prev_lineno:
                    self._lines_done.append(n.lineno)
                    prev_lineno = n.lineno
                    new_body.append(tracer_ast(n.lineno))
                new_body.append(n)
            node.body = new_body
        return self.generic_visit(node)


class CodeEditor(CodeInput):
    def __init__(self, **kwargs):
        self._highlight_line = None
        self.ac_begin = False
        self.ac_current = 0
        self.ac_position = None
        self.ac_completions = []
        super(CodeEditor, self).__init__(**kwargs)


#    def _get_bbcode(self, ntext):
#        print('_get_bbcode', ntext)
#        return super(CodeEditor, self)._get_bbcode(ntext)

#    def _create_line_label(self, text, hint=False):
#        print('_create_line_label', text, hint)
#        return super(CodeEditor, self)._create_line_label(text, hint)

    def highlight_line(self, line_num):
        self._highlight_line = line_num
        self._trigger_update_graphics()

    def _update_graphics(self, *largs):
        super(CodeInput, self)._update_graphics(*largs)
        self._update_graphics_highlight()

    def _update_graphics_highlight(self):
        if not self._highlight_line:
            return
        self.canvas.remove_group('highlight')
        dy = self.line_height + self.line_spacing
        rects = self._lines_rects
        padding_top = self.padding[1]
        padding_bottom = self.padding[3]
        _top = self.top
        y = _top - padding_top + self.scroll_y
        miny = self.y + padding_bottom
        maxy = _top - padding_top
        draw_highlight = self._draw_highlight
        line_num = self._highlight_line - 1
        # pass only the selection lines[]
        # passing all the lines can get slow when dealing with a lot of text
        y -= line_num * dy
        _lines = self._lines
        _get_text_width = self._get_text_width
        tab_width = self.tab_width
        _label_cached = self._label_cached
        width = self.width
        padding_left = self.padding[0]
        padding_right = self.padding[2]
        x = self.x
        canvas_add = self.canvas.add
        highlight_color = (.9, .1, .1, .3)
        # value = _lines[line_num]
        if miny <= y <= maxy + dy:
            r = rects[line_num]
            draw_highlight(r.pos, r.size, line_num, _lines, _get_text_width,
                           tab_width, _label_cached, width, padding_left,
                           padding_right, x, canvas_add, highlight_color)
        y -= dy
        self._position_handles('both')

    def _draw_highlight(self, *largs):
        pos, size, line_num,\
            _lines, _get_text_width, tab_width, _label_cached, width,\
            padding_left, padding_right, x, canvas_add, selection_color = largs
        # Draw the current selection on the widget.
        x, y = pos
        w, h = size
        x1 = x
        x2 = x + w
        lines = _lines[line_num]
        x1 -= self.scroll_x
        x2 = (x - self.scroll_x) + _get_text_width(lines, tab_width,
                                                   _label_cached)
        width_minus_padding = width - (padding_right + padding_left)
        maxx = x + width_minus_padding
        if x1 > maxx:
            return
        x1 = max(x1, x)
        x2 = min(x2, x + width_minus_padding)
        canvas_add(Color(*selection_color, group='highlight'))
        canvas_add(
            Rectangle(
                pos=(x1, pos[1]), size=(x2 - x1, size[1]), group='highlight'))

    def _split_smart(self, text):
        # Disable word wrapping (because of broken highlight)
        lines = text.split(u'\n')
        lines_flags = [0] + [FL_IS_LINEBREAK] * (len(lines) - 1)
        return lines, lines_flags

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        # print('keyboard_on_key_down', keycode, text, modifiers)
        key, key_str = keycode
        if key == 9:  # Tab
            cc, cr = self.cursor
            _lines = self._lines
            text = _lines[cr]
            # cursor_index = self.cursor_index()
            # text_last_line = _lines[cr - 1]
            # print(111, repr(text[:cc]), repr(text[cc:]))
            if text[:cc].lstrip() == '' and not modifiers:
                self.insert_text('    ')
                self.ac_begin = False
            else:
                # print("AC", self.ac_begin, cc, cr, self.ac_position, self.ac_current, self.ac_completions)
                if self.ac_begin:
                    if self.ac_completions[self.ac_current].complete:
                        self.do_undo()
                    self.ac_current += 1
                    self.ac_current %= len(self.ac_completions)
                else:
                    self.ac_completions = autocomp(self.text, self.namespace,
                                                   cr, cc)
                    print(
                        "ACS: ", '\n'.join(
                            sorted(
                                [ac.full_name for ac in self.ac_completions])))
                    # print('= ' * 40)
                    # print('AC BEGIN', _lines[cr], self.text.splitlines()[cr], cr, cc, self.ac_completions)
                    if self.ac_completions:
                        self.ac_begin = True
                        self.ac_position = self.cursor
                        self.ac_current = 0
                #print('ac:', self.ac_state, self.ac_completions)
                if self.ac_completions:
                    ac = self.ac_completions[self.ac_current]
                    self.insert_text(ac.complete)
                    #self.ac_state[2] = len(ac.complete)
            return True
        self.ac_begin = False
        super(CodeInput, self).keyboard_on_key_down(window, keycode, text,
                                                    modifiers)


class Ball(Widget):
    def goto(self, x, y):
        self.pos = (x, y)


class VarSlider(GridLayout):
    var_name = StringProperty('a')
    value = NumericProperty(0)
    min = NumericProperty(-10)
    max = NumericProperty(10)
    step = NumericProperty(0.01)
    type = OptionProperty('int', options=['float', 'int'])

    _VALID_ID = re.compile(r"^[^\d\W]\w*")

    def _to_numtype(self, v):
        try:
            if self.type == 'float':
                return round(float(v), 1)
            else:
                return int(v)
        except ValueError:
            return self.min

    def _str(self, v):
        if self.type == 'float':
            return '{:.1f}'.format(v)
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


#    def __init__(self, **kwargs):
#        super(CodeEditor, self).__init__(**kwargs)


class OurSandbox(ScatterPlane):
    ball = ObjectProperty(None)

    #    points = ListProperty([90, -30, 0, 70])
    #    turtles = []

    def __init__(self, **kwargs):
        # self.turtles = []
        super(OurSandbox, self).__init__(**kwargs)

    # def add_turtle(self):
    #     t = Turtle()
    #     self.turtles.append(t)
    #     return t
#    def goto(self, x, y):
#        r = self.ball.size[0]
#        self.ball.pos = (x-r/2, y-r/2)
#        self.points.append((x, y))
#        if len(self.points) > 70:
#            self.points.pop(0)
#
#    def prpos(self):
#        print('POS:', self.ball.pos)
#        return self.ball.pos

    def on_touch_down(self, touch):
        if touch.is_mouse_scrolling:
            scale = self.scale + (0.05
                                  if touch.button == 'scrolldown' else -0.05)
            if (self.scale_min and scale < self.scale_min) \
                    or (self.scale_max and scale > self.scale_max):
                return
            rescale = scale * 1.0 / self.scale
            self.apply_transform(
                Matrix().scale(rescale, rescale, rescale),
                post_multiply=True,
                anchor=self.to_local(*touch.pos))
            return self.dispatch('on_transform_with_touch', touch)
        return super().on_touch_down(touch)


#    velocity_x = NumericProperty(0)
#    velocity_y = NumericProperty(0)
#    velocity = ReferenceListProperty(velocity_x, velocity_y)

#    def move(self):
#        self.pos = Vector(*self.velocity) + self.pos


class MapViewer(StencilView):
    def on_touch_down(self, touch):
        if not self.collide_point(*touch.pos):  # touch is not within bounds
            return False
        return super(MapViewer,
                     self).on_touch_down(touch)  # delegate to stencil

    def on_touch_move(self, touch):
        if not self.collide_point(*touch.pos):  # touch is not within bounds
            return False
        return super(MapViewer,
                     self).on_touch_move(touch)  # delegate to stencil

    def on_touch_up(self, touch):
        if not self.collide_point(*touch.pos):
            return False
        return super(MapViewer, self).on_touch_up(touch)


class Playground(FloatLayout):

    init_code = StringProperty('''#penup()
#for i in range(10*n):
#    pensize(i/50)
#    color(sin(i/m)/2+0.5,cos(i/m)/2+0.5,b/10+1)
#    goto(300*sin(i/(a*5)), 300*cos(i/15))
#    pendown()
forward(100)
right(45)
forward(70)
bob = Turtle()
bob.color('yellow')
bob.left(45)
bob.forward(90)
''')
    run_code = StringProperty('''
#x, y = t.pos()
#t.goto(300*sin(steps/(4)), 300*cos(steps/5))
bob.left(1)
bob.forward(1)
''')

    #    var_a = NumericProperty(0.5)
    #    var_b = NumericProperty(0.5)
    #    var_m = NumericProperty(10)
    #    var_n = NumericProperty(10)
    vars = DictProperty({})

    console = StringProperty('')

    sandbox = ObjectProperty(None)
    init_editor = ObjectProperty(None)
    rpanel = ObjectProperty(None)
    run_to_cursor = BooleanProperty(False)

    init_code_compiled = None
    run_code_compiled = None

    #    ball = ObjectProperty(None)

    def add_turtle_instance(self):
        turtle = Turtle()  # self.sandbox.add_turtle()
        self._the_turtle = turtle
        for v in dir(turtle):
            if v[0] != '_':
                self._globals[v] = getattr(turtle, v)

    def __init__(self, **kwargs):
        super(Playground, self).__init__(**kwargs)
        #        self.test_canvas = RenderContext()
        #        s = self.test_canvas.shader
        self._run_vars = None

        self._globals = dict()
        for v in 'random randint uniform choice seed sin cos atan2 \
                sqrt ceil floor degrees radians log pi exp'.split(
        ):
            self._globals[v] = eval(v)

        def _dump_vars(v, lineno):
            global _vars
            for k, v in v.copy().items():
                if any(
                    [isinstance(v, t)
                     for t in [int, float, str, dict, tuple]]) and k[0] != '_':
                    self._run_vars[lineno][k].append(
                        v if getsizeof(v) <= _MAX_VAR_SIZE else '<LARGE>')

        self._globals['_dump_vars'] = _dump_vars
        self._globals['Turtle'] = Turtle  # self.sandbox.add_turtle
        self.add_turtle_instance()
        try:
            self._globals['cam16ucs_to_srgb'] = mycolors.cam16ucs_to_srgb
            self._globals['jzazbz_to_srgb'] = mycolors.jzazbz_to_srgb
            self._globals['srgb_to_cam16ucs'] = mycolors.srgb_to_cam16ucs
            #for v in dir(mycolors):
            #    if v[0] != '_':
            #        self._globals[v] = getattr(mycolors, v)
        except:
            pass

        self.init_editor.namespace = self._globals

        vs1 = VarSlider(var_name='a', type='float')
        vs2 = VarSlider(var_name='b', type='float')
        vs3 = VarSlider(var_name='c', type='float')
        vs4 = VarSlider(var_name='l', min=0, max=50)
        vs5 = VarSlider(var_name='m', min=0, max=100)
        vs6 = VarSlider(var_name='n', min=0, max=150)
        self.rpanel.add_widget(vs1, 1)
        self.rpanel.add_widget(vs2, 1)
        self.rpanel.add_widget(vs3, 1)
        self.rpanel.add_widget(vs4, 1)
        self.rpanel.add_widget(vs5, 1)
        self.rpanel.add_widget(vs6, 1)

        #        self._globals = {}
        self.steps = 0
        self.trigger_exec_init = Clock.create_trigger(self.execute_init, -1)
        self.bind(init_code=self.compile_init)
        self.bind(run_to_cursor=self.compile_init)
        self.init_editor.bind(cursor_row=self.on_init_editor_cursor_row)

        #        self.bind(var_a=self.trigger_exec_init)
        #        self.bind(var_b=self.trigger_exec_init)

        def _set_var(wid, value):
            self.vars[wid.var_name] = value
            self.trigger_exec_init()

        vs1.bind(value=_set_var)
        vs2.bind(value=_set_var)
        vs3.bind(value=_set_var)
        vs4.bind(value=_set_var)
        vs5.bind(value=_set_var)
        vs6.bind(value=_set_var)
        vs1.value = 1.2
        vs2.value = 3.4
        vs3.value = 4.2
        vs4.value = 15
        vs5.value = 50
        vs6.value = 75

        #        self.bind(var_m=self.trigger_exec_init)
        #        self.bind(var_n=self.trigger_exec_init)
        self.compile_init()

        self.graphics_instructions = []
        self.bind(run_code=self.compile_run)
        self.compile_run()
        self.trigger_exec_run = Clock.create_trigger(self.execute_run, -1)
        self.run_schedule = None  # Clock.schedule_interval(self.trigger_exec_run, 1.0 / 60.0)

    def on_init_editor_cursor_row(self, *largs):
        if self.run_to_cursor:
            self.compile_init()

    def update_sandbox(self, redraw=True):
        if redraw:
            self.graphics_instructions = []
            self.sandbox.canvas.clear()
            with self.sandbox.canvas:
                for t in Turtle.turtles():
                    for color, width, points in t._lines:
                        # if self.run_to_cursor:
                        # color = *color[:3], 0.5
                        # igroup = InstructionGroup()
                        ci = Color(*color)
                        li = Line(
                            points=points,
                            width=width,
                            joint='round',
                            cap='round')
                        self.graphics_instructions.append((deepcopy(
                            (color, width, points)), ci, li))
                    for shape, pos, size, color in t._stamps:
                        # if self.run_to_cursor:
                        # color = *color[:3], 0.5
                        with self.sandbox.canvas:
                            Color(*color)
                            if shape == 'Ellipse':
                                Ellipse(
                                    pos=pos - 0.5 * Vec2d(size[0], size[1]),
                                    size=(size[0], size[1]))
                            elif shape == 'Rectangle':
                                Rectangle(
                                    pos=pos - 0.5 * Vec2d(size[0], size[1]),
                                    size=(size[0], size[1]))

                    PushMatrix()
                    size = t._shapesize
                    Translate(*t._position)
                    Scale(size)
                    Rotate(angle=t.heading(), axis=(0, 0, 1), origin=(0, 0))
                    Color(*t._pencolor)
                    Triangle(points=[0, -10, 30, 0, 0, 10])
                    PopMatrix()
        else:
            i = 0
            instrs = len(self.graphics_instructions)
            # st_ch = 0
            # st_n = 0
            with self.sandbox.canvas:
                for t in Turtle.turtles():
                    for color, width, points in t._lines:
                        if i < instrs:
                            line, ci, li = self.graphics_instructions[i]
                            # print(line)
                            if line != (color, width, points):
                                # print("CHANGE", points, li.points)
                                li.points = points
                                self.graphics_instructions[i] = (deepcopy(
                                    (color, width, points)), ci, li)
                                # st_ch += 1
                        else:
                            # st_n += 1
                            ci = Color(*color)
                            li = Line(
                                points=points,
                                width=width,
                                joint='round',
                                cap='round')
                            self.graphics_instructions.append((deepcopy(
                                (color, width, points)), ci, li))
                        i += 1
            # print("STATS:", instrs, st_ch, st_n)

    def compile_init(self, *largs):
        #        self._run_vars = defaultdict(lambda: defaultdict(list))
        print('= ' * 40)
        print(self.init_code)

        try:
            #            tree = ast.parse(self.init_code)
            #            Tracer2().visit(tree)
            #            ast.fix_missing_locations(tree)
            #            self.init_code_compiled = compile(tree, '<code-input>', 'exec')

            if self.run_to_cursor:
                print("ROW:", self.init_editor.cursor_row)
                tree = ast.Module([
                    node for node in ast.parse(self.init_code).body
                    if node.lineno <= self.init_editor.cursor_row
                ])
                self.init_code_compiled = compile(tree, '<code-input>', 'exec')
            else:
                self.init_code_compiled = compile(self.init_code,
                                                  '<code-input>', 'exec')
        except Exception as e:
            print(e)
            print('* ' * 40)
            exc_type, exc_obj, exc_tb = exc_info()
            line_num = None
            #            print(44, exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno)
            if hasattr(exc_obj, 'lineno'
                       ):  #exc_type and issubclass(exc_type, SyntaxError):
                print('exc_obj', exc_obj.lineno, exc_obj.filename)
                line_num = exc_obj.lineno

            while exc_tb:
                #                print(55, exc_tb.tb_lineno, exc_tb.tb_frame.f_code.co_filename)
                if exc_tb.tb_frame.f_code.co_filename == '<code-input>':
                    line_num = exc_tb.tb_lineno
                exc_tb = exc_tb.tb_next

            self.init_editor.highlight_line(line_num)

#            fname = exc_tb.tb_frame.f_code.co_filename
#            print(333, exc_tb.tb_lineno, fname)
#            self.init_editor._highlight_line = exc_tb.tb_lineno # line_num #exc_info()[2].tb_lineno

#            etype, value, tb = exc_info()
#            texc = traceback.TracebackException(exc_type, exc_obj, exc_tb, limit=None)
#            print('00', list(texc.format_exception_only()))
##            print(77, list(texc.format(chain=True)))
#            print(88, texc.stack)
#            print(99, ''.join(texc.stack.format()))
#            line_num = None
#            for fs in reversed(texc.stack):
#                print(44, repr(fs.filename))
#                if fs.filename == '<code-input>':
#                    line_num = fs.lineno
#                    break
#            print('line_num', line_num)
#            for s in reversed(traceback.extract_stack()):
#                print(55, s)
#            print(66, traceback.format_exc())
#            print(333, self.init_editor._highlight_line)
#            print(self.init_editor) #e)
#            print(type(self.init_editor))
#            print(exc_info()[2].tb_lineno)
#            traceback.print_stack()
#            tb = exc_info()[2]
#            print(22, tb, tb.tb_next)
#            while tb:
#                print('1111', tb, tb.tb_lineno, tb.tb_frame.f_code.co_filename, tb.tb_frame.f_lineno)
#                tb = tb.tb_next
##            stack = traceback.extract_stack()[:-1]  # last one would be full_stack()
##            if exc is not None:  # i.e. an exception is present
##                del stack[-1]       # remove call of full_stack, the printed exception
##                                    # will contain the caught exception caller instead
#            trc = 'Traceback (most recent call last):\n'
#            stackstr = '' # trc + ''.join(traceback.format_list(stack))
##            if tb is not None:
#            stackstr += '  ' + traceback.format_exc().lstrip(trc)
#            print(2222, stackstr)
##            print(traceback.format_exc())
#            print(exc.tb_frame.f_lineno)
#            print(exc)
#            print(exc.tv)
#            print(dir(self.init_editor))
        else:
            self.init_editor.highlight_line(None)
            self.trigger_exec_init()

    def execute_init(self, *largs):
        if not self.init_code_compiled:
            return
#        init_code = self.init_code
#        run_code = self.run_code

#        print('-->', init_code)

        self._globals.update(self.vars)
        Turtle.clear_turtles()
        self.add_turtle_instance()
        self._run_vars = defaultdict(lambda: defaultdict(list))
        seed(123)
        try:
            exec(self.init_code_compiled, self._globals, None)
            #self._globals.update(globals())
            #self._globals.update(locals())
#            cons_out = ''
#            for k, v in globals().copy().items():
#                print(123, k, type(v), repr(v)[:50])
#                if any([isinstance(v, t) for t in [int, float, str, dict, tuple]]) and k[0] != '_':
#                    #print(k, type(v), repr(v)[:80], sep='\t')
#                    cons_out += k + '\t' + repr(v)[:80] + '\n'
#            self.console = cons_out
#            print('cons_out', cons_out)

#            self.sandbox.points = self.sandbox.turtles[-1]._points
#            self.sandbox.ball.pos = tuple(self.sandbox.turtles[0]._position)

        except Exception as e:
            #            cons_out = ''
            #            for k, v in globals().copy().items():
            #                if any([isinstance(v, t) for t in [int, float, str, dict, tuple]]) and k[0] != '_':
            #                    #print(k, type(v), repr(v)[:80], sep='\t')
            #                    cons_out += k + '\t' + repr(v)[:80] + '\n'
            #            self.console = cons_out
            print(e)
            exc_type, exc_obj, exc_tb = exc_info()
            line_num = None
            #            print(44, exc_tb.tb_frame.f_code.co_filename, exc_tb.tb_lineno)
            if hasattr(exc_obj, 'lineno'
                       ):  #exc_type and issubclass(exc_type, SyntaxError):
                print('exc_obj', exc_obj.lineno, exc_obj.filename)
                line_num = exc_obj.lineno

            while exc_tb:
                #                print(55, exc_tb.tb_lineno, exc_tb.tb_frame.f_code.co_filename)
                if exc_tb.tb_frame.f_code.co_filename == '<code-input>':
                    line_num = exc_tb.tb_lineno
                exc_tb = exc_tb.tb_next

            self.init_editor.highlight_line(line_num)

        else:
            #            pass
            self.init_editor.highlight_line(None)
            # FIXME: add scene spdiff
            self.update_sandbox()
            if self.run_schedule:
                Clock.unschedule(self.run_schedule)
            self.run_schedule = Clock.schedule_interval(
                self.trigger_exec_run, 1.0 / 30.0)

            #Ellipse(pos=t._position - 0.5*Vec2d(size, size), size=(size, size))

#        last_line = max(self._run_vars.keys())[-1]
#        print(33333, _run_vars[last_line])
#        cons_out = ''
#        for k, vs in _run_vars[last_line].items():
#            v = vs[-1]
#            if any([isinstance(v, t) for t in [int, float, str, dict, tuple]]) and k[0] != '_':
#                #print(k, type(v), repr(v)[:80], sep='\t')
#                cons_out += k + '\t' + repr(v)[:80] + '\n'
#        self.console = cons_out

    def compile_run(self, *largs):
        try:
            self.run_code_compiled = compile(self.run_code, '<run-code-input>',
                                             'exec')
        except:
            pass  # FIXME

    def execute_run(self, *largs):
        self.steps += 1
        # glob = self._globals #globals()
        # glob.update(self.vars)
        #        glob['goto'] = self.sandbox.goto
        #        glob['prpos'] = self.sandbox.prpos
        # glob['steps'] = self.steps
        self._globals.update(self.vars)
        self._globals['steps'] = self.steps
        #        glob['a'] = self.var_a
        #        glob['b'] = self.var_b
        #        glob['m'] = self.var_m
        #        glob['n'] = self.var_n
        try:
            #exec(init_code, glob)
            # exec(self.run_code_compiled, glob)
            exec(self.run_code_compiled, self._globals, None)


#            exec(self.run_code, glob)
        except Exception as e:
            print(e)
        else:
            self.update_sandbox(False)
