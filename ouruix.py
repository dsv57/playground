#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import random, randint, uniform, choice, seed
#from numpy import sin, cos, arctan2, sqrt, ceil, floor, degrees, radians, log, pi, exp
from math import sin, cos, atan2, sqrt, ceil, floor, degrees, radians, log, pi, exp
from copy import deepcopy
from time import process_time, time
from os.path import exists
# from sys import exc_info
import re

# Code analysis
# import sys  # for sys.path (autocomp)
import ast
# import jedi
# from sys import getsizeof
from collections import defaultdict, namedtuple

from kivy.uix.textinput import FL_IS_LINEBREAK

from kivy.uix.behaviors import FocusBehavior
from kivy.uix.codeinput import CodeInput
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.actionbar import ActionItem
from kivy.uix.stencilview import StencilView
from kivy.uix.scatter import ScatterPlane
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, Line, Rectangle, Ellipse, Triangle, \
        PushMatrix, PopMatrix, RoundedRectangle
from kivy.graphics.transformation import Matrix
from kivy.graphics.context_instructions import Rotate, Translate, Scale
from kivy.properties import StringProperty, NumericProperty, \
        ListProperty, ObjectProperty, BooleanProperty, \
        OptionProperty, DictProperty
from kivy.clock import Clock
from kivy.cache import Cache
from kivy.utils import escape_markup
from kivy.core.text.markup import MarkupLabel
from kivy.core.window import Window # request keyboard

import pymunk

from ourturtle import Turtle
from sprite import Sprite, OurImage, Vector
from codean import autocomp, CodeRunner, Break, COMMON_CODE
from sokoban.sokoban import Sokoban

try:
    import mycolors
except:
    pass

# https://github.com/kivy/kivy/wiki/Working-with-Python-threads-inside-a-Kivy-application

F_UPDATE = 'update'
F_ON_KEY_PRESS = 'on_key_press'
F_ON_KEY_RELEASE = 'on_key_release'

R_TURN = re.compile(r'^(\s*)(right|left|up|down)\(([0-9]*)\)$')

def whos(vars, max_repr=40):
    w_types = (int, float, str, list, dict, tuple)
    def w_repr(v):
        r = repr(v)
        return r if len(r) < max_repr else r[:max_repr-3] + '...'
    return [(k, type(v).__qualname__, w_repr(v))
            for k, v in vars.items()
            if isinstance(v, w_types) and k[0] != '_']

class Key(namedtuple('Key', ['keycode', 'key', 'text'])):
    def __eq__(self, b):
        if b is not None and b in (self.keycode, self.key, self.text):
            return True
        return False

    def __str__(self):
        return f"Key '{self.key or self.text}' ({self.keycode})"

class ActionStepSlider(BoxLayout, ActionItem):
    step = NumericProperty(0)
    max_step = NumericProperty(0)

class CodeEditor(CodeInput, FocusBehavior):
    def __init__(self, **kwargs):
        # self._highlight_line = None
        self.hightlight_styles = {
            'error': (True, (.9, .1, .1, .4)),
            'run': (False, (.1, .9, .1, 1.0))
        }
        self._highlight = defaultdict(set)
        # self._highlight['run'].add(3)
        self.namespace = {}
        self.ac_begin = False
        self.ac_current = 0
        self.ac_position = None
        self.ac_completions = []
        self.register_event_type('on_key_down')
        super(CodeEditor, self).__init__(**kwargs)


#    def _get_bbcode(self, ntext):
#        print('_get_bbcode', ntext)
#        return super(CodeEditor, self)._get_bbcode(ntext)

#    def _create_line_label(self, text, hint=False):
#        print('_create_line_label', text, hint)
#        return super(CodeEditor, self)._create_line_label(text, hint)

    def highlight_line(self, line_num, style='error', add=False):
        if line_num:
            if add:
                if isinstance(line_num, int):
                    self._highlight[style].add(line_num)
                else:
                    self._highlight[style].update(line_num)
            else:
                if isinstance(line_num, int):
                    line_num = [line_num]
                self._highlight[style] = set(line_num)
        else:
            self._highlight[style].clear()
        self._trigger_update_graphics()

    def _update_graphics(self, *largs):
        super(CodeInput, self)._update_graphics(*largs)
        self._update_graphics_highlight()

    def _update_graphics_highlight(self):
        if not self._highlight:
            return
        for style in self._highlight:
            self.canvas.remove_group('hl-'+style)
            for line_num in self._highlight[style]:
                dy = self.line_height + self.line_spacing
                padding_top = self.padding[1]
                padding_bottom = self.padding[3]
                y = self.top - padding_top + self.scroll_y
                miny = self.y + padding_bottom
                maxy = self.top - padding_top
                line_num -= 1
                # pass only the selection lines[]
                # passing all the lines can get slow when dealing with a lot of text
                y -= line_num * dy
                if miny <= y <= maxy + dy:
                    self._draw_highlight(line_num, style)
        self._position_handles('both')

    def _draw_highlight(self, line_num, style):
        if min(len(self._lines_rects), len(self._lines)) <= line_num:
            return
        fill, highlight_color = self.hightlight_styles[style]
        rect = self._lines_rects[line_num]
        pos = rect.pos
        size = rect.size
        # Draw the current selection on the widget.
        x, y = pos
        w, h = size
        x1 = x
        x2 = x + w
        lines = self._lines[line_num]
        x1 -= self.scroll_x
        x2 = (x - self.scroll_x) + self._get_text_width(lines, self.tab_width,
                                                   self._label_cached)
        width_minus_padding = self.width - (self.padding[2] + self.padding[0])
        maxx = x + width_minus_padding
        if x1 > maxx:
            return
        x1 = max(x1, x)
        x2 = min(x2, x + width_minus_padding)

        self.canvas.add(Color(*highlight_color, group='hl-'+style))
        # self.canvas.add(
        #     Rectangle(
        #         pos=(x1, pos[1]), size=(x2 - x1, size[1]), group='highlight'))
        group='hl-'+style
        with self.canvas:
            Color(*highlight_color, group=group)
            if fill:
                RoundedRectangle(
                    pos=(x1, pos[1]), size=(x2 - x1, size[1]),
                    radius=(4, 4), segments=3, group=group)
            else:
                Line(
                    rounded_rectangle=(x1, pos[1], x2 - x1, size[1], 4),
                    group=group)
        # self.canvas.add(
        #     Line(rounded_rectangle=(x1, pos[1], x2 - x1, size[1], 4), width=1.3, group='highlight'))


    def _split_smart(self, text):
        # Disable word wrapping (because of broken syntax highlight)
        lines = text.split(u'\n')
        lines_flags = [0] + [FL_IS_LINEBREAK] * (len(lines) - 1)
        return lines, lines_flags

    def _create_line_label(self, text, hint=False):
        # Fix empty lines bug
        # Create a label from a text, using line options
        ntext = text.replace(u'\n', u'').replace(u'\t', u' ' * self.tab_width)
        ntext = self._get_bbcode(ntext)
        kw = self._get_line_options()
        cid = u'{}\0{}\0{}'.format(ntext, self.password, kw)
        texture = Cache.get('textinput.label', cid)

        if texture is None:
            # FIXME right now, we can't render very long line...
            # if we move on "VBO" version as fallback, we won't need to
            # do this.
            # try to find the maximum text we can handle
            label = MarkupLabel(text=ntext, **kw)
            label.refresh()
            # ok, we found it.
            texture = label.texture
            Cache.append('textinput.label', cid, texture)
            label.text = ''
        return texture

    def _auto_indent(self, substring):
        index = self.cursor_index()
        _text = self._get_text(encode=False)
        if index > 0:
            line_start = _text.rfind('\n', 0, index)
            if line_start > -1:
                line = _text[line_start + 1:index]
                indent = self.re_indent.match(line).group()
                substring += indent
        if len(_text) > 0 and _text[index-1] == ':':
            substring += ' ' * self.tab_width
        return substring

    def do_backspace(self, from_undo=False, mode='bkspc'):
        # Clever backspace: remove up to previous indent level.
        if self.readonly:
            return
        cc, cr = self.cursor
        _lines = self._lines
        text = _lines[cr]
        cursor_index = self.cursor_index()
        tab = self.tab_width
        if cc > 0 and text[:cc].lstrip() == '':
            indent = (cc - 1) // tab
            remove = cc - indent * tab
            new_text = ' ' * indent * tab + text[cc:]
            substring = ' ' * remove
            self._set_line_text(cr, new_text)
            self.cursor = self.get_cursor_from_index(cursor_index - remove)
            # handle undo and redo
            self._set_undo_redo_bkspc(
                cursor_index,
                cursor_index - remove,
                substring, from_undo)
            return
        super(CodeEditor, self).do_backspace(from_undo, mode)

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        # print('keyboard_on_key_down', keycode, text, modifiers)
        key, key_str = keycode
        print('kk', keycode, text, modifiers)
        if key == 9:  # Tab
            if modifiers == ['ctrl']:
                if self.focus_next:
                    self.focus_next.focus = True
                return True
            cc, cr = self.cursor
            _lines = self._lines
            text = _lines[cr]
            before_cursor = text[:cc].lstrip()
            tab = self.tab_width
            # cursor_index = self.cursor_index()
            # text_last_line = _lines[cr - 1]
            # print(111, repr(text[:cc]), repr(text[cc:]))
            if self._selection and modifiers in [[], ['shift']]:
                a, b = self._selection_from, self._selection_to
                cc_from, cr_from = self.get_cursor_from_index(a)
                cc_to, cr_to = self.get_cursor_from_index(b)
                for cr in range(min(cr_from, cr_to), max(cr_from, cr_to) + 1):
                    line = _lines[cr]
                    if not modifiers:
                        new_text = ' ' * tab + line
                        self._set_line_text(cr, new_text)
                        if cr == cr_from: cc_from += tab
                        if cr == cr_to: cc_to += tab
                    else:
                        spaces = len(line) - len(line.lstrip())
                        indent = (spaces - 1) // tab
                        if indent >= 0:
                            remove = spaces - indent * tab
                            new_text = line[remove:]
                            self._set_line_text(cr, new_text)
                            if cr == cr_from: cc_from -= remove
                            if cr == cr_to: cc_to -= remove
                self._selection_from = self.cursor_index((cc_from, cr_from))
                self._selection_to = self.cursor_index((cc_to, cr_to))
                self._selection_finished = True
                self._update_selection(True)
                self._update_graphics_selection()
                return True
            elif not self._selection and before_cursor == '' and not modifiers:
                self.insert_text(' ' * tab)
                self.ac_begin = False
                return True
            elif not self._selection and before_cursor != '' and not modifiers:
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
                # print('ac:', self.ac_state, self.ac_completions)
                if self.ac_completions:
                    ac = self.ac_completions[self.ac_current]
                    self.insert_text(ac.complete)
                    # self.ac_state[2] = len(ac.complete)
                return True

        elif modifiers == ['alt'] and key_str in ['up', 'down', 'right', 'left']:
            print(self._lines)
            cc, cr = self.cursor
            empty_line = self._lines[cr].strip() == ''
            if empty_line:
                cr -= 1
            # if self._lines[cr].strip():
            #     self.do_cursor_movement('cursor_end')
            #     self.insert_text('\n')
            #     cc, cr = self.cursor
            # l1 = self._lines[:cr]
            # l2 = self._lines[cr:]
            # self._lines = l1 + [key_str+'()'] + l2

            # self._selection_from = self._selection_to = self.cursor_index()
            # self._selection = True
            # self._selection_finished = False
            space = ''
            if cr >= 0:
                prev_line = self._lines[cr]
                m = R_TURN.match(prev_line)
                if m:
                    space, cmd, step = m.groups()
                    if cmd == key_str:
                        if step:
                            step = str(int(step)+1)
                        else:
                            step = '2'
                        self._set_line_text(cr, space + cmd + '(' + step + ')')
                        # self._lines[cr-1] = space + cmd + '(' + step + ')'
                        return True
            key_str = space + key_str
            if not empty_line:
                self.do_cursor_movement('cursor_end')
                key_str = '\n' + key_str
            self.insert_text(f'{key_str}()')
            # self.cursor = (cc, cr+1)
            return True

        if self.dispatch('on_key_down', window, keycode, text, modifiers):
            return True

        self.ac_begin = False
        return super(CodeInput, self).keyboard_on_key_down(
            window, keycode, text, modifiers)

    def on_key_down(self, window, keycode, text, modifiers):
        pass

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


class OurSandbox(FocusBehavior, ScatterPlane):

    def __init__(self, **kwargs):
        super(OurSandbox, self).__init__(**kwargs)
        # self.space = pymunk.Space()
        # self.space.gravity = (0.0, -900.0)
        # self.space.sleep_time_threshold = 0.3
        self.register_event_type('on_key_down')
        self.register_event_type('on_key_up')
        self.space = None
        # self._keyboard = None

        # self._keyboard = Window.request_keyboard(
        #     self._keyboard_closed, self, 'text')
        # if self._keyboard.widget:
        #     pass
        # self._keyboard.bind(on_key_down=self.on_key_down, on_key_up=self.on_key_up)
        # print(dir(ScatterPlane))

    # def on_focus(self, instance, value, *largs):
    #     print('Touch')
    #     if not self._keyboard:
    #         self._keyboard = Window.request_keyboard(
    #             self._keyboard_closed, self, 'text')
    #         if self._keyboard.widget:
    #             pass
    #     self._keyboard.bind(on_key_down=self.on_key_down, on_key_up=self.on_key_up)

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        '''We call super before doing anything else to enable tab cycling
        by FocusBehavior. If we wanted to use tab for ourselves, we could just
        not call it, or call it if we didn't need tab.
        '''
        print('DOWN', keycode, text, modifiers)
        if self.dispatch('on_key_down', window, keycode, text, modifiers):
            return True
        return super(OurSandbox, self).keyboard_on_key_down(
            window, keycode, text, modifiers)

    def on_key_down(self, window, keycode, text, modifiers):
        pass

    def keyboard_on_key_up(self, window, keycode):
        if self.dispatch('on_key_up', window, keycode):
            return True
        return super(OurSandbox, self).keyboard_on_key_up(
            window, keycode)

    def on_key_up(self, window, keycode):
        pass

    # def _keyboard_closed(self):
    #     print('UNBIND')
    #     self._keyboard.unbind(on_key_down=self.on_key_down, on_key_up=self.on_key_up)
    #     self._keyboard = None

    # def on_key_down(self, keyboard, keycode, text, modifiers):
    #     print('DOWN', keycode[1] or text, modifiers)
    #     return True

    # def on_key_up(self, keyboard, keycode, *args):
    #     # print('UP', chr(keycode[0]), args)
    #     return True


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

#for i in range(10*n):
#    pensize(i/50)
#    color(sin(i/m)/2+0.5,cos(i/m)/2+0.5,b/10+1)
#    goto(300*sin(i/(a*5)), 300*cos(i/15))
#    pendown()
# forward(100)
# right(45)
# forward(70)
# bob = Turtle()
# bob.color('yellow')
# bob.left(45)
# bob.forward(390)
# bob.pensize(3)

# im = cam16.from_xyz100(srgb.to_xyz100(srgb.from_srgb1(image.reshape((image.shape[0] * image.shape[1], 3)).T)))[[True, True,False,True,False,False,False]].T.reshape(image.shape)
# im[...,2] += a
# im[...,2] %= 400
# image_sr = srgb.to_srgb1(srgb.from_xyz100(cam16.to_xyz100(im.reshape((image.shape[0] * image.shape[1], 3)).T,'JCh'))).T
# img.set_image(image_sr)
# im = img.imc16.copy()

# import numpy as np
# im = np.array([img.imc16[...,0].T, img.imc16[...,5].T, img.imc16[...,3].T]).T
# im[...,2] += a
# im[...,2] %= 360
# im[...,1] *= b
# im[...,0] *= c
# img.set_imc16(im, 'Jsh')
# def update():
#   pass


    code = StringProperty('''ball = add_sprite('circle', x=0, y=-120, body_type=0, color='green')
ball.apply_impulse((50000, 0))

platform = add_sprite('platform', x=250, y=-120, body_type=1, color='red')
def on_key_press(key, modifiers):
    if key == 'up':
        platform.velocity += 0, 15
    if key == 'down':
        platform.velocity -= 0, 15

    print(key, modifiers)

def update(dt):
    pass

''')

    vars = DictProperty({})

    status = ObjectProperty((None, ))

    status_text = StringProperty('')
    console = StringProperty('')
    watches = StringProperty('')
    replay_step = NumericProperty(0)
    replay_steps = NumericProperty(0)

    sandbox = ObjectProperty(None)
    code_editor = ObjectProperty(None)
    rpanel = ObjectProperty(None)
    textout = ObjectProperty(None)
    run_to_cursor = BooleanProperty(False)

    # ball = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Playground, self).__init__(**kwargs)

        self._run_vars = None
        self._last_update_time = None

        globs = dict()
        for v in 'random randint uniform choice seed sin cos atan2 \
                sqrt ceil floor degrees radians log exp'.split():
            globs[v] = eval(v)

        def _dump_vars(v, lineno):
            global _vars
            for k, v in v.copy().items():
                if any(
                    [isinstance(v, t)
                     for t in [int, float, str, dict, tuple]]) and k[0] != '_':
                    self._run_vars[lineno][k].append(
                        v if getsizeof(v) <= _MAX_VAR_SIZE else '<LARGE>')

        globs['Turtle'] = Turtle
        try:
            globs['cam16_to_srgb'] = mycolors.cam16_to_srgb
            globs['cam16ucs_to_srgb'] = mycolors.cam16ucs_to_srgb
            globs['jzazbz_to_srgb'] = mycolors.jzazbz_to_srgb
            globs['srgb_to_cam16ucs'] = mycolors.srgb_to_cam16ucs
            globs['lab_to_cam16ucs'] = mycolors.lab_to_cam16ucs
            # for v in dir(mycolors):
            #     if v[0] != '_':
            #         self._globals[v] = getattr(mycolors, v)
        except:
            pass

        def _add_sprite(*largs, **kvargs):
            sp = Sprite(*largs, **kvargs)
            self.sandbox.add_widget(sp)
            return sp
        globs['add_sprite'] = _add_sprite

        def _add_line(*largs, **kvargs):
            # self.sandbox.add_widget(line)
            with self.sandbox.canvas:
                line = Line(*largs, **kvargs)
            return line
        globs['Line'] = _add_line

        self.sokoban = Sokoban()
        def sokoban_go(dx, dy):
            def go(step=1):
                self.sokoban.move_player(dx, dy, step)
            return go
        globs['right'] = sokoban_go(1, 0)
        globs['left'] = sokoban_go(-1, 0)
        globs['up'] = sokoban_go(0, 1)
        globs['down'] = sokoban_go(0, -1)

        # self._segments = []
        # def _add_segment(point_a, point_b, radius=0.4, color='khaki'):
        #     self._segments.append((point_a, point_b, radius, color))
        # globs['Segment'] = _add_segment
        self._gravity = Vector(0, 0)
        def _set_gravity(g=(0, -900)):
            if isinstance(g, (int, float)):
                g = (0, g)
            self._gravity = Vector(g)
        globs['set_gravity'] = _set_gravity

        self.trigger_exec_update = Clock.create_trigger(self.execute_update, -1)
        self.update_schedule = None

        self.runner = CodeRunner(globals=globs, special_funcs=[F_UPDATE, F_ON_KEY_PRESS, F_ON_KEY_RELEASE])

        self.code_editor.namespace = self.runner.globals  # FIXME?

        # FIXME
        # vs1 = VarSlider(var_name='a', min=0, max=360, type='float')
        # vs2 = VarSlider(var_name='b', type='float')
        # vs3 = VarSlider(var_name='c', type='float')
        # vs4 = VarSlider(var_name='l', min=0, max=50)
        # vs5 = VarSlider(var_name='m', min=0, max=100)
        # vs6 = VarSlider(var_name='n', min=0, max=150)
        # self.rpanel.add_widget(vs1, 1)
        # self.rpanel.add_widget(vs2, 1)
        # self.rpanel.add_widget(vs3, 1)
        # self.rpanel.add_widget(vs4, 1)
        # self.rpanel.add_widget(vs5, 1)
        # self.rpanel.add_widget(vs6, 1)

        # self.status = None
        self._kb_events = []
        self.step = self.prev_step = 0
        self.trigger_exec = Clock.create_trigger(self.execute_code, -1)
        self.bind(code=self.compile_code)
        self.bind(run_to_cursor=self.compile_code)
        self.sandbox.bind(on_key_down=self.sandbox_on_key_down)
        self.sandbox.bind(on_key_up=self.sandbox_on_key_up)
        self.code_editor.bind(cursor_row=self.on_code_editor_cursor_row)
        self.code_editor.bind(on_key_down=self.code_editor_on_key_down)
        self.code_editor.focus_next = self.sandbox
        self.sandbox.focus_next = self.code_editor

        def _set_var(wid, value):
            self.vars[wid.var_name] = value
            if wid.var_name in self.runner.common_vars:
                self.trigger_exec()

        if exists('source.py'):
            with open('source.py') as f:
                self.code = f.read()

        # FIXME
        # vs1.bind(value=_set_var)
        # vs2.bind(value=_set_var)
        # vs3.bind(value=_set_var)
        # vs4.bind(value=_set_var)
        # vs5.bind(value=_set_var)
        # vs6.bind(value=_set_var)
        # vs1.value = 1.2
        # vs2.value = 3.4
        # vs3.value = 4.2
        # vs4.value = 15
        # vs5.value = 50
        # vs6.value = 75

        self.compile_code()

        self.graphics_instructions = []
        # self.bind(run_code=self.compile_run)
        # self.compile_run()

    def code_editor_on_key_down(self, widget, window, keycode, text, modifiers):
        print('code_editor_on_key_down', keycode, text, modifiers)
        if keycode[1] == 'f5':
            self.step = 0
            self.prev_step = 0
            self.runner.reset()
            self.trigger_exec()
            return True

    def sandbox_on_key_down(self, widget, window, keycode, text, modifiers):
        if keycode[1] == 'f5':
            self.step = 0
            self.prev_step = 0
            self.runner.reset()
            self.trigger_exec()
            return True
        self._kb_events.append((
            'down', time(),
            Key(keycode=keycode[0], key=keycode[1], text=text), modifiers))

    def sandbox_on_key_up(self, widget, window, keycode):
        self._kb_events.append((
            'up', time(),
            Key(keycode=keycode[0], key=keycode[1], text=None), None))

    def on_replay_step(self, *largs):
        if self.sokoban:
            code_lines = self.sokoban.replay(self.replay_step)
            self.update_sandbox()
            self.code_editor.highlight_line(code_lines, 'run')

    def on_code_editor_cursor_row(self, *largs):
        if self.run_to_cursor:
            self.compile_code()

    def on_status(self, *largs):
        status = self.status[0]
        if status == 'ERROR':
            exc = self.status[1]
            exc_name = exc.__class__.__name__ if exc else "Unknown Error"
            self.status_text = f'[b][color=f92672]{exc_name}[/color]: [/b]'
            if isinstance(exc, SyntaxError):
                code = exc.text.replace('\n', '⏎')  #.replace('\t', ' ' * 4).replace(' ', '˽')
                pos = exc.offset - 1
                code_before = escape_markup(code[:pos].lstrip())
                code_hl  = escape_markup(code[pos].replace(' ', '_'))
                code_after = escape_markup(code[pos+1:].rstrip()) if len(code) > pos else ""
                code = f'[color=e6db74]{code_before}[/color][b][color=f92672]{code_hl}[/color][/b][color=e6db74]{code_after}[/color]'
                self.status_text += f'{escape_markup(exc.msg)}: {code}'
            else:
                msg = str(exc) or '???'
                self.status_text += escape_markup(msg)
        elif status == 'BREAK':
            self.status_text = '[color=e6db74][b]Break[/b][/color]'
        elif status == 'EXEC':
            pass
        elif status == 'COMPLETE':
            self.status_text = '[color=a6e22e][b]Completed[/b][/color]'
        elif status == 'RUN':
            self.status_text = '[color=a6e22e][b]Run[/b][/color]'
        else:
            pass

    def update_sandbox(self, redraw=True):
        try:
            if redraw:
                self.graphics_instructions = []
                # self.sandbox.canvas.clear()

                # self.sokoban.draw_level(self.sandbox)  # FIXME

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
                                        pos=pos - 0.5 * Vector(size[0], size[1]),
                                        size=(size[0], size[1]))
                                elif shape == 'Rectangle':
                                    Rectangle(
                                        pos=pos - 0.5 * Vector(size[0], size[1]),
                                        size=(size[0], size[1]))

                        PushMatrix()
                        size = t._shapesize
                        Translate(*t._position)
                        Scale(size)
                        Rotate(angle=t.heading(), axis=(0, 0, 1), origin=(0, 0))
                        Color(*t._pencolor)
                        # Triangle(points=[0, -10, 30, 0, 0, 10])  # FIXME
                        PopMatrix()
                    # for s in Sprite._instances:
                    #     s.draw()
                        # self.sandbox.add_widget(s)
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
        except Exception as e:
            print('update_sandbox:', e)

    def compile_code(self, *largs):
        if self.update_schedule:
            self.update_schedule.cancel()
        breakpoint = None
        if self.run_to_cursor:
            breakpoint = self.code_editor.cursor_row + 1

        try:
            changed = self.runner.parse(self.code, breakpoint)
            if changed:
                try:
                    with open('source.py', 'w') as f:
                        f.write(self.code)
                except Exception as e:
                    print("Cannot save file:", e)
            if COMMON_CODE in changed:
                self.runner.reset()

            self.runner.compile(changed)
        except Exception as e:
            print('E:', e)
            print('* ' * 40)
            line_num = self.runner.exception_lineno(e)
            self.code_editor.highlight_line(None, 'run')
            self.code_editor.highlight_line(line_num)
            self.status = ('ERROR', e)
        else:
            self.code_editor.highlight_line(None)
            if COMMON_CODE in changed:
                print('-='*30)
                print(self.code)
                print('-='*30)
                self.trigger_exec()
                changed.remove(COMMON_CODE)
            print('EXEC Changed:', changed)
            try:
                if self.runner.execute(changed) and F_UPDATE in self.runner.globals and not self.update_schedule.is_triggered:
                    self.update_schedule.cancel()
                    self.update_schedule()
            except Exception as e:
                print('E3:', e)
                self.status = ('ERROR', None)

    def execute_code(self, *largs):
        print('execute_code')
        self.status = ('EXEC',)
        self.runner.reset(globals=self.vars)
        self.prev_step = max(self.step, self.prev_step)
        self.step = 0
        # self.runner.set_globals(self.vars, False)
        Turtle.clear_turtles()
        self._segments = []
        # Sprite.clear_sprites()  # FIXME

        turtle = Turtle()  # self.sandbox.add_turtle()
        self._the_turtle = turtle
        self.sokoban.load_level()
        # for v in dir(turtle):  # FIXME
        #     if v[0] != '_':
        #         self.runner.globals[v] = getattr(turtle, v)


        # Clear the Right Way (Thank you Mark Vasilkov)
        # saved = self.sandbox.children[:]
        self.sandbox.clear_widgets()
        self.sandbox.canvas.clear()
        Sprite.clear_sprites()
        if self.sandbox.space:
            self.sandbox.space.remove(*self.sandbox.space.shapes, *self.sandbox.space.bodies)
        self.sandbox.space = pymunk.Space()
        self.sandbox.space.gravity = self._gravity
        self.sandbox.space.sleep_time_threshold = 3.0
        self.sandbox.space.replay_mode = False
        # for widget in saved:
            # self.sandbox.add_widget(widget)
        # self.sandbox.add_widget(Sprite('sokoban/images/player.png', x=0, y=250, body_type=0))
        # self.sandbox.add_widget(Sprite('turtle', x=-50, y=50, body_type=0))
        # self.sandbox.add_widget(Sprite('circle', x=0, y=0, body_type=0))
        # self.sandbox.add_widget(Sprite('circle', x=0, y=40, body_type=0))
        # self.sandbox.add_widget(Sprite('platform', x=-300, y=80, body_type=1))
        # self.sandbox.add_widget(Sprite('square', x=0, y=120, body_type=0))
        # self.sandbox.add_widget(Sprite('circle', x=0, y=160, body_type=0))
        # self.sandbox.add_widget(Sprite('platform', x=300, y=200, body_type=1))

        static_body = self.sandbox.space.static_body
        # for a, b, radius, color in self._segments:
        # static_lines = [pymunk.Segment(static_body, (-311.0, 280.0-400), (0.0, 246.0-400), 0.0),
                        # pymunk.Segment(static_body, (0.0, 246.0-400), (607.0, 343.0-400), 0.0)
                        # ]
        # for line in static_lines:
        #     line.elasticity = 0.95
        #     line.friction = 0.9
        # self.sandbox.space.add(static_lines)

        seed(123)
        ok = False
        try:
            ok = self.runner.execute()
        except Exception as e:
            print('E2:', e)

        watches = ''
        for v, t, r in whos(self.runner.globals):
            watches += f'{v}\t{t}\t{r}\n'

        if ok and F_UPDATE in self.runner.globals and self.prev_step > 0:
            # print('Replay:', prev_step)
            t_start = time()
            self._last_update_time = time() - self.prev_step * 1/30
            for i in range(self.prev_step):
                self.execute_update(0.0, True)
            Sprite.update_from_pymunk(False)
            print('Replay time:', (time() - t_start) * 1000, 'ms')


        out = self.runner.text_stream.getvalue()
        self.console = out
        print('out:', out)
        print('- ' * 40)

        # FIXME: add scene spdiff
        self.update_sandbox()
        if self.sokoban:
            self.replay_step = len(self.sokoban.log)
            self.replay_steps = len(self.sokoban.log)

        # self.code_editor.highlight_line(None)
        if not ok:
            if self.update_schedule:
                self.update_schedule.cancel()
            if self.runner.exception:
                # for l in self.runner.traceback.format():
                #     print(l[:300])
                # print('STACK', self.runner.traceback.stack)
                # stack = None
                exc, exc_str, traceback = self.runner.exception
                print('EXC:', exc_str)
                is_break = isinstance(exc, Break)
                if is_break:
                    self.status = ('BREAK', exc)
                    hl_style = 'run'
                    # self.code_editor.highlight_line(None, 'run')
                else:
                    self.status = ('ERROR', exc)
                    hl_style = 'error'
                # print('Br Line:', self.runner.breakpoint)
                self.code_editor.highlight_line(None)
                self.code_editor.highlight_line(None, 'run')
                # self.code_editor.highlight_line(self.runner.breakpoint, 'run')
                for filename, lineno, name, line, locals in traceback:
                    print('TRACE:', filename, lineno, name, repr(line), repr(locals)[:80]) # filename, lineno, name, line, locals)
                    if filename == '<code-input>':
                        if name != '<module>':
                            watched_locals = whos(locals)
                            if watched_locals:
                                watches += f'== {name} ==\n'
                                for v, t, r in watched_locals:
                                    watches += f'{v}\t{t}\t{r}\n'
                        self.code_editor.highlight_line(lineno, hl_style, add=True)
                        # print('LINES +', lineno, self.code_editor._highlight)

            else:
                self.status = ('ERROR', None)
                print('Unhandled exception')
                self.code_editor.highlight_line(None) # FIXME
        # else:
            # self.code_editor.highlight_line(None)
        else:
            self.status = ('COMPLETE',)
            self.code_editor.highlight_line(None)
            # player_pos = [-x * 36 for x in self.sokoban._player_pos]
            if self.sokoban and self.sokoban.boxes_remaining == 0:
                print('Level completed:', self.sokoban.level)
                self.sokoban.level += 1
                self.sandbox.clear_widgets()
                # self.sokoban.draw_level(self.sandbox)
            if F_UPDATE in self.runner.globals:  # and (not self.update_schedule or not self.update_schedule.is_triggered):
                self._last_update_time = time()
                if self.update_schedule:
                    self.update_schedule.cancel()
                self.update_schedule = Clock.schedule_interval(self.trigger_exec_update, 1.0 / 60.0)

        self.watches = watches
        print('= ' * 40)

        # print("highlight", self.code_editor._highlight)
        # for k, v in self.runner.globals.items():
        #     if any([isinstance(v, t) for t in [int, float, str, dict, tuple]]) and k[0] != '_':
        #         print(k, type(v), repr(v)[:80], sep='\t')
        # print('= ' * 40)

        # if ok and 'update' in self.runner.globals:
        #     self._last_update_time = time()
        #     self.update_schedule = Clock.schedule_interval(self.trigger_exec_update, 1.0 / 30.0)

    def execute_update(self, dt, replay=False):
        self.sandbox.space.replay_mode = replay
        self.step += 1
        self.runner.globals.update(self.vars)
        self.runner.globals['step'] = self.step
        if not replay:
            ts_pos = self.runner.text_stream.tell()
            now = time()
        else:
            now = self._last_update_time + self.step * 1 / 30
        dt = now - self._last_update_time

        try:
            while self._kb_events:
                ev, t, key, modifiers = self._kb_events[0]
                if ev == 'down':
                    self.runner.call_if_exists(F_ON_KEY_PRESS, key, modifiers)
                elif ev == 'up':
                    self.runner.call_if_exists(F_ON_KEY_RELEASE, key)
                self._kb_events.pop(0)
            self.runner.call(F_UPDATE, dt)

        except Exception as e:
            self.update_schedule.cancel()
            watches = ''
            for v, t, r in whos(self.runner.globals):
                watches += f'{v}\t{t}\t{r}\n'
            if self.runner.exception:
                exc, exc_str, traceback = self.runner.exception
            else:
                exc = e
                if hasattr(e, 'message'):
                    exc_str = e.message
                else:
                    exc_str = str(e) or e.__class__.__name__
                traceback = self.runner._trace(e.__traceback__)
            print('EXC:', exc_str)
            is_break = isinstance(exc, Break)
            if is_break:
                self.status = ('BREAK', exc)
                hl_style = 'run'
            else:
                self.status = ('ERROR', exc)
                hl_style = 'error'
            # print('E4', e)
            # lineno = self.runner.exception_lineno(e)
            self.code_editor.highlight_line(None)
            self.code_editor.highlight_line(None, 'run')
            # self.code_editor.highlight_line(self.runner.breakpoint, 'run')
            for filename, lineno, name, line, locals in traceback:
                print('TRACE:', filename, lineno, name, repr(line), repr(locals)[:80]) # filename, lineno, name, line, locals)
                if filename == '<code-input>':
                    if name != '<module>':
                        watched_locals = whos(locals)
                        if watched_locals:
                            watches += f'== {name} ==\n'
                            for v, t, r in watched_locals:
                                watches += f'{v}\t{t}\t{r}\n'
                    self.code_editor.highlight_line(lineno, hl_style, add=True)
            self.watches = watches
        else:
            self._last_update_time = now
            self.sandbox.space.step(1. / 60.)
            self.sandbox.space.step(1. / 60.)
            if not replay:
                Sprite.update_from_pymunk()
                self.update_sandbox(False)
                if self.status[0] != 'RUN':
                    self.status = ('RUN',)

        if not replay:
            self.runner.text_stream.seek(ts_pos)
            out = self.runner.text_stream.read()
            if out:
                print(out)
                print('* ' * 20)
                self.console = (self.console + out)[-3000:]
