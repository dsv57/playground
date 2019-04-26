#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import random, randint, uniform, choice, seed
#from numpy import sin, cos, arctan2, sqrt, ceil, floor, degrees, radians, log, pi, exp
from math import sin, cos, atan2, sqrt, ceil, floor, degrees, radians, log, pi, exp
from copy import deepcopy
from time import process_time, time
from weakref import WeakValueDictionary
from itertools import chain
from os.path import exists
# from sys import exc_info
import re

# Code analysis
# import sys  # for sys.path (autocomp)
import ast
# import jedi
# from sys import getsizeof
from collections import defaultdict, namedtuple
from traceback import print_exc

from kivy.uix.textinput import FL_IS_LINEBREAK

from kivy.uix.behaviors import FocusBehavior
from kivy.uix.codeinput import CodeInput
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.actionbar import ActionItem
from kivy.uix.stencilview import StencilView
from kivy.uix.scatter import ScatterPlane
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.boxlayout import BoxLayout
from kivy.core.image import Image as CoreImage
from kivy.graphics import Color, Line, Rectangle, Ellipse, Triangle, \
        PushMatrix, PopMatrix, RoundedRectangle, RenderContext, Mesh, \
        ClearBuffers, ClearColor, Callback, BindTexture, Fbo, ClearBuffers, ClearColor
from kivy.graphics.texture import Texture
from kivy.graphics.transformation import Matrix
from kivy.graphics.context_instructions import Rotate, Translate, Scale, MatrixInstruction, Transform as KvTransform
from kivy.properties import StringProperty, NumericProperty, \
        ListProperty, ObjectProperty, BooleanProperty, \
        OptionProperty, DictProperty, AliasProperty
from kivy.clock import Clock
from kivy.cache import Cache
from kivy.utils import escape_markup
from kivy.core.text.markup import MarkupLabel
from kivy.core.window import Window # request keyboard
from kivy.graphics.opengl import glEnable, glDisable, glFinish
from kivy.resources import resource_find
# from kivy.graphics.fbo import Fbo
from kivy.animation import Animation

import pymunk

from ourturtle import Turtle
from sprite import Sprite
from codean import autocomp, CodeRunner, Break, COMMON_CODE
from sokoban.sokoban import Sokoban

from playground.color import _srgb_to_linear, _parse_srgb, _global_update_colors, Color as OurColor
from playground.shapes import Stroke, Physics, Shape, Circle, Rectangle, KeepRefs, Image as OurImage
from playground.geometry import Vector, VectorRef, Transform

try:
    import mycolors
except:
    pass

# https://github.com/kivy/kivy/wiki/Working-with-Python-threads-inside-a-Kivy-application

GL_VERTEX_PROGRAM_POINT_SIZE = 34370
GL_FRAMEBUFFER_SRGB_EXT = 36281

F_UPDATE = 'update'
F_ON_KEY_PRESS = 'on_key_press'
F_ON_KEY_RELEASE = 'on_key_release'

TRANSITION_TIME = 0.4 # * 2
TRANSITION_IN = 'in_back'
TRANSITION_OUT = 'out_back'

R_TURN = re.compile(r'^(\s*)(right|left|up|down)\(([0-9]*)\)$')

grace_hopper = Image(source='grace_hopper.jpg', mipmap=True, anim_delay=0.04166) #, keep_data=True)
# print(grace_hopper, grace_hopper.texture, grace_hopper.texture.tex_coords, grace_hopper.texture.uvpos, grace_hopper.texture.uvsize)
# grace_hopper.texture.blit_buffer(pbuffer=_srgba_bytes_to_linear(grace_hopper.texture.pixels), colorfmt='rgba')
# print(grace_hopper.texture.pixels)
# raise Exception
# texture = Texture.create_from_data(grace_hopper.data)
# print(texture, texture.)

# def texture_to_linear_srgb(texture):

#     fbo = Fbo(size=texture.size)
#     fbo.shader.fs = '''
#     $HEADER$

#     vec3 to_linear(vec3 srgb){
#         vec3 cutoff = vec3(lessThan(srgb, vec3(12.92 * 0.0031308)));
#         vec3 higher = pow((srgb + 0.055) / 1.055, vec3(2.4));
#         vec3 lower = srgb / vec3(12.92);
#         return mix(higher, lower, cutoff);
#     }

#     void main (void) {
#         vec4 srgb = texture2D(texture0, tex_coord0);
#         gl_FragColor = vec4(1., 0., 0., 1.); //to_linear(srgb.rgb), srgb.w);
#     }
#     '''
#     with fbo:
#         Color(1, 1, 1)
#         Rectangle(size=texture.size, texture=texture, tex_coords=texture.tex_coords)
#     fbo.draw()

#     return fbo.texture

# def radial_gradient(border_color=(1, 1, 0), center_color=(1, 0, 0),
#         size=(64, 64)):

#     fbo = Fbo(size=size, clear_color=(.1, 1, .2, 1))
#     fbo.shader.fs = '''
#     $HEADER$
#     uniform vec3 border_color;
#     uniform vec3 center_color;
#     void main (void) {
#         float d = clamp(distance(tex_coord0, vec2(0.5, 0.5)), 0., 1.);
#         gl_FragColor = vec4(1., 1., 0., 1.); //vec4(mix(center_color, border_color, d), 1);
#     }
#     '''

#     # use the shader on the entire surface
#     fbo['border_color'] = list(map(float, border_color))
#     fbo['center_color'] = list(map(float, center_color))
#     with fbo:
#         ClearColor(1, 1, 1, 1)
#         ClearBuffers()
#         Color(1, 1, 0)
#         Rectangle(size=size)
#     fbo.draw()

#     return fbo.texture


# def create_tex(*args):
#     center_color = 255, 255, 0
#     border_color = 100, 0, 0

#     size = (64, 64)
#     tex = Texture.create(size=size)

#     sx_2 = size[0] // 2
#     sy_2 = size[1] // 2

#     buf = bytearray()
#     for x in range(-sx_2, sx_2):
#         for y in range(-sy_2, sy_2):
#             a = x / (1.0 * sx_2)
#             b = y / (1.0 * sy_2)
#             d = (a ** 2 + b ** 2) ** .5

#             for c in (0, 1, 2):
#                 buf += bytearray((max(0,
#                                min(255,
#                                    int(center_color[c] * (1 - d)) +
#                                    int(border_color[c] * d))),))

#     tex.blit_buffer(bytes(buf), colorfmt='rgb', bufferfmt='ubyte')
#     return tex



# grace_hopper.texture = texture_to_linear_srgb(grace_hopper.texture)
# glFinish()
# grad = radial_gradient()
# glFinish()
# print('GRAD', grad, grad.tex_coords)

def debounce(wait):
    """ Decorator that will postpone a functions
        execution until after wait seconds
        have elapsed since the last time it was invoked. """

    def decorator(fn):
        def debounced(*args, **kwargs):
            def call_it(*t):
                debounced._timer = None
                debounced._last_call = time()
                return fn(*args, **kwargs)

            time_since_last_call = time() - debounced._last_call
            if time_since_last_call >= wait:
                return call_it()

            if debounced._timer is None:
                # debounced._timer = threading.Timer(wait - time_since_last_call, call_it)
                debounced._timer = Clock.schedule_once(call_it, wait - time_since_last_call)
                # debounced._timer.start()

        debounced._timer = None
        debounced._last_call = 0

        return debounced

    return decorator

def whos(vars, max_repr=40):
    w_types = (int, float, str, list, dict, tuple)
    w_types += (OurColor, Stroke, Physics, Shape, Circle, Rectangle, KeepRefs, OurImage, Vector, VectorRef, Transform)
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

from pygments import styles
from pygments.formatters import BBCodeFormatter

class CodeEditor(CodeInput, FocusBehavior):
    def __init__(self, **kwargs):
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
        self.cursor = 0, 0

    # Kivy bug workaround
    def get_cursor_from_xy(self, x, y):
        # print('get_cursor_from_xy', x, y, type(x), type(y))
        return super(CodeEditor, self).get_cursor_from_xy(int(x), y)

    def on_style(self, *args):
        self.formatter = BBCodeFormatter(style=self.style)
        bg_color, alpha = _parse_srgb(self.formatter.style.background_color)
        self.background_color = bg_color + (alpha or 1,)
        self._trigger_update_graphics()

    def on_style_name(self, *args):
        self.style = styles.get_style_by_name(self.style_name)
        bg_color, alpha = _parse_srgb(self.style.background_color)
        self.background_color = bg_color + (alpha or 1,)
        self._trigger_refresh_text()

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

    def _update_graphics(self, *largs):  # FIXME: Not needed?
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
        # print('kk', keycode, text, modifiers)
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
                            if cr == cr_from: cc_from = max(0, cc_from - remove)
                            if cr == cr_to: cc_to = max(0, cc_to - remove)
                self._selection_from = self.cursor_index((cc_from, cr_from))
                self._selection_to = self.cursor_index((cc_to, cr_to))
                self._selection_finished = True
                self._update_selection(True)
                self._update_graphics_selection()
                # TODO: Add undo/redo
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
                    # print(
                    #     "ACS: ", '\n'.join(
                    #         sorted(
                    #             [ac.full_name for ac in self.ac_completions])))
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

        # Sokoban
        # elif modifiers == ['alt'] and key_str in ['up', 'down', 'right', 'left']:
        #     print(self._lines)
        #     cc, cr = self.cursor
        #     empty_line = self._lines[cr].strip() == ''
        #     if empty_line:
        #         cr -= 1
        #     space = ''
        #     if cr >= 0:
        #         prev_line = self._lines[cr]
        #         m = R_TURN.match(prev_line)
        #         if m:
        #             space, cmd, step = m.groups()
        #             if cmd == key_str:
        #                 if step:
        #                     step = str(int(step)+1)
        #                 else:
        #                     step = '2'
        #                 self._set_line_text(cr, space + cmd + '(' + step + ')')
        #                 # self._lines[cr-1] = space + cmd + '(' + step + ')'
        #                 return True
        #     key_str = space + key_str
        #     if not empty_line:
        #         self.do_cursor_movement('cursor_end')
        #         key_str = '\n' + key_str
        #     self.insert_text(f'{key_str}()')
        #     return True
        # Comment line or selection by Ctrl-/
        elif key == 47 and modifiers == ['ctrl']:
            cc, cr = self.cursor
            _lines = self._lines

            if self._selection:
                a, b = self._selection_from, self._selection_to
                cc_from, cr_from = self.get_cursor_from_index(a)
                cc_to, cr_to = self.get_cursor_from_index(b)
            else:
                cc_from, cr_from = self.cursor
                cc_to, cr_to = cc_from, cr_from
            uncomment = True #_lines[min(cr_from, cr_to)].lstrip().startswith('#')
            indent = 1000
            for cr in range(min(cr_from, cr_to), max(cr_from, cr_to) + 1):
                line = _lines[cr]
                if not line.lstrip().startswith('#'):
                    uncomment = False
                indent = min(indent, len(line) - len(line.lstrip()))

            for cr in range(min(cr_from, cr_to), max(cr_from, cr_to) + 1):
                line = _lines[cr]
                if not uncomment:
                    new_text = f'{" " * indent}# {line[indent:]}'
                    self._set_line_text(cr, new_text)
                    if cr == cr_from: cc_from += 2
                    if cr == cr_to: cc_to += 2
                else:
                    new_text = re.sub(r'^(\s*)# ?', r'\1', line)
                    removed = len(line) - len(new_text)
                    self._set_line_text(cr, new_text)
                    if cr == cr_from: cc_from = max(0, cc_from - removed)
                    if cr == cr_to: cc_to = max(0, cc_to - removed)
            self._selection_from = self.cursor_index((cc_from, cr_from))
            self._selection_to = self.cursor_index((cc_to, cr_to))
            self._selection_finished = True
            self._update_selection(True)
            self._update_graphics_selection()
            # TODO: Add undo/redo
            return True

        if self.dispatch('on_key_down', window, keycode, text, modifiers):
            return True

        self.ac_begin = False
        return super(CodeInput, self).keyboard_on_key_down(
            window, keycode, text, modifiers)

    def do_cursor_movement(self, action, control=False, alt=False):
        if action == 'cursor_home' and not control:
            cc, cr = self.cursor
            _lines = self._lines
            text = _lines[cr]
            indent = len(text) - len(text.lstrip())
            if cc != indent:
                self.cursor = (indent, cr)
                return
        super(CodeInput, self).do_cursor_movement(action, control, alt)

    def _get_cursor(self):
        return self._cursor

    def _set_cursor(self, pos):
        ret = super(CodeInput, self)._set_cursor(pos)
        cursor = self.cursor
        padding_left = self.padding[0]
        padding_right = self.padding[2]
        viewport_width = self.width - padding_left - padding_right
        sx = self.scroll_x
        offset = self.cursor_offset()
        if offset > viewport_width + sx - 25:
            self.scroll_x = offset - viewport_width + 25
        if offset < min(sx + 25, viewport_width):
            self.scroll_x = offset + 25
        return ret

    cursor = AliasProperty(_get_cursor, _set_cursor)

    def on_size(self, instance, value):
        self._trigger_refresh_text()
        self._refresh_hint_text()
        cursor = self.cursor
        padding_left = self.padding[0]
        padding_right = self.padding[2]
        viewport_width = self.width - padding_left - padding_right
        sx = self.scroll_x
        offset = self.cursor_offset()
        if offset < viewport_width + sx - 25:
            self.scroll_x = max(0, offset - viewport_width + 25)

    def on_double_tap(self):
        # last_identifier = re.compile(r'(\w+)(?!.*\w.*)')
        ci = self.cursor_index()
        cc = self.cursor_col
        line = self._lines[self.cursor_row]
        len_line = len(line)
        # start = max(0, len(line[:cc]) - line[:cc].rfind(u' ') - 1)
        # end = line[cc:].find(u' ')
        # end = end if end > - 1 else (len_line - cc)
        start = ci - cc
        end = ci - cc + len(line)
        words = [m.span() for m in re.finditer(r'\w+', line)]
        if words:
            s1, s2 = zip(*words)
            nonwords = zip(s2, s1[1:])
            for span in (*words, *nonwords): #words + list(nonwords):
                if span[0] <= cc < span[1]:
                    end = start + span[1]
                    start += span[0]
        Clock.schedule_once(lambda dt: self.select_text(start, end))

    def on_key_down(self, window, keycode, text, modifiers):
        pass

    def on_cursor_row(self, *largs):
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

    texture = ObjectProperty(None)

    def __init__(self, **kwargs):
        # self.canvas = RenderContext()  # ?
        self.canvas = RenderContext(use_parent_projection=True,
                                    use_parent_modelview=True,
                                    use_parent_frag_modelview=True)

        with self.canvas.before:
            self.cb = Callback(self.setup_gl_context)
        # with self.canvas:
        #     self.fbo = Fbo(size=self.size)
        #     self.fbo_color = Color(1, 1, 1, 1)
        #     self.fbo_rect = Rectangle()
        with self.canvas.after:
            self.cb = Callback(self.reset_gl_context)
        # with self.fbo:
        #     ClearColor(0, 0, 0, 0)
        #     ClearBuffers()
            # Color(.7, 0, .7, 1)
            # Rectangle(pos=(0,0),size=(300,500))
        # self.texture = self.fbo.texture
        self.canvas.shader.fs = open(resource_find('srgb_to_linear.glsl')).read()
        super(OurSandbox, self).__init__(**kwargs)
        # self.canvas.shader.source = resource_find('shader2.glsl')
        self.rc1 = RenderContext(use_parent_modelview=True, use_parent_projection=True)
        self.rc2 = RenderContext(use_parent_modelview=True, use_parent_projection=True)
        self.rc1.shader.source = resource_find('shader1.glsl')
        self.rc2.shader.source = resource_find('shader2.glsl')
        self.rc1['texture1'] = 1
        self.rc2['texture1'] = 1
        # x, y, w, *stroke, *fill, a1, a2, *tr
        self.rc1_vfmt = (
            (b'size',   2, 'float'),
            (b'center', 2, 'float'),
            (b'width',  1, 'float'),
            (b'stroke', 4, 'float'),
            (b'fill',   4, 'float'),
            (b'angle_start', 1, 'float'),
            (b'angle_end',   1, 'float'),
            (b'transform',   4, 'float'),
            (b'tex_coords0', 2, 'float'),
            (b'tex_coords1', 2, 'float'),
        )
        self.rc2_vfmt = (
            (b'size',   2, 'float'),
            (b'center', 2, 'float'),
            (b'radius', 1, 'float'),
            (b'width',  1, 'float'),
            (b'stroke', 4, 'float'),
            (b'fill',   4, 'float'),
            (b'transform', 4, 'float'),
            (b'tex_coords0', 2, 'float'),
            (b'tex_coords1', 2, 'float'),
        )
        # with self.rc1:
        #     self.mesh1 = self.make_ellipses()
        # with self.rc2:
        #     self.mesh2 = self.make_rects()

        # self.add_widget(Button(text='Hello'))
        # self.add_widget(Image(source='/home/user/pics/birthday-72.jpg'))
        # self.add_widget(Image(source='grad1.png'))

            # glDisable(GL_FRAMEBUFFER_SRGB_EXT)
        # self.canvas.add(self.rc1)
        # self.canvas.add(self.rc2)
        self.shapes_by_trace = dict()
        self.shapes_by_id = dict()
        self.images = defaultdict(list)
        self.image_meshes = defaultdict(list)
        self.code = []  # Store code for diffing scene

        self.transition_time = TRANSITION_TIME

        # Clock.schedule_interval(self.update_shader, 1 / 60.)
        glEnable(GL_VERTEX_PROGRAM_POINT_SIZE)

        self.register_event_type('on_key_down')
        self.register_event_type('on_key_up')
        self.space = None

    def setup_gl_context(self, *args):
        glEnable(GL_FRAMEBUFFER_SRGB_EXT)

    def reset_gl_context(self, *args):
        glDisable(GL_FRAMEBUFFER_SRGB_EXT)

    def update_shader(self, *largs):
        # self.canvas['projection_mat'] = Window.render_context['projection_mat']
        for rc in [self.rc1, self.rc2, self.canvas]: #, self.fbo]: #
            rc['modelview_mat'] = self.transform #Window.render_context['modelview_mat']
            rc['resolution'] = list(map(float, self.size))
            rc['time'] = Clock.get_boottime()
            rc['scale'] = self.transform[0]
            rc['texture1'] = 1
        self.canvas.ask_update()

    # def add_widget(self, *largs):
    #     # trick to attach graphics instruction to fbo instead of canvas
    #     canvas = self.canvas
    #     self.canvas = self.fbo
    #     ret = super(OurSandbox, self).add_widget(*largs)
    #     self.canvas = canvas
    #     return ret

    # def remove_widget(self, *largs):
    #     canvas = self.canvas
    #     self.canvas = self.fbo
    #     super(OurSandbox, self).remove_widget(*largs)
    #     self.canvas = canvas

    # def on_size(self, instance, value):
    #     self.fbo.size = value
    #     self.texture = self.fbo.texture
    #     self.fbo_rect.size = value

    # def on_pos(self, instance, value):
    #     self.fbo_rect.pos = value

    # def on_texture(self, instance, value):
    #     self.fbo_rect.texture = value

    # def on_alpha(self, instance, value):
    #     self.fbo_color.rgba = (1, 1, 1, value)

    def make_ellipses(self):
        step = 10
        istep = (pi * 2) / float(step)
        meshes = []
        # indices = [0, 1, 3, 2]
        indices = [0, 3, 1, 2]
        tr = 1, 0, 0, 1
        tex_coords = grace_hopper.texture.tex_coords
        for i in range(600):
            # x = 100 + cos(istep * i) * 100
            # y = 100 + sin(istep * i) * 100
            x = randint(0, 2000)
            y = randint(0, 2000)
            a = randint(10, 100)
            b = randint(10, 100)
            w = randint(0, min(a, b) // 2)
            stroke = *OurColor(random()*30 + 40, 50, random()*360, mode='Jsh').linear_srgb, random() # random(), random(), random(), random()
            fill = *OurColor(random()*30 + 40, 50, random()*360, mode='Jsh').linear_srgb, random() # # random(), random(), random(), random()
            a1 = 2 * pi * random()
            a2 = 2 * pi * random()
            v0 = x, y, +a, -b, w, *stroke, *fill, a1, a2, *tr, *tex_coords[0:2]
            v1 = x, y, -a, -b, w, *stroke, *fill, a1, a2, *tr, *tex_coords[2:4]
            v2 = x, y, -a, +b, w, *stroke, *fill, a1, a2, *tr, *tex_coords[4:6]
            v3 = x, y, +a, +b, w, *stroke, *fill, a1, a2, *tr, *tex_coords[6:8]
            vertices = v0 + v1 + v2 + v3
            meshes.append(Mesh(fmt=self.rc1_vfmt, mode='triangle_strip', vertices=vertices, indices=indices, texture=grace_hopper.texture))
        return meshes

    def make_rects(self):
        step = 10
        istep = (pi * 2) / float(step)
        meshes = []
        indices = [0, 1, 3, 2]
        tr = 1, 0, 0, 1
        for i in range(900):
            # x = 100 + cos(istep * i) * 100
            # y = 100 + sin(istep * i) * 100
            x = randint(0, 2000)
            y = randint(0, 2000)
            a = randint(10, 100)
            b = randint(10, 100)
            w = randint(0, min(a, b) // 2)
            r = randint(0, min(a, b) // 2)
            stroke = *OurColor(random()*30 + 40, 50, random()*360, mode='Jsh').linear_srgb, random() # random(), random(), random(), random()
            fill = *OurColor(random()*30 + 40, 50, random()*360, mode='Jsh').linear_srgb, random() # # random(), random(), random(), random()
            v0 = x, y, -a, -b, r, w, *stroke, *fill, *tr
            v1 = x, y, -a, +b, r, w, *stroke, *fill, *tr
            v2 = x, y, +a, +b, r, w, *stroke, *fill, *tr
            v3 = x, y, +a, -b, r, w, *stroke, *fill, *tr
            vertices = v0 + v1 + v2 + v3
            meshes.append(Mesh(fmt=self.rc2_vfmt, mode='triangle_strip', vertices=vertices, indices=indices))
        return meshes

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        '''We call super before doing anything else to enable tab cycling
        by FocusBehavior. If we wanted to use tab for ourselves, we could just
        not call it, or call it if we didn't need tab.
        '''
        print('DOWN', keycode, text, modifiers, '\n', self.transform)

        if keycode[1] == 'f2':
            self.rc1.shader.source = resource_find('shader1.glsl')
            self.rc2.shader.source = resource_find('shader2.glsl')
            self.canvas.shader.fs = open(resource_find('srgb_to_linear.glsl')).read()
            print('projection_mat\n', self.canvas['projection_mat'])
            print('modelview_mat\n', self.canvas['modelview_mat'])
            print('frag_modelview_mat\n', self.canvas['frag_modelview_mat'])
        elif self.dispatch('on_key_down', window, keycode, text, modifiers):
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
    code_editor_scrlv = ObjectProperty(None)
    rpanel = ObjectProperty(None)
    textout = ObjectProperty(None)
    run_to_cursor = BooleanProperty(False)

    # ball = ObjectProperty(None)

    def __init__(self, **kwargs):
        # self.canvas = RenderContext(use_parent_projection=True,
        #                             use_parent_modelview=True,
        #                             use_parent_frag_modelview=True)
        # self.canvas.shader.fs = open(resource_find('srgb_to_linear.glsl')).read()
        super(Playground, self).__init__(**kwargs)
        # with self.canvas.before:
        #     glEnable(GL_FRAMEBUFFER_SRGB_EXT)
        # glEnable(GL_FRAMEBUFFER_SRGB_EXT)

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
            # self.sandbox.add_widget(Button(text='Abc'))
            return sp
        globs['add_sprite'] = _add_sprite

        def _add_line(*largs, **kvargs):
            # self.sandbox.add_widget(line)
            with self.sandbox.canvas:
                line = Line(*largs, **kvargs)
            return line
        globs['Line'] = _add_line

        # self.sokoban = Sokoban()
        # def sokoban_go(dx, dy):
        #     def go(step=1):
        #         self.sokoban.move_player(dx, dy, step)
        #     return go
        # globs['right'] = sokoban_go(1, 0)
        # globs['left'] = sokoban_go(-1, 0)
        # globs['up'] = sokoban_go(0, 1)
        # globs['down'] = sokoban_go(0, -1)

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

        self._show_clipped = True
        def _show_clipped_colors(show=True):
            self._show_clipped = show
        globs['show_clipped_colors'] = _show_clipped_colors

        self.trigger_exec_update = Clock.create_trigger(self.execute_update, -1)
        self.update_schedule = None

        self.runner = CodeRunner(globals=globs, special_funcs=[F_UPDATE, F_ON_KEY_PRESS, F_ON_KEY_RELEASE])

        self.code_editor.namespace = self.runner.globals  # FIXME?

        # FIXME
        vs1 = VarSlider(var_name='a', min=0, max=360, type='float')
        # vs2 = VarSlider(var_name='b', type='float')
        # vs3 = VarSlider(var_name='c', type='float')
        # vs4 = VarSlider(var_name='l', min=0, max=50)
        # vs5 = VarSlider(var_name='m', min=0, max=100)
        # vs6 = VarSlider(var_name='n', min=0, max=150)
        self.rpanel.add_widget(vs1, 1)
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

        #@debounce(0.2)
        def _set_var(wid, value):
            self.sandbox.transition_time = TRANSITION_TIME / 2 #0.2
            self.sandbox.transition_in = 'in_cubic'
            self.sandbox.transition_out = 'out_cubic'
            self.vars[wid.var_name] = value
            if wid.var_name in self.runner.common_vars:
                self.trigger_exec()

        if exists('source.py'):
            with open('source.py') as f:
                self.code = f.read()
            def _reset_cursor(*t):
                self.code_editor.cursor = 0, 0
            Clock.schedule_once(_reset_cursor, 0)
                # self.code_editor.scroll_x = 0
                # self.code_editor.scroll_y = 0
                # self.code_editor.cursor = 0, 0

        # FIXME
        vs1.bind(value=_set_var)
        # vs2.bind(value=_set_var)
        # vs3.bind(value=_set_var)
        # vs4.bind(value=_set_var)
        # vs5.bind(value=_set_var)
        # vs6.bind(value=_set_var)
        vs1.value = 36.
        # vs2.value = 3.4
        # vs3.value = 4.2
        # vs4.value = 15
        # vs5.value = 50
        # vs6.value = 75

        self.compile_code()

        self.graphics_instructions = []
        # self.bind(run_code=self.compile_run)
        # self.compile_run()

    def code_editor_change_scroll_y(self):
        ti = self.code_editor
        scrlv = self.code_editor_scrlv
        y_cursor = ti.cursor_pos[1]
        y_bar = scrlv.scroll_y * (ti.height-scrlv.height)
        if ti.height > scrlv.height:
            if y_cursor >= y_bar + scrlv.height:
                dy = y_cursor - (y_bar + scrlv.height)
                scrlv.scroll_y = scrlv.scroll_y + scrlv.convert_distance_to_scroll(0, dy)[1]
            if y_cursor - ti.line_height <= y_bar:
                dy = (y_cursor - ti.line_height) - y_bar
                scrlv.scroll_y = scrlv.scroll_y + scrlv.convert_distance_to_scroll(0, dy)[1]

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
        pass
        # if self.sokoban:
        #     code_lines = self.sokoban.replay(self.replay_step)
        #     self.update_sandbox()
        #     self.code_editor.highlight_line(code_lines, 'run')

    def on_code_editor_cursor_row(self, *largs):
        # self.code_editor_change_scroll_y()
        if self.run_to_cursor:
            self.compile_code()

    def on_status(self, *largs):
        status = self.status[0]
        if status == 'ERROR':
            exc = self.status[1]
            exc_name = exc.__class__.__name__ if exc else "Unknown Error"
            self.status_text = f'[b][color=f92672]{exc_name}[/color]: [/b]'
            if isinstance(exc, SyntaxError):
                code = (exc.text or self.code.splitlines()[exc.lineno - 1]).replace('\n', '⏎')  #.replace('\t', ' ' * 4).replace(' ', '˽')
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
        from difflib import SequenceMatcher
        # redraw = True
        matcher_opcodes = None
        indices = [0, 3, 1, 2] #[0,3,1,2] #[0, 1, 3, 2]
        # [u, v, u + w, v, u + w, v + h, u, v + h]
        tex_coords_fill = tex_coords_stroke = 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0
        # print('self.sandbox.shapes_by_trace', self.sandbox.shapes_by_trace)
        def update_texture(image, texture):
            # print("Hi!", self.sandbox.images, image.source)
            if image.source in self.sandbox.image_meshes:
                for mesh in self.sandbox.image_meshes[image.source]:
                    mesh.texture = texture
            else:
                print('DEL', image, image.source)
                del image
        def match_lines(lines):
            match = []
            for lineno in lines:
                lineno -= 1
                for tag, i1, i2, j1, j2 in matcher_opcodes:
                    if i1 <= lineno < i2:
                        if tag in ('equal', 'replace'):
                            match.append(1 + lineno + j1 - i1)
                        else:
                            match.append(-1)
                        break
            return tuple(match)

        try:
            t1 = process_time()
            _global_update_colors()
            t2 = process_time()
            new_shapes = []
            # new_render_shapes = dict
            # shapes = []
            old_shapes = set([shape for shape in self.sandbox.shapes_by_id.values()])
            # old_shapes = {id(s[0][1][0]): (s[0][0], s[0][1][0]) for s in self.sandbox.shapes_by_id.values()}
            # old_shapes = set([(s[0][0], id(s[0][1][0])) for s in self.sandbox.shapes_by_id.values()])
            shape_ids = []
            shape_traces = []
            # old_shapes = set(self.sandbox.shapes_by_id.values())
            if redraw:
                Shape._trace_counters.clear()
                self.sandbox.shapes_by_id = dict()
                self.sandbox.images = defaultdict(list)
                self.sandbox.image_meshes = defaultdict(list)

                old_code = self.sandbox.code
                new_code = self.code.splitlines()
                matcher = SequenceMatcher(lambda x: x in ' \t', new_code, old_code)
                matcher_opcodes = matcher.get_opcodes()
                self.sandbox.code = new_code
            # else:
            #     self.sandbox.shapes_by_trace = dict()
            # if redraw:
            #     self.sandbox.shapes_by_id = dict()
            #     self.sandbox.images = defaultdict(list)
            #     self.sandbox.image_meshes = defaultdict(list)
            # with:
                # if redraw:
                #     self.ellipses = self.sandbox.make_ellipses()
            for shape in Shape.get_instances(True):
                # print(shape, shape._trace, shape._trace_iter)
                # textured = shape.fill is not None and isinstance(shape.fill, OurImage)
                # if textured and not redraw and id(shape) in self.sandbox.shapes and \
                #         shape.fill.source in self.sandbox.images:
                #     image = self.sandbox.images[shape.fill.source]
                #     if image.anim_available:
                #         print('Update texture!')
                #         self.sandbox.shapes[id(shape)][0].texture = image.texture
                shape_ids.append(id(shape))
                # self.sandbox.rendered_shapes.append(shape)
                render_shape = None #if redraw else self.sandbox.shapes_by_id.get(id(shape))
                shape_trace = (shape._trace, shape._trace_iter) if shape._trace else None
                if shape_trace is not None:
                    shape_traces.append(shape_trace)

                if redraw:
                    if shape_trace is not None:
                        print('shape_trace', shape_trace)
                        print('shape_trace matched:', match_lines(shape_trace[0]))
                        mathed_trace = match_lines(shape_trace[0]), shape_trace[1]
                        print('self.sandbox.shapes_by_trace.keys()', self.sandbox.shapes_by_trace.keys())
                        render_shape = self.sandbox.shapes_by_trace.get(mathed_trace)
                        self.sandbox.shapes_by_id[id(shape)] = render_shape
                else:
                    render_shape = self.sandbox.shapes_by_id.get(id(shape))
                # if not redraw and render_shape is not None and not shape._is_modified:
                #     continue

                w = 0
                stroke = 0, 0, 0, 0
                fill = 0, 0, 0, 0
                image_fill = None
                image_stroke = None
                texture_fill = None
                texture_stroke = None
                vertices = []
                render_context = None
                vfmt = None
                mesh = None
                if shape.stroke is not None and shape.stroke.fill is not None:
                    if isinstance(shape.stroke.fill, OurImage):
                        stroke = 1, 1, 1, 1
                        source = shape.stroke.fill.source
                        image_stroke = self.sandbox.images.get(source)
                        if not image_stroke:
                            image_stroke = Image(
                                source=source, mipmap=True,
                                anim_delay=shape.stroke.fill.anim_delay)
                            self.sandbox.images[source] = image_stroke
                            # self.sandbox.images[shape.fill.source] = image
                            image.bind(texture=update_texture) # TODO: Animation
                            # TODO: If not exists (image_stroke.texture is None)...
                        texture_stroke = image_stroke.texture
                        tex_coords_stroke = texture_stroke.tex_coords
                    else:
                        stroke = shape.stroke.fill.linear_srgba
                    # if not self._show_clipped and shape.stroke.fill.is_clipped:
                    #     continue
                    w = shape.stroke.width
                if shape.fill is not None:
                    if isinstance(shape.fill, OurImage):
                        fill = 1, 1, 1, 1
                        source = shape.fill.source
                        image_fill = self.sandbox.images.get(source)
                        if not image_fill: #shape.fill.source not in self.sandbox.images:
                            image_fill = Image(
                                source=source, mipmap=True,
                                anim_delay=shape.fill.anim_delay)
                            self.sandbox.images[source] = image_fill
                            # self.sandbox.images[shape.fill.source] = image
                            # image_fill.bind(texture=update_texture)
                            # TODO: If not exists (image_stroke.texture is None)...
                        texture_fill = image_fill.texture
                        if texture_fill is not None:
                            tex_coords_fill = texture_fill.tex_coords
                            # TODO: Mark figure.
                    else:
                        # if not self._show_clipped and shape.fill.is_clipped:
                        #     continue
                        fill = shape.fill.linear_srgba
                if shape.opacity != 100.0:
                    stroke = *stroke[:3], stroke[3] * shape.opacity / 100
                    fill = *fill[:3], fill[3] * shape.opacity / 100
                # if shape.fill and shape.fill.is_clipped:
                #     print('Clipped!')
                tr = 1, 0, 0, 1
                if shape.transform:
                    tr = tuple(shape.transform)
                    x += tr[4]
                    y += tr[5]
                    tr = tr[:4]

                if isinstance(shape, Circle):
                    render_context = self.sandbox.rc1
                    vfmt = self.sandbox.rc1_vfmt

                    x, y = shape.center
                    a = b = shape.radius * 2
                    a1 = radians(shape.angle_start)
                    a2 = radians(shape.angle_end)
                    v_attrs = x, y, w, *stroke, *fill, a1, a2, *tr
                    v0 = +a, -b, *v_attrs, *tex_coords_fill[0:2], *tex_coords_stroke[0:2]
                    v1 = -a, -b, *v_attrs, *tex_coords_fill[2:4], *tex_coords_stroke[2:4]
                    v2 = -a, +b, *v_attrs, *tex_coords_fill[4:6], *tex_coords_stroke[4:6]
                    v3 = +a, +b, *v_attrs, *tex_coords_fill[6:8], *tex_coords_stroke[6:8]
                    vertices = v0 + v1 + v2 + v3

                elif isinstance(shape, Rectangle):
                    render_context = self.sandbox.rc2
                    vfmt = self.sandbox.rc2_vfmt

                    x, y = shape.corner
                    a, b = shape.size
                    r = shape.radius
                    v_attrs = x, y, r, w, *stroke, *fill, *tr
                    v0 = +a, -b, *v_attrs, *tex_coords_fill[0:2], *tex_coords_stroke[0:2]
                    v1 = -a, -b, *v_attrs, *tex_coords_fill[2:4], *tex_coords_stroke[2:4]
                    v2 = -a, +b, *v_attrs, *tex_coords_fill[4:6], *tex_coords_stroke[4:6]
                    v3 = +a, +b, *v_attrs, *tex_coords_fill[6:8], *tex_coords_stroke[6:8]
                    vertices = v0 + v1 + v2 + v3

                else:
                    raise NotImplementedError

                new_shapes.append((
                    id(shape), render_context, vfmt, texture_stroke, texture_fill, vertices,
                    indices, 'triangle_strip', (image_fill,)
                ))

                with render_context:
                    BindTexture(texture=texture_stroke, index=1)
                if render_shape is None:  # id(shape) not in self.sandbox.shapes:
                    with render_context:
                        # BindTexture(texture=texture_stroke, index=1)
                        if redraw and self.sandbox.transition_time > 0:
                            v_len = len(vertices) // 4
                            # Set initial size to 0, 0
                            initial_vs = tuple(0 if i % v_len in (0, 1) else v for i, v in enumerate(vertices))
                        else:
                            initial_vs = vertices
                        mesh = Mesh(
                            fmt=vfmt, mode='triangle_strip', vertices=initial_vs,
                            indices=indices, texture=texture_fill)
                    self.sandbox.shapes_by_id[id(shape)] = ((render_context, (mesh, )), )
                    if image_fill is not None:
                        self.sandbox.image_meshes[image_fill.source].append(mesh)
                else:
                    mesh = render_shape[0][1][0]
                    mesh.texture = texture_fill

                if shape_trace:
                    self.sandbox.shapes_by_trace[shape_trace] = ((render_context, (mesh, )), )

                # else:
                    # self.sandbox.shapes[id(shape)][0].vertices = vertices

                    # SO [1]: If an animation is repeated for many interactions (like a contextual
                    # menu), a slower, more-perceivable animation (600ms) is going to feel quite
                    # tedious to most of your users. Micro-animations (like a nav bar or a context
                    # menu) of ~250ms will be noticeable by most people, but just noticeable enough
                    # that they won't feel like they're waiting for it.
                    #
                    # SO [2]: The sweet spot that shows up time after time in game and UI design is
                    # 250-300ms. For transitions that bounce or are elastic, 400-500ms lets the
                    # motion read better.
                # mesh.vertices = vertices

                if redraw and self.sandbox.transition_time > 0:
                    def _on_complete(*lt):
                        if mesh is not None:
                            mesh.vertices = vertices
                    Animation.stop_all(mesh)
                    anim = Animation(vertices=vertices, t=self.sandbox.transition_out,
                        duration=self.sandbox.transition_time)
                    anim.bind(on_complete=_on_complete)
                    anim.start(mesh)
                else:
                    mesh.vertices = vertices

            def remove_garbage(shapes):
                for shape in shapes: #chain(shapes.get(oid, ()) for oid in oids):
                    for context, instructions in shape:
                        for inst in instructions:
                            context.remove(inst)
                # for oid in oids:
                #     self.sandbox.shapes_by_id.pop(oid, None)
            # to_remove = set(old_render_shapes) - set(shape_ids)
            # old_shapes = {id(s[0][1][0]): (s[0][0], s[0][1][0]) for shape in self.sandbox.shapes_by_id.values()}
            # print('old_shapes', len(old_shapes))
            # print('new shapes', len([id(s[0][1][0]) for s in self.sandbox.shapes_by_id.values()]))
            # to_remove = set(old_shapes) - set([id(s[0][1][0]) for s in self.sandbox.shapes_by_id.values()])
            to_remove = old_shapes - set([shape for shape in self.sandbox.shapes_by_id.values()])


            for shape_trace in set(self.sandbox.shapes_by_trace) - set(shape_traces):
                self.sandbox.shapes_by_trace.pop(shape_trace)

            # for sid in to_remove:
            #     context, mesh = old_shapes[sid]
            if self.sandbox.transition_time > 0:
                for shape in to_remove:
                    for context, instructions in shape:
                        for mesh in instructions:
                            Animation.stop_all(mesh)
                            vertices = mesh.vertices
                            v_len = len(vertices) // 4
                            # Set initial size to 0, 0
                            initial_vs = tuple(0 if i % v_len in (0, 1) else v for i, v in enumerate(vertices))
                            Animation(vertices=initial_vs, t=self.sandbox.transition_in,
                                duration=self.sandbox.transition_time).start(mesh)
                Clock.schedule_once(lambda _:  remove_garbage(to_remove), self.sandbox.transition_time + 1 / 60.)
            else:
                remove_garbage(to_remove)

            # print('old_render_shapes', len(list(old_shapes)))
            # print('self.sandbox.shapes_by_id.values()', len(self.sandbox.shapes_by_id.values()))
            # print('to_remove', len(to_remove))
            # print('self.sandbox.shapes_by_trace AFTER', len(self.sandbox.shapes_by_trace))

            # for oid in to_remove:
                # self.sandbox.shapes_by_id.pop(oid, None)
            # for oid, shape in self.sandbox.shapes_by_id.items():
            #     if shape in to_remove:
            #         # print('Removing', oid, shape)
            #         self.sandbox.shapes_by_id.pop(oid)

            # remove_garbage(to_remove)

            t3 = process_time()

            # print('T1-2', t2-t1)
            # print('T2-3', t3-t2)
            # print('T', t3-t1)
        except Exception as e:
            print('E at update_sandbox:')
            print_exc()

        # try:
        #     if redraw:
        #         self.graphics_instructions = []
        #         # self.sokoban.draw_level(self.sandbox)  # FIXME
        #         with self.sandbox.canvas:
        #             pass
        #             # Color(0.3, 0, 0)
        #             # Ellipse(pos=(100, 100), size=(200, 200))
        #             # Color(0.0732, 0, 0)
        #             # Ellipse(pos=(300, 100), size=(200, 200))

        #             # self.mesh = self.sandbox.build_mesh()


        #             for shape in Circle.get_instances():
        #                 stroke = shape.stroke
        #                 fill = shape.fill
        #                 trans = shape.transform
        #                 if trans:
        #                     PushMatrix()
        #                     # print('trans', fill, stroke, type(trans), trans, 111, shape.transform)
        #                     a, b, c, d, tx, ty = trans
        #                     mat = Matrix()  # .translate(500, 200, 0)
        #                     mat.set(array=[
        #                         [a, b, 0, 0],
        #                         [c, d, 0, 0],
        #                         [0, 0, 1, 0],
        #                         [tx, ty, 0, 1]])
        #                     # print('MATRIX', mat) # Matrix().rotate(radians(30),0,0,1))
        #                     KvTransform().transform(mat)
        #                     # Translate(0, 250, 0)
        #                 cx, cy = shape.center
        #                 r = shape.radius
        #                 pos = cx - r, cy - r
        #                 size = 2*r, 2*r
        #                 if fill:
        #                     Color(*fill.srgb)
        #                     Ellipse(pos=Vector(pos) - (0,0), size=size)
        #                 if stroke and stroke.fill:
        #                     Color(*stroke.fill.srgb)
        #                     dashes = dict(
        #                         dash_length=stroke.dashes[0],
        #                         dash_offset=stroke.dashes[1]
        #                     ) if stroke.dashes else {}
        #                     print('dashes', dashes)
        #                     line = Line(
        #                         ellipse=(*pos, *size),
        #                         width=stroke.width,
        #                         cap=stroke.cap,
        #                         joint=stroke.joint, **dashes)
        #                 if trans:
        #                     PopMatrix()

        #             for t in Turtle.turtles():
        #                 for color, width, points in t._lines:
        #                     # if self.run_to_cursor:
        #                     # color = *color[:3], 0.5
        #                     # igroup = InstructionGroup()
        #                     ci = Color(*color)
        #                     li = Line(
        #                         points=points,
        #                         width=width,
        #                         joint='round',
        #                         cap='round')
        #                     self.graphics_instructions.append((deepcopy(
        #                         (color, width, points)), ci, li))
        #                 for shape, pos, size, color in t._stamps:
        #                     # if self.run_to_cursor:
        #                     # color = *color[:3], 0.5
        #                     with self.sandbox.canvas:
        #                         # Color(0.3, 0, 0)
        #                         # Ellipse(pos=(100, 100), size=(200, 200))
        #                         # Color(0.0732, 0, 0)
        #                         # Ellipse(pos=(300, 100), size=(200, 200))

        #                         Color(*color)
        #                         if shape == 'Ellipse':
        #                             Ellipse(
        #                                 pos=pos - 0.5 * Vector(size[0], size[1]),
        #                                 size=(size[0], size[1]))
        #                         elif shape == 'Rectangle':
        #                             Rectangle(
        #                                 pos=pos - 0.5 * Vector(size[0], size[1]),
        #                                 size=(size[0], size[1]))

        #                 # PushMatrix()
        #                 # size = t._shapesize
        #                 # Translate(*t._position)
        #                 # Scale(size)
        #                 # Rotate(angle=t.heading(), axis=(0, 0, 1), origin=(0, 0))
        #                 # Color(*t._pencolor)
        #                 # Triangle(points=[0, -10, 30, 0, 0, 10])  # FIXME
        #                 # PopMatrix()
        #             # for s in Sprite._instances:
        #             #     s.draw()
        #                 # self.sandbox.add_widget(s)
        #     else:
        #         i = 0
        #         instrs = len(self.graphics_instructions)
        #         # st_ch = 0
        #         # st_n = 0
        #         with self.sandbox.canvas:
        #             for t in Turtle.turtles():
        #                 for color, width, points in t._lines:
        #                     if i < instrs:
        #                         line, ci, li = self.graphics_instructions[i]
        #                         # print(line)
        #                         if line != (color, width, points):
        #                             # print("CHANGE", points, li.points)
        #                             li.points = points
        #                             self.graphics_instructions[i] = (deepcopy(
        #                                 (color, width, points)), ci, li)
        #                             # st_ch += 1
        #                     else:
        #                         # st_n += 1
        #                         ci = Color(*color)
        #                         li = Line(
        #                             points=points,
        #                             width=width,
        #                             joint='round',
        #                             cap='round')
        #                         self.graphics_instructions.append((deepcopy(
        #                             (color, width, points)), ci, li))
        #                     i += 1
        #         # print("STATS:", instrs, st_ch, st_n)
        # except KeyError as e:
        #     print('update_sandbox:', e)
        # Reset transition time
        self.sandbox.update_shader()
        self.sandbox.transition_time = TRANSITION_TIME
        self.sandbox.transition_in = TRANSITION_IN
        self.sandbox.transition_out = TRANSITION_OUT

    def compile_code(self, *largs):
        # if self.update_schedule is not None:
        #     self.update_schedule.cancel()
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

            # self.runner.compile() # changed) FIXME
        except Exception as e:
            print('E:', e)
            print_exc()
            print('* ' * 40)
            line_num = self.runner.exception_lineno(e)
            self.code_editor.highlight_line(None, 'run')
            self.code_editor.highlight_line(line_num)
            self.status = ('ERROR', e)
        else:
            self.code_editor.highlight_line(None)
            if COMMON_CODE in changed:
                # print('-='*30)
                # print(self.code)
                # print('-='*30)
                self.trigger_exec()
                changed.remove(COMMON_CODE)
            # print('EXEC Changed:', changed)
            try:
                if self.runner.execute(changed) and F_UPDATE in self.runner.globals \
                        and not (self.update_schedule and self.update_schedule.is_triggered):
                    if self.trigger_exec_update:
                        self.trigger_exec_update.cancel()
                    if self.update_schedule is not None:
                        self.update_schedule.cancel()
                        self.update_schedule()
            except Exception as e:
                print('E3:', e)
                print_exc()
                self.status = ('ERROR', None)

    def execute_code(self, *largs):
        print('execute_code')
        self.status = ('EXEC',)
        if self.update_schedule:
            self.update_schedule.cancel()
        if self.trigger_exec_update:
            self.trigger_exec_update.cancel()
        self._gravity = Vector(0, 0)
        self._show_clipped = True
        self.runner.reset(globals=self.vars)
        self.prev_step = max(self.step, self.prev_step)
        self.step = 0
        # self.runner.set_globals(self.vars, False)
        Turtle.clear_turtles()
        self._segments = []
        # Sprite.clear_sprites()  # FIXME

        turtle = Turtle()
        self._the_turtle = turtle
        # self.sokoban.load_level()
        # for v in dir(turtle):  # FIXME
        #     if v[0] != '_':
        #         self.runner.globals[v] = getattr(turtle, v)

        KeepRefs._clear_instances()

        # Clear the Right Way (Thank you Mark Vasilkov)
        saved = self.sandbox.children[:]
        self.sandbox.clear_widgets()
        self.sandbox.canvas.clear()
        # self.sandbox.rc1.clear()
        # self.sandbox.rc2.clear()
        Sprite.clear_sprites()
        if self.sandbox.space:
            self.sandbox.space.remove(*self.sandbox.space.shapes, *self.sandbox.space.bodies)
        self.sandbox.space = pymunk.Space()
        self.sandbox.space.gravity = self._gravity
        self.sandbox.space.sleep_time_threshold = 3.0
        self.sandbox.space.replay_mode = False
        for widget in saved:
            print('SAVED:', widget)
            self.sandbox.add_widget(widget)
        self.sandbox.canvas.add(self.sandbox.rc1)
        self.sandbox.canvas.add(self.sandbox.rc2)

        # self.sandbox.fbo.add(self.sandbox.rc1)
        # self.sandbox.fbo.add(self.sandbox.rc2)

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

        ok = False
        start = process_time()
        seed(0)
        try:
            ok = self.runner.execute()
        except Exception as e:
            print('E2:')
            print_exc()
        print('Exec Time:', process_time() - start)

        watches = ''
        for v, t, r in whos(self.runner.globals):
            watches += f'{v + " " * (8 - len(v))}  {r}\n'
            # watches += f'{v + " " * (8 - len(v))} {t + " " * (5 - len(t))}  {r}\n'

        if False: # ok and F_UPDATE in self.runner.globals and self.prev_step > 0:
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
        # self.update_sandbox()
        # if self.sokoban:
        #     self.replay_step = len(self.sokoban.log)
        #     self.replay_steps = len(self.sokoban.log)

        # self.code_editor.highlight_line(None)
        if not ok:
            # if self.update_schedule:
            #     self.update_schedule.cancel()
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
                    print('TRACE:', filename, lineno, name, repr(line), repr(locals)[:800]) # filename, lineno, name, line, locals)
                    # if filename == '<code-input>':
                    #     if name != '<module>':
                    watched_locals = whos(locals)
                    if watched_locals:
                        watches += f'== {name} ==\n'
                        for v, t, r in watched_locals:
                            watches += f'{v + " " * (8 - len(v))}  {r}\n'
                            # watches += f'{v}\t{t}\t{r}\n'
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
            if F_UPDATE in self.runner.globals:
                self._last_update_time = time()
            # self.update_sandbox()
            Clock.schedule_once(lambda _: self.update_sandbox(), 0)
            # if self.sokoban and self.sokoban.boxes_remaining == 0:
            #     print('Level completed:', self.sokoban.level)
            #     self.sokoban.level += 1
            #     self.sandbox.clear_widgets()
            if F_UPDATE in self.runner.globals:  # and (not self.update_schedule or not self.update_schedule.is_triggered):
                # self.sandbox.transition_time = 0
                def run_update(*t):
                    self.update_schedule = Clock.schedule_interval(self.trigger_exec_update, 1.0 / 60.0)
                self.update_schedule = Clock.schedule_once(run_update, self.sandbox.transition_time)

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
            print_exc()
            if self.update_schedule:
                self.update_schedule.cancel()
            if self.trigger_exec_update:
                self.trigger_exec_update.cancel()
            watches = ''
            for v, t, r in whos(self.runner.globals):
                # watches += f'{v}\t{t}\t{r}\n'
                watches += f'{v + " " * (8 - len(v))}  {r}\n'
            if self.runner.exception:
                exc, exc_str, traceback = self.runner.exception
            else:
                exc = e
                if hasattr(e, 'message'):
                    exc_str = e.message
                else:
                    exc_str = str(e) or e.__class__.__name__
                traceback = self.runner._trace(e.__traceback__)
            print('EXC2:', exc_str)
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
                                # watches += f'{v}\t{t}\t{r}\n'
                                watches += f'{v + " " * (8 - len(v))}  {r}\n'
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
