#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from random import random, randint, uniform, choice, seed
#from numpy import sin, cos, arctan2, sqrt, ceil, floor, degrees, radians, log, pi, exp
from math import sin, cos, atan2, sqrt, ceil, floor, degrees, radians, log, pi, exp
from copy import deepcopy
from time import process_time
# from sys import exc_info
import re

# Code analysis
# import sys  # for sys.path (autocomp)
import ast
# import jedi
# from sys import getsizeof
from collections import defaultdict

from kivy.uix.textinput import FL_IS_LINEBREAK

from kivy.uix.behaviors import FocusBehavior
from kivy.uix.codeinput import CodeInput
from kivy.uix.widget import Widget
from kivy.uix.image import Image
from kivy.uix.stencilview import StencilView
from kivy.uix.scatter import ScatterPlane
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.gridlayout import GridLayout
from kivy.graphics import Color, Line, Rectangle, Ellipse, Triangle, \
        PushMatrix, PopMatrix, RoundedRectangle
from kivy.graphics.transformation import Matrix
from kivy.graphics.context_instructions import Rotate, Translate, Scale
from kivy.properties import StringProperty, NumericProperty, \
        ListProperty, ObjectProperty, BooleanProperty, \
        OptionProperty, DictProperty
from kivy.clock import Clock
from kivy.cache import Cache
from kivy.core.text.markup import MarkupLabel
from kivy.core.window import Window # request keyboard

import pymunk

from ourturtle import Turtle, Vec2d
from sprite import Sprite, OurImage
from codean import autocomp, CodeRunner, Break, COMMON_CODE
from sokoban.level import Level

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


class CodeEditor(CodeInput):
    def __init__(self, **kwargs):
        # self._highlight_line = None
        self.hightlight_styles = {
            'error': (True, (.9, .1, .1, .4))
        }
        self._highlight = defaultdict(lambda: [])
        self._highlight['error'].append(3)
        self.namespace = {}
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

    def highlight_line(self, line_num, style='error', add=False):
        if line_num:
            if add:
                self._highlight[style].append(line_num)
            else:
                self._highlight[style] = [line_num]
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
                highlight_color = (.9, .1, .1, .5)
                if miny <= y <= maxy + dy:
                    self._draw_highlight(line_num, style)
        self._position_handles('both')

    def _draw_highlight(self, line_num, style):
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
                    width=1.3, group=group)
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
            # print('_auto_indent', repr(index), repr(_text), repr(line_start), repr(_text[line_start + 1:index]))
            if line_start > -1:
                line = _text[line_start + 1:index]
                indent = self.re_indent.match(line).group()
                if line[-1] == ':':
                    indent += ' ' * self.tab_width
                substring += indent
        return substring

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        # print('keyboard_on_key_down', keycode, text, modifiers)
        key, key_str = keycode
        print('kk', keycode, text, modifiers)
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
                    space, cmd, steps = m.groups()
                    if cmd == key_str:
                        if steps:
                            steps = str(int(steps)+1)
                        else:
                            steps = '2'
                        self._set_line_text(cr, space + cmd + '(' + steps + ')')
                        # self._lines[cr-1] = space + cmd + '(' + steps + ')'
                        return True
            key_str = space + key_str
            if not empty_line:
                self.do_cursor_movement('cursor_end')
                key_str = '\n' + key_str
            self.insert_text(f'{key_str}()')
            # self.cursor = (cc, cr+1)
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


class OurSandbox(FocusBehavior, ScatterPlane):

    def __init__(self, **kwargs):
        super(OurSandbox, self).__init__(**kwargs)
        self.space = pymunk.Space()
        self._keyboard = None

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
        print('DOWN2', keycode[1] or text, modifiers)
        if super(OurSandbox, self).keyboard_on_key_down(window, keycode,
                                                         text, modifiers):
            return True
        #self.text = keycode[1]
        return True

    def _keyboard_closed(self):
        print('UNBIND')
        self._keyboard.unbind(on_key_down=self.on_key_down, on_key_up=self.on_key_up)
        self._keyboard = None

    def on_key_down(self, keyboard, keycode, text, modifiers):
        print('DOWN', keycode[1] or text, modifiers)
        return True

    def on_key_up(self, keyboard, keycode, *args):
        # print('UP', chr(keycode[0]), args)
        return True


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


    init_code = StringProperty('''
up()
left(4)
right()
up(3)
left()
up()
left(2)
down()
left(2)
down(3)
''')

      # print(123)
  # bob.setheading(a)
  # bob.forward(1)
  # bob.right(2)
  # 128907
#     run_code = StringProperty('''
# #x, y = t.pos()
# #t.goto(300*sin(steps/(4)), 300*cos(steps/5))
# #bob.left(1)
# #bob.forward(1)
# ''')

    #    var_a = NumericProperty(0.5)
    #    var_b = NumericProperty(0.5)
    #    var_m = NumericProperty(10)
    #    var_n = NumericProperty(10)
    vars = DictProperty({})

    console = StringProperty('')
    watches = StringProperty('')

    sandbox = ObjectProperty(None)
    init_editor = ObjectProperty(None)
    rpanel = ObjectProperty(None)
    textout = ObjectProperty(None)
    run_to_cursor = BooleanProperty(False)

    # ball = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(Playground, self).__init__(**kwargs)

        self._run_vars = None

        globs = dict()
        for v in 'random randint uniform choice seed sin cos atan2 \
                sqrt ceil floor degrees radians log exp'.split(
        ):
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

        wall = 'sokoban/images/wall.png'
        box = 'sokoban/images/box.png'
        box_on_target = 'sokoban/images/box_on_target.png'
        space = 'sokoban/images/space.png'
        target = 'sokoban/images/target.png'
        player = 'sokoban/images/player.png'
        # player = 'sokoban/images/beetle-robot.png'
        self._sokoban_images = {'#': wall, ' ': space, '$': box, '.': target, '@': player, '*': box_on_target}

        self._sokobal_level_number = 1
        self._sokoban_level = Level('our', 1)
        self._sokoban_tiles = None
        # draw_level(my_level.get_matrix())
        self._sokoban_target_found = False
        self._sokoban_dir = 'U'
        self._sokoban_player_pos = self._sokoban_level.get_player_position()

        def sokoban_go(direction):
            def go(steps=1):
                self.sokoban_move_player(direction, steps)
            return go
        globs['right'] = sokoban_go('R')
        globs['left'] = sokoban_go('L')
        globs['up'] = sokoban_go('U')
        globs['down'] = sokoban_go('D')
        # globs['right'] = lambda: self.sokoban_turn(1) # setattr(self, '_sokoban_dir', {'R': 'D', 'L': 'U', 'U': 'R', 'D': 'L'}[self._sokoban_dir])
        # globs['left'] = lambda: self.sokoban_turn(-1) # setattr(self, '_sokoban_dir', {'R': 'U', 'L': 'D', 'U': 'L', 'D': 'R'}[self._sokoban_dir])
        # globs['forward'] = self.sokoban_move_player

        # self.sandbox.add_widget(Sprite('images/simple_cv_joint_animated.gif'))  # orc.gif
        # self.sandbox.add_widget(Sprite('images/bird.zip')) #orc.gif
        # self.sandbox.add_widget(Sprite('images/cube.zip')) #orc.gif
        # self.sandbox.add_widget(Sprite('turtle'))
        # img = Our'grace_hopper.jpg')
        # globs['image'] = img.image
        # globs['img'] = img

        self.runner = CodeRunner(globals=globs, special_funcs=['update'])

        self.init_editor.namespace = self.runner.globals  # FIXME?

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

        self.steps = 0
        self.trigger_exec = Clock.create_trigger(self.execute_code, -1)
        self.bind(init_code=self.compile_code)
        self.bind(run_to_cursor=self.compile_code)
        self.init_editor.bind(cursor_row=self.on_init_editor_cursor_row)

        def _set_var(wid, value):
            self.vars[wid.var_name] = value
            if wid.var_name in self.runner.common_vars:
                self.trigger_exec()

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
        self.trigger_exec_run = Clock.create_trigger(self.execute_run, -1)
        self.run_schedule = None  # Clock.schedule_interval(self.trigger_exec_run, 1.0 / 60.0)

    def sokoban_turn(self, dir):
        if dir == 1:
            self._sokoban_dir = {'R': 'D', 'L': 'U', 'U': 'R', 'D': 'L'}[self._sokoban_dir]
        else:
            self._sokoban_dir = {'R': 'U', 'L': 'D', 'U': 'L', 'D': 'R'}[self._sokoban_dir]

    def sokoban_draw_level(self, update=False):
        images = self._sokoban_images
        matrix = self._sokoban_level.get_matrix()
        if not update:
            self._sokoban_tiles = []
        for i, row in enumerate(matrix):
            if not update:
                self._sokoban_tiles.append([])
            for j, c in enumerate(row):
                # rot = {'R': -90, 'L': 90, 'U': 0, 'D': 180}[self._sokoban_dir] if c == '@' else 0
                if update:
                    tile = self._sokoban_tiles[i][j]
                    image = tile.shapes[0]
                    if image.source != images[c]:
                        image.source = images[c]
                        # tile.clear_widgets()
                        # img = Image(source=images[c])
                        # tile.add_widget(img)
                        # tile.shapes[0] = img
                        # with self._sokoban_tiles[i][j].canvas:
                        #     Color(1,0,0)
                        #     Rectangle(pos=(0, 0), size=(20,20))
                        # with self._sokoban_tiles[i][j].canvas.after:
                        #     Color(1,0,0)
                        #     Rectangle(pos=(0, 0), size=(10,20))

                        # self._sokoban_tiles[i][j].x += 10
                        # self._sokoban_tiles[i][j].position = (30, 30)
                        # image.reload()
                        print('Reload', images[c])

                    # print('Upd:', self._sokoban_tiles[i][j].shapes[0].source, images[c])
                    # self._sokoban_tiles[i][j].shapes[0].source = images[c]
                else:
                    tile = Sprite(images[c], x=36*j, y=36*(len(matrix)-i), trace=False)
                    self._sokoban_tiles[-1].append(tile)
                    self.sandbox.add_widget(tile) # Sprite(images[c], x=36*j, y=36*(len(matrix)-i), trace=False)) # rotation=0,

    def sokoban_move_player(self, direction, steps=1):
        # direction = self._sokoban_dir
        my_level = self._sokoban_level
        target_found = self._sokoban_target_found
        matrix = my_level.get_matrix()

        my_level.add_to_history(matrix)

        # print boxes
        # print(my_level.get_boxes())

        if steps < 0:
            direction = {'R': 'L', 'L': 'R', 'U': 'D', 'D': 'U'}[direction]
            steps = -steps

        while steps > 0:
            x, y = self._sokoban_player_pos
            # print('target_found 1', target_found)

            if direction == "L":
                # print("######### Moving Left #########")

                # if is_space
                if matrix[y][x-1] == " ":
                    # print("OK Space Found")
                    matrix[y][x-1] = "@"
                    self._sokoban_player_pos = (x-1, y)
                    if target_found == True:
                        matrix[y][x] = "."
                        target_found = False
                    else:
                        matrix[y][x] = " "

                # if is_box
                elif matrix[y][x-1] == "$":
                    # print("Box Found")
                    if matrix[y][x-2] == " ":
                        matrix[y][x-2] = "$"
                        matrix[y][x-1] = "@"
                        self._sokoban_player_pos = (x-1, y)
                        if target_found == True:
                            matrix[y][x] = "."
                            target_found = False
                        else:
                            matrix[y][x] = " "
                    elif matrix[y][x-2] == ".":
                        matrix[y][x-2] = "*"
                        matrix[y][x-1] = "@"
                        self._sokoban_player_pos = (x-1, y)
                        if target_found == True:
                            matrix[y][x] = "."
                            target_found = False
                        else:
                            matrix[y][x] = " "
                    else:
                        raise Exception('cannot go there')

                # if is_box_on_target
                elif matrix[y][x-1] == "*":
                    # print("Box on target Found")
                    if matrix[y][x-2] == " ":
                        matrix[y][x-2] = "$"
                        matrix[y][x-1] = "@"
                        self._sokoban_player_pos = (x-1, y)
                        if target_found == True:
                            matrix[y][x] = "."
                        else:
                            matrix[y][x] = " "
                        target_found = True

                    elif matrix[y][x-2] == ".":
                        matrix[y][x-2] = "*"
                        matrix[y][x-1] = "@"
                        self._sokoban_player_pos = (x-1, y)
                        if target_found == True:
                            matrix[y][x] = "."
                        else:
                            matrix[y][x] = " "
                        target_found = True

                    else:
                        raise Exception('cannot go there')

                # if is_target
                elif matrix[y][x-1] == ".":
                    # print("Target Found")
                    matrix[y][x-1] = "@"
                    self._sokoban_player_pos = (x-1, y)
                    if target_found == True:
                        matrix[y][x] = "."
                    else:
                        matrix[y][x] = " "
                    target_found = True

                # else
                else:
                    # print("There is a wall here")
                    raise Exception('cannot go there')

            elif direction == "R":
                # print("######### Moving Right #########")

                # if is_space
                if matrix[y][x+1] == " ":
                    # print("OK Space Found")
                    matrix[y][x+1] = "@"
                    self._sokoban_player_pos = (x+1, y)
                    if target_found == True:
                        matrix[y][x] = "."
                        target_found = False
                    else:
                        matrix[y][x] = " "

                # if is_box
                elif matrix[y][x+1] == "$":
                    # print("Box Found")
                    if matrix[y][x+2] == " ":
                        matrix[y][x+2] = "$"
                        matrix[y][x+1] = "@"
                        self._sokoban_player_pos = (x+1, y)
                        if target_found == True:
                            matrix[y][x] = "."
                            target_found = False
                        else:
                            matrix[y][x] = " "

                    elif matrix[y][x+2] == ".":
                        matrix[y][x+2] = "*"
                        matrix[y][x+1] = "@"
                        self._sokoban_player_pos = (x+1, y)
                        if target_found == True:
                            matrix[y][x] = "."
                            target_found = False
                        else:
                            matrix[y][x] = " "

                    else:
                        raise Exception('cannot go there')

                # if is_box_on_target
                elif matrix[y][x+1] == "*":
                    # print("Box on target Found")
                    if matrix[y][x+2] == " ":
                        matrix[y][x+2] = "$"
                        matrix[y][x+1] = "@"
                        self._sokoban_player_pos = (x+1, y)
                        if target_found == True:
                            matrix[y][x] = "."
                        else:
                            matrix[y][x] = " "
                        target_found = True

                    elif matrix[y][x+2] == ".":
                        matrix[y][x+2] = "*"
                        matrix[y][x+1] = "@"
                        self._sokoban_player_pos = (x+1, y)
                        if target_found == True:
                            matrix[y][x] = "."
                        else:
                            matrix[y][x] = " "
                        target_found = True

                    else:
                        raise Exception('cannot go there')

                # if is_target
                elif matrix[y][x+1] == ".":
                    # print("Target Found")
                    matrix[y][x+1] = "@"
                    self._sokoban_player_pos = (x+1, y)
                    if target_found == True:
                        matrix[y][x] = "."
                    else:
                        matrix[y][x] = " "
                    target_found = True

                # else
                else:
                    # print("There is a wall here")
                    raise Exception('cannot go there')

            elif direction == "D":
                # print("######### Moving Down #########")

                # if is_space
                if matrix[y+1][x] == " ":
                    # print("OK Space Found")
                    matrix[y+1][x] = "@"
                    self._sokoban_player_pos = (x, y+1)
                    if target_found == True:
                        matrix[y][x] = "."
                        target_found = False
                    else:
                        matrix[y][x] = " "

                # if is_box
                elif matrix[y+1][x] == "$":
                    # print("Box Found")
                    if matrix[y+2][x] == " ":
                        matrix[y+2][x] = "$"
                        matrix[y+1][x] = "@"
                        self._sokoban_player_pos = (x, y+1)
                        if target_found == True:
                            matrix[y][x] = "."
                            target_found = False
                        else:
                            matrix[y][x] = " "

                    elif matrix[y+2][x] == ".":
                        matrix[y+2][x] = "*"
                        matrix[y+1][x] = "@"
                        self._sokoban_player_pos = (x, y+1)
                        if target_found == True:
                            matrix[y][x] = "."
                            target_found = False
                        else:
                            matrix[y][x] = " "

                    else:
                        raise Exception('cannot go there')

                # if is_box_on_target
                elif matrix[y+1][x] == "*":
                    # print("Box on target Found")
                    if matrix[y+2][x] == " ":
                        matrix[y+2][x] = "$"
                        matrix[y+1][x] = "@"
                        self._sokoban_player_pos = (x, y+1)
                        if target_found == True:
                            matrix[y][x] = "."
                        else:
                            matrix[y][x] = " "
                        target_found = True

                    elif matrix[y+2][x] == ".":
                        matrix[y+2][x] = "*"
                        matrix[y+1][x] = "@"
                        self._sokoban_player_pos = (x, y+1)
                        if target_found == True:
                            matrix[y][x] = "."
                        else:
                            matrix[y][x] = " "
                        target_found = True

                    else:
                        raise Exception('cannot go there')

                # if is_target
                elif matrix[y+1][x] == ".":
                    # print("Target Found")
                    matrix[y+1][x] = "@"
                    self._sokoban_player_pos = (x, y+1)
                    if target_found == True:
                        matrix[y][x] = "."
                    else:
                        matrix[y][x] = " "
                    target_found = True

                # else
                else:
                    # print("There is a wall here")
                    raise Exception('cannot go there')

            elif direction == "U":
                # print("######### Moving Up #########")

                # if is_space
                if matrix[y-1][x] == " ":
                    # print("OK Space Found")
                    matrix[y-1][x] = "@"
                    self._sokoban_player_pos = (x, y-1)
                    if target_found == True:
                        matrix[y][x] = "."
                        target_found = False
                    else:
                        matrix[y][x] = " "

                # if is_box
                elif matrix[y-1][x] == "$":
                    # print("Box Found")
                    if matrix[y-2][x] == " ":
                        matrix[y-2][x] = "$"
                        matrix[y-1][x] = "@"
                        self._sokoban_player_pos = (x, y-1)
                        if target_found == True:
                            matrix[y][x] = "."
                            target_found = False
                        else:
                            matrix[y][x] = " "

                    elif matrix[y-2][x] == ".":
                        matrix[y-2][x] = "*"
                        matrix[y-1][x] = "@"
                        self._sokoban_player_pos = (x, y-1)
                        if target_found == True:
                            matrix[y][x] = "."
                            target_found = False
                        else:
                            matrix[y][x] = " "

                    else:
                        raise Exception('cannot go there')

                # if is_box_on_target
                elif matrix[y-1][x] == "*":
                    # print("Box on target Found")
                    if matrix[y-2][x] == " ":
                        matrix[y-2][x] = "$"
                        matrix[y-1][x] = "@"
                        self._sokoban_player_pos = (x, y-1)
                        if target_found == True:
                            matrix[y][x] = "."
                        else:
                            matrix[y][x] = " "
                        target_found = True

                    elif matrix[y-2][x] == ".":
                        matrix[y-2][x] = "*"
                        matrix[y-1][x] = "@"
                        self._sokoban_player_pos = (x, y-1)
                        if target_found == True:
                            matrix[y][x] = "."
                        else:
                            matrix[y][x] = " "
                        target_found = True

                    else:
                        raise Exception('cannot go there')

                # if is_target
                elif matrix[y-1][x] == ".":
                    # print("Target Found")
                    matrix[y-1][x] = "@"
                    self._sokoban_player_pos = (x, y-1)
                    if target_found == True:
                        matrix[y][x] = "."
                    else:
                        matrix[y][x] = " "
                    target_found = True

                # else
                else:
                    # print("There is a wall here")
                    raise Exception('cannot go there')
            # print('target_found 2', target_found)
            self._sokoban_target_found = target_found
            steps -= 1

        # draw_level(matrix)

        # print("Boxes remaining: " + str(len(my_level.get_boxes())))

        if len(my_level.get_boxes()) == 0:
            # my_environment.screen.fill((0, 0, 0))
            print("Level Completed")
            self._sokobal_level_number += 1
            self._sokoban_level = Level('our', self._sokobal_level_number)
            self._sokoban_tiles = None
            self._sokoban_target_found = False
            self.sandbox.clear_widgets()
            self.sokoban_draw_level()
            # global current_level
            # current_level += 1
            # init_level(level_set,current_level)

    def on_init_editor_cursor_row(self, *largs):
        if self.run_to_cursor:
            self.compile_code()

    def update_sandbox(self, redraw=True):
        try:
            if redraw:
                self.graphics_instructions = []
                # self.sandbox.canvas.clear()

                # Clear the Right Way (Thank you Mark Vasilkov)
                saved = self.sandbox.children[:]
                self.sandbox.clear_widgets()
                self.sandbox.canvas.clear()
                self.sandbox.space.remove(*self.sandbox.space.shapes)
                self.sandbox.space.remove(*self.sandbox.space.bodies)
                for widget in saved:
                    self.sandbox.add_widget(widget)

                self.sokoban_draw_level(self._sokoban_tiles is not None)

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
        #        self._run_vars = defaultdict(lambda: defaultdict(list))
        # print('= ' * 40)
        # print(self.init_code)

        # if self.run_schedule:
        #     Clock.unschedule(self.run_schedule)

        breakpoint = None
        if self.run_to_cursor:
            breakpoint = self.init_editor.cursor_row + 1

        try:
            changed = self.runner.parse(self.init_code, breakpoint)
            if COMMON_CODE in changed:
                self.runner.reset()

            self.runner.compile(changed)
        except Exception as e:
            print('E:', e)
            print('* ' * 40)
            line_num = self.runner.exception_lineno(e)
            self.init_editor.highlight_line(line_num)
        else:
            # self.init_editor.highlight_line(None)
            if COMMON_CODE in changed:
                print('-='*30)
                print(self.init_code)
                print('-='*30)
                self.trigger_exec()
                changed.remove(COMMON_CODE)
            print('EXEC Changed', changed)
            try:
                self.runner.execute(changed)
            except Exception as e:
                print('E3:', e)


    def execute_code(self, *largs):
        print('execute_code')
        self.runner.reset(globals=self.vars)
        # self.runner.set_globals(self.vars, False)
        Turtle.clear_turtles()
        # Sprite.clear_sprites()  # FIXME

        turtle = Turtle()  # self.sandbox.add_turtle()
        self._the_turtle = turtle
        self._sokoban_level = Level('our', self._sokobal_level_number) # our
        # draw_level(my_level.get_matrix())
        self._sokoban_target_found = False
        self._sokoban_dir = 'U'
        self._sokoban_player_pos = self._sokoban_level.get_player_position()
        # for v in dir(turtle):  # FIXME
        #     if v[0] != '_':
        #         self.runner.globals[v] = getattr(turtle, v)

        # self._run_vars = defaultdict(lambda: defaultdict(list))
        seed(123)
        ok = False
        try:
            ok = self.runner.execute()

        except Exception as e:
            print('E2:', e)

        out = self.runner.text_stream.getvalue()
        self.console = out
        print('out:', out)
        print('- ' * 40)
        # self.init_editor.highlight_line(None)
        watches = ''
        for v, t, r in whos(self.runner.globals):
            watches += f'{v}\t{t}\t{r}\n'

        # FIXME: add scene spdiff
        self.update_sandbox()


        self.init_editor.highlight_line(None)
        if not ok:
            if self.run_schedule:
                Clock.unschedule(self.run_schedule)
            if self.runner.traceback:
                # print('STACK', self.runner.traceback.stack)
                stack = None
                for stack in self.runner.traceback.stack:
                    if stack.filename == '<code-input>':
                        watched_locals = whos(stack.locals)
                        if watched_locals:
                            watches += f'== {stack.name} ==\n'
                            for v, t, r in watched_locals:
                                watches += f'{v}\t{t}\t{r}\n'
                        self.init_editor.highlight_line(stack.lineno, add=True)
                        print('LINES +', stack.lineno, self.init_editor._highlight)

                # if not stack:
                #     print('Unhandled 2')
                #     self.init_editor.highlight_line(None)
                # else:
                #     if self.runner.traceback.exc_type is not Break:
                #         # print("LINENO", stack.lineno)
                #         self.init_editor.highlight_line(stack.lineno)
                #     else:
                #         self.init_editor.highlight_line(None)

                # stack = self.runner.traceback.stack[-1]
                # line_num = self.runner.exception_lineno(e)

                # for k, v in stack.locals.items():
                #     if k[0] != '_' and v[0] != '<':
                #         watches += f'{k}\t{v[:40]}\n'
                #         # print(k, type(v), repr(v)[:80], sep='\t')
                #         # watches += f'{k}\t{type(v).__qualname__}\t{repr(v)[:80]}\n'

            else:
                print('Unhandled exception')
                # self.init_editor.highlight_line(None) # FIXME
        # else:
            # self.init_editor.highlight_line(None)
        # else:

        self.watches = watches
        print('= ' * 40)


        # for k, v in self.runner.globals.items():
        #     if any([isinstance(v, t) for t in [int, float, str, dict, tuple]]) and k[0] != '_':
        #         print(k, type(v), repr(v)[:80], sep='\t')
        # print('= ' * 40)

        if ok and 'update' in self.runner.globals:
            self.run_schedule = Clock.schedule_interval(self.trigger_exec_run, 1.0 / 30.0)


    def execute_run(self, *largs):
        self.steps += 1
        self.runner.globals.update(self.vars)
        self.runner.globals['steps'] = self.steps
        ts_pos = self.runner.text_stream.tell()

        try:
            self.runner.call_if_exists('update')

        except Exception as e:
            print(e)

        else:
            self.update_sandbox(False)

        self.runner.text_stream.seek(ts_pos)
        out = self.runner.text_stream.read()
        if out:
            print(out)
            print('* ' * 20)
