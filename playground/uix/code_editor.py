import re
from collections import defaultdict

from pygments import styles
from pygments.formatters import BBCodeFormatter

from kivy.uix.textinput import FL_IS_LINEBREAK
from kivy.uix.behaviors import FocusBehavior
from kivy.uix.codeinput import CodeInput
from kivy.graphics import Color, Line, RoundedRectangle
from kivy.properties import AliasProperty
from kivy.clock import Clock
from kivy.cache import Cache
from kivy.core.text.markup import MarkupLabel


from playground.codean import autocomp
from playground.color import _parse_srgb


R_TURN = re.compile(r"^(\s*)(right|left|up|down)\(([0-9]*)\)$")


class CodeEditor(CodeInput, FocusBehavior):
    def __init__(self, **kwargs):
        self.hightlight_styles = {"error": (True, (0.9, 0.1, 0.1, 0.4)), "run": (False, (0.1, 0.9, 0.1, 1.0))}
        self._highlight = defaultdict(set)
        # self._highlight['run'].add(3)
        self.namespace = {}
        self.ac_begin = False
        self.ac_current = 0
        self.ac_position = None
        self.ac_completions = []
        self.register_event_type("on_key_down")

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

    # def _get_bbcode(self, ntext):
    #     print('_get_bbcode', ntext)
    #     return super(CodeEditor, self)._get_bbcode(ntext)

    # def _create_line_label(self, text, hint=False):
    #     print('_create_line_label', text, hint)
    #     return super(CodeEditor, self)._create_line_label(text, hint)

    def highlight_line(self, line_num, style="error", add=False):
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
            self.canvas.remove_group("hl-" + style)
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
        self._position_handles("both")

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
        x2 = (x - self.scroll_x) + self._get_text_width(lines, self.tab_width, self._label_cached)
        width_minus_padding = self.width - (self.padding[2] + self.padding[0])
        maxx = x + width_minus_padding
        if x1 > maxx:
            return
        x1 = max(x1, x)
        x2 = min(x2, x + width_minus_padding)

        self.canvas.add(Color(*highlight_color, group="hl-" + style))
        # self.canvas.add(
        #     Rectangle(
        #         pos=(x1, pos[1]), size=(x2 - x1, size[1]), group='highlight'))
        group = "hl-" + style
        with self.canvas:
            Color(*highlight_color, group=group)
            if fill:
                RoundedRectangle(pos=(x1, pos[1]), size=(x2 - x1, size[1]), radius=(4, 4), segments=3, group=group)
            else:
                Line(rounded_rectangle=(x1, pos[1], x2 - x1, size[1], 4), group=group)
        # self.canvas.add(
        #     Line(rounded_rectangle=(x1, pos[1], x2 - x1, size[1], 4), width=1.3, group='highlight'))

    def _split_smart(self, text):
        # Disable word wrapping (because of broken syntax highlight)
        lines = text.split("\n")
        lines_flags = [0] + [FL_IS_LINEBREAK] * (len(lines) - 1)
        return lines, lines_flags

    def _create_line_label(self, text, hint=False):
        # Fix empty lines bug
        # Create a label from a text, using line options
        ntext = text.replace("\n", "").replace("\t", " " * self.tab_width)
        ntext = self._get_bbcode(ntext)
        kw = self._get_line_options()
        cid = "{}\0{}\0{}".format(ntext, self.password, kw)
        texture = Cache.get("textinput.label", cid)

        if texture is None:
            # FIXME right now, we can't render very long line...
            # if we move on "VBO" version as fallback, we won't need to
            # do this.
            # try to find the maximum text we can handle
            label = MarkupLabel(text=ntext, **kw)
            label.refresh()
            # ok, we found it.
            texture = label.texture
            Cache.append("textinput.label", cid, texture)
            label.text = ""
        return texture

    def _auto_indent(self, substring):
        index = self.cursor_index()
        _text = self._get_text() # (encode=False)
        if index > 0:
            line_start = _text.rfind("\n", 0, index)
            if line_start > -1:
                line = _text[line_start + 1 : index]
                indent = self.re_indent.match(line).group()
                substring += indent
        if len(_text) > 0 and _text[index - 1] == ":":
            substring += " " * self.tab_width
        return substring

    def do_backspace(self, from_undo=False, mode="bkspc"):
        # Clever backspace: remove up to previous indent level.
        if self.readonly:
            return
        cc, cr = self.cursor
        _lines = self._lines
        text = _lines[cr]
        cursor_index = self.cursor_index()
        tab = self.tab_width
        if cc > 0 and text[:cc].lstrip() == "":
            indent = (cc - 1) // tab
            remove = cc - indent * tab
            new_text = " " * indent * tab + text[cc:]
            substring = " " * remove
            self._set_line_text(cr, new_text)
            self.cursor = self.get_cursor_from_index(cursor_index - remove)
            # handle undo and redo
            self._set_unredo_bkspc(cursor_index, cursor_index - remove, substring, from_undo)
            return
        super(CodeEditor, self).do_backspace(from_undo, mode)

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        # print('keyboard_on_key_down', keycode, text, modifiers)
        key, key_str = keycode
        # print('kk', keycode, text, modifiers)
        if key == 9:  # Tab
            if modifiers == ["ctrl"]:
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
            if self._selection and modifiers in [[], ["shift"]]:
                a, b = self._selection_from, self._selection_to
                cc_from, cr_from = self.get_cursor_from_index(a)
                cc_to, cr_to = self.get_cursor_from_index(b)
                for cr in range(min(cr_from, cr_to), max(cr_from, cr_to) + 1):
                    line = _lines[cr]
                    if not modifiers:
                        new_text = " " * tab + line
                        self._set_line_text(cr, new_text)
                        if cr == cr_from:
                            cc_from += tab
                        if cr == cr_to:
                            cc_to += tab
                    else:
                        spaces = len(line) - len(line.lstrip())
                        indent = (spaces - 1) // tab
                        if indent >= 0:
                            remove = spaces - indent * tab
                            new_text = line[remove:]
                            self._set_line_text(cr, new_text)
                            if cr == cr_from:
                                cc_from = max(0, cc_from - remove)
                            if cr == cr_to:
                                cc_to = max(0, cc_to - remove)
                self._selection_from = self.cursor_index((cc_from, cr_from))
                self._selection_to = self.cursor_index((cc_to, cr_to))
                self._selection_finished = True
                self._update_selection(True)
                self._update_graphics_selection()
                # TODO: Add undo/redo
                return True
            elif not self._selection and before_cursor == "" and not modifiers:
                self.insert_text(" " * tab)
                self.ac_begin = False
                return True
            elif not self._selection and before_cursor != "" and not modifiers:
                # print("AC", self.ac_begin, cc, cr, self.ac_position, self.ac_current, self.ac_completions)
                if self.ac_begin:
                    if self.ac_completions[self.ac_current].complete:
                        self.do_undo()
                    self.ac_current += 1
                    self.ac_current %= len(self.ac_completions)
                else:
                    self.ac_completions = autocomp(self.text, self.namespace, cr, cc)
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
        elif key == 47 and modifiers == ["ctrl"]:
            cc, cr = self.cursor
            _lines = self._lines

            if self._selection:
                a, b = self._selection_from, self._selection_to
                cc_from, cr_from = self.get_cursor_from_index(a)
                cc_to, cr_to = self.get_cursor_from_index(b)
            else:
                cc_from, cr_from = self.cursor
                cc_to, cr_to = cc_from, cr_from
            uncomment = True  # _lines[min(cr_from, cr_to)].lstrip().startswith('#')
            indent = 1000
            for cr in range(min(cr_from, cr_to), max(cr_from, cr_to) + 1):
                line = _lines[cr]
                if not line.lstrip().startswith("#"):
                    uncomment = False
                indent = min(indent, len(line) - len(line.lstrip()))

            for cr in range(min(cr_from, cr_to), max(cr_from, cr_to) + 1):
                line = _lines[cr]
                if not uncomment:
                    new_text = f'{" " * indent}# {line[indent:]}'
                    self._set_line_text(cr, new_text)
                    if cr == cr_from:
                        cc_from += 2
                    if cr == cr_to:
                        cc_to += 2
                else:
                    new_text = re.sub(r"^(\s*)# ?", r"\1", line)
                    removed = len(line) - len(new_text)
                    self._set_line_text(cr, new_text)
                    if cr == cr_from:
                        cc_from = max(0, cc_from - removed)
                    if cr == cr_to:
                        cc_to = max(0, cc_to - removed)
            self._selection_from = self.cursor_index((cc_from, cr_from))
            self._selection_to = self.cursor_index((cc_to, cr_to))
            self._selection_finished = True
            self._update_selection(True)
            self._update_graphics_selection()
            # TODO: Add undo/redo
            return True

        if self.dispatch("on_key_down", window, keycode, text, modifiers):
            return True

        self.ac_begin = False
        return super(CodeInput, self).keyboard_on_key_down(window, keycode, text, modifiers)

    def do_cursor_movement(self, action, control=False, alt=False):
        if action == "cursor_home" and not control:
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
        if offset > viewport_width + sx - 50:
            self.scroll_x = offset - viewport_width + 50
        if offset < min(sx + 50, viewport_width):
            self.scroll_x = max(0, offset - 50)  # + 25 # FIXME TODO BUG
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
        words = [m.span() for m in re.finditer(r"\w+", line)]
        if words:
            s1, s2 = zip(*words)
            nonwords = zip(s2, s1[1:])
            for span in (*words, *nonwords):  # words + list(nonwords):
                if span[0] <= cc < span[1]:
                    end = start + span[1]
                    start += span[0]
        Clock.schedule_once(lambda dt: self.select_text(start, end))

    def on_key_down(self, window, keycode, text, modifiers):
        pass

    def on_cursor_row(self, *largs):
        pass
