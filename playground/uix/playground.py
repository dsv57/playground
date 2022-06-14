from random import random, randint, uniform, choice, seed
from math import sin, cos, atan2, sqrt, ceil, floor, degrees, radians, log, pi, exp
from time import process_time, time
from os.path import exists
from collections import defaultdict, namedtuple
from traceback import print_exc

import pymunk

from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.graphics import (
    Color,
    Line,
    Triangle,
    PushMatrix,
    PopMatrix,
    Mesh,
    BindTexture,
)
from kivy.graphics.transformation import Matrix

# from kivy.graphics.context_instructions import Rotate, Translate, Scale, Transform as KvTransform
from kivy.properties import (
    StringProperty,
    NumericProperty,
    ObjectProperty,
    BooleanProperty,
    DictProperty,
)
from kivy.clock import Clock
from kivy.utils import escape_markup
from kivy.animation import Animation

from playground.sprite import Sprite
from playground.ourturtle import Turtle
from playground.codean import CodeRunner, Break, COMMON_CODE
from playground.uix.var_slider import VarSlider
from playground.color import _global_update_colors, Color as OurColor
from playground.shapes import (
    Stroke,
    Physics,
    Shape,
    Circle,
    Rectangle,
    KeepRefs,
    Image as OurImage,
    Line as OurLine,
)
from playground.geometry import Vector, VectorRef, Transform

try:
    import mycolors
except:
    pass


F_UPDATE = "update"
F_ON_KEY_PRESS = "on_key_press"
F_ON_KEY_RELEASE = "on_key_release"

FPS = 60

TRANSITION_TIME = 0.4  # * 2 # FIXME: Import from common config
TRANSITION_IN = "in_back"
TRANSITION_OUT = "out_back"


def whos(vars, max_repr=40):
    w_types = (int, float, str, list, dict, tuple)
    w_types += (
        OurColor,
        Stroke,
        Physics,
        Shape,
        Circle,
        Rectangle,
        KeepRefs,
        OurImage,
        Vector,
        VectorRef,
        Transform,
    )

    def w_repr(v):
        r = repr(v)
        return r if len(r) < max_repr else r[: max_repr - 3] + "..."

    return [
        (k, type(v).__qualname__, w_repr(v))
        for k, v in vars.items()
        if isinstance(v, w_types) and k[0] != "_"
    ]


class Key(namedtuple("Key", ["keycode", "key", "text"])):
    def __eq__(self, b):
        if b is not None and b in (self.keycode, self.key, self.text):
            return True
        return False

    def __str__(self):
        return f"Key '{self.key or self.text}' ({self.keycode})"


class Playground(FloatLayout):

    code = (
        StringProperty()
    )  # '''ball = add_sprite('circle', x=0, y=-120, body_type=0, color='green')
    # ball.apply_impulse((50000, 0))

    # platform = add_sprite('platform', x=250, y=-120, body_type=1, color='red')
    # def on_key_press(key, modifiers):
    #     if key == 'up':
    #         platform.velocity += 0, 15
    #     if key == 'down':
    #         platform.velocity -= 0, 15

    #     print(key, modifiers)

    # def update(dt):
    #     pass

    # ''')

    vars = DictProperty({})

    status = ObjectProperty((None,))

    status_text = StringProperty("")
    console = StringProperty("")
    watches = StringProperty("")
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
        for (
            v
        ) in "random randint uniform choice seed sin cos atan2 \
                sqrt ceil floor degrees radians log exp".split():
            globs[v] = eval(v)

        def _dump_vars(v, lineno):
            global _vars
            for k, v in v.copy().items():
                if (
                    any([isinstance(v, t) for t in [int, float, str, dict, tuple]])
                    and k[0] != "_"
                ):
                    self._run_vars[lineno][k].append(
                        v if getsizeof(v) <= _MAX_VAR_SIZE else "<LARGE>"
                    )

        globs["Turtle"] = Turtle
        try:
            globs["cam16_to_srgb"] = mycolors.cam16_to_srgb
            globs["cam16ucs_to_srgb"] = mycolors.cam16ucs_to_srgb
            globs["jzazbz_to_srgb"] = mycolors.jzazbz_to_srgb
            globs["srgb_to_cam16ucs"] = mycolors.srgb_to_cam16ucs
            globs["lab_to_cam16ucs"] = mycolors.lab_to_cam16ucs
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

        globs["add_sprite"] = _add_sprite

        def _add_line(*largs, **kvargs):
            # self.sandbox.add_widget(line)
            with self.sandbox.canvas:
                line = Line(*largs, **kvargs)
            return line

        globs["Line"] = _add_line

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

        globs["set_gravity"] = _set_gravity

        self._show_clipped = True

        def _show_clipped_colors(show=True):
            self._show_clipped = show

        globs["show_clipped_colors"] = _show_clipped_colors

        self.trigger_exec_update = Clock.create_trigger(self.execute_update, -1)
        self.update_schedule = None

        self.runner = CodeRunner(
            globals=globs, special_funcs=[F_UPDATE, F_ON_KEY_PRESS, F_ON_KEY_RELEASE]
        )

        self.code_editor.namespace = self.runner.globals  # FIXME?

        # FIXME
        vs1 = VarSlider(var_name="a", min=0, max=360, type="float")
        vs2 = VarSlider(var_name="b", min=0, max=360, type="float")
        vs3 = VarSlider(var_name="c", min=0, max=360, type="float")
        # vs4 = VarSlider(var_name='l', min=0, max=50)
        # vs5 = VarSlider(var_name='m', min=0, max=100)
        # vs6 = VarSlider(var_name='n', min=0, max=150)
        self.rpanel.add_widget(vs1, 1)
        self.rpanel.add_widget(vs2, 1)
        self.rpanel.add_widget(vs3, 1)
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

        # @debounce(0.2)
        def _set_var(wid, value):
            self.sandbox.transition_time = TRANSITION_TIME / 2  # 0.2
            self.sandbox.transition_in = "in_cubic"
            self.sandbox.transition_out = "out_cubic"
            self.vars[wid.var_name] = value
            if wid.var_name in self.runner.common_vars:
                self.trigger_exec()

        if exists("source.py"):
            with open("source.py") as f:
                self.code = f.read()

            def _reset_cursor(*t):
                self.code_editor.cursor = 0, 0

            Clock.schedule_once(_reset_cursor, 0)
            # self.code_editor.scroll_x = 0
            # self.code_editor.scroll_y = 0
            # self.code_editor.cursor = 0, 0

        # FIXME
        vs1.bind(value=_set_var)
        vs2.bind(value=_set_var)
        vs3.bind(value=_set_var)
        # vs4.bind(value=_set_var)
        # vs5.bind(value=_set_var)
        # vs6.bind(value=_set_var)
        vs1.value = 36.0
        vs2.value = 72.0
        vs3.value = 108.0
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
        y_bar = scrlv.scroll_y * (ti.height - scrlv.height)
        if ti.height > scrlv.height:
            if y_cursor >= y_bar + scrlv.height:
                dy = y_cursor - (y_bar + scrlv.height)
                scrlv.scroll_y = (
                    scrlv.scroll_y + scrlv.convert_distance_to_scroll(0, dy)[1]
                )
            if y_cursor - ti.line_height <= y_bar:
                dy = (y_cursor - ti.line_height) - y_bar
                scrlv.scroll_y = (
                    scrlv.scroll_y + scrlv.convert_distance_to_scroll(0, dy)[1]
                )

    def code_editor_on_key_down(self, widget, window, keycode, text, modifiers):
        # print('code_editor_on_key_down', keycode, text, modifiers)
        if keycode[1] == "f5":
            self.step = 0
            self.prev_step = 0
            self.runner.reset()
            self.trigger_exec()
            return True

    def sandbox_on_key_down(self, widget, window, keycode, text, modifiers):
        if keycode[1] == "f5":
            self.step = 0
            self.prev_step = 0
            self.runner.reset()
            self.trigger_exec()
            return True
        self._kb_events.append(
            (
                "down",
                time(),
                Key(keycode=keycode[0], key=keycode[1], text=text),
                modifiers,
            )
        )

    def sandbox_on_key_up(self, widget, window, keycode):
        self._kb_events.append(
            ("up", time(), Key(keycode=keycode[0], key=keycode[1], text=None), None)
        )

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
        if status == "ERROR":
            exc = self.status[1]
            exc_name = exc.__class__.__name__ if exc else "Unknown Error"
            self.status_text = f"[b][color=f92672]{exc_name}[/color]: [/b]"
            if isinstance(exc, SyntaxError):
                code = (exc.text or self.code.splitlines()[exc.lineno - 1]).replace(
                    "\n", "⏎"
                )  # .replace('\t', ' ' * 4).replace(' ', '˽')
                pos = exc.offset - 1
                code_before = escape_markup(code[:pos].lstrip())
                code_hl = escape_markup(code[pos].replace(" ", "_"))
                code_after = (
                    escape_markup(code[pos + 1 :].rstrip()) if len(code) > pos else ""
                )
                code = f"[color=e6db74]{code_before}[/color][b][color=f92672]{code_hl}[/color][/b][color=e6db74]{code_after}[/color]"
                self.status_text += f"{escape_markup(exc.msg)}: {code}"
            else:
                msg = str(exc) or "???"
                self.status_text += escape_markup(msg)
        elif status == "BREAK":
            self.status_text = "[color=e6db74][b]Break[/b][/color]"
        elif status == "EXEC":
            pass
        elif status == "COMPLETE":
            self.status_text = "[color=a6e22e][b]Completed[/b][/color]"
        elif status == "RUN":
            self.status_text = "[color=a6e22e][b]Run[/b][/color]"
        else:
            pass

    def update_sandbox(self, redraw=True):
        from difflib import SequenceMatcher

        matcher_opcodes = None
        tex_coords_fill = tex_coords_stroke = 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0

        def update_texture(image, texture):
            if image.source in self.sandbox.image_meshes:
                for mesh in self.sandbox.image_meshes[image.source]:
                    mesh.texture = texture
            # else:
            #     print('DEL', image, image.source)
            #     del image

        def match_lines(lines):
            match = []
            for lineno in lines:
                lineno -= 1
                for tag, i1, i2, j1, j2 in matcher_opcodes:
                    if i1 <= lineno < i2:
                        if tag in ("equal", "replace"):
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
                matcher = SequenceMatcher(lambda x: x in " \t", new_code, old_code)
                matcher_opcodes = matcher.get_opcodes()
                self.sandbox.code = new_code

            for shape in Shape.get_instances(True):
                shape_ids.append(id(shape))
                render_shape = (
                    None  # if redraw else self.sandbox.shapes_by_id.get(id(shape))
                )
                shape_trace = (
                    (shape._trace, shape._trace_iter) if shape._trace else None
                )
                if shape_trace is not None:
                    shape_traces.append(shape_trace)

                if redraw:
                    if shape_trace is not None:
                        # print('shape_trace', shape_trace)
                        # print('shape_trace matched:', match_lines(shape_trace[0]))
                        mathed_trace = match_lines(shape_trace[0]), shape_trace[1]
                        # print('self.sandbox.shapes_by_trace.keys()', self.sandbox.shapes_by_trace.keys())
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
                render_context = None
                vfmt = None
                mesh = None
                vertices = []
                indices = None
                if shape.stroke is not None and shape.stroke.fill is not None:
                    if isinstance(shape.stroke.fill, OurImage):
                        stroke = 1, 1, 1, 1
                        source = shape.stroke.fill.source
                        image_stroke = self.sandbox.images.get(source)
                        if not image_stroke:
                            image_stroke = Image(
                                source=source,
                                mipmap=True,
                                anim_delay=shape.stroke.fill.anim_delay,
                            )
                            self.sandbox.images[source] = image_stroke
                            # self.sandbox.images[shape.fill.source] = image
                            # image.bind(texture=update_texture)  # TODO: Animation
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
                        if (
                            not image_fill
                        ):  # shape.fill.source not in self.sandbox.images:
                            image_fill = Image(
                                source=source,
                                mipmap=True,
                                anim_delay=shape.fill.anim_delay,
                            )
                            self.sandbox.images[source] = image_fill
                            # self.sandbox.images[shape.fill.source] = image
                            # image_fill.bind(texture=update_texture)
                            # TODO: If not exists (image_stroke.texture is None)...
                        texture_fill = image_fill.texture
                        if texture_fill is not None:
                            # texture_fill = texture_fill.get_region(0,0,186,186)
                            shape_width, shape_height = shape.size
                            shape_ratio = shape_width / shape_height
                            texture_width, texture_height = texture_fill.size
                            texture_ratio = texture_width / texture_height
                            if shape_ratio != texture_ratio:
                                if shape_ratio > texture_ratio:
                                    texture_height = texture_width / shape_ratio
                                else:
                                    texture_width = texture_height * shape_ratio
                                texture_x = (texture_fill.width - texture_width) / 2
                                texture_y = (texture_fill.height - texture_height) / 2
                                # print('TEX', texture_x, texture_y, texture_width, texture_height)
                                texture_fill = texture_fill.get_region(
                                    texture_x, texture_y, texture_width, texture_height
                                )
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
                tr = 1, 0, 0, 1, 0, 0
                if shape.transform:
                    tr = tuple(shape.transform)
                    # x += tr[4]
                    # y += tr[5]
                    # tr = tr[:4]

                if isinstance(shape, Circle):
                    render_context = self.sandbox.rc1
                    vfmt = self.sandbox.rc1_vfmt

                    x, y = shape.center
                    a = b = shape.radius * 2
                    a1 = radians(shape.angle_start)
                    a2 = radians(shape.angle_end)
                    v_attrs = x + tr[4], y + tr[5], w, *stroke, *fill, a1, a2, *tr[:4]
                    v0 = (
                        -a,
                        -b,
                        *v_attrs,
                        *tex_coords_fill[0:2],
                        *tex_coords_stroke[0:2],
                    )
                    v1 = (
                        -a,
                        +b,
                        *v_attrs,
                        *tex_coords_fill[6:8],
                        *tex_coords_stroke[6:8],
                    )
                    v2 = (
                        +a,
                        -b,
                        *v_attrs,
                        *tex_coords_fill[2:4],
                        *tex_coords_stroke[2:4],
                    )
                    v3 = (
                        +a,
                        +b,
                        *v_attrs,
                        *tex_coords_fill[4:6],
                        *tex_coords_stroke[4:6],
                    )
                    vertices = v0 + v1 + v2 + v3
                    indices = [0, 1, 2, 3]

                elif isinstance(shape, Rectangle):
                    render_context = self.sandbox.rc2
                    vfmt = self.sandbox.rc2_vfmt

                    x, y = shape.corner
                    a, b = shape.size
                    r = shape.radius
                    v_attrs = x + tr[4], y + tr[5], r, w, *stroke, *fill, *tr[:4]
                    v0 = (
                        -a,
                        -b,
                        *v_attrs,
                        *tex_coords_fill[0:2],
                        *tex_coords_stroke[0:2],
                    )
                    v1 = (
                        -a,
                        +b,
                        *v_attrs,
                        *tex_coords_fill[6:8],
                        *tex_coords_stroke[6:8],
                    )
                    v2 = (
                        +a,
                        -b,
                        *v_attrs,
                        *tex_coords_fill[2:4],
                        *tex_coords_stroke[2:4],
                    )
                    v3 = (
                        +a,
                        +b,
                        *v_attrs,
                        *tex_coords_fill[4:6],
                        *tex_coords_stroke[4:6],
                    )
                    vertices = v0 + v1 + v2 + v3
                    indices = [0, 1, 2, 3]

                elif isinstance(shape, OurLine):
                    render_context = self.sandbox.rc3
                    vfmt = self.sandbox.dls._vbuffer.vfmt

                    self.sandbox.dls.clear()
                    self.sandbox.dls.append(
                        shape.points,
                        translate=(tr[4], tr[5]),
                        color=(1, 0, 0, 1),
                        linewidth=w + 2,
                        dash_pattern="solid",
                    )
                    vertices = self.sandbox.dls.vertices.tolist()
                    indices = self.sandbox.dls.indices.tolist()
                    self.sandbox.dls.bind_textures()

                else:
                    raise NotImplementedError

                new_shapes.append(
                    (
                        id(shape),
                        render_context,
                        vfmt,
                        texture_stroke,
                        texture_fill,
                        vertices,
                        indices,
                        "triangle_strip",
                        (image_fill,),
                    )
                )

                with render_context:
                    BindTexture(texture=texture_stroke, index=1)
                if render_shape is None:  # id(shape) not in self.sandbox.shapes:
                    with render_context:
                        # BindTexture(texture=texture_stroke, index=1)
                        if redraw and self.sandbox.transition_time > 0:
                            v_len = sum([v[1] for v in vfmt])  # len(vertices) // 4
                            # Set initial size to 0, 0
                            initial_vs = tuple(
                                0 if i % v_len in (0, 1) else v
                                for i, v in enumerate(vertices)
                            )
                        else:
                            initial_vs = vertices
                        mesh = Mesh(
                            fmt=vfmt,
                            mode="triangle_strip",
                            vertices=initial_vs,
                            indices=indices,
                            texture=texture_fill,
                        )
                    self.sandbox.shapes_by_id[id(shape)] = ((render_context, (mesh,)),)
                    if image_fill is not None:
                        self.sandbox.image_meshes[image_fill.source].append(mesh)
                else:
                    mesh = render_shape[0][1][0]
                    mesh.texture = texture_fill

                if shape_trace:
                    self.sandbox.shapes_by_trace[shape_trace] = (
                        (render_context, (mesh,)),
                    )

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
                    # FIXME: Fails when line becomes shorter on animation.py:
                    #        361: return tp([_calculate(a[x], b[x], t) for x in range(len(a))])
                    anim = Animation(
                        vertices=vertices,
                        t=self.sandbox.transition_out,
                        duration=self.sandbox.transition_time,
                    )
                    anim.bind(on_complete=_on_complete)
                    anim.start(mesh)
                else:
                    mesh.vertices = vertices

            def remove_garbage(shapes):
                for shape in shapes:  # chain(shapes.get(oid, ()) for oid in oids):
                    for context, instructions in shape:
                        for inst in instructions:
                            context.remove(
                                inst
                            )  # FIXME: May hang there when removing objects in editor

            to_remove = old_shapes - set(
                [shape for shape in self.sandbox.shapes_by_id.values()]
            )

            for shape_trace in set(self.sandbox.shapes_by_trace) - set(shape_traces):
                self.sandbox.shapes_by_trace.pop(shape_trace)

            if self.sandbox.transition_time > 0:
                for shape in to_remove:
                    for context, instructions in shape:
                        for mesh in instructions:
                            Animation.stop_all(mesh)
                            vertices = mesh.vertices
                            v_len = len(vertices) // 4
                            # Set initial size to 0, 0
                            initial_vs = tuple(
                                0 if i % v_len in (0, 1) else v
                                for i, v in enumerate(vertices)
                            )
                            Animation(
                                vertices=initial_vs,
                                t=self.sandbox.transition_in,
                                duration=self.sandbox.transition_time,
                            ).start(mesh)
                Clock.schedule_once(
                    lambda _: remove_garbage(to_remove),
                    self.sandbox.transition_time + 1 / FPS,
                )
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
            print("E at update_sandbox:")
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
                    with open("source.py", "w") as f:
                        f.write(self.code)
                except Exception as e:
                    print("Cannot save file:", e)
            if COMMON_CODE in changed:
                self.runner.reset()

            # self.runner.compile() # changed) FIXME
        except Exception as e:
            print("E:", e)
            print_exc()
            print("* " * 40)
            line_num = self.runner.exception_lineno(e)
            self.code_editor.highlight_line(None, "run")
            self.code_editor.highlight_line(line_num)
            self.status = ("ERROR", e)
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
                if (
                    self.runner.execute(changed)
                    and F_UPDATE in self.runner.globals
                    and not (self.update_schedule and self.update_schedule.is_triggered)
                ):
                    if self.trigger_exec_update:
                        self.trigger_exec_update.cancel()
                    if self.update_schedule is not None:
                        self.update_schedule.cancel()
                        self.update_schedule()
            except Exception as e:
                print("E3:", e)
                print_exc()
                self.status = ("ERROR", None)

    def execute_code(self, *largs):
        print("execute_code")
        self.status = ("EXEC",)
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
            self.sandbox.space.remove(
                *self.sandbox.space.shapes, *self.sandbox.space.bodies
            )
        self.sandbox.space = pymunk.Space()
        # self.sandbox.space.gravity = self._gravity
        self.sandbox.space.sleep_time_threshold = 3.0
        self.sandbox.space.replay_mode = False
        for widget in saved:
            print("SAVED:", widget)
            self.sandbox.add_widget(widget)
        self.sandbox.canvas.add(self.sandbox.rc1)
        self.sandbox.canvas.add(self.sandbox.rc2)
        self.sandbox.canvas.add(self.sandbox.rc3)

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
            print("E2:")
            print_exc()
        print("Exec Time:", process_time() - start)

        watches = ""
        for v, t, r in whos(self.runner.globals):
            watches += f'{v + " " * (8 - len(v))}  {r}\n'
            # watches += f'{v + " " * (8 - len(v))} {t + " " * (5 - len(t))}  {r}\n'

        if False:  # ok and F_UPDATE in self.runner.globals and self.prev_step > 0:
            # print('Replay:', prev_step)
            t_start = time()
            self._last_update_time = time() - self.prev_step * 1 / 30
            for i in range(self.prev_step):
                self.execute_update(0.0, True)
            Sprite.update_from_pymunk(False)
            print("Replay time:", (time() - t_start) * 1000, "ms")

        out = self.runner.text_stream.getvalue()
        self.console = out
        print("out:", out)
        print("- " * 40)

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
                print("EXC:", exc_str)
                is_break = isinstance(exc, Break)
                if is_break:
                    self.status = ("BREAK", exc)
                    hl_style = "run"
                    # self.code_editor.highlight_line(None, 'run')
                else:
                    self.status = ("ERROR", exc)
                    hl_style = "error"
                # print('Br Line:', self.runner.breakpoint)
                self.code_editor.highlight_line(None)
                self.code_editor.highlight_line(None, "run")
                # self.code_editor.highlight_line(self.runner.breakpoint, 'run')
                for filename, lineno, name, line, locals in traceback:
                    print(
                        "TRACE:", filename, lineno, name, repr(line), repr(locals)[:800]
                    )  # filename, lineno, name, line, locals)
                    # if filename == '<code-input>':
                    #     if name != '<module>':
                    watched_locals = whos(locals)
                    if watched_locals:
                        watches += f"== {name} ==\n"
                        for v, t, r in watched_locals:
                            watches += f'{v + " " * (8 - len(v))}  {r}\n'
                            # watches += f'{v}\t{t}\t{r}\n'
                        self.code_editor.highlight_line(lineno, hl_style, add=True)
                        # print('LINES +', lineno, self.code_editor._highlight)

            else:
                self.status = ("ERROR", None)
                print("Unhandled exception")
                self.code_editor.highlight_line(None)  # FIXME
        # else:
        # self.code_editor.highlight_line(None)
        else:
            self.status = ("COMPLETE",)
            self.code_editor.highlight_line(None)
            if F_UPDATE in self.runner.globals:
                self._last_update_time = time()
            # self.update_sandbox()
            Clock.schedule_once(lambda _: self.update_sandbox(), 0)
            # if self.sokoban and self.sokoban.boxes_remaining == 0:
            #     print('Level completed:', self.sokoban.level)
            #     self.sokoban.level += 1
            #     self.sandbox.clear_widgets()
            if (
                F_UPDATE in self.runner.globals
            ):  # and (not self.update_schedule or not self.update_schedule.is_triggered):
                # self.sandbox.transition_time = 0
                def run_update(*t):
                    self.update_schedule = Clock.schedule_interval(
                        self.trigger_exec_update, 1 / FPS
                    )

                self.update_schedule = Clock.schedule_once(
                    run_update, self.sandbox.transition_time
                )

        self.watches = watches
        print("= " * 40)

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
        self.runner.globals["step"] = self.step
        if not replay:
            ts_pos = self.runner.text_stream.tell()
            now = time()
        else:
            now = self._last_update_time + self.step * 1 / 30
        dt = now - self._last_update_time

        try:
            while self._kb_events:
                ev, t, key, modifiers = self._kb_events[0]
                if ev == "down":
                    self.runner.call_if_exists(F_ON_KEY_PRESS, key, modifiers)
                elif ev == "up":
                    self.runner.call_if_exists(F_ON_KEY_RELEASE, key)
                self._kb_events.pop(0)
            self.runner.call(F_UPDATE, dt)

        except Exception as e:
            print_exc()
            if self.update_schedule:
                self.update_schedule.cancel()
            if self.trigger_exec_update:
                self.trigger_exec_update.cancel()
            watches = ""
            for v, t, r in whos(self.runner.globals):
                # watches += f'{v}\t{t}\t{r}\n'
                watches += f'{v + " " * (8 - len(v))}  {r}\n'
            if self.runner.exception:
                exc, exc_str, traceback = self.runner.exception
            else:
                exc = e
                if hasattr(e, "message"):
                    exc_str = e.message
                else:
                    exc_str = str(e) or e.__class__.__name__
                traceback = self.runner._trace(e.__traceback__)
            print("EXC2:", exc_str)
            is_break = isinstance(exc, Break)
            if is_break:
                self.status = ("BREAK", exc)
                hl_style = "run"
            else:
                self.status = ("ERROR", exc)
                hl_style = "error"
            # print('E4', e)
            # lineno = self.runner.exception_lineno(e)
            self.code_editor.highlight_line(None)
            self.code_editor.highlight_line(None, "run")
            # self.code_editor.highlight_line(self.runner.breakpoint, 'run')
            for filename, lineno, name, line, locals in traceback:
                print(
                    "TRACE:", filename, lineno, name, repr(line), repr(locals)[:80]
                )  # filename, lineno, name, line, locals)
                if filename == "<code-input>":
                    if name != "<module>":
                        watched_locals = whos(locals)
                        if watched_locals:
                            watches += f"== {name} ==\n"
                            for v, t, r in watched_locals:
                                # watches += f'{v}\t{t}\t{r}\n'
                                watches += f'{v + " " * (8 - len(v))}  {r}\n'
                    self.code_editor.highlight_line(lineno, hl_style, add=True)
            self.watches = watches
        else:
            self._last_update_time = now
            self.sandbox.space.step(1 / FPS / 2)
            self.sandbox.space.step(1 / FPS / 2)
            if not replay:
                Sprite.update_from_pymunk()
                self.update_sandbox(False)
                if self.status[0] != "RUN":
                    self.status = ("RUN",)

        if not replay:
            self.runner.text_stream.seek(ts_pos)
            out = self.runner.text_stream.read()
            if out:
                print(out)
                print("* " * 20)
                self.console = (self.console + out)[-3000:]
