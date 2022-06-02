from collections import defaultdict

from kivy.uix.behaviors import FocusBehavior
from kivy.uix.image import Image
from kivy.uix.scatter import ScatterPlane
from kivy.graphics import Color, Rectangle, RenderContext, Mesh, ClearBuffers, ClearColor, Callback, Fbo
from kivy.graphics.transformation import Matrix
from kivy.properties import ObjectProperty
from kivy.clock import Clock
from kivy.graphics.opengl import glEnable, glDisable  # glFinish
from kivy.resources import resource_find, resource_paths

from playground import utils
from rougier.dash_lines_2D import DashLines
from playground.color import Color as OurColor


GL_VERTEX_PROGRAM_POINT_SIZE = 34370
GL_FRAMEBUFFER_SRGB_EXT = 36281

TRANSITION_TIME = 0.4  # FIXME: Import from common config

utils.resource_paths = resource_paths


class OurSandbox(FocusBehavior, ScatterPlane):

    texture = ObjectProperty(None)

    def __init__(self, **kwargs):
        # self.canvas = RenderContext()  # ?
        self.canvas = RenderContext(
            use_parent_projection=True, use_parent_modelview=True, use_parent_frag_modelview=True
        )

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
        self.canvas.shader.fs = open(resource_find("srgb_to_linear.glsl")).read()
        super(OurSandbox, self).__init__(**kwargs)
        # self.canvas.shader.source = resource_find('shader2.glsl')
        self.rc1 = RenderContext(use_parent_modelview=True, use_parent_projection=True)
        self.rc2 = RenderContext(use_parent_modelview=True, use_parent_projection=True)
        self.rc1.shader.source = resource_find("shader1.glsl")
        self.rc2.shader.source = resource_find("shader2.glsl")
        self.rc1["texture1"] = 1
        self.rc2["texture1"] = 1

        self.dls = DashLines()
        self.rc3 = self.dls.context
        # lw = 20
        # x0,y0 = 500.0, 500.0
        # coils = 3 #12
        # rho_max = 450.
        # theta_max = coils * 2 * pi
        # rho_step = rho_max / theta_max

        # P=[]
        # chord = 1
        # theta = 1 + chord / rho_step
        # while theta <= theta_max:
        #     rho = rho_step * theta
        #     x = rho * cos( theta )
        #     y = rho * sin( theta )
        #     P.append( (x,y) )
        #     theta += chord / rho
        #     chord += .05

        # self.dls.append(P, translate=(x0,y0),
        #                   color=(1,0,0,1), linewidth=lw+2, dash_pattern = 'solid')
        # self.dls.draw()

        # x, y, w, *stroke, *fill, a1, a2, *tr
        self.rc1_vfmt = (
            (b"size", 2, "float"),
            (b"center", 2, "float"),
            (b"width", 1, "float"),
            (b"stroke", 4, "float"),
            (b"fill", 4, "float"),
            (b"angle_start", 1, "float"),
            (b"angle_end", 1, "float"),
            (b"transform", 4, "float"),
            (b"tex_coords0", 2, "float"),
            (b"tex_coords1", 2, "float"),
        )
        self.rc2_vfmt = (
            (b"size", 2, "float"),
            (b"center", 2, "float"),
            (b"radius", 1, "float"),
            (b"width", 1, "float"),
            (b"stroke", 4, "float"),
            (b"fill", 4, "float"),
            (b"transform", 4, "float"),
            (b"tex_coords0", 2, "float"),
            (b"tex_coords1", 2, "float"),
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

        self.register_event_type("on_key_down")
        self.register_event_type("on_key_up")
        self.space = None

    def setup_gl_context(self, *args):
        glEnable(GL_FRAMEBUFFER_SRGB_EXT)

    def reset_gl_context(self, *args):
        glDisable(GL_FRAMEBUFFER_SRGB_EXT)

    def update_shader(self, *largs):
        # self.canvas['projection_mat'] = Window.render_context['projection_mat']
        for rc in [self.rc1, self.rc2, self.canvas]:  # , self.fbo]: #
            rc["modelview_mat"] = self.transform  # Window.render_context['modelview_mat']
            rc["resolution"] = list(map(float, self.size))
            rc["time"] = Clock.get_boottime()
            rc["scale"] = self.transform[0]
            rc["texture1"] = 1
        # self.rc3['modelview_mat'] = self.transform
        # self.rc3['resolution'] = list(map(float, self.size))
        # self.rc3['scale'] = self.transform[0]
        self.dls.set_uniforms(
            {
                "modelview_mat": self.transform,
                # 'resolution': list(map(float, self.size)),
                "u_scale": self.transform[0],
            }
        )
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
            stroke = (
                *OurColor(random() * 30 + 40, 50, random() * 360, mode="Jsh").linear_srgb,
                random(),
            )  # random(), random(), random(), random()
            fill = (
                *OurColor(random() * 30 + 40, 50, random() * 360, mode="Jsh").linear_srgb,
                random(),
            )  # # random(), random(), random(), random()
            a1 = 2 * pi * random()
            a2 = 2 * pi * random()
            v0 = x, y, +a, -b, w, *stroke, *fill, a1, a2, *tr, *tex_coords[0:2]
            v1 = x, y, -a, -b, w, *stroke, *fill, a1, a2, *tr, *tex_coords[2:4]
            v2 = x, y, -a, +b, w, *stroke, *fill, a1, a2, *tr, *tex_coords[4:6]
            v3 = x, y, +a, +b, w, *stroke, *fill, a1, a2, *tr, *tex_coords[6:8]
            vertices = v0 + v1 + v2 + v3
            meshes.append(
                Mesh(
                    fmt=self.rc1_vfmt,
                    mode="triangle_strip",
                    vertices=vertices,
                    indices=indices,
                    texture=grace_hopper.texture,
                )
            )
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
            stroke = (
                *OurColor(random() * 30 + 40, 50, random() * 360, mode="Jsh").linear_srgb,
                random(),
            )  # random(), random(), random(), random()
            fill = (
                *OurColor(random() * 30 + 40, 50, random() * 360, mode="Jsh").linear_srgb,
                random(),
            )  # # random(), random(), random(), random()
            v0 = x, y, -a, -b, r, w, *stroke, *fill, *tr
            v1 = x, y, -a, +b, r, w, *stroke, *fill, *tr
            v2 = x, y, +a, +b, r, w, *stroke, *fill, *tr
            v3 = x, y, +a, -b, r, w, *stroke, *fill, *tr
            vertices = v0 + v1 + v2 + v3
            meshes.append(Mesh(fmt=self.rc2_vfmt, mode="triangle_strip", vertices=vertices, indices=indices))
        return meshes

    def keyboard_on_key_down(self, window, keycode, text, modifiers):
        """We call super before doing anything else to enable tab cycling
        by FocusBehavior. If we wanted to use tab for ourselves, we could just
        not call it, or call it if we didn't need tab.
        """
        print("DOWN", keycode, text, modifiers, "\n", self.transform)

        if keycode[1] == "f2":
            self.rc1.shader.source = resource_find("shader1.glsl")
            self.rc2.shader.source = resource_find("shader2.glsl")
            self.rc3.shader.source = resource_find("dash-lines-2D.glsl")
            self.canvas.shader.fs = open(resource_find("srgb_to_linear.glsl")).read()
            print("projection_mat\n", self.canvas["projection_mat"])
            print("modelview_mat\n", self.canvas["modelview_mat"])
            print("frag_modelview_mat\n", self.canvas["frag_modelview_mat"])
        elif self.dispatch("on_key_down", window, keycode, text, modifiers):
            return True
        return super(OurSandbox, self).keyboard_on_key_down(window, keycode, text, modifiers)

    def on_key_down(self, window, keycode, text, modifiers):
        pass

    def keyboard_on_key_up(self, window, keycode):
        if self.dispatch("on_key_up", window, keycode):
            return True
        return super(OurSandbox, self).keyboard_on_key_up(window, keycode)

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
            scale = self.scale + (0.05 if touch.button == "scrolldown" else -0.05)
            if (self.scale_min and scale < self.scale_min) or (self.scale_max and scale > self.scale_max):
                return
            rescale = scale * 1.0 / self.scale
            self.apply_transform(
                Matrix().scale(rescale, rescale, rescale), post_multiply=True, anchor=self.to_local(*touch.pos)
            )
            self.update_shader()
            return self.dispatch("on_transform_with_touch", touch)
        return super().on_touch_down(touch)
