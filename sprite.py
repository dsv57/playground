#!/usr/bin/python3

from math import *
from random import *
from os import stat
from filecmp import _sig as file_sig
import pickle
# import hashlib

import kivy
# kivy.require('1.9.1')

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.uix.label import Label
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.scatter import Scatter
from kivy.uix.scatterlayout import ScatterLayout
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.image import Image  # as ImageWidget
from kivy.clock import Clock
from kivy.graphics import Ellipse, Line, Color, Triangle, Quad, Rectangle, \
    Mesh, PushMatrix, PopMatrix
from kivy.graphics.tesselator import Tesselator, WINDING_ODD, TYPE_POLYGONS
from kivy.graphics.context_instructions import Rotate, Translate, Scale
from kivy.core.window import Window
# from kivy.core.image import Image as CoreImage
from kivy.properties import StringProperty, NumericProperty, \
    ListProperty, ObjectProperty, AliasProperty

import pymunk
import pymunk.autogeometry
from pymunk.vec2d import Vec2d

from named_colors import COLORS

# from kivy.uix.image import Image


def trace_image(img, threshold=3, simplify_tolerance=0.7, cache=True):
    lines = None
    trace_sig = file_sig(
        stat(img.filename)), img.size, threshold, simplify_tolerance
    print('IMG', img.width, img.height)
    print('SIG', trace_sig, img.filename)
    if cache:
        try:
            with open(img.filename + '.contour', 'rb') as f:
                lines, fsig = pickle.load(f)
                if fsig == trace_sig:
                    return lines
        except:
            pass

    bb = pymunk.BB(0, 0, img.width - 1, img.height - 1)

    def sample_func(point):
        try:
            color = img.read_pixel(point.x, point.y)
            if point.y == 120:
                print(point.x, sum(color[:3]) / 3 * 255)
            return sum(color[:3]) / 3 * 255  # color[3]*255
        except Exception as e:
            print(e)  # FIXME
            return 0

    line_set = pymunk.autogeometry.PolylineSet()

    def segment_func(v0, v1):
        line_set.collect_segment(v0, v1)

    pymunk.autogeometry.march_soft(bb, img.width, img.height, threshold,
                                   segment_func, sample_func)

    lines = []
    for line in line_set:
        line = pymunk.autogeometry.simplify_curves(line, simplify_tolerance)

        max_x = 0
        min_x = 1000
        max_y = 0
        min_y = 1000
        for l in line:
            max_x = max(max_x, l.x)
            min_x = min(min_x, l.x)
            max_y = max(max_y, l.y)
            min_y = min(min_y, l.y)
        w, h = max_x - min_x, max_y - min_y

        center = Vec2d(min_x + w / 2., min_y + h / 2.)
        # t = pymunk.Transform(a=1.0, d=1.0, tx=-center.x, ty=-center.y)

        line = [Vec2d(l.x, img.height - l.y) for l in line]
        lines.append(line)
        # return lines
        # print('Lines:', len(lines))
    if cache and lines:
        try:
            with open(img.filename + '.contour', 'wb') as f:
                pickle.dump((lines, trace_sig), f)
        except:
            pass

    print('lines', len(lines), [len(l) for l in lines])
    return lines

    # for line in lines:
    #     for i in range(len(line)-1):
    #         shape = pymunk.Segment(self.space.static_body, line[i], line[i+1], 1)
    #         shape.friction = friction
    #         self.space.add(shape)
    #         if show_geometry:
    #             self.draw(shape, (1,0,0))

    # return lines


def ellipse_from_circle(shape):
    pos = shape.body.position - (shape.radius, shape.radius) + shape.offset
    e = Ellipse(pos=pos, size=[shape.radius * 2, shape.radius * 2])
    circle_edge = shape.body.position + shape.offset + Vec2d(
        shape.radius, 0).rotated(shape.body.angle)
    Color(*COLORS['dark slate gray'])  # (.17,.24,.31)
    line = Line(points=[
        shape.body.position.x + shape.offset.x, shape.body.position.y +
        shape.offset.y, circle_edge.x, circle_edge.y
    ])
    return e, line


def points_from_poly(shape):
    body = shape.body
    ps = [p.rotated(body.angle) + body.position for p in shape.get_vertices()]
    vs = []
    for p in ps:
        vs += [p.x, p.y]
    return vs


def vertices_from_poly(shape):
    body = shape.body
    ps = [p.rotated(body.angle) + body.position for p in shape.get_vertices()]
    vs = []
    for p in ps:
        vs += [p.x, p.y, 0, 0]
    return vs


def points_from_segment(segm):
    return [segm.a.x, segm.a.y, segm.b.x, segm.b.y]


def points_from_constraint(cons):
    if isinstance(cons, pymunk.constraint.RatchetJoint):
        return []
    if isinstance(cons, pymunk.constraint.GrooveJoint):
        return []
    if isinstance(cons, pymunk.constraint.RotaryLimitJoint):
        return []
    if isinstance(cons, pymunk.constraint.DampedRotarySpring):
        return []
    else:
        p1 = cons.anchor_a.rotated(cons.a.angle) + cons.a.position
        p2 = cons.anchor_b.rotated(cons.b.angle) + cons.b.position
        return [p1.x, p1.y, p2.x, p2.y]


def draw_pymunk_shape(canvas, shape, color=None):
    with canvas:
        if color:
            Color(*color)
        else:
            Color(1., 1., 1.)
        if isinstance(shape, pymunk.Circle):
            return ellipse_from_circle(shape)
        if isinstance(shape, pymunk.Segment):
            return Line(points=points_from_segment(shape), width=shape.radius)
        elif isinstance(shape, pymunk.Poly):
            vs = shape.get_vertices()
            if len(vs) == 3:
                return Triangle(points=points_from_poly(shape))
            elif len(vs) == 4:
                return Quad(points=points_from_poly(shape))
            else:
                return Mesh(
                    vertices=vertices_from_poly(shape),
                    indices=range(len(vs)),
                    mode="triangle_fan"
                )  # Line(points=self.points_from_poly(shape), width=3)
        elif isinstance(shape, pymunk.constraint.Constraint):
            points = points_from_constraint(shape)
            if len(points) > 0:
                shape.kivy = Line(points=points)


class Sprite(Scatter):

    _instances = set()

    _shapes = {
        "arrow": {
            'type': 'polygon',
            'points': [(-10, 0), (10, 0), (0, 10)]
        },
        "turtle": {
            'type':
            'polygon',
            'points': [(0, 16), (-2, 14), (-1, 10), (-4, 7), (-7, 9), (-9, 8),
                       (-6, 5), (-7, 1), (-5, -3), (-8, -6), (-6, -8), (-4,
                                                                        -5),
                       (0, -7), (4, -5), (6, -8), (8, -6), (5, -3), (7, 1),
                       (6, 5), (9, 8), (7, 9), (4, 7), (1, 10), (2, 14)]
        },
        "circle": {
            'type': 'circle',
            'radius': 10
        },
        "square": {
            'type': 'polygon',
            'points': [(10, -10), (10, 10), (-10, 10), (-10, -10)]
        },
        "triangle": {
            'type': 'polygon',
            'points': [(10, -5.77), (0, 11.55), (-10, -5.77)]
        },
        "classic": {
            'type': 'polygon',
            'points': [(0, 0), (-5, -9), (0, -7), (5, -9)]
        }
    }

    space = ObjectProperty(None)

    def __init__(self, source, x=0, y=0, scale=1, **kwargs):
        super(Sprite, self).__init__()

        self.body = pymunk.Body(body_type=pymunk.Body.KINEMATIC)

        self.source = source
        self.bottom_left = (0, 0)
        if source not in self._shapes:
            self._type = 'image'
            imag = Image(
                source=source, keep_data=True,
                anim_delay=0.05)  # , pos_hint={'center_x': 0, 'center_y': 0})
            self.add_widget(imag)
            imag.size = imag.texture_size
            self.size = imag.texture_size
            # print(222222, imag.texture_size, imag.norm_image_size)
            self.shapes = [imag]
            self.pymunk_shapes = []
            self.bottom_left = (0, 0)
            # TODO: self.bottom_left

            lines = trace_image(imag._coreimage, cache=False)
            with self.canvas:
                for line in lines:
                    poly = pymunk.Poly(self.body, line)
                    # self.parent.space.add(poly)
                    self.pymunk_shapes.append(poly)

                    Color(1, 0, 0)
                    self.shapes.append(
                        Line(points=[
                            coord for point in line for coord in point
                        ]))
                    print("LL", len(line), max([p.x for p in line]),
                          min([p.x for p in line]), max([p.y for p in line]),
                          min([p.y for p in line]))
                    # for i in range(len(line)-1):
                    #     shape = pymunk.Segment(self.body, line[i], line[i+1], 1)
                    #     self.parent.space.add(shape)
            print('------------')
        else:
            shape = self._shapes[source]
            self._type = shape['type']
            self.shapes = []
            self.pymunk_shapes = []

            xs, ys = tuple(zip(*shape['points']))
            min_x = min(xs)
            min_y = min(ys)
            self.size = (max(xs) - min_x, max(ys) - min_y)
            self.bottom_left = (min_x, min_y)
            # self.pos = self.bottom_left
            self.position = (0, 0)  # (x, y)
            # print(11111, self.position, self.bottom_left, self.pos)

            tess = Tesselator()
            tess.add_contour([
                coord for point in shape['points'] for coord in point
            ])  # (point[0]+x,point[1]+y)]) #shape['points'])
            tess.tesselate(WINDING_ODD, TYPE_POLYGONS)
            with self.canvas:
                PushMatrix()
                Translate(-min_x, -min_y)
                Color(1, 0, 0)
                for vertices, indices in tess.meshes:
                    self.shapes.append(
                        Mesh(
                            vertices=vertices,
                            indices=indices,
                            mode="triangle_fan"))

                    vs = []
                    for i in range(len(vertices) // 4):
                        vs.append((vertices[4 * i], vertices[4 * i + 1]))
                    poly = pymunk.Poly(self.body, vs)
                    # self.parent.space.add(poly)
                    self.pymunk_shapes.append(poly)
                PopMatrix()

        self._register()

    # def draw(self):
    #     with self.canvas:
    #         for shape in self._shapes:
    #             self.add_widget(shape)
    def on_parent(self, instance, value):
        print('on_parent', instance, value)
        if value and value is not Sprite.space:
            Sprite.space = self.parent.space
            for shape in self.pymunk_shapes:
                print("OP", shape)
                Sprite.space.add(shape)
        # return super().on_parent(instance, value)

    def collide_point(self, x, y):
        # print('bbox', self.bbox)
        if super().collide_point(x, y):
            for shape in self.pymunk_shapes:
                pq = shape.point_query((x, y))
                if pq[0] < 0:
                    return True
        return False  # super().collide_point(x, y) #False

    def on_position(self, instance, position):
        print('On position', position)
        self.position = position
        # self.body.position = position # self.to_parent(*pos)
        # if self.space:
        #     self.space.reindex_shapes_for_body(self.body)
        # print("POS", pos, self.to_local(*pos), self.to_parent(*pos))
        # if Sprite.space:
        #     pq = Sprite.space.point_query((x, y), 0.0, pymunk.ShapeFilter())
        #     print('PQ:', pq)
        #     return pq

    # def on_touch_down(self, touch):
    #     print(touch)
    #     print(self.collide_point(*touch.pos))

    #         # return self.dispatch('on_transform_with_touch', touch)
    #     return super().on_touch_down(touch)

    def _get_position(self):
        return (self.bbox[0][0] - self.bottom_left[0],
                self.bbox[0][1] - self.bottom_left[1])

    def _set_position(self, position):
        self.body.position = position
        if self.space:
            self.space.reindex_shapes_for_body(self.body)
        return super()._set_pos((position[0] + self.bottom_left[0],
                                 position[1] + self.bottom_left[1]))

    position = AliasProperty(_get_position, _set_position, bind=('bbox', ))

    def _get_xcor(self):
        return self.bbox[0][0] - self.bottom_left[0]

    def _set_xcor(self, x):
        return self._set_x(x + self.bottom_left[0])

    xcor = AliasProperty(_get_xcor, _set_xcor, bind=('bbox', ))

    def _get_ycor(self):
        return self.bbox[0][1] - self.bottom_left[1]

    def _set_ycor(self, y):
        return self._set_y(y + self.bottom_left[1])

    ycor = AliasProperty(_get_ycor, _set_ycor, bind=('bbox', ))

    def _set_pos(self, pos):
        return self._set_position(pos[0] - self.bottom_left[0],
                                  pos[1] - self.bottom_left[1])

    def _set_x(self, x):
        self.body.position.x = x + self.bottom_left[0]
        if self.space:
            self.space.reindex_shapes_for_body(self.body)
        return super()._set_x(x)

    def _set_y(self, y):
        self.body.position.y = y + self.bottom_left[1]
        if self.space:
            self.space.reindex_shapes_for_body(self.body)
        return super()._set_y(y)

    def _register(self):
        Sprite._instances.add(self)

    def _unregister(self):
        if self in Sprite._instances:
            Sprite._instances.remove(self)

    @staticmethod
    def clear_sprites():
        for sprite in Sprite._instances:
            Sprite.space.remove(*sprite.pymunk_shapes)
        # TODO: Remove bodies
        Sprite._instances = set()


# #class Sandbox(RelativeLayout):
# class Sandbox(ScatterLayout):

# #    def __init__(self, **kwargs):
# #        super(Sandbox, self).__init__(**kwargs)
# #
# #        self.running = True

#     def __init__(self, **kwargs):
#         super(Sandbox, self).__init__(**kwargs)
#         self._target_scale = 1
#         self._keyboard = Window.request_keyboard(
#             self._keyboard_closed, self, 'text')
#         if self._keyboard.widget:
#             pass
#         self._keyboard.bind(on_key_down=self.on_key_down)

#     def _keyboard_closed(self):
#         self._keyboard.unbind(on_key_down=self.on_key_down)
#         self._keyboard = None

#     def on_key_down(self, keyboard, keycode, text, modifiers):
#         return True

#     def make_segment(self, p1, p2, width=1, density=0.4, friction=1, elasticity=0.1, color='gray20', body=None):
#         if isinstance(color, str):
#             color = COLORS[color]
#         if body is None:
#             body = pymunk.Body()
#             self.space.add(body)
#         elif isinstance(body, pymunk.Shape):
#             body = body.body

#         s = pymunk.Segment(body, p1, p2, width)
#         s.density = density
#         s.friction = 1.0
#         s.elasticity = elasticity
#         self.space.add(s)
#         self.draw(s, color)

#         return s

#     def make_box(self, size=(40,20), density=0.4, friction=1, elasticity=0.1, color='gray20', position=None, body=None):
#         if isinstance(color, str):
#             color = COLORS[color]
#         if body is None:
#             body = pymunk.Body()
#             if position is not None:
#                 body.position = position
#             self.space.add(body)
#         elif isinstance(body, pymunk.Shape):
#             body = body.body
#         s = pymunk.Poly.create_box(body, size)
#         s.density = density
#         s.friction = friction
#         s.elasticity = elasticity
#         self.space.add(s)
#         self.draw(s, color)
#         return s

#     def make_poly(self, vs, density=0.4, friction=1, elasticity=0.1, color='sienna3', position=None, body=None):
#         if isinstance(color, str):
#             color = COLORS[color]
#         tr = None
#         if body is not None and position is not None:
#             tr = pymunk.Transform(tx=position[0], ty=position[1])
#         if body is None:
#             body = pymunk.Body()
#             if position is not None:
#                 body.position = position
#             self.space.add(body)
#         elif isinstance(body, pymunk.Shape):
#             body = body.body
#         s = pymunk.Poly(body, vs, transform=tr)
#         s.density = density
#         s.friction = friction
#         s.elasticity = elasticity
#         self.space.add(s)
#         self.draw(s, color)
#         return s

#     def make_spring(self, a, b, anchor_a=(0,0), anchor_b=(0,0), rest_length=None, stiffness=100, damping=10):
#         a = a.body if isinstance(a, pymunk.Shape) else a
#         b = b.body if isinstance(b, pymunk.Shape) else b

#         if rest_length is None:
#             rest_length = (a.position+anchor_a).get_distance(b.position+anchor_b) * 0.9

#         spring = pymunk.DampedSpring(a, b, anchor_a, anchor_b, rest_length, stiffness, damping)
#         self.space.add(spring)
#         self.draw(spring)
#         return spring

#     def make_pin_joint(self, a, b, anchor_a=(0,0), anchor_b=(0,0)):
#         a = a.body if isinstance(a, pymunk.Shape) else a
#         b = b.body if isinstance(b, pymunk.Shape) else b

#         joint = pymunk.PinJoint(a, b, anchor_a, anchor_b)
#         self.space.add(joint)
#         self.draw(joint)
#         return joint

#     def make_motor(self, a, b, speed=-4, max_force=300*1000*1000):
#         a = a.body if isinstance(a, pymunk.Shape) else a
#         b = b.body if isinstance(b, pymunk.Shape) else b
#         motor = pymunk.SimpleMotor(a, b, speed)
#         motor.max_force = max_force
#         self.space.add(motor)
#         return motor

#     def make_ball(self, radius=25, density=0.2, friction=1.5, elasticity=0.99, color='sienna3', position=None, body=None):
#         if isinstance(color, str):
#             color = COLORS[color]
#         offset = (0, 0)
#         if body is not None and position is not None:
#             offset = position
#         if body is None:
#             body = pymunk.Body()
#             if position is not None:
#                 body.position = position
#             self.space.add(body)
#         elif isinstance(body, pymunk.Shape):
#             body = body.body
#         ball_s = pymunk.Circle(body, radius, offset)
#         ball_s.density = density
#         ball_s.friction = friction
#         ball_s.elasticity = elasticity
#         ball_s.color = color
#         self.space.add(ball_s)
#         self.draw(ball_s, color)
#         return ball_s

#     def init(self):
#         self.step = 1/60.
#         self.touches = {}
#         self.space = pymunk.Space()
#         # self.start()

#     def reset(self, *args):
#         self.clear_widgets()
#         self.update_event.cancel()
#         self.canvas.clear()
#         self.start()

#     # # # # # # # # # # Update # # # # # # # # # #

#     def update(self, dt):
#         stepdelay = 25
#         for x in range(6):
#             for i in range(3):
#                 self.space.step(self.step/2/3)
#                 self.space.step(self.step/2/3)
#             self.space.steps += 1
#             if len(self.events) > 0 and self.space.steps-stepdelay > self.events[0][0]:
#                 _, f = self.events.pop(0)
#                 f(self.space)

#         for shape in self.space.shapes:
#             if hasattr(shape, "kivy") and not shape.body.is_sleeping:
#                 if isinstance(shape, pymunk.Circle):
#                     body = shape.body
#                     shape.kivy[0].pos = body.position - (shape.radius, shape.radius) + shape.offset
#                     circle_edge = body.position + shape.offset + Vec2d(shape.radius, 0).rotated(body.angle)
#                     shape.kivy[1].points = [body.position.x + shape.offset.x, body.position.y + shape.offset.y, circle_edge.x, circle_edge.y]
#                 if isinstance(shape, pymunk.Segment):
#                     body = shape.body
#                     p1 = body.position + shape.a.cpvrotate(body.rotation_vector)
#                     p2 = body.position + shape.b.cpvrotate(body.rotation_vector)
#                     shape.kivy.points = p1.x, p1.y, p2.x, p2.y
#                 if isinstance(shape, pymunk.Poly):
#                     if isinstance(shape.kivy, Mesh):
#                         shape.kivy.vertices = self.vertices_from_poly(shape)
#                     else:
#                         shape.kivy.points = self.points_from_poly(shape)
#         for cons in self.space.constraints:
#             if hasattr(cons, "kivy"):
#                 cons.kivy.points = self.points_from_constraint(cons)

#     # # # # # # # # # # Events # # # # # # # # # #

#                 body = pymunk.Body()
#                 self.space.add(body)

#                 tess = Tesselator()
#                 tess.add_contour(touch.ud['current_line'].points)
#                 tess.tesselate(WINDING_ODD, TYPE_POLYGONS)
#                 for vertices, indices in tess.meshes:
#                     #print('Indices:', list(indices))
#                     vs = []
#                     for i in range(len(vertices) // 4):
#                         vs.append((vertices[4*i], vertices[4*i+1]))
#                     #print(vs)
#                     poly = pymunk.Poly(body, vs)
#                     poly.density = 1
#                     poly.friction = 1
#                     self.space.add(poly)
#                     self.draw(poly)

#                 self.canvas.remove(touch.ud['current_line'])

#     # # # # # # # # # # Kivy Drawing # # # # # # # # # #
