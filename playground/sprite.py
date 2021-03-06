#!/usr/bin/python3

from math import *
from random import *
from weakref import WeakSet
from os import stat
from filecmp import _sig as file_sig
import pickle

# kivy.require('1.9.1')

from kivy.uix.scatter import Scatter
from kivy.uix.image import Image  # as ImageWidget
from kivy.graphics import (
    Ellipse,
    Line,
    Color,
    Triangle,
    Quad,
    Rectangle,
    Mesh,
    PushMatrix,
    PopMatrix,
)
from kivy.graphics.tesselator import Tesselator, WINDING_ODD, TYPE_POLYGONS
from kivy.graphics.context_instructions import Translate
from kivy.graphics.transformation import Matrix

# from kivy.core.image import Image as CoreImage
from kivy.properties import (
    StringProperty,
    NumericProperty,
    ListProperty,
    ObjectProperty,
    AliasProperty,
)

import numpy as np  # OurImage

import pymunk
import pymunk.autogeometry
from pymunk import Body

# from pymunk.vec2d import Vec2d
from .named_colors import COLORS

from playground.geometry import Vector


def trace_image(img, threshold=3, simplify_tolerance=0.7, cache=True):
    lines = None
    trace_sig = file_sig(stat(img.filename)), img.size, threshold, simplify_tolerance
    # print('IMG', img.width, img.height)
    # print('SIG', trace_sig, img.filename)
    if cache:
        try:
            with open(img.filename + ".contour", "rb") as f:
                lines, fsig = pickle.load(f)
                if fsig == trace_sig:
                    return lines
        except:
            pass

    bb = pymunk.BB(0, 0, img.width - 1, img.height - 1)

    def sample_alpha(point):
        color = img.read_pixel(point.x, point.y)
        return color[3] * 255

    def sample_black(point):
        color = img.read_pixel(point.x, point.y)
        return sum(color[:3]) / 3

    line_set = pymunk.autogeometry.PolylineSet()

    def segment_func(v0, v1):
        line_set.collect_segment(v0, v1)

    if len(img.read_pixel(0, 0)) == 4:
        sample_func = sample_alpha
    else:
        sample_func = sample_black
    pymunk.autogeometry.march_soft(
        bb, img.width, img.height, threshold, segment_func, sample_func
    )

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
        # w, h = max_x - min_x, max_y - min_y

        # center = (min_x + w / 2.0, min_y + h / 2.0)
        # t = pymunk.Transform(a=1.0, d=1.0, tx=-center.x, ty=-center.y)

        line = [(l.x, img.height - l.y) for l in line]
        lines.append(line)
        # return lines
        # print('Lines:', len(lines))
    if cache and lines:
        try:
            with open(img.filename + ".contour", "wb") as f:
                pickle.dump((lines, trace_sig), f)
        except:
            pass

    # print('lines', len(lines), [len(l) for l in lines])
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
    circle_edge = (
        shape.body.position
        + shape.offset
        + Vector(shape.radius, 0).rotated(shape.body.angle)
    )
    Color(*COLORS["dark slate gray"])  # (.17,.24,.31)
    line = Line(
        points=[
            shape.body.position.x + shape.offset.x,
            shape.body.position.y + shape.offset.y,
            circle_edge.x,
            circle_edge.y,
        ]
    )
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
            Color(1.0, 1.0, 1.0)
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
                    mode="triangle_fan",
                )  # Line(points=self.points_from_poly(shape), width=3)
        elif isinstance(shape, pymunk.constraint.Constraint):
            points = points_from_constraint(shape)
            if len(points) > 0:
                shape.kivy = Line(points=points)


from .colorio import CAM16UCS, SrgbLinear
from .colorio.illuminants import whitepoints_cie1931


srgb = SrgbLinear()
cam16ucs = CAM16UCS(0.69, 20, 20, True, whitepoints_cie1931["D65"])
cam16 = cam16ucs.cam16


class OurImage(Image):

    # image = NumericProperty()

    def __init__(self, **kwargs):
        super(OurImage, self).__init__(**kwargs)
        texture = self._coreimage.texture
        self.reference_size = texture.size
        self.texture = texture
        image = (
            np.fromstring(self.texture.pixels, dtype="ubyte")
            .reshape(*texture.size, 4)[..., :3]
            .astype("float")
            / 255
        )
        self._image = image
        width, height = texture.size
        # self.imc16 = cam16.from_xyz100(srgb.to_xyz100(srgb.from_srgb1(image.reshape((image.shape[0] * image.shape[1], 3)).T)))[[True, True,False,True,False,False,False]].T.reshape(image.shape)
        image_flat_rgb = image.reshape((width * height, 3)).T
        image_flat_cam16 = cam16.from_xyz100(
            srgb.to_xyz100(srgb.from_srgb1(image_flat_rgb))
        )
        J, C, H, h, M, s, Q = image_flat_cam16.T.reshape(width, height, 7).T
        self.imc16 = image_flat_cam16.T.reshape(width, height, 7)

    def _get_image(self):
        return self._image.copy()

    def _set_image(self, image):
        # print('on_image ' * 30)
        buf = (image * 255).flatten().astype("ubyte")
        self.texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")

    image = AliasProperty(_get_image, _set_image)

    def set_image(self, image):
        # print('on_image ' * 30)
        buf = (image * 255).flatten().astype("ubyte")
        self.texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")

    def set_imc16(self, im, descr="JCh"):
        image_sr = srgb.to_srgb1(
            srgb.from_xyz100(
                cam16.to_xyz100(
                    im.reshape((self.image.shape[0] * self.image.shape[1], 3)).T, descr
                )
            )
        ).T  # JCh
        self.set_image(image_sr.clip(0, 1))


# Vector libs:
#   https://github.com/exyte/Macaw/wiki/Getting-started + https://blog.exyte.com/replicating-apple-design-awarded-applications-70e5df4c4b94
#   http://dmitrybaranovskiy.github.io/raphael/
#   http://paperjs.org/tutorials/geometry/point-size-and-rectangle/ + http://paperjs.org/reference/path/

# class Locus {
#     open func bounds() -> Rect {
#         return Rect()
#     }
#     open func stroke(with: Stroke) -> Shape {
#         return Shape(form: self, stroke: with)
#     }
#     open func fill(with: Fill) -> Shape {
#         return Shape(form: self, fill: with)
#     }
#     open func stroke(fill: Fill = Color.black, width: Double = 1, cap: LineCap = .butt, join: LineJoin = .miter, dashes: [Double] = []) -> Shape {
#         return Shape(form: self, stroke: Stroke(fill: fill, width: width, cap: cap, join: join, dashes: dashes))
#     }
# }


# def __init__(self, *args, **kwargs):
#     corner = None
#     size = None
#     if args and isinstance(args[0], Iterable):
#         corner = args[0]
#     if len(args) > 1 and isinstance(args[1], Iterable):
#         size = args[1]
#     if len(args) in [3, 4] and all([isinstance(a, Number) for a in args]):
#         corner = args[:2]
#         size = args[2:]
#         if len(size) == 1:
#             size *= 2
#     if size is None:
#         size = kwargs.pop('size', None)
#         if isinstance(size, Number):
#             size = size, size
#         if size is None and ('width' in kwargs or 'height' in kwargs):
#             size = kwargs.pop('width', 50), kwargs.pop('height', 50)
#     if corner is None:
#         corner = kwargs.pop('corner', None)
#         if corner is None:
#             if 'x' in kwargs or 'y' in kwargs:
#                 corner = kwargs.pop('x', 0), kwargs.pop('y', 0)
#             elif size and 'center' in kwargs:
#                 corner = Vector(kwargs.pop('center')) - Vector(size) / 2
#             # elif size and ('center_x' in kwargs or 'center_y' in kwargs):
#             #     corner = Vector(kwargs.get('center_x', 0), kwargs.get('center_y', 0)) - size / 2
#     if corner and size is None and 'to' in kwargs:
#         size = Vector(kwargs.pop('to')) - corner

#     if kwargs:
#         raise Exception(f'Bad arguments: {", ".join(kwargs)}')

#     self.corner = Vector(corner or (0, 0))
#     self.size = Vector(size or (50, 50))


class Sprite(Scatter):

    _instances = WeakSet()

    _shapes = {
        "arrow": {"type": "polygon", "points": [(-10, 0), (10, 0), (0, 10)]},
        "turtle": {
            "type": "polygon",
            "points": [
                (0, 16),
                (-2, 14),
                (-1, 10),
                (-4, 7),
                (-7, 9),
                (-9, 8),
                (-6, 5),
                (-7, 1),
                (-5, -3),
                (-8, -6),
                (-6, -8),
                (-4, -5),
                (0, -7),
                (4, -5),
                (6, -8),
                (8, -6),
                (5, -3),
                (7, 1),
                (6, 5),
                (9, 8),
                (7, 9),
                (4, 7),
                (1, 10),
                (2, 14),
            ],
        },
        "circle": {"type": "circle", "radius": 25},
        "square": {
            "type": "polygon",
            "points": [(10, -10), (10, 10), (-10, 10), (-10, -10)],
        },
        "platform": {
            "type": "polygon",
            "points": [(-10, -50), (10, -50), (10, 50), (-10, 50)],
        },
        "triangle": {
            "type": "polygon",
            "points": [(10, -5.77), (0, 11.55), (-10, -5.77)],
        },
        "classic": {"type": "polygon", "points": [(0, 0), (-5, -9), (0, -7), (5, -9)]},
    }

    # space = ObjectProperty(None)

    def __init__(
        self,
        source,
        x=0,
        y=0,
        scale=1,
        rotation=0,
        color="old lace",
        trace=True,
        body_type=Body.KINEMATIC,
        **kwargs
    ):
        super(Sprite, self).__init__()

        self.body = pymunk.Body(body_type=body_type)
        density = kwargs.get("density", 0.4)
        friction = kwargs.get("friction", 1.0)
        elasticity = kwargs.get("elasticity", 1.0)
        self.space = None
        self.source = source
        self.bottom_left = Vector(0, 0)
        self._selection_line = None
        self.image = None
        self.is_moved = False
        if source not in Sprite._shapes:
            self._type = "image"
            imag = Image(
                source=source,
                keep_data=trace,
                anim_delay=kwargs.get("anim_delay", 0.05),
            )
            self.add_widget(imag)
            imag.size = imag.texture_size
            self.size = imag.texture_size
            self.shapes = [imag]
            self.pymunk_shapes = []

            lines = trace_image(imag._coreimage, cache=False) if trace else []
            print("lines:", len(lines))
            if len(lines) > 1:
                raise Exception("cannot add fragmented image")
            if not lines:
                lines = [
                    [(0, 0), (0, imag.size[1]), imag.size, (imag.size[0], 0), (0, 0)]
                ]
            self.contour = lines[0] if lines else []
            with self.canvas:
                for line in lines:
                    poly = pymunk.Poly(self.body, line)
                    # self.parent.space.add(poly)
                    if body_type == Body.DYNAMIC:
                        poly.density = density
                    poly.friction = friction
                    poly.elasticity = elasticity
                    self.pymunk_shapes.append(poly)

                    # Color(1., 0., 0.)
                    # self.shapes.append(
                    #     Line(points=[
                    #         coord for point in line for coord in point
                    #     ]))
                    # print("LL", len(line), max([p.x for p in line]),
                    #       min([p.x for p in line]), max([p.y for p in line]),
                    #       min([p.y for p in line]))
                    # for i in range(len(line)-1):
                    #     shape = pymunk.Segment(self.body, line[i], line[i+1], 1)
                    #     self.parent.space.add(shape)
            # print('------------')
        else:  # Polygon shape
            shape = Sprite._shapes[source]
            self._type = shape["type"]
            self.shapes = []
            self.pymunk_shapes = []

            if shape["type"] == "polygon":
                self.contour = shape["points"]
                self.contour.append(self.contour[0])
                # print('contour', self.contour)
                xs, ys = tuple(zip(*shape["points"]))
                min_x = min(xs)
                min_y = min(ys)
                self.size = (max(xs) - min_x, max(ys) - min_y)
                self.bottom_left = Vector(min_x, min_y)

                tess = Tesselator()
                tess.add_contour(
                    [coord for point in shape["points"] for coord in point]
                )  # (point[0]+x,point[1]+y)]) #shape['points'])
                tess.tesselate(WINDING_ODD, TYPE_POLYGONS)
                with self.canvas:
                    PushMatrix()
                    Translate(-min_x, -min_y)
                    if isinstance(color, str):
                        Color(*(COLORS.get(color) or (1, 0, 0)))
                    else:
                        Color(*color)
                    for vertices, indices in tess.meshes:
                        # offset_vs = [v - (min_x, min_y)[i % 2] for i, v in enumerate(vertices)]
                        self.shapes.append(
                            Mesh(
                                vertices=vertices, indices=indices, mode="triangle_fan"
                            )
                        )

                        vs = list(zip(vertices[::4], vertices[1::4]))
                        poly = pymunk.Poly(self.body, vs)
                        if body_type == Body.DYNAMIC:
                            poly.density = density
                        poly.friction = friction
                        poly.elasticity = elasticity
                        self.pymunk_shapes.append(poly)
                    PopMatrix()

            elif shape["type"] == "circle":
                r = shape["radius"]
                self.contour = [(-r, -r), (-r, r), (r, r), (r, -r), (-r, -r)]
                self.size = (2 * r, 2 * r)
                self.bottom_left = Vector(-r, -r)
                with self.canvas:
                    if isinstance(color, str):
                        Color(*(COLORS.get(color) or (1, 0, 0)))
                    else:
                        Color(*color)
                    self.shapes.append(Ellipse(pos=(0, 0), size=(r * 2, r * 2)))
                    Color(0, 0, 0)
                    self.shapes.append(Line(points=[r, r, 2 * r, r], width=2))
                    cir = pymunk.Circle(self.body, r, (0, 0))
                    if body_type == Body.DYNAMIC:
                        cir.density = density
                    cir.friction = friction
                    cir.elasticity = elasticity
                    self.pymunk_shapes.append(cir)
            else:
                raise Exception("unkown shape type")

        self.position = x, y
        self.rotation = rotation
        self._register()

    def apply_force(self, force, point=(0, 0)):
        self.body.apply_force_at_local_point(force, point)

    def apply_impulse(self, force, point=(0, 0)):
        self.body.apply_impulse_at_local_point(force, point)

    @property
    def velocity(self):
        return self.body.velocity

    @velocity.setter
    def velocity(self, velocity):
        self.body.velocity = velocity

    @property
    def angular_velocity(self):
        return self.body.angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity):
        self.body.angular_velocity = angular_velocity

    # def draw(self):
    #     with self.canvas:
    #         for shape in self._shapes:
    #             self.add_widget(shape)
    def on_parent(self, instance, value):
        # print('on_parent', instance, value)
        # if value: print('...', value.space, self.space, self.space and self.space.shapes, self.pymunk_shapes)
        if value:  # FIXME FIXME FIXME and value.space is not self.space:
            self.space = value.space
            for shape in self.pymunk_shapes:
                # print("OP", shape)
                # print('SHapes:', self.space.shapes)
                if shape.body not in self.space.bodies:
                    self.space.add(shape.body)
                if shape not in self.space.shapes:
                    self.space.add(shape)

    def collide_point(self, x, y):
        # print('bbox', self.bbox)
        kivy_collide = super(Sprite, self).collide_point(x, y)
        if kivy_collide:
            for shape in self.pymunk_shapes:
                pq = shape.point_query((x, y))
                if pq[0] < 0:
                    # print('Collide!')
                    return True
        return kivy_collide  # False

    def on_position(self, instance, position):
        pass
        # print('On position', position)
        # self.position = position
        # if self._selection_line:
        # contour = [(p[0] + self.position[0], p[1] + self.position[1]) for p in self.contour]
        # self._selection_line.points = contour

        # print('ON POS', self.space, self.space and self.space.shapes, self.pymunk_shapes)
        # self.body.position = position # self.to_parent(*pos)
        # if self.space:
        #     self.space.reindex_shapes_for_body(self.body)
        # print("POS", pos, self.to_local(*pos), self.to_parent(*pos))
        # if Sprite.space:
        #     pq = Sprite.space.point_query((x, y), 0.0, pymunk.ShapeFilter())
        #     print('PQ:', pq)
        #     return pq

    def on_touch_up(self, touch):
        # print(touch)
        if touch.grab_current is not self:
            return
        # assert(repr(self) in touch.ud)
        touch.ungrab(self)
        self.is_moved = False

        # print('t. up', touch.dpos, touch.pos, touch.ppos)
        # print(touch.dpos == (0.0, 0.0), self.collide_point(touch.x, touch.y))
        # if touch.dpos == (0.0, 0.0) and self.collide_point(touch.x, touch.y):
        #     if self._selection_line:
        #         print('Clear')
        #         self.canvas.after.clear() #remove(self._selection_line)
        #         self._selection_line = None
        #     else:
        #         print('Add')
        #         with self.canvas.after:
        #             Color(0.2, 0.3, 1)
        #             x, y = self.position #self.bottom_left
        #             contour = [(p[0] + x, p[1] + y) for p in self.contour]
        #             # print('cont',contour)
        #             line = Line(points=contour, width=1.5)
        #             print('line', line)
        #             self._selection_line = line
        # # return True
        # return super(Sprite, self).on_touch_up(touch)

    def on_touch_move(self, touch):
        if touch.grab_current is not self:
            return

        self.position += Vector(touch.dpos) / 2  # touch.pos # touch.dx, touch.dy

    def on_touch_down(self, touch):
        # if not self.collide_point(touch.x, touch.y):
        # return False
        # if repr(self) in touch.ud:
        #     return False
        # touch.grab(self)
        # touch.ud[repr(self)] = True
        # return True

        # return super(Sprite, self).on_touch_down(touch)
        if self.collide_point(touch.x, touch.y):
            print(
                "touch", touch, touch.dpos, (touch.ox, touch.oy), touch.pos, touch.ppos
            )
            touch.grab(self)
            self.is_moved = True
            self.body.velocity = 0, 0
            self.body.angular_velocity = 0
            if self._selection_line:
                self.canvas.remove(self._selection_line)
                self._selection_line = None
            else:
                with self.canvas:  # .after:
                    Color(0.2, 0.3, 1)
                    x, y = -self.bottom_left  # self.position  # self.bottom_left
                    print("source:", self.source)
                    contour = [(p[0] + x, p[1] + y) for p in self.contour]
                    self._selection_line = Line(
                        points=contour, dash_offset=5, dash_length=10, width=1.5
                    )
            return super(Sprite, self).on_touch_down(touch)
        else:
            return False
        # touch.grab(self)

    # def on_touch_down(self, touch):
    #     print(touch)
    #     print(self.collide_point(*touch.pos))

    #         # return self.dispatch('on_transform_with_touch', touch)
    #     return super().on_touch_down(touch)

    def _get_position(self):
        if self.body:
            return self.body.position
        else:
            return Vector(self.bbox[0]) + self.offset

    def _set_position(self, position):
        if self.body:
            self.body.position = position
            # self.body.velocity = 0, 0
            # self.body.angular_velocity = 0
            if self.space:
                self.space.reindex_shapes_for_body(self.body)
                if self.space.replay_mode:
                    return
        pos = Vector(position)
        _pos = self.to_parent(*-self.bottom_left)
        if pos == _pos:
            return
        t = pos - _pos
        trans = Matrix().translate(t.x, t.y, 0)
        self.apply_transform(trans)

    position = AliasProperty(_get_position, _set_position, bind=("bbox",))

    def _get_xcor(self):
        if self.body:
            return self.body.position.x
        return self.bbox[0][0] + self.offset.x

    def _set_xcor(self, x):
        self._set_x(x - self.offset.x)

    xcor = AliasProperty(_get_xcor, _set_xcor, bind=("bbox",))

    def _get_ycor(self):
        if self.body:
            return self.body.position.y
        return self.bbox[0][1] + self.offset.y

    def _set_ycor(self, y):
        self._set_y(y - self.offset.y)

    ycor = AliasProperty(_get_ycor, _set_ycor, bind=("bbox",))

    def _get_bbox(self):
        # FIXME: self.size?
        if self.body:
            xmin, ymin = xmax, ymax = self.body.local_to_world(self.bottom_left)
            for point in [(self.width, 0), (0, self.height), self.size]:
                x, y = self.body.local_to_world(Vector(point) + self.bottom_left)
                if x < xmin:
                    xmin = x
                if y < ymin:
                    ymin = y
                if x > xmax:
                    xmax = x
                if y > ymax:
                    ymax = y
            return Vector(xmin, ymin), Vector(xmax - xmin, ymax - ymin)
        return super(Sprite, self).bbox

    bbox = AliasProperty(_get_bbox, None, bind=("transform", "width", "height"))

    def _get_center(self):
        return super(Sprite, self)._get_center

    def _set_center(self, center):
        if center == self.center:
            return False
        t = Vector(*center) - self.center
        trans = Matrix().translate(t.x, t.y, 0)
        self.apply_transform(trans)
        if self.body:
            self.body.position += t

    center = AliasProperty(_get_center, _set_center, bind=("bbox",))

    def _set_pos(self, pos):
        self._set_position(Vector(pos) + self.offset)

    def _get_pos(self):
        if self.body:
            return self.body.position - self.offset

    pos = AliasProperty(_get_pos, _set_pos, bind=("bbox",))

    def _get_x(self):
        if self.body:
            return self.body.position.x - self.offset.x
        return self.bbox[0][0]

    def _set_x(self, x):
        if self.body:
            self.body.position.x = x + self.offset.x
            if self.space:
                self.space.reindex_shapes_for_body(self.body)
                if self.space.replay_mode:
                    return
        super(Sprite, self)._set_x(x)

    x = AliasProperty(_get_x, _set_x, bind=("bbox",))

    def _get_y(self):
        if self.body:
            return self.body.position.y - self.offset.y
        return self.bbox[0][1]

    def _set_y(self, y):
        if self.body:
            self.body.position.y = y + self.offset.y
            if self.space:
                self.space.reindex_shapes_for_body(self.body)
                if self.space.replay_mode:
                    return
        super(Sprite, self)._set_y(y)

    y = AliasProperty(_get_y, _set_y, bind=("bbox",))

    def _register(self):
        Sprite._instances.add(self)

    def _unregister(self):
        if self in Sprite._instances:
            Sprite._instances.remove(self)

    @staticmethod
    def clear_sprites():
        for sprite in Sprite._instances:
            if sprite.space:
                sprite.space.remove(*sprite.pymunk_shapes)
                if sprite.body.body_type == Body.DYNAMIC:
                    sprite.space.remove(sprite.body)
        Sprite._instances = WeakSet()

    @property
    def offset(self):
        if self.body:
            return self.body.position - self.bbox[0]
        cx, cy = self.to_parent(*-self.bottom_left)
        return Vector(cx - self.x, cy - self.y)

    def _get_rotation(self):
        if self.body:
            return degrees(self.body.angle)
        return (
            360
            - (Vector(self.to_parent(0, 10)) - self.to_parent(0, 0)).angle_between(
                (0, 10)
            )
        ) % 360

    def _set_rotation(self, rotation):
        # TODO: Add is_symmetric optimization
        if not self.space or not self.space.replay_mode:
            angle_change = self.rotation - rotation
            if angle_change != 0:
                r = Matrix().rotate(-radians(angle_change), 0, 0, 1)
                self.apply_transform(r, post_multiply=True, anchor=-self.bottom_left)
        if self.body:
            self.body.angle = radians(rotation)
            self.body.position = self.position
            if self.space:
                self.space.reindex_shapes_for_body(self.body)

    rotation = AliasProperty(_get_rotation, _set_rotation, bind=("x", "y", "transform"))

    def _update(self):
        if self.is_moved:
            self.body.position = self.position
            self.body.velocity = 0, 0
            self.body.angular_velocity = 0
        else:
            if abs(self.body.position.x) > 10000 or abs(self.body.position.y) > 10000:
                self.space.remove(*self.pymunk_shapes)
                self.space.remove(self.body)
                return
            _rotation = (
                360
                - (Vector(self.to_parent(0, 10)) - self.to_parent(0, 0)).angle_between(
                    (0, 10)
                )
            ) % 360
            rotation = degrees(self.body.angle)
            angle_change = _rotation - rotation
            if angle_change != 0.0:
                r = Matrix().rotate(-radians(angle_change), 0, 0, 1)
                self.apply_transform(
                    r, post_multiply=True, anchor=-self.body.center_of_gravity
                )
            pos = Vector(self.body.position)
            _pos = self.to_parent(*-self.bottom_left)
            if pos == _pos:
                return
            t = pos - _pos
            trans = Matrix().translate(t.x, t.y, 0)
            self.apply_transform(trans)

    @staticmethod
    def update_from_pymunk(ignore_sleeping=True):
        for sprite in Sprite._instances:
            if sprite.body.body_type != Body.STATIC and (
                not ignore_sleeping or not sprite.body.is_sleeping
            ):
                sprite._update()

        # for shape in self.sandbox.space.shapes:
        #     if hasattr(shape, "kivy") and not shape.body.is_sleeping:
        #         if isinstance(shape, pymunk.Circle):
        #             body = shape.body
        #             shape.kivy[0].pos = body.position - (shape.radius, shape.radius) + shape.offset
        #             circle_edge = body.position + shape.offset + Vector(shape.radius, 0).rotated(body.angle)
        #             shape.kivy[1].points = [body.position.x + shape.offset.x, body.position.y + shape.offset.y, circle_edge.x, circle_edge.y]
        #         if isinstance(shape, pymunk.Segment):
        #             body = shape.body
        #             p1 = body.position + shape.a.cpvrotate(body.rotation_vector)
        #             p2 = body.position + shape.b.cpvrotate(body.rotation_vector)
        #             shape.kivy.points = p1.x, p1.y, p2.x, p2.y
        #         if isinstance(shape, pymunk.Poly):
        #             if isinstance(shape.kivy, Mesh):
        #                 shape.kivy.vertices = self.vertices_from_poly(shape)
        #             else:
        #                 shape.kivy.points = self.points_from_poly(shape)
        # for cons in self.space.constraints:
        #     if hasattr(cons, "kivy"):
        #         cons.kivy.points = self.points_from_constraint(cons)


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
#                     circle_edge = body.position + shape.offset + Vector(shape.radius, 0).rotated(body.angle)
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
