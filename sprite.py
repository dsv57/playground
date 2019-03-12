#!/usr/bin/python3

from math import *
from random import *
from os import stat
from filecmp import _sig as file_sig
from collections import OrderedDict
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
from kivy.graphics.transformation import Matrix
from kivy.core.window import Window
# from kivy.core.image import Image as CoreImage
from kivy.properties import StringProperty, NumericProperty, \
    ListProperty, ObjectProperty, AliasProperty
from kivy.vector import Vector

import numpy as np  # OurImage

import pymunk
import pymunk.autogeometry
from pymunk import Body
from pymunk.vec2d import Vec2d

from named_colors import COLORS

# from kivy.uix.image import Image


def trace_image(img, threshold=3, simplify_tolerance=0.7, cache=True):
    lines = None
    trace_sig = file_sig(
        stat(img.filename)), img.size, threshold, simplify_tolerance
    # print('IMG', img.width, img.height)
    # print('SIG', trace_sig, img.filename)
    if cache:
        try:
            with open(img.filename + '.contour', 'rb') as f:
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

        center = (min_x + w / 2., min_y + h / 2.)
        # t = pymunk.Transform(a=1.0, d=1.0, tx=-center.x, ty=-center.y)

        line = [(l.x, img.height - l.y) for l in line]
        lines.append(line)
        # return lines
        # print('Lines:', len(lines))
    if cache and lines:
        try:
            with open(img.filename + '.contour', 'wb') as f:
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


import numpy as np
from numpy import sin, cos, arctan2 as atan2, \
                  sqrt, ceil, floor, degrees, radians, log, pi, exp, transpose
from colorio import CIELAB, CAM16UCS, CAM16, JzAzBz, SrgbLinear
from colorio.illuminants import whitepoints_cie1931

L_A = 64 / pi / 5
srgb = SrgbLinear()
lab = CIELAB()
cam16 = CAM16(0.69, 20, L_A)
cam16ucs = CAM16UCS(0.69, 20, L_A)


class Colors:

    def __init__(self, colors=None, description='sRGB', **kwargs):
        #self.cam16 = CAM16(0.69, 20, L_A)
        c = kwargs.get('c', 0.69)
        Y_b = kwargs.get('Y_b', 20)
        L_A = kwargs.get('L_A', 64 / pi / 5)
        exact_inversion = kwargs.get('exact_inversion', True)
        whitepoint = kwargs.get('whitepoint', 'D65')
        if isinstance(whitepoint, str):
            whitepoint = whitepoints_cie1931[whitepoint]

        cam16ucs = CAM16UCS(c, Y_b, L_A, exact_inversion, whitepoint)
        cam16 = cam16ucs.cam16
        self._cam16ucs = cam16ucs
        self._cam16 = cam16
        self._srgb = SrgbLinear()
        self._srgb_colors = None
        self._original_colors = None
        self._dimensions = 'Jsh'

        if colors:
            self.set(colors, description)

    def reset(self):
        self._colors = self._original_colors.copy()

    def set(self, colors, description='sRGB'):
        cam16ucs = self._cam16ucs
        cam16 = cam16ucs.cam16
        if isinstance(colors, str):
            colors = [colors]
        if isinstance(colors, bytes): #'Image' in globals() and isinstance(colors, (Image, CoreImage, Texture)):
            # if isinstance(colors, CoreImage):
            #     texture = colors.texture
            # elif isinstance(colors, Image):
            #     texture = colors._coreimage.texture
            # else:
            #     texture = colors
            width, height = texture.size
            length = len(colors) // 4
            image_srgb = np.fromstring(colors, dtype='ubyte').reshape(length, 4)[...,:3].astype('float') / 255
            image_flat_srgb = image_srgb.reshape((length, 3)).T

            colors = image_flat_srgb
            # self._rgb_image = image
            self._srgb_colors = image.reshape((length, 3)).T
        elif isinstance(colors, list):
            # self._size = (len(colors),)
            self._srgb_colors = None
            if isinstance(colors[0], (tuple, list)):
                if all([all([isinstance(c, int) for c in cc]) and len(cc) == 3 for cc in colors]):
                    colors = np.array(colors).astype('float')
                    if description == 'sRGB':
                        colors /= 255
                elif all([all([isinstance(c, float) for c in cc]) and len(cc) == 3 for cc in colors]):
                    colors = np.array(colors)
                else:
                    raise Exception('bad colors list')
            elif isinstance(colors[0], str) and description == 'sRGB':
                colors_srgb = []
                for color in colors:
                    if color.startswith('#'):
                        if len(color) == 7:
                            colors_srgb.append([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)])
                        elif len(color) == 4:
                            colors_srgb.append([16 * int(h, 16) / 255 for h in color[1:]])
                    elif color in COLORS:
                        colors_srgb.append(COLORS[color])
                    else:
                        raise Exception('bad colors list')
                colors = np.array(colors_srgb).astype('float') / 255
            else:
                raise Exception('bad colors list')
        else:
            raise Exception('bad colors list')

        colors = np.transpose(colors)
        if description == 'sRGB':
            xyz = srgb.to_xyz100(srgb.from_srgb1(colors))
        elif description == 'CIELAB':
            xyz = CIELAB().to_xyz100(colors)
        elif description == 'CIELUV':
            xyz = CIELUV().to_xyz100(colors)
        elif description == 'CIELCH':
            xyz = CIELCH().to_xyz100(colors)
        elif description == 'XYZ':
            xyz = colors
        elif description[0] in 'JQ' and description[1] in 'CMs' and description[2] in 'Hh':
            xyz = cam16.to_xyz100(colors, description)
        elif description in ['CAM16UCS', 'CAM16-UCS']:
            xyz = cam16ucs.to_xyz100(colors)

        self._xyz = xyz
        self._colors = cam16.from_xyz100(xyz)
        if not self._original_colors:
            self._original_colors = self._colors.copy()

    @property
    def lightness(self):
        return self._colors[0]

    @property
    def chroma(self):
        return self._colors[1]

    @property
    def hue_quadrature(self):
        return self._colors[2]

    @property
    def hue(self):
        return self._colors[3]

    @property
    def colorfulness(self):
        return self._colors[4]

    @property
    def saturation(self):
        return self._colors[5]

    @property
    def brightness(self):
        return self._colors[6]

    _dims = {
        'lightness': ('J', 0, 0),
        'brightness': ('Q', 6, 0),
        'chroma': ('C', 1, 1),
        'colorfulness': ('M', 4, 1),
        'saturation': ('s', 5, 1),
        'hue quadrature': ('H', 2, 2),
        'hue': ('h', 3, 2),
    }

    _dims_ltr = {  # {_dims[d][0]: _dims[d][1:]+(d,) for d in _dims}
        'J': (0, 0, 'lightness'),
        'Q': (6, 0, 'brightness'),
        'C': (1, 1, 'chroma'),
        'M': (4, 1, 'colorfulness'),
        's': (5, 1, 'saturation'),
        'H': (2, 2, 'hue quadrature'),
        'h': (3, 2, 'hue')
    }

    def _select_dims(self, ds):
        # J, C, H, h, M, s, Q
        if isinstance(ds, str) and len(ds) == 3 and ds[0] in 'JQ' and ds[1] in 'CMs' and ds[2] in 'Hh':
            return ds

        dims = Colors._dims
        dims_ltr = Colors._dims_ltr
        dstr = list(self._dimensions)
        if isinstance(ds, str) and ds in dims:
            ds = [ds]
        if isinstance(ds, list):
            for d in ds:
                if len(d) == 1:
                    ltr = d
                    dim_n = dims_ltr[d][0]
                else:
                    ltr, _, dim_n = dims.get(d)
                dstr[dim_n] = ltr
        else:
            for ltr in ds:
                dim_n = dims_ltr[ltr][0]
                dstr[dim_n] = ltr

        self._dimensions = ''.join(dstr)
        return dstr


    def set_dimension(self, dimension, value, preserve=None):
        if len(dimension) == '1':
            dim = Colors._dims_ltr[ltr][0]
        else:
            dim = Colors._dims[dimension][1]
        self._colors[dim] = value
        if preserve:
            self._select_dims(preserve)
        self._select_dims(dimension)


    def get_something(self):
        srgb = self._srgb
        colors = self._colors
        length = len(colors)
        srgb_colors = srgb.to_srgb1(srgb.from_xyz100(cam16.to_xyz100(im.reshape((length, 3)).T, self._dimensions))).T # JCh
        self._srgb_colors = srgb_colors.clip(0, 1, srgb_colors)
        return srgb_colors


    # J, C, H, h, M, s, Q = cam16.from_xyz100(xyz)
    # JQ CMs Hh

    # self.K_L = 1.0
    # self.c1 = 0.007
    # self.c2 = 0.0228
    # params = {
    #     "LCD": (0.77, 0.007, 0.0053),
    #     "SCD": (1.24, 0.007, 0.0363),
    #     "UCS": (1.00, 0.007, 0.0228),
    # }


class OurImage(Image):

    # image = NumericProperty()

    def __init__(self, **kwargs):
        super(OurImage, self).__init__(**kwargs)
        texture = self._coreimage.texture
        self.reference_size = texture.size
        self.texture = texture
        image = np.fromstring(self.texture.pixels, dtype='ubyte').reshape(*texture.size, 4)[..., :3].astype('float') / 255
        self._image = image
        width, height = texture.size
        # self.imc16 = cam16.from_xyz100(srgb.to_xyz100(srgb.from_srgb1(image.reshape((image.shape[0] * image.shape[1], 3)).T)))[[True, True,False,True,False,False,False]].T.reshape(image.shape)
        image_flat_rgb = image.reshape((width * height, 3)).T
        image_flat_cam16 = cam16.from_xyz100(srgb.to_xyz100(srgb.from_srgb1(image_flat_rgb)))
        J, C, H, h, M, s, Q = image_flat_cam16.T.reshape(width, height, 7).T
        self.imc16 = image_flat_cam16.T.reshape(width, height, 7)

    def _get_image(self):
        return self._image.copy()

    def _set_image(self, image):
        #print('on_image ' * 30)
        buf = (image * 255).flatten().astype('ubyte')
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

    image = AliasProperty(_get_image, _set_image)

    def set_image(self, image):
        # print('on_image ' * 30)
        buf = (image * 255).flatten().astype('ubyte')
        self.texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')

    def set_imc16(self, im, descr='JCh'):
        image_sr = srgb.to_srgb1(srgb.from_xyz100(cam16.to_xyz100(im.reshape((self.image.shape[0] * self.image.shape[1], 3)).T, descr))).T # JCh
        self.set_image(image_sr.clip(0, 1))


class Sprite(Scatter):

    _instances = set()

    _shapes = {
        "arrow": {
            'type': 'polygon',
            'points': [(-10, 0), (10, 0), (0, 10)]
        },
        "turtle": {
            'type': 'polygon',
            'points': [(0, 16), (-2, 14), (-1, 10), (-4, 7), (-7, 9), (-9, 8),
                       (-6, 5), (-7, 1), (-5, -3), (-8, -6), (-6, -8),
                       (-4, -5), (0, -7), (4, -5), (6, -8), (8, -6), (5, -3),
                       (7, 1), (6, 5), (9, 8), (7, 9), (4, 7), (1, 10),
                       (2, 14)]
        },
        "circle": {
            'type': 'circle',
            'radius': 25
        },
        "square": {
            'type': 'polygon',
            'points': [(50, -10), (50, 10), (-10, 10), (-10, -10)]
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

    # space = ObjectProperty(None)

    def __init__(self, source, x=0, y=0, scale=1, rotation=0, trace=True,
                 body_type=Body.KINEMATIC, **kwargs):
        super(Sprite, self).__init__()

        self.body = pymunk.Body(body_type=body_type)
        density = kwargs.get('density', 0.4)
        friction = kwargs.get('friction', 1.0)
        elasticity = kwargs.get('elasticity', 0.1)
        self.space = None
        self.source = source
        self.bottom_left = Vec2d(0, 0)
        self._selection_line = None
        self.image = None
        self.is_moved = False
        if source not in Sprite._shapes:
            self._type = 'image'
            imag = Image(
                source=source, keep_data=trace,
                anim_delay=kwargs.get('anim_delay', 0.05))
            self.add_widget(imag)
            imag.size = imag.texture_size
            self.size = imag.texture_size
            self.shapes = [imag]
            self.pymunk_shapes = []

            lines = trace_image(imag._coreimage, cache=False) if trace else []
            print('lines:', len(lines))
            if len(lines) > 1:
                raise Exception('cannot add fragmented image')
            if not lines:
                lines = [[(0, 0), (0, imag.size[1]),
                          imag.size, (imag.size[0], 0), (0, 0)]]
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
            self._type = shape['type']
            self.shapes = []
            self.pymunk_shapes = []

            if shape['type'] == 'polygon':
                self.contour = shape['points']
                xs, ys = tuple(zip(*shape['points']))
                min_x = min(xs)
                min_y = min(ys)
                self.size = (max(xs) - min_x, max(ys) - min_y)
                self.bottom_left += min_x, min_y

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
                        if body_type == Body.DYNAMIC:
                            poly.density = density
                        poly.friction = friction
                        poly.elasticity = elasticity
                        self.pymunk_shapes.append(poly)
                    PopMatrix()

            elif shape['type'] == 'circle':
                r = shape['radius']
                self.contour = [(-r, -r), (-r, r), (r, r), (r, -r), (-r, -r)]
                self.size = (2 * r, 2 * r)
                self.bottom_left += -r, -r
                with self.canvas:
                    Color(0, 1, 0)
                    self.shapes.append(
                        Ellipse(pos=(0, 0), size=(r * 2, r * 2)))
                    cir = pymunk.Circle(self.body, r, (0, 0))
                    if body_type == Body.DYNAMIC:
                        cir.density = density
                    cir.friction = friction
                    cir.elasticity = elasticity
                    self.pymunk_shapes.append(cir)
            else:
                raise Exception('unkown shape type')

        self.position = x, y
        self.rotation = rotation
        self._register()

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

        self.position += Vec2d(touch.dpos) / 2  # touch.pos # touch.dx, touch.dy

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
            print('touch', touch, touch.dpos, (touch.ox, touch.oy), touch.pos, touch.ppos)
            touch.grab(self)
            self.is_moved = True
            self.body.velocity = 0, 0
            self.body.angular_velocity = 0
            if self._selection_line:
                self.canvas.remove(self._selection_line)
                self._selection_line = None
            else:
                with self.canvas: #.after:
                    Color(0.2, 0.3, 1)
                    x, y = -self.bottom_left #self.position  # self.bottom_left
                    print('source:', self.source)
                    contour = [(p[0] + x, p[1] + y) for p in self.contour]
                    self._selection_line = Line(
                        points=contour, dash_offset=5,
                        dash_length=10, width=1.5)
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
            return Vec2d(self.bbox[0]) + self.offset

    def _set_position(self, position):
        if self.body:
            self.body.position = position
            # self.body.velocity = 0, 0
            # self.body.angular_velocity = 0
            if self.space:
                self.space.reindex_shapes_for_body(self.body)
                if self.space.replay_mode:
                    return
        pos = Vec2d(position)
        _pos = self.to_parent(*-self.bottom_left)
        if pos == _pos:
            return
        t = pos - _pos
        trans = Matrix().translate(t.x, t.y, 0)
        self.apply_transform(trans)
    position = AliasProperty(_get_position, _set_position, bind=('bbox', ))

    def _get_xcor(self):
        if self.body:
            return self.body.position.x
        return self.bbox[0][0] + self.offset.x

    def _set_xcor(self, x):
        self._set_x(x - self.offset.x)
    xcor = AliasProperty(_get_xcor, _set_xcor, bind=('bbox', ))

    def _get_ycor(self):
        if self.body:
            return self.body.position.y
        return self.bbox[0][1] + self.offset.y

    def _set_ycor(self, y):
        self._set_y(y - self.offset.y)
    ycor = AliasProperty(_get_ycor, _set_ycor, bind=('bbox', ))

    def _get_bbox(self):
        # FIXME: self.size?
        if self.body:
            xmin, ymin = xmax, ymax = self.body.local_to_world(self.bottom_left)
            for point in [(self.width, 0), (0, self.height), self.size]:
                x, y = self.body.local_to_world(Vec2d(point) + self.bottom_left)
                if x < xmin:
                    xmin = x
                if y < ymin:
                    ymin = y
                if x > xmax:
                    xmax = x
                if y > ymax:
                    ymax = y
            return Vec2d(xmin, ymin), Vec2d(xmax - xmin, ymax - ymin)
        return super(Sprite, self).bbox
    bbox = AliasProperty(_get_bbox, None, bind=(
        'transform', 'width', 'height'))

    def _get_center(self):
        return super(Sprite, self)._get_center

    def _set_center(self, center):
        if center == self.center:
            return False
        t = Vec2d(*center) - self.center
        trans = Matrix().translate(t.x, t.y, 0)
        self.apply_transform(trans)
        if body:
            self.body.position += t
    center = AliasProperty(_get_center, _set_center, bind=('bbox', ))

    def _set_pos(self, pos):
        self._set_position(Vec2d(pos) + self.offset)

    def _get_pos(self):
        if self.body:
            return self.body.position - self.offset
    pos = AliasProperty(_get_pos, _set_pos, bind=('bbox', ))

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
    x = AliasProperty(_get_x, _set_x, bind=('bbox', ))

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
    y = AliasProperty(_get_y, _set_y, bind=('bbox', ))

    def _register(self):
        Sprite._instances.add(self)

    def _unregister(self):
        if self in Sprite._instances:
            Sprite._instances.remove(self)

    @staticmethod
    def clear_sprites():
        print('CLEAR ' * 20)
        for sprite in Sprite._instances:
            if sprite.space:
                sprite.space.remove(*sprite.pymunk_shapes, sprite.body)
        Sprite._instances = set()

    @property
    def offset(self):
        if self.body:
            return self.body.position - self.bbox[0]
        cx, cy = self.to_parent(*-self.bottom_left)
        return Vec2d(cx - self.x, cy - self.y)

    def _get_rotation(self):
        if self.body:
            return degrees(self.body.angle)
        return (360 - (Vec2d(self.to_parent(0, 10)) - self.to_parent(0, 0)).get_angle_degrees_between((0, 10))) % 360

    def _set_rotation(self, rotation):
        # TODO: Add is_symmetric optimization
        if not self.space or not self.space.replay_mode:
            angle_change = self.rotation - rotation
            if angle_change != 0:
                r = Matrix().rotate(-radians(angle_change), 0, 0, 1)
                self.apply_transform(r, post_multiply=True,
                                     anchor=-self.bottom_left)
        if self.body:
            self.body.angle = radians(rotation)
            self.body.position = self.position
            if self.space:
                self.space.reindex_shapes_for_body(self.body)

    rotation = AliasProperty(_get_rotation, _set_rotation, bind=(
        'x', 'y', 'transform'))

    def _update(self):
        if self.is_moved:
            self.body.position = self.position
            self.body.velocity = 0, 0
            self.body.angular_velocity = 0
        else:
            _rotation = (360 - (Vec2d(self.to_parent(0, 10)) - self.to_parent(0, 0)).get_angle_degrees_between((0, 10))) % 360
            rotation = degrees(self.body.angle)
            angle_change = _rotation - rotation
            if angle_change != 0.0:
                r = Matrix().rotate(-radians(angle_change), 0, 0, 1)
                self.apply_transform(r, post_multiply=True,
                                     anchor=-self.body.center_of_gravity)
            pos = Vec2d(self.body.position)
            _pos = self.to_parent(*-self.bottom_left)
            if pos == _pos:
                return
            t = pos - _pos
            trans = Matrix().translate(t.x, t.y, 0)
            self.apply_transform(trans)

    @staticmethod
    def update_from_pymunk(ignore_sleeping=True):
        for sprite in Sprite._instances:
            if sprite.body.body_type != Body.STATIC and (not ignore_sleeping or not sprite.body.is_sleeping):
                sprite._update()







        # for shape in self.sandbox.space.shapes:
        #     if hasattr(shape, "kivy") and not shape.body.is_sleeping:
        #         if isinstance(shape, pymunk.Circle):
        #             body = shape.body
        #             shape.kivy[0].pos = body.position - (shape.radius, shape.radius) + shape.offset
        #             circle_edge = body.position + shape.offset + Vec2d(shape.radius, 0).rotated(body.angle)
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
