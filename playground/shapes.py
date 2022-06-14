from numbers import Number
from collections import defaultdict
from collections.abc import Iterable
import inspect

from pymunk import Body

# from pymunk import Transform as PymunkTransform

from .utils import KeepRefs, SetterProperty
from .color import Color
from .geometry import Vector, VectorRefProperty, VectorList, Transform


__all__ = ["Stroke", "Shape", "Rectangle", "Circle", "Line", "Image", "Physics"]


# class KeepWeakRefs(object):
#     __refs__ = defaultdict(WeakValueDictionary)
#     def __init__(self):
#         self.__refs__[self.__class__][id(self)] = self

#     @classmethod
#     def get_instances(cls, subclasses=False):
#         refs = cls.__refs__
#         if subclasses:
#             return chain.from_iterable(refs[c].values() for c in refs if issubclass(c, cls))
#         return refs[cls].values()


# join = { 'miter' : 0,
#          'round' : 1,
#          'bevel' : 2 }
_linejoins = ("miter", "round", "bevel")
_linecaps = (
    "",
    "none",
    ".",
    "round",
    ")",
    "(",
    "o",
    "triangle in",
    "<",
    "triangle out",
    ">",
    "square",
    "[",
    "]",
    "=",
    "butt",
    "|",
)


class Stroke(object):
    # __slots__ = ('fill', 'width', 'cap', 'joint', 'dashes', '_modified')

    def __init__(
        self, fill="white", width=1.0, caps="round", join="round", dashes=None
    ):
        self.fill = fill
        self.width = width
        self.caps = caps
        self.join = join
        self.dashes = dashes
        self._modified = False

    def __setattr__(self, name, value):
        if name[0] != "_":
            self.__dict__["_modified"] = True
            if (
                name == "fill"
                and value is not None
                and not isinstance(value, (Color, Image))
            ):
                value = Color(value)
            elif name == "width":
                value = float(value)
            elif name == "caps":
                if isinstance(value, str):
                    if value in _linecaps:
                        value = (value, value)
                    else:
                        linecaps = '", "'.join(_linecaps)
                        raise TypeError(
                            f'Stroke caps must be one of: "{linecaps}" or pair'
                        )
                elif (
                    len(value) != 2
                    or value[0] not in _linecaps
                    or value[1] not in _linecaps
                ):
                    linecaps = '", "'.join(_linecaps)
                    raise TypeError(f'Stroke caps must be one of: "{linecaps}" or pair')
            elif name == "join":
                if value not in _linejoins:
                    linejoins = '", "'.join(_linejoins)
                    raise TypeError(f'Stroke joins must be one of: "{linejoins}"')
            elif name not in ("fill", "width", "dashes"):
                raise TypeError(f"got an unexpected keyword argument '{name}'")
        super(Stroke, self).__setattr__(name, value)

    def __iter__(self):
        yield self.fill
        yield self.width
        yield self.caps
        yield self.join
        yield self.dashes

    def __len__(self):
        return 5

    def __repr__(self):
        return f"Stroke({self.fill, self.width})"

    def __eq__(self, other):
        if isinstance(other, Iterable) and len(other) == 5:
            return tuple(self) == tuple(other)
        return False

    def __ne__(self, other):
        if isinstance(other, Iterable) and len(other) == 5:
            return tuple(self) != tuple(other)
        return True

    def __hash__(self):
        return hash(tuple(self))


_body_type = {
    "static": Body.STATIC,
    "kinematic": Body.KINEMATIC,
    "dynamic": Body.DYNAMIC,
}
_body_type_str = {t: s for s, t in _body_type.items()}


class Physics(object):
    __slots__ = ("body",)
    # 'sensor', 'collision_type', 'filter', 'elasticity', 'friction',
    # 'surface_velocity', 'mass', 'density', 'body', 'type', '_modified')
    # STATIC = Body.STATIC
    # KINEMATIC = Body.KINEMATIC
    # DYNAMIC = Body.DYNAMIC

    def __init__(
        self, type="kinematic", density=1.0, elasticity=1.0, friction=1.0, **kwargs
    ):
        self.density = density
        self.elasticity = elasticity
        self.friction = friction
        if "body" not in kwargs:
            self.body = Body(body_type=_body_type[type])
        for param, value in kwargs.items():
            setattr(self, param, value)
        self._pm_shapes = []
        self._modified = False

    def __eq__(self, other):
        if isinstance(other, Physics):
            return (
                self._pm_shapes == other._pm_shapes
                and self.body == other.body
                and self.density == other.density
                and self.elasticity == other.elasticity
                and self.friction == other.friction
            )
        return False

    def __hash__(self):
        return hash(
            (self.density, self.elasticity, self.friction, self.body, self._pm_shapes)
        )

    def __setattr__(self, name, value):
        self.__dict__["_modified"] = True
        if name == "mass":
            area = [shape.area for shape in self._pm_shapes]
            total_area = sum(area)
            for i, shape in enumerate(self._pm_shapes):
                shape.mass = value * area[i] / total_area
        elif name == "type":
            self.body.body_type = _body_type[value]
        elif name == "moment":
            self.body.moment = value
        else:
            if name not in (
                "density",
                "elasticity",
                "friction",
                "body",
                "mass",
                "type",
                "moment",
            ):
                raise TypeError(f"got an unexpected keyword argument '{name}'")
            for shape in self._pm_shapes:
                setattr(shape, name, value)
        # if name[0] != '_':
        # self._modified = True
        # if name in ['elasticity', 'friction', 'surface_velocity', 'mass', 'density']:
        #     value = float(value)
        # elif name == 'width':
        #     value = float(value)
        super(Physics, self).__setattr__(name, value)

    def __getattr__(self, name):
        if name == "mass":
            return sum([shape.mass for shape in self._pm_shapes])
        elif name == "type":
            return _body_type_str[self.body.body_type]
        elif name == "moment":
            if len(self._pm_shapes) == 1:
                return self._pm_shapes[0].moment
            raise NotImplementedError  # TODO
        elif name == "body":
            return self.body
        elif name[0] != "_":
            # if self._pm_shapes:
            return getattr(self._pm_shapes[0], name)
            # else:
            # raise AssertionError
        # if name[0] != '_':
        # self._modified = True
        # if name in ['elasticity', 'friction', 'surface_velocity', 'mass', 'density']:
        #     value = float(value)
        # elif name == 'width':
        #     value = float(value)
        return super(Physics, self).__getattr__(name)


# Shape:
#      form
#   v  stroke
#   v  fill
#   v  place
#
#   v  physics
#
#      opacity
#      opaque
#      clip
#      mask
#      effect
#      visible
#      tag


class Shape(KeepRefs):
    # __slots__ = ('stroke', 'fill', 'physics', '_modified')

    trace = True
    _trace_counters = defaultdict(lambda: 0)

    def __init__(self, **kwargs):
        # for attr in 'stroke', 'fill', 'physics', 'transform':
        #     self.__dict__[attr] = None
        args = kwargs.copy()
        self.stroke = args.pop("stroke", None)
        self.fill = args.pop("fill", None)
        self.opacity = args.pop("opacity", 100.0)
        self.physics = args.pop("physics", None)
        self.transform = args.pop("transform", None)
        if args:
            arg = tuple(args.keys())[0]
            raise TypeError(f"got an unexpected keyword argument '{arg}'")

        self._modified = False
        self._trace = None
        self._trace_iter = None
        trace = []
        if Shape.trace:
            f = inspect.currentframe()
            depth = 0
            while f is not None:
                filename = f.f_code.co_filename
                lineno = f.f_lineno
                user_code = filename == "<code-input>"
                if depth > 4 and not user_code:
                    break
                # print('depth', depth, filename, lineno)
                if user_code:
                    trace.append(lineno)
                depth += 1
                f = f.f_back
            trace = tuple(trace)
            self._trace = trace
            self._trace_iter = Shape._trace_counters[trace]
            Shape._trace_counters[trace] += 1

        super(Shape, self).__init__()

    def __eq__(self, other):
        if isinstance(other, Shape):
            return (
                self.stroke == other.stroke
                and self.fill == other.fill
                and self.opacity == other.opacity
                and self.physics == other.physics
                and self.transform == other.transform
            )
        return False

    # def __hash__(self):
    #     return hash((self.stroke, self.fill, self.opacity, self.physics, self.transform))

    def _set_modified(self, *largs):
        self._modified = True

    def _is_modified(self, reset=True):
        if (
            self._modified
            or (self.stroke and self.stroke._modified)
            or (self.fill and self.fill._modified)
            or (self.physics and self.physics._modified)
        ):
            if reset:
                self._modified = False
                if self.stroke:
                    self.stroke._modified = False
                if self.fill:
                    self.fill._modified = False
                if self.physics:
                    self.physics._modified = False
            return True
        return False

    @property
    def size(self):
        return (0, 0)

    @SetterProperty
    def stroke(self, value):
        if isinstance(value, dict):
            value = Stroke(**value)
        elif isinstance(value, Iterable):
            value = Stroke(*value)
        elif value is not None and not isinstance(value, Stroke):
            raise AttributeError("must be Stroke, dict or tuple")
        # stroke = self.__dict__.get('stroke')
        # if stroke != value: FIXME
        self.__dict__["stroke"] = value
        self._set_modified()
        # elif stroke is None:
        # self.__dict__['stroke'] = None

    @SetterProperty
    def fill(self, value):
        if value is not None and not isinstance(value, (Color, Image)):
            value = Color(value)
        # fill = self.__dict__.get('fill')
        # if fill != value: FIXME
        self._set_modified()
        self.__dict__["fill"] = value
        # elif fill is None:
        # self.__dict__['fill'] = None

    @SetterProperty
    def opacity(self, value):
        self._set_modified()
        self.__dict__["opacity"] = float(value) if value is not None else 100.0

    @SetterProperty
    def physics(self, value):
        self._set_modified()
        if isinstance(value, dict):
            value = Physics(**value)
        elif isinstance(value, Iterable):
            value = Physics(*value)
        elif value is not None and not isinstance(value, Physics):
            raise AttributeError("'physics' attribute must be Physics, dict or tuple")
        self.__dict__["physics"] = value

    @SetterProperty
    def transform(self, value):
        if value:
            if not isinstance(value, Transform):
                value = Transform(value)
            if self.__dict__.get("transform") != value:
                self._set_modified()
                self.__dict__["transform"] = value
        else:
            self.__dict__["transform"] = None


class Line(Shape):
    def __init__(self, points=None, stroke=("white", 1.0), **kwargs):
        self.points = VectorList(points or [(0, 0), (0, 50)])
        super(Line, self).__init__(stroke=stroke, **kwargs)

    def _is_modified(self, reset=True):
        return super(Line, self)._is_modified(reset)

    def __repr__(self):
        return f"Line({self.points}, stroke={self.stroke})"

    def __eq__(self, other):
        if isinstance(other, Line):
            return self.points == other.points and super(Line, self) == super(
                Line, other
            )
        return False

    # def __hash__(self):
    #     return hash((self.points, super(Line, self).__hash__()))

    @SetterProperty
    def points(self, value):
        self._set_modified()
        self.__dict__["points"] = VectorList(value)

    @property
    def fill(self):
        return None

    @fill.setter
    def fill(self, value):
        if value is not None:
            raise NotImplementedError


class Circle(Shape):
    def __init__(
        self,
        center=(0, 0),
        radius=10,
        stroke=None,
        fill="white",
        angle_start=0,
        angle_end=360,
        physics=None,
        **kwargs,
    ):
        self._center = Vector(center)
        self.radius = radius
        self.angle_start = angle_start
        self.angle_end = angle_end
        super(Circle, self).__init__(
            stroke=stroke, fill=fill, physics=physics, **kwargs
        )

    def _is_modified(self, reset=True):
        return super(Circle, self)._is_modified(reset)

    def __repr__(self):
        return f"Circle(({self._center.x}, {self._center.y}), radius={self.radius})"

    def __eq__(self, other):
        if isinstance(other, Circle):
            return (
                self._center == other._center
                and self.radius == other.radius
                and self.angle_start == other.angle_start
                and self.angle_end == other.angle_end
                and super(Circle, self) == super(Circle, other)
            )
        return False

    # def __hash__(self):
    #     return hash((self._center, self.radius, self.angle_start,
    #         self.angle_end, super(Circle, self).__hash__()))

    @property
    def center(self):
        return self._center.ref(self._set_modified)

    @center.setter
    def center(self, value):
        self._center = Vector(value)
        self._set_modified()

    @SetterProperty
    def radius(self, value):
        self._set_modified()
        self.__dict__["radius"] = float(value)

    @property
    def size(self):
        return (2 * self.radius, 2 * self.radius)

    @size.setter
    def size(self, value):
        if isinstance(value, Iterable) and len(value) == 2:
            if value[0] == value[1]:
                self.__dict__["radius"] = float(value[0] / 2)
            else:
                raise AttributeError("circle must have equal width and height")
        else:
            raise TypeError("size must be iterable of length 2")

    @SetterProperty
    def angle_start(self, value):
        self._set_modified()
        self.__dict__["angle_start"] = float(value)

    @SetterProperty
    def angle_end(self, value):
        self._set_modified()
        self.__dict__["angle_end"] = float(value)


class Rectangle(Shape):
    def __init__(
        self,
        corner=(0, 0),
        size=(50, 50),
        radius=0,
        stroke=None,
        fill="white",
        **kwargs,
    ):
        self._corner = Vector(corner)
        if isinstance(size, Number):
            size = size, size
        self._size = Vector(size)
        self.__dict__["radius"] = float(radius)

        super(Rectangle, self).__init__(stroke=stroke, fill=fill, **kwargs)

        # self.corner = self._corner.ref()
        # self.size = self._size.ref()

    def _is_modified(self, reset=True):
        return super(Circle, self)._is_modified(reset)

    def __repr__(self):
        return f"Rectangle(({self._corner.x}, {self._corner.y}), ({self._size.x}, {self._size.y}))"

    def __eq__(self, other):
        if isinstance(other, Rectangle):
            return (
                self._corner == other._corner
                and self._size == other._size
                and self.radius == other.radius
            )
        return False

    # def __hash__(self):
    #     return hash((self._corner, self._size, self.radius, super(Rectangle, self).__hash__()))

    @classmethod
    def from_center(cls, center=(0, 0), size=(50, 50)):
        if isinstance(size, Number):
            size = size, size
        return cls(Vector(center) - Vector(size) / 2, size)

    @classmethod
    def from_corners(cls, from_corner=(0, 0), to_corner=(50, 50)):
        return cls(from_corner, Vector(to_corner) - from_corner)

    @property
    def corner(self):
        return self._corner.ref(self._set_modified)

    @corner.setter
    def corner(self, value):
        self._corner = Vector(value)
        self._set_modified()

    @property
    def size(self):
        return self._size.ref(self._set_modified)

    @size.setter
    def size(self, value):
        self._size = Vector(value)
        self._set_modified()

    def _get_center(self):
        return self._corner + self._size / 2

    def _set_center(self, center):
        self._corner = center - self._size / 2
        self._set_modified()

    center = VectorRefProperty(_get_center, _set_center)

    @SetterProperty
    def radius(self, value):
        self._set_modified()
        self.__dict__["radius"] = float(value)

    # corner = VectorRefProperty('_corner')
    # size = VectorRefProperty('_size')

    # @property
    # def width(self):
    #     return self.size.x

    # @width.setter
    # def width(self, width):
    #     self.size.x = width

    # @property
    # def height(self):
    #     return self.size.y

    # @height.setter
    # def height(self, height):
    #     self.size.y = height


class Image:
    def __init__(self, source, anim_delay=1 / 24):
        self.__dict__["source"] = source
        self.__dict__["anim_delay"] = anim_delay
        self._modified = False

    @SetterProperty
    def source(self, value):
        self._modified = True
        self.__dict__["source"] = value

    @SetterProperty
    def anim_delay(self, value):
        self._modified = True
        self.__dict__["anim_delay"] = float(value)
