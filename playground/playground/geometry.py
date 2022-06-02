import operator
from math import sin, cos, tan, atan2, radians, degrees, hypot
from numbers import Number
from collections.abc import Iterable


__all__ = ["Vector", "VectorRef", "VectorRefProperty", "VectorList", "Transform"]


def _as_tuple(other):
    if isinstance(other, Vector):
        x, y = other
    elif isinstance(other, Iterable):
        x = other[0]
        y = other[1]
    else:
        x = y = float(other)
    return x, y


class Vector(object):
    __slots__ = ("x", "y")

    def __init__(self, x_or_pair=None, y=None):
        if x_or_pair != None:
            if y == None:
                if isinstance(x_or_pair, Vector):
                    self.x, self.y = x_or_pair
                if hasattr(x_or_pair, "x") and hasattr(x_or_pair, "y"):
                    self.x = float(x_or_pair.x)
                    self.y = float(x_or_pair.y)
                else:
                    self.x = float(x_or_pair[0])
                    self.y = float(x_or_pair[1])
            else:
                self.x = float(x_or_pair)
                self.y = float(y)
        else:
            self.x = 0.0
            self.y = 0.0

    def __getitem__(self, i):
        if i == 0:
            return self.x
        elif i == 1:
            return self.y
        raise IndexError()

    def __setitem__(self, i, value):
        if i == 0:
            self.x = value
        elif i == 1:
            self.y = value
        else:
            raise IndexError()
        # UPDATE

    def __repr__(self):
        return "Vector(%s, %s)" % (self.x, self.y)

    def __str__(self):
        return str(tuple(self))

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2

    def __format__(self, fmt_spec=""):
        if fmt_spec.endswith("p"):
            fmt_spec = fmt_spec[:-1]
            coords = (abs(self), self.angle())
            outer_fmt = "<{}, {}>"
        else:
            coords = self
            outer_fmt = "({}, {})"
            components = (format(c, fmt_spec) for c in coords)
        return outer_fmt.format(*components)

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        if isinstance(other, Iterable) and len(other) == 2:
            x2, y2 = _as_tuple(other)
            return self.x == x2 and self.y == y2
        else:
            return False

    def __ne__(self, other):
        if isinstance(other, Iterable) and len(other) == 2:
            x2, y2 = _as_tuple(other)
            return self.x != x2 or self.y != y2
        else:
            return True

    def __nonzero__(self):
        return self.x != 0.0 or self.y != 0.0

    # Generic operator handlers
    def _o2(self, other, f):
        "Any two-operator operation where the left operand is a Vector"
        x2, y2 = _as_tuple(other)
        return Vector(f(self.x, x2), f(self.y, y2))

    def _r_o2(self, other, f):
        "Any two-operator operation where the right operand is a Vector"
        x2, y2 = _as_tuple(other)
        return Vector(f(x2, self.x), f(y2, self.y))

    def _io(self, other, f):
        "inplace operator"
        x2, y2 = _as_tuple(other)
        self.x = f(self.x, x2)
        self.y = f(self.y, y2)
        # UPDATE
        return self

    def __add__(self, other):
        x2, y2 = _as_tuple(other)
        return Vector(self.x + x2, self.y + y2)

    __radd__ = __add__

    def __iadd__(self, other):
        x2, y2 = _as_tuple(other)
        self.x += x2
        self.y += y2
        # UPDATE
        return self

    def __sub__(self, other):
        x2, y2 = _as_tuple(other)
        return Vector(self.x - x2, self.y - y2)

    def __rsub__(self, other):
        x2, y2 = _as_tuple(other)
        return Vector(x2 - self.x, y2 - self.y)

    def __isub__(self, other):
        x2, y2 = _as_tuple(other)
        self.x -= x2
        self.y -= y2
        # UPDATE
        return self

    def __mul__(self, other):
        x2, y2 = _as_tuple(other)
        return Vector(self.x * x2, self.y * y2)

    __rmul__ = __mul__

    def __imul__(self, other):
        x2, y2 = _as_tuple(other)
        self.x *= x2
        self.y *= y2
        # UPDATE
        return self

    def __div__(self, other):
        return self._o2(other, operator.div)

    def __rdiv__(self, other):
        return self._r_o2(other, operator.div)

    def __idiv__(self, other):
        return self._io(other, operator.div)

    def __floordiv__(self, other):
        return self._o2(other, operator.floordiv)

    def __rfloordiv__(self, other):
        return self._r_o2(other, operator.floordiv)

    def __ifloordiv__(self, other):
        return self._io(other, operator.floordiv)

    def __truediv__(self, other):
        return self._o2(other, operator.truediv)

    def __rtruediv__(self, other):
        return self._r_o2(other, operator.truediv)

    def __itruediv__(self, other):
        return self._io(other, operator.truediv)

    def __mod__(self, other):
        return self._o2(other, operator.mod)

    def __rmod__(self, other):
        return self._r_o2(other, operator.mod)

    def __divmod__(self, other):
        return self._o2(other, divmod)

    def __rdivmod__(self, other):
        return self._r_o2(other, divmod)

    def __pow__(self, other):
        return self._o2(other, operator.pow)

    def __rpow__(self, other):
        return self._r_o2(other, operator.pow)

    def __lshift__(self, other):
        return self._o2(other, operator.lshift)

    def __rlshift__(self, other):
        return self._r_o2(other, operator.lshift)

    def __rshift__(self, other):
        return self._o2(other, operator.rshift)

    def __rrshift__(self, other):
        return self._r_o2(other, operator.rshift)

    def __and__(self, other):
        return self._o2(other, operator.and_)

    __rand__ = __and__

    def __or__(self, other):
        return self._o2(other, operator.or_)

    __ror__ = __or__

    def _xor__(self, other):
        return self._o2(other, operator.xor)

    __rxor__ = _xor__

    def __neg__(self):
        return Vector(operator.neg(self.x), operator.neg(self.y))

    def __pos__(self):
        return Vector(operator.pos(self.x), operator.pos(self.y))

    def __abs__(self):
        return Vector(abs(self.x), abs(self.y))

    def __invert__(self):
        return Vector(-self.x, -self.y)

    def ref(self, set_hook=None):
        def fset(x, y):
            self.x = x
            self.y = y
            if set_hook is not None:
                set_hook(self, x, y)

        return VectorRef(lambda: self, fset)

    def rotate(self, angle):
        c = cos(radians(angle))
        s = sin(radians(angle))
        x = self.x * c - self.y * s
        y = self.x * s + self.y * c
        self.x = x
        self.y = y
        # UPDATE

    @staticmethod
    def polar(angle, radius=1):
        r = float(radius)
        a = float(radians(angle))
        return Vector(r * cos(a), r * sin(a))

    # @classmethod
    # def polar_degrees(cls, angle, radius=1):
    #     return cls.polar(radians(angle), radius)

    # def __eq__(self, other):
    # return tuple(self) == tuple(other)

    # def get_length_sqrd(self):
    #     return self.x * self.x + self.y * self.y

    @property
    def length(self):
        return hypot(self.x, self.y)

    @length.setter
    def length(self, value):
        length = self.get_length()
        if length != 0:
            self.x *= value / length
            self.y *= value / length
        else:
            self.x = value
        # UPDATE

    def distance(self, other):
        return hypot(self.x - other[0], self.y - other[1])

    # def get_dist_sqrd(self, other):
    #     dx = self.x - other[0]
    #     dy = self.y - other[1]
    #     return dx * dx + dy * dy

    def rotated(self, angle):
        """Create and return a new vector by rotating this vector by
        angle degrees.

        :return: Rotated vector
        """
        c = cos(radians(angle))
        s = sin(radians(angle))
        x = self.x * c - self.y * s
        y = self.x * s + self.y * c
        return Vector(x, y)

    @property
    def angle(self):
        if self.length == 0:
            return 0
        return degrees(atan2(self.y, self.x))

    @angle.setter
    def angle(self, angle):
        self.x = self.length
        self.y = 0
        self.rotate(angle)

    def angle_between(self, other):
        """Get the angle between the vector and the other in radians

        :return: The angle
        """
        x2, y2 = _as_tuple(other)
        cross = self.x * y2 - self.y * x2
        dot = self.x * x2 + self.y * y2
        return degrees(atan2(cross, dot))

    def normalized(self):
        """Get a normalized copy of the vector
        Note: This function will return 0 if the length of the vector is 0.

        :return: A normalized vector
        """
        length = self.length
        if length != 0:
            return self / length
        return Vector(self)

    def normalize(self):
        """Normalize the vector and return its length before the normalization

        :return: The length before the normalization
        """
        length = self.length
        if length != 0:
            self.x /= length
            self.y /= length
        return length

    def perpendicular(self):
        return Vector(-self.y, self.x)

    # def perpendicular_normal(self):
    #     length = self.length
    #     if length != 0:
    #         return Vector(-self.y/length, self.x/length)
    #     return Vector(self)

    def dot(self, other):
        """The dot product between the vector and other vector
            v1.dot(v2) -> v1.x*v2.x + v1.y*v2.y

        :return: The dot product
        """
        x2, y2 = _as_tuple(other)
        return float(self.x * x2 + self.y * y2)

    def projection(self, other):
        x2, y2 = _as_tuple(other)
        other_length_sqrd = x2 * x2 + y2 * y2
        projected_length_times_other_length = self.dot(other)
        return other * (projected_length_times_other_length / other_length_sqrd)

    def cross(self, other):
        """The cross product between the vector and other vector
            v1.cross(v2) -> v1.x*v2.y - v1.y*v2.x

        :return: The cross product
        """
        x2, y2 = _as_tuple(other)
        return self.x * y2 - self.y * y1

    def interpolate_to(self, other, range):
        x2, y2 = _as_tuple(other)
        return Vector(self.x + (x2 - self.x) * range, self.y + (y2 - self.y) * range)

    # def convert_to_basis(self, x_vector, y_vector):
    #     x = self.dot(x_vector)/x_vector.get_length_sqrd()
    #     y = self.dot(y_vector)/y_vector.get_length_sqrd()
    #     return Vector(x, y)

    # def __get_int_xy(self):
    #     return int(self.x), int(self.y)
    # int_tuple = property(__get_int_xy,
    #     doc="""Return the x and y values of this vector as ints""")

    # @staticmethod
    # def zero():
    #     """A vector of zero length"""
    #     return Vector(0, 0)

    # @staticmethod
    # def unit():
    #     """A unit vector pointing right"""
    #     return Vector(1, 0)

    # @staticmethod
    # def ones():
    #     """A vector where both x and y is 1"""
    #     return Vector(1, 1)

    # # Extra functions, mainly for chipmunk
    # def cpvrotate(self, other):
    #     """Uses complex multiplication to rotate this vector by the other. """
    #     return Vector(self.x*other.x - self.y*other.y, self.x*other.y + self.y*other.x)
    # def cpvunrotate(self, other):
    #     """The inverse of cpvrotate"""
    #     return Vector(self.x*other.x + self.y*other.y, self.y*other.x - self.x*other.y)

    # Pickle
    def __reduce__(self):
        callable = Vector
        args = (self.x, self.y)
        return (callable, args)


class VectorRef(Vector):
    __slots__ = ("_fget", "_fset")

    def __init__(self, fget, fset=None):
        self._fget = fget
        if not fset:

            def fset(x, y):
                raise AttributeError("can't set attribute")

        self._fset = fset

    def __get__(self, obj, objtype=None):
        return self

    @classmethod
    def from_property(cls, obj, property, readonly=False):
        fget = lambda: getattr(obj, property)
        fset = None if readonly else lambda x, y: setattr(obj, property, (x, y))
        return cls(fget, fset)

    @property
    def x(self):
        x, y = self._fget()
        return x

    @x.setter
    def x(self, x):
        self._fset(x, self.y)

    @property
    def y(self):
        x, y = self._fget()
        return y

    @y.setter
    def y(self, y):
        self._fset(self.x, y)

    def set(self, x, y):
        self._fset(x, y)

    def __repr__(self):
        x, y = self._fget()
        return "VectorRef(%s, %s)" % (x, y)

    def __format__(self, fmt_spec=""):
        return Vector(self).__format__(self, fmt_spec)

    def __iter__(self):
        x, y = self._fget()
        yield x
        yield y

    def __hash__(self):
        return hash((self._fget, self._fset))

    def __eq__(self, other):
        x, y = self._fget()
        if isinstance(other, Iterable) and len(other) == 2:
            x2, y2 = _as_tuple(other)
            return x == x2 and y == y2
        else:
            return False

    def __ne__(self, other):
        x, y = self._fget()
        if isinstance(other, Iterable) and len(other) == 2:
            x2, y2 = _as_tuple(other)
            return x != x2 or y != y2
        else:
            return True

    def __nonzero__(self):
        x, y = self._fget()
        return x != 0.0 or y != 0.0

    def copy(self):
        x, y = self._fget()
        return Vector(x, y)

    # Generic operator handlers
    def _o2(self, other, f):
        "Any two-operator operation where the left operand is a Vector"
        return Vector(self)._o2(other, f)

    def _r_o2(self, other, f):
        "Any two-operator operation where the right operand is a Vector"
        return Vector(self)._r_o2(other, f)

    def _io(self, other, f):
        "inplace operator"
        x, y = self._fget()
        x2, y2 = _as_tuple(other)
        self._fset(f(x, x2), f(y, y2))
        return self

    def __add__(self, other):
        return Vector(self) + other

    __radd__ = __add__

    def __iadd__(self, other):
        x, y = self._fget()
        x2, y2 = _as_tuple(other)
        self._fset(x + x2, y + y2)
        return self

    def __sub__(self, other):
        return Vector(self) - Vector(other)

    def __rsub__(self, other):
        return Vector(other) - Vector(self)

    def __isub__(self, other):
        x, y = self._fget()
        x2, y2 = _as_tuple(other)
        self._fset(x - x2, y - y2)
        return self

    def __mul__(self, other):
        return Vector(self) * other

    __rmul__ = __mul__

    def __imul__(self, other):
        x, y = self._fget()
        x2, y2 = _as_tuple(other)
        self._fset(x * x2, y * y2)
        return self

    def rotate(self, angle):
        c = cos(radians(angle))
        s = sin(radians(angle))
        x, y = self._fget()
        new_x = x * c - y * s
        new_y = x * s + y * c
        self._fset(new_x, new_y)

    @property
    def length(self):
        x, y = self._fget()
        return hypot(x, y)

    @length.setter
    def length(self, value):
        x, y = self._fget()
        length = hypot(x, y)
        if length != 0.0:
            self._fset(x * value / length, y2)
        else:
            self._fset(value, 0)
        # UPDATE

    def distance(self, other):
        x, y = self._fget()
        x2, y2 = _as_tuple(other)
        return hypot(x - x2, y - y2)

    def rotated(self, angle):
        """Create and return a new vector by rotating this vector by
        angle degrees.

        :return: Rotated vector
        """
        x, y = self._fget()
        c = cos(radians(angle))
        s = sin(radians(angle))
        x2 = x * c - y * s
        y2 = x * s + y * c
        return Vector(x2, y2)

    @property
    def angle(self):
        if self.length == 0:
            return 0
        x, y = self._fget()
        return degrees(atan2(y, x))

    @angle.setter
    def angle(self, angle):
        x, y = self._fget()
        length = hypot(x, y)
        x = length * cos(radians(angle))
        y = length * sin(radians(angle))
        self._fset(x, y)

    def angle_between(self, other):
        """Get the angle between the vector and the other in radians

        :return: The angle
        """
        x, y = self._fget()
        x2, y2 = _as_tuple(other)
        cross = x * y2 - y * x2
        dot = x * x2 + y * y2
        return degrees(atan2(cross, dot))

    def normalized(self):
        """Get a normalized copy of the vector
        Note: This function will return 0 if the length of the vector is 0.

        :return: A normalized vector
        """
        length = self.length
        if length != 0:
            return self / length
        return Vector(self)

    def normalize(self):
        """Normalize the vector and return its length before the normalization

        :return: The length before the normalization
        """
        x, y = self._fget()
        length = hypot(x, y)
        if length != 0:
            self /= length
        return length

    def perpendicular(self):
        return Vector(-self.y, self.x)

    # def perpendicular_normal(self):
    #     length = self.length
    #     if length != 0:
    #         return Vector(-self.y/length, self.x/length)
    #     return Vector(self)

    def dot(self, other):
        """The dot product between the vector and other vector
            v1.dot(v2) -> v1.x*v2.x + v1.y*v2.y

        :return: The dot product
        """
        x, y = self._fget()
        x2, y2 = _as_tuple(other)
        return float(x * x2 + y * y2)

    def projection(self, other):
        x2, y2 = _as_tuple(other)
        other_length_sqrd = x2 * x2 + y2 * y2
        projected_length_times_other_length = self.dot(other)
        return other * (projected_length_times_other_length / other_length_sqrd)

    def cross(self, other):
        """The cross product between the vector and other vector
            v1.cross(v2) -> v1.x*v2.y - v1.y*v2.x

        :return: The cross product
        """
        x, y = self._fget()
        x2, y2 = _as_tuple(other)
        return x * y2 - y * y1

    def interpolate_to(self, other, range):
        x, y = self._fget()
        x2, y2 = _as_tuple(other)
        return Vector(x + (x2 - x) * range, y + (y2 - y) * range)


class VectorRefProperty(object):
    def __init__(self, fget_or_vector, fset=None, doc=None):
        if isinstance(fget_or_vector, str):
            vec = fget_or_vector
            self.fget = lambda obj: getattr(obj, vec)
            self.fset = lambda obj, value: setattr(obj, vec, Vector(value))
        else:
            fget = fget_or_vector
            self.fget = fget
            self.fset = fset
            if doc is None and fget is not None:
                doc = fget.__doc__
        self.__doc__ = doc

    def __get__(self, obj, objtype=None):
        if self.fset:
            fset = lambda x, y: self.fset(obj, (x, y))
        else:
            fset = None
        return VectorRef(lambda: self.fget(obj), fset)

    def __set__(self, obj, value):
        if self.fset is None:
            raise AttributeError("can't set attribute")
        self.fset(obj, value)


from collections.abc import MutableSequence


def _flatten_point_list(points):
    points = list(points)
    if points and isinstance(points[0], Iterable):
        if not all([len(p) == 2 for p in points]):
            raise TypeError("must be list of points (pairs of numbers)")
        return [float(coord) for point in points for coord in point]
    return list(map(float, points))


def _to_point(item):
    if len(item) != 2:
        raise TypeError("point must be pair of numbers")
    return (float(item[0]), float(item[1]))


class VectorList(MutableSequence):
    """A more or less complete user-defined wrapper around list objects."""

    def __init__(self, initlist=None):
        self._data = _flatten_point_list(initlist or [])
        self._modified = False

    def _is_modified(self, reset=True):
        if self._modified:
            if reset:
                self._modified = False
            return True
        return False

    @property
    def _points(self):
        return zip(self._data[::2], self._data[1::2])

    @property
    def _pointlist(self):
        return list(zip(self._data[::2], self._data[1::2]))

    @property
    def _vectors(self):
        return list(map(Vector, self._points))

    def __repr__(self):
        return repr(self._pointlist)

    def __eq__(self, other):
        return self._data == self.__cast(other)

    def __cast(self, other):
        return other._data if isinstance(other, VectorList) else _flatten_point_list(other)

    def __contains__(self, item):
        if len(item) == 2:
            return _to_point(item) in self._points
        return False

    def __len__(self):
        return len(self._data) // 2

    def __iter__(self):
        for point in self._points:
            yield Vector(point)

    def __getitem__(self, i):
        def fget():
            x = self._data[2 * i]
            y = self._data[2 * i + 1]
            return x, y

        def fset(x, y):
            self[i] = x, y

        return VectorRef(fget, fset)

    def __setitem__(self, i, item):
        self._modified = True
        self._data[2 * i], self._data[2 * i + 1] = _to_point(item)

    def __delitem__(self, i):
        self._modified = True
        del self._data[2 * i]
        del self._data[2 * i]

    def __add__(self, other):
        if isinstance(other, VectorList):
            return self._vectors + other._vectors
        return self._vectors + list(map(Vector, other))

    def __radd__(self, other):
        if isinstance(other, VectorList):
            return other._vectors + self._vectors
        return list(map(Vector, other)) + self._vectors

    def __iadd__(self, other):
        self._modified = True
        if isinstance(other, VectorList):
            self._data += other._data
        else:
            self._data += _flatten_point_list(other)
        return self

    def __mul__(self, n):
        return self._vectors * n

    __rmul__ = __mul__

    def __imul__(self, n):
        self._modified = True
        self._data *= n
        return self

    def append(self, item):
        self._modified = True
        self._data += _to_point(item)

    def insert(self, i, item):
        self._modified = True
        self._data = self._data[: i // 2] + _to_point(item) + self._data[i // 2 :]

    def pop(self, i=-1):
        self._modified = True
        x = self._data.pop(i // 2)
        y = self._data.pop(i // 2)
        if i < 0:
            return Vector(y, x)
        return Vector(x, y)

    def remove(self, item):
        self._modified = True
        points = self._pointlist
        points.remove(_to_point(item))
        self._data = _flatten_point_list(points)

    def clear(self):
        self._modified = True
        self._data.clear()

    def copy(self):
        return self._vectors

    def count(self, item):
        return self._pointlist.count(_to_point(item))

    def index(self, item, *args):
        return self._pointlist.index(_to_point(item), *args)

    def reverse(self):
        self._modified = True
        points = self._pointlist
        points.reverse()
        self._data = _flatten_point_list(points)

    def sort(self, *args, **kwds):
        self._modified = True
        points = self._pointlist
        points.sort(*args, **kwds)
        self._data = _flatten_point_list(points)

    def extend(self, other):
        self._modified = True
        if isinstance(other, VectorList):
            self._data.extend(other._data)
        else:
            self._data.extend(_flatten_point_list(other))


class Transform(object):
    __slots__ = ("a", "b", "c", "d", "tx", "ty")

    def __init__(
        self, *largs, translate=None, rotate=None, scale=None, skew=None, anchor=(0, 0)
    ):  # a=None, b=None, c=None, d=None, tx=None, ty=None):
        if largs:
            self.a, self.b, self.c, self.d, self.tx, self.ty = largs
        else:
            self.a = 1.0
            self.b = 0.0
            self.c = 0.0
            self.d = 1.0
            self.tx = 0.0
            self.ty = 0.0
        if rotate:
            self.rotate(rotate, anchor)
        if scale:
            if isinstance(scale, Iterable) and len(scale) == 2:
                self.scale(*scale, anchor=anchor)
            else:
                self.scale(scale, anchor=anchor)
        if skew:
            if isinstance(skew, Iterable) and len(skew) == 2:
                self.skew(*skew, anchor=anchor)
            else:
                self.skew(skew, anchor=anchor)
        if translate:
            self += translate

    @property
    def matrix(self):
        return [[self.a, self.b, self.tx], [self.c, self.d, self.ty], [0.0, 0.0, 1.0]]

    @matrix.setter
    def matrix(self, matrix):
        self.a = float(matrix[0][0])
        self.b = float(matrix[0][1])
        self.c = float(matrix[1][0])
        self.d = float(matrix[1][1])
        self.tx = float(matrix[0][2])
        self.ty = float(matrix[1][2])

    def __getitem__(self, i):
        if isinstance(i, int):
            return (self.a, self.b, self.c, self.d, self.tx, self.ty)[i]
        elif isinstance(i, tuple) and len(i) == 2:
            if i[0] == 0:
                return (self.a, self.b, self.tx)[i[1]]
            elif i[0] == 1:
                return (self.c, self.d, self.ty)[i[1]]
            elif i[0] == 2:
                return (1.0, 0.0, 0.0)[i[1]]
        raise IndexError()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(a={self.a}, b={self.b}, " f"c={self.c}, d={self.d}, tx={self.tx}, ty={self.ty})"
        )

    def __iter__(self):
        yield self.a
        yield self.b
        yield self.c
        yield self.d
        yield self.tx
        yield self.ty

    def __len__(self):
        return 6

    def __hash__(self):
        return hash((self.a, self.b, self.c, self.d, self.tx, self.ty))

    def __eq__(self, other):  # FIXME: isinstance
        if isinstance(other, Iterable) and len(other) >= 2:
            return all(self[i, j] == other[i, j] for i in range(2) for j in range(3))
        return False

    def __ne__(self, other):  # FIXME: isinstance
        if isinstance(other, Iterable) and len(other) >= 2:
            return any(self[i, j] != other[i, j] for i in range(2) for j in range(3))
        return True

    def __nonzero__(self):
        return tuple(self) != 1.0, 0.0, 0.0, 1.0, 0.0, 0.0

    def __add__(self, other):
        if isinstance(other, Iterable) and len(other) == 2:
            return Transform(*self, translate=other)
        raise TypeError("unsupported operand type")

    def __iadd__(self, other):
        if isinstance(other, Iterable) and len(other) == 2:
            self.translate(*other)
            return self
        raise TypeError("unsupported operand type")

    def __sub__(self, other):
        if isinstance(other, Iterable) and len(other) == 2:
            return Transform(*self, translate=(-other[0], -other[1]))
        raise TypeError("unsupported operand type")

    def __isub__(self, other):
        if isinstance(other, Iterable) and len(other) == 2:
            tx, ty = other
            self.translate(-tx, -ty)
            return self
        raise TypeError("unsupported operand type")

    def translate(self, tx, ty):
        self.tx += float(tx)
        self.ty += float(ty)

    def rotate(self, angle, anchor=(0, 0)):
        co = cos(radians(angle))
        si = sin(radians(angle))
        a, b, c, d = co, si, -si, co
        ax, ay = anchor
        self @= (a, b, c, d, (a * ax + b * ay - ax), (c * ax + d * ay - ay))

    def scale(self, sx, sy=None, anchor=(0, 0)):
        if sy == None:
            # if isinstance(sx, Iterable) and len(sx) == 2:
            #     sx, sy = sx
            # else:
            sy = sx
        a, b, c, d = sx, 0, 0, sy
        ax, ay = anchor
        self @= a, b, c, d, a * ax + b * ay - ax, c * ax + d * ay - ay

    def skew(self, ax, ay=0, anchor=(0, 0)):
        a, b, c, d = 1, tan(radians(ay)), tan(radians(ax)), 1
        ax, ay = anchor
        self @= a, b, c, d, a * ax + b * ay - ax, c * ax + d * ay - ay

    def _reflect_unit(self, ux, uy, anchor=(0, 0)):
        a = 2.0 * ux * ux - 1.0
        b = 2.0 * ux * uy
        c = 2.0 * ux * uy
        d = 2.0 * uy * uy - 1.0
        ax, ay = anchor
        self @= a, b, c, d, a * ax + b * ay - ax, c * ax + d * ay - ay

    def reflect(self, x, y, anchor=(0, 0)):
        h = hypot(x, y)
        self._reflect_unit(x / h, y / h, anchor)

    def reflect_angle(self, angle, anchor=(0, 0)):
        self._reflect_unit(cos(radians(angle)), sin(radians(angle)), anchor)

    # reflection

    def combine(self, other, apply_before=False, anchor=(0, 0)):
        a, b, c, d, tx, ty = other
        ax, ay = anchor

        if apply_before:
            t = Transform(a, b, c, d, a * ax + b * ay + tx - ax, c * ax + d * ay + ty - ay)
            t @= self
            self.a = t.a
            self.b = t.b
            self.c = t.c
            self.d = t.d
            self.tx = t.tx
            self.ty = t.ty
        else:
            self @= a, b, c, d, a * ax + b * ay + tx - ax, c * ax + d * ay + ty - ay

    def __imatmul__(self, other):
        a, b, c, d, tx, ty = other  # FIXME
        self.a, self.b = self.a * a + self.b * c, self.a * b + self.b * d  ##1*0.5 + 0*0.866
        self.c, self.d = a * self.c + c * self.d, b * self.c + self.d * d  ##
        self.tx += self.a * tx + self.b * ty
        self.ty += self.c * tx + self.d * ty
        return self

    def __matmul__(self, other):
        trans = Transform(*self)
        trans @= other
        return trans

    @classmethod
    def identity(cls):
        """The identity transform"""
        return cls(1, 0, 0, 1, 0, 0)
