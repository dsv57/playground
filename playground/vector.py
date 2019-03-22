import operator
from math import sin, cos, radians, degrees, atan2
from numbers import Number
from collections.abc import Iterable


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
    __slots__ = ('x', 'y')

    def __init__(self, x_or_pair=None, y=None):
        if x_or_pair != None:
            if y == None:
                if isinstance(x_or_pair, Vector):
                    self.x, self.y = x_or_pair
                if hasattr(x_or_pair, 'x') and hasattr(x_or_pair, 'y'):
                    self.x = float(x_or_pair.x)
                    self.y = float(x_or_pair.y)
                else:
                    self.x = float(x_or_pair[0])
                    self.y = float(x_or_pair[1])
            else:
                self.x = float(x_or_pair)
                self.y = float(y)
        else:
            self.x = 0.
            self.y = 0.

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
        return 'Vector(%s, %s)' % (self.x, self.y)

    def __str__(self):
        return str(tuple(self))

    def __iter__(self):
        yield self.x
        yield self.y

    def __len__(self):
        return 2

    def __format__(self, fmt_spec=''):
        if fmt_spec.endswith('p'):
            fmt_spec = fmt_spec[:-1]
            coords = (abs(self), self.angle())
            outer_fmt = '<{}, {}>'
        else:
            coords = self
            outer_fmt = '({}, {})'
            components = (format(c, fmt_spec) for c in coords)
        return outer_fmt.format(*components)

    def __hash__(self):
        return hash(self.x) ^ hash(self.y)

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
        return Vector(f(self.x, x2),
                      f(self.y, y2))
 
    def _r_o2(self, other, f):
        "Any two-operator operation where the right operand is a Vector"
        x2, y2 = _as_tuple(other)
        return Vector(f(x2, self.x),
                      f(y2, self.y))

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
        if (self.length == 0):
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
    
    @staticmethod
    def zero():
        """A vector of zero length"""
        return Vector(0, 0)
        
    @staticmethod
    def unit():
        """A unit vector pointing up"""
        return Vector(0, 1)
        
    @staticmethod
    def ones():
        """A vector where both x and y is 1"""
        return Vector(1, 1)
 
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
    __slots__ = ('_fget', '_fset')

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
        return 'VectorRef(%s, %s)' % (x, y)

    def __format__(self, fmt_spec=''):
        return Vector(self).__format__(self, fmt_spec)

    def __iter__(self):
        x, y = self._fget()
        yield x
        yield y

    def __hash__(self):
        x, y = self._fget()
        return hash(x) ^ hash(y)

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
        length = self.length
        self._fset(length * c, length * s)

    def rotated(self, angle):
        v = Vector(self)
        v.rotate(angle)
        return v

    @property
    def length(self):
        x, y = self._fget()
        return hypot(x, y)

    @length.setter
    def length(self, value):
        x, y = self._fget()
        length = hypot(x, y)
        if length != 0.:
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
        if (self.length == 0):
            return 0
        x, y = self._fget()
        return degrees(atan2(y, x))

    @angle.setter
    def angle(self, angle):
        x, y = self._fget()
        length = hypot(x,y)
        self.x = self.length
        self.y = 0
        self.rotate(angle)

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




