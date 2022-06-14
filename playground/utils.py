from collections import defaultdict  # , OurList
from collections.abc import MutableSequence
from itertools import chain
from weakref import WeakValueDictionary


class KeepRefs(object):
    __refs__ = defaultdict(list)

    def __init__(self):
        self.__refs__[self.__class__].append(self)

    def __del__(self):
        if self in self.__refs__[self.__class__]:
            self.__refs__[self.__class__].remove(self)

    @classmethod
    def get_instances(cls, subclasses=False):
        refs = cls.__refs__
        if subclasses:
            return chain.from_iterable(refs[c] for c in refs if issubclass(c, cls))
        return refs[cls]

    @classmethod
    def _clear_instances(cls):
        cls.__refs__.clear()


class KeepWeakRefs(object):
    __refs__ = defaultdict(WeakValueDictionary)

    def __init__(self):
        self.__refs__[self.__class__][id(self)] = self

    @classmethod
    def get_instances(cls, subclasses=False):
        refs = cls.__refs__
        if subclasses:
            return chain.from_iterable(refs[c].values() for c in refs if issubclass(c, cls))
        return refs[cls].values()


class SetterProperty(object):
    def __init__(self, func, doc=None):
        self.func = func
        self.__doc__ = doc if doc is not None else func.__doc__

    def __set__(self, obj, value):
        return self.func(obj, value)


class OurList(MutableSequence):
    """A more or less complete user-defined wrapper around list objects."""

    def __init__(self, initlist=None):
        self._data = []
        if initlist is not None:
            # XXX should this accept an arbitrary sequence?
            if type(initlist) == type(self._data):
                self._data[:] = initlist
            elif isinstance(initlist, OurList):
                self._data[:] = initlist._data[:]
            else:
                self._data = list(initlist)
        self._modified = False

    def __repr__(self):
        return repr(self._data)

    def __lt__(self, other):
        return self._data < self.__cast(other)

    def __le__(self, other):
        return self._data <= self.__cast(other)

    def __eq__(self, other):
        return self._data == self.__cast(other)

    def __gt__(self, other):
        return self._data > self.__cast(other)

    def __ge__(self, other):
        return self._data >= self.__cast(other)

    def __cast(self, other):
        return other._data if isinstance(other, OurList) else other

    def __contains__(self, item):
        return item in self._data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, i):
        return self._data[i]

    def __setitem__(self, i, item):
        self._modified = True
        self._data[i] = item

    def __delitem__(self, i):
        self._modified = True
        del self._data[i]

    def __add__(self, other):
        if isinstance(other, OurList):
            return self._data + other._data
        elif isinstance(other, type(self._data)):
            return self._data + other
        return self._data + list(other)

    def __radd__(self, other):
        if isinstance(other, OurList):
            return other._data + self._data
        elif isinstance(other, type(self._data)):
            return other + self._data
        return list(other) + self._data

    def __iadd__(self, other):
        self._modified = True
        if isinstance(other, OurList):
            self._data += other._data
        elif isinstance(other, type(self._data)):
            self._data += other
        else:
            self._data += list(other)
        return self

    def __mul__(self, n):
        return self._data * n

    __rmul__ = __mul__

    def __imul__(self, n):
        self._modified = True
        self._data *= n
        return self

    def append(self, item):
        self._modified = True
        self._data.append(item)

    def insert(self, i, item):
        self._modified = True
        self._data.insert(i, item)

    def pop(self, i=-1):
        self._modified = True
        return self._data.pop(i)

    def remove(self, item):
        self._modified = True
        self._data.remove(item)

    def clear(self):
        self._modified = True
        self._data.clear()

    def copy(self):
        return list(self._data)

    def count(self, item):
        return self._data.count(item)

    def index(self, item, *args):
        return self._data.index(item, *args)

    def reverse(self):
        self._modified = True
        self._data.reverse()

    def sort(self, *args, **kwds):
        self._modified = True
        self._data.sort(*args, **kwds)

    def extend(self, other):
        self._modified = True
        if isinstance(other, OurList):
            self._data.extend(other._data)
        else:
            self._data.extend(other)


# from os.path import join, dirname, exists, abspath
# from sys import argv

# resource_paths = ['.', dirname(argv[0])]

# def resource_find(filename):
#     '''Search for a resource in the list of paths.
#     '''
#     if not filename:
#         return
#     if filename[:8] == 'atlas://':
#         return filename
#     if exists(abspath(filename)):
#         return abspath(filename)
#     for path in reversed(resource_paths):
#         output = abspath(join(path, filename))
#         if exists(output):
#             return output
#     if filename[:5] == 'data:':
#         return filename
