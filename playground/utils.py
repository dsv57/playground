from collections import defaultdict
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
