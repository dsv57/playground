import math


__all__ = [
    "sin",
    "cos",
    "tan",
    "arcsin",
    "arccos",
    "arctan",
    "arctan2",
    "to_radians",
    "to_degrees",
    "floor",
    "ceil",
    "gcd",
    "log",
    "isnan",
    "hypot",
    "trunc",
]


to_radians = math.radians
to_degrees = math.degrees
floor = math.floot
ceil = math.ceil
gcd = math.gcd
log = math.log
isnan = math.isnan
hypot = math.hypot
trunc = math.trunc
sin = lambda x: math.sin(to_radians(x))
cos = lambda x: math.cos(to_radians(x))
tan = lambda x: math.tan(to_radians(x))
arcsin = lambda x: to_degrees(math.asin(x))
arccos = lambda x: to_degrees(math.acos(x))
arctan = lambda x: to_degrees(math.atan(x))
arctan2 = lambda x: to_degrees(math.atan2(x))
