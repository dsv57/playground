#! /usr/bin/python3
"""
Copyright (C) 2005 Aaron Spike, aaron@ekips.org

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""

# import traceback

from math import pi, sin, cos, atan2, degrees, radians
from turtle import TurtleGraphicsError  # Vec2d

from pymunk.vec2d import Vec2d

from named_colors import COLORS


class Turtle:
    _instances = set()

    def __init__(self):
        self.reset()
        self._register()

    def reset(self):
        self._shown = True
        self._drawing = True
        self._pencolor = (0.4, 0.4, 1)  # COLORS['black']
        self._pensize = 1
        self._shapesize = 1

        self._position = Vec2d(0, 0)
        self._lines = []  # [(self._pencolor, self._pensize, [])]
        self._stamps = []
        self._orient = Vec2d(1, 0)
        self._z = 0

        self._new_line()

    def _new_line(self):
        if self._lines and len(self._lines[-1][2]) < 3:
            self._lines.pop()
        self._lines.append((self._pencolor, self._pensize, list(self._position)))

    def dot(self, r=15, color=None):
        if not color:
            color = self._pencolor
        else:
            color = self._color(color)
        self._stamps.append(("Ellipse", self._position, (r, r), color))

    def rect(self, w=25, h=25, color=None):
        if not color:
            color = self._pencolor
        else:
            color = self._color(color)
        self._stamps.append(("Rectangle", self._position, (float(w), float(h)), color))

    def shapesize(self, size):
        self._shapesize = float(size)

    def forward(self, distance):
        self._go(distance)

    def backward(self, distance):
        self._go(-distance)

    def right(self, deg):
        self._rotate(-deg)

    def left(self, deg):
        self._rotate(deg)

    def penup(self):
        self._drawing = False
        if self._lines and len(self._lines[-1][2]) == 2:
            self._lines.pop()

    #        self._new = False

    def pendown(self):
        if not self._drawing:
            self._new_line()
            self._drawing = True

    def pentoggle(self):
        if self._drawing:
            self.penup()
        else:
            self.pendown()

    def pensize(self, width=None):
        if width:
            prev_size = self._pensize
            self._pensize = int(width)
            if self._drawing is True and self._pensize != prev_size:
                self._new_line()
        else:
            return self._pensize

    def color(self, *args):
        prev_c = self._pencolor
        self._pencolor = self._color(*args)

        if self._drawing is True and self._pencolor != prev_c:
            self._new_line()

    def home(self):
        self.goto(0, 0)
        self._orient = Vec2d(1, 0)

    #    def clean(self):
    #        self._path = ''

    def clear(self):
        self.clean()
        self.home()

    def heading(self):
        # x, y = self._orient
        # return round(atan2(y, x)*180.0/pi, 10) % 360.0
        return self._orient.get_angle_degrees()

    def goto(self, x, y=None):
        if y is None and isinstance(x, (list, tuple)) and len(x) == 2:
            self._goto(Vec2d(*map(float, x)))
        else:
            self._goto(Vec2d(float(x), float(y)))

    #        if self.__new:
    #            self.__path += "M"+",".join([str(i) for i in self.__pos])
    #            self.__new = False
    #        self.__pos = [x, y]
    #        if self.__draw:
    #            self.__path += "L"+",".join([str(i) for i in self.__pos])

    def position(self):
        return self._position

    def setheading(self, deg):
        # self._orient = Vec2d(1, 0)
        # self._rotate(deg)
        self._orient.angle_degrees = deg

    def zlevel(self, z=None):
        if z:
            self._z = z
        else:
            return self._z

    @staticmethod
    def turtles():
        return sorted(Turtle._instances, key=lambda t: t._z)

    @staticmethod
    def clear_turtles():
        Turtle._instances = set()

    # fd = forward
    # bk = backward
    # rt = right
    # lt = left
    # pu = penup
    # pd = pendown
    # setpos = goto
    # position = pos

    #
    # Private
    #
    def _register(self):
        Turtle._instances.add(self)

    def _unregister(self):
        if self in Turtle._instances:
            Turtle._instances.remove(self)

    def _color(self, *color):
        if len(color) == 1 and isinstance(color[0], (tuple, list)):
            color = color[0]

        if len(color) in [3, 4] and all([isinstance(c, (float, int)) for c in color]):
            return color
        elif len(color) == 1 and isinstance(color[0], str):
            color = color[0]
            if color.startswith("#"):
                if len(color) == 7:
                    return [int(color[i : i + 2], 16) / 255 for i in (1, 3, 5)]
                elif len(color) == 4:
                    return [16 * int(h, 16) / 255 for h in color[1:]]
            elif color in COLORS:
                return COLORS[color]

        raise TurtleGraphicsError("bad colorstring: %s" % color)

    def _rotate(self, angle):
        self._orient.rotate_degrees(angle)

    def _go(self, distance):
        ende = self._position + self._orient * distance
        self._goto(ende)

    def _goto(self, end):
        if self._drawing:
            self._lines[-1][2].extend(list(end))
        self._position = end


# vim: expandtab shiftwidth=4 tabstop=4 softtabstop=4 textwidth=99
